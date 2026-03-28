"""
SEC 10-K Filing Extractor  v2.0
================================
Fetches, cleans, chunks, and catalogs sections from the latest 10-K filings.

New in v2:
  1. Rate limiting     – token-bucket throttle between SEC requests
  2. Metadata tracking – filing date, accession number, source URL saved to JSON
  3. Deduplication     – skips files that were already extracted in a prior run
  4. Text normalization– unicode fixes, whitespace collapsing, legal boilerplate trim
  5. Document chunking – each section split into fixed-size chunks for NLP/embedding
  6. Structured catalog– sec_data/catalog.csv with ticker, section, date, words, path
  7. Error categories  – NETWORK / MISSING_SECTION / PARSE / UNKNOWN errors labelled

Usage:
    python sec_fetcher_v2.py

Dependencies:
    pip install edgartools
"""

from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from edgar import set_identity, Company

# ══════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════

IDENTITY      = "Muhammad Adil Usmani (L1F22BSCS0399@ucp.edu.pk)"
OUTPUT_DIR    = Path("sec_data")
MAX_WORKERS   = 4          # parallel ticker threads
RETRY_LIMIT   = 3          # retries per failure
RETRY_DELAY   = 2.0        # seconds between retries

# ── Rate limiter (token bucket) ──────────────────────────────
RATE_LIMIT_RPS   = 5       # max requests per second to SEC EDGAR
RATE_LIMIT_BURST = 10      # burst capacity

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE    = 1_000      # words per chunk
CHUNK_OVERLAP = 100        # overlap between consecutive chunks

TICKERS = ["JPM", "PYPL"]

TARGET_SECTIONS: dict[str, str] = {
    "Item 1A": "Risk_Factors",
    "Item 7":  "MDA",
}

CATALOG_FILE  = OUTPUT_DIR / "catalog.csv"
CATALOG_COLS  = [
    "ticker", "section_key", "section_label",
    "filing_date", "accession_number", "source_url",
    "word_count", "chunk_count", "skipped",
    "error_category", "error_detail",
    "text_file", "chunks_dir", "metadata_file",
    "extracted_at",
]


# ══════════════════════════════════════════════════════════════
# 1. LOGGING
# ══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 2. RATE LIMITER  (token-bucket, thread-safe)
# ══════════════════════════════════════════════════════════════

class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate      = rate          # tokens added per second
        self._burst     = float(burst)
        self._tokens    = float(burst)
        self._last_time = time.monotonic()
        self._lock      = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_time
            self._last_time = now
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)

            if self._tokens >= 1:
                self._tokens -= 1
                return

            wait = (1 - self._tokens) / self._rate

        time.sleep(wait)
        with self._lock:
            self._tokens = max(0, self._tokens - 1)


_limiter = RateLimiter(rate=RATE_LIMIT_RPS, burst=RATE_LIMIT_BURST)


# ══════════════════════════════════════════════════════════════
# 3. ERROR TAXONOMY
# ══════════════════════════════════════════════════════════════

class ErrorCategory(str, Enum):
    NETWORK         = "NETWORK"           # connection / HTTP / timeout
    MISSING_SECTION = "MISSING_SECTION"   # section absent in filing
    PARSE           = "PARSE"             # HTML/text parsing failure
    NO_FILING       = "NO_FILING"         # company has no 10-K at all
    UNKNOWN         = "UNKNOWN"           # catch-all

def _categorize(exc: Exception) -> ErrorCategory:
    msg = str(exc).lower()
    if any(k in msg for k in ("connection", "timeout", "http", "ssl",
                               "network", "requests", "urlopen", "remote")):
        return ErrorCategory.NETWORK
    if any(k in msg for k in ("parse", "decode", "html", "beautifulsoup",
                               "attribute", "index", "key")):
        return ErrorCategory.PARSE
    return ErrorCategory.UNKNOWN


# ══════════════════════════════════════════════════════════════
# 4. DATA MODELS
# ══════════════════════════════════════════════════════════════

@dataclass
class FilingMeta:
    ticker:           str
    section_key:      str
    section_label:    str
    filing_date:      str  = ""
    accession_number: str  = ""
    source_url:       str  = ""
    word_count:       int  = 0
    chunk_count:      int  = 0
    skipped:          bool = False
    error_category:   str  = ""
    error_detail:     str  = ""
    text_file:        str  = ""
    chunks_dir:       str  = ""
    metadata_file:    str  = ""
    extracted_at:     str  = ""


@dataclass
class Summary:
    results: list[FilingMeta] = field(default_factory=list)

    def record(self, meta: FilingMeta) -> None:
        self.results.append(meta)

    @property
    def succeeded(self) -> list[FilingMeta]:
        return [r for r in self.results if not r.error_category and not r.skipped]

    @property
    def skipped(self) -> list[FilingMeta]:
        return [r for r in self.results if r.skipped]

    @property
    def failed(self) -> list[FilingMeta]:
        return [r for r in self.results if r.error_category]

    def print_report(self) -> None:
        log.info("═" * 60)
        log.info("FINAL REPORT")
        log.info("  ✓ Extracted : %d", len(self.succeeded))
        log.info("  ⊘ Skipped   : %d  (already on disk)", len(self.skipped))
        log.info("  ✗ Failed    : %d", len(self.failed))

        if self.failed:
            log.warning("\n  Failure details:")
            for r in self.failed:
                log.warning("    [%s | %s]  %-18s %s",
                             r.ticker, r.section_key,
                             r.error_category, r.error_detail)
        log.info("═" * 60)


# ══════════════════════════════════════════════════════════════
# 5. TEXT UTILITIES
# ══════════════════════════════════════════════════════════════

# Common legal boilerplate phrases that pad filings without NLP value
_BOILERPLATE_PATTERNS = [
    r"table of contents",
    r"see notes? to (the )?consolidated financial statements",
    r"incorporated herein by reference",
    r"forward[- ]looking statements?",
    r"this page intentionally left blank",
]
_BOILERPLATE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _BOILERPLATE_PATTERNS),
    flags=re.IGNORECASE,
)

def normalize_text(raw: str) -> str:
    """
    1. NFC-normalize unicode  (fixes smart quotes, em-dashes, etc.)
    2. Replace common non-ASCII characters with ASCII equivalents
    3. Collapse runs of whitespace / blank lines
    4. Strip leading/trailing whitespace on every line
    5. Remove common legal boilerplate lines
    """
    # 1. NFC unicode normalization
    text = unicodedata.normalize("NFC", raw)

    # 2. Common substitutions
    replacements = {
        "\u2019": "'",  "\u2018": "'",
        "\u201c": '"',  "\u201d": '"',
        "\u2013": "-",  "\u2014": "-",
        "\u00a0": " ",  "\u2022": "*",
        "\u00ae": "(R)", "\u2122": "(TM)",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # 3 & 4. Strip per-line whitespace then collapse multiple blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned_lines: list[str] = []
    blank_run = 0
    for ln in lines:
        if not ln:
            blank_run += 1
            if blank_run <= 1:          # allow at most one blank separator
                cleaned_lines.append("")
        else:
            blank_run = 0
            # 5. Drop pure boilerplate lines
            if not _BOILERPLATE_RE.fullmatch(ln.lower().strip(".")):
                cleaned_lines.append(ln)

    return "\n".join(cleaned_lines).strip()


def chunk_text(text: str,
               chunk_size: int   = CHUNK_SIZE,
               overlap: int      = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into word-based chunks with a sliding overlap window.
    Returns a list of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap   # slide forward by (size - overlap)

    return chunks


# ══════════════════════════════════════════════════════════════
# 6. DEDUPLICATION HELPERS
# ══════════════════════════════════════════════════════════════

def _already_extracted(text_file: Path, meta_file: Path) -> bool:
    """Return True if both the text file and its metadata JSON exist."""
    return text_file.exists() and meta_file.exists()


def sanitize(name: str) -> str:
    return name.replace(" ", "").replace("/", "-")


# ══════════════════════════════════════════════════════════════
# 7. CORE FETCHING LOGIC
# ══════════════════════════════════════════════════════════════

def fetch_ticker(ticker: str, output_dir: Path) -> list[FilingMeta]:
    """Fetch all target sections for one ticker, with retries."""
    results: list[FilingMeta] = []

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            _limiter.acquire()   # ← rate limiting applied here
            log.info("[%s] Fetching 10-K (attempt %d/%d)…",
                     ticker, attempt, RETRY_LIMIT)

            company = Company(ticker)

            _limiter.acquire()
            filings = company.get_filings(form="10-K").latest(1)

            filing = None
            if filings:
                if isinstance(filings, (list, tuple)):
                    filing = filings[0] if filings else None
                else:
                    try:
                        filing = filings[0]
                    except Exception:
                        filing = filings

            if not filing:
                log.warning("[%s] No 10-K filing found.", ticker)
                for sk, sl in TARGET_SECTIONS.items():
                    results.append(FilingMeta(
                        ticker=ticker, section_key=sk, section_label=sl,
                        error_category=ErrorCategory.NO_FILING,
                        error_detail="Company has no 10-K on EDGAR",
                    ))
                return results

            filing_date = str(getattr(filing, "filing_date", ""))
            accession   = str(getattr(filing, "accession_number", ""))
            source_url  = str(getattr(filing, "filing_url",
                              getattr(filing, "url", "")))

            log.info("[%s] Got 10-K  date=%s  accession=%s",
                     ticker, filing_date, accession)

            _limiter.acquire()
            ten_k = filing.obj()

            for section_key, section_label in TARGET_SECTIONS.items():
                meta = _extract_section(
                    ten_k=ten_k,
                    ticker=ticker,
                    section_key=section_key,
                    section_label=section_label,
                    filing_date=filing_date,
                    accession=accession,
                    source_url=source_url,
                    output_dir=output_dir,
                )
                results.append(meta)

            return results          # done – skip remaining retry attempts

        except Exception as exc:
            category = _categorize(exc)
            log.warning("[%s] Attempt %d/%d  [%s]  %s",
                        ticker, attempt, RETRY_LIMIT, category.value, exc)
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)   # exponential back-off

    # All retries exhausted
    for sk, sl in TARGET_SECTIONS.items():
        results.append(FilingMeta(
            ticker=ticker, section_key=sk, section_label=sl,
            error_category=ErrorCategory.NETWORK,
            error_detail=f"All {RETRY_LIMIT} attempts failed",
        ))
    return results


def _extract_section(
    ten_k,
    ticker:        str,
    section_key:   str,
    section_label: str,
    filing_date:   str,
    accession:     str,
    source_url:    str,
    output_dir:    Path,
) -> FilingMeta:
    """Extract, normalise, chunk, and persist one section."""

    meta = FilingMeta(
        ticker=ticker,
        section_key=section_key,
        section_label=section_label,
        filing_date=filing_date,
        accession_number=accession,
        source_url=source_url,
        extracted_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    base_name   = f"{ticker}_{sanitize(section_key)}"
    text_file   = output_dir / f"{base_name}.txt"
    chunks_dir  = output_dir / f"{base_name}_chunks"
    meta_file   = output_dir / f"{base_name}_meta.json"

    # ── 3. Deduplication ────────────────────────────────────
    if _already_extracted(text_file, meta_file):
        log.info("[%s] '%s' already on disk — skipping.", ticker, section_key)
        meta.skipped = True
        meta.text_file      = str(text_file)
        meta.chunks_dir     = str(chunks_dir)
        meta.metadata_file  = str(meta_file)
        return meta

    # ── Extract raw text ────────────────────────────────────
    try:
        raw: Optional[str] = ten_k[section_key]
    except Exception as exc:
        meta.error_category = ErrorCategory.PARSE
        meta.error_detail   = f"Indexing error: {exc}"
        log.error("[%s] Parse error on '%s': %s", ticker, section_key, exc)
        return meta

    if not raw or not raw.strip():
        meta.error_category = ErrorCategory.MISSING_SECTION
        meta.error_detail   = "Section returned empty or None"
        log.warning("[%s] '%s' is empty/missing.", ticker, section_key)
        return meta

    # ── 4. Text normalisation ────────────────────────────────
    text = normalize_text(raw)

    # ── 6. Save clean text ───────────────────────────────────
    text_file.write_text(text, encoding="utf-8")
    meta.word_count = len(text.split())
    meta.text_file  = str(text_file)

    # ── 5. Chunking ──────────────────────────────────────────
    chunks = chunk_text(text)
    meta.chunk_count = len(chunks)
    meta.chunks_dir  = str(chunks_dir)

    chunks_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks, 1):
        chunk_file = chunks_dir / f"chunk_{i:04d}.txt"
        chunk_file.write_text(chunk, encoding="utf-8")

    # ── 2. Metadata JSON ─────────────────────────────────────
    meta.metadata_file = str(meta_file)
    meta_file.write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info("[%s] '%s' → %d words, %d chunks  [%s]",
             ticker, section_key, meta.word_count, meta.chunk_count,
             text_file.name)
    return meta


# ══════════════════════════════════════════════════════════════
# 8. CATALOG (CSV)
# ══════════════════════════════════════════════════════════════

def _append_to_catalog(results: list[FilingMeta]) -> None:
    """Append results to the master catalog CSV (creates header if new)."""
    write_header = not CATALOG_FILE.exists()
    with CATALOG_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CATALOG_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for meta in results:
            writer.writerow(asdict(meta))
    log.info("Catalog updated → %s", CATALOG_FILE)


# ══════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Attach file log handler now that the directory is ready
    fh = logging.FileHandler(OUTPUT_DIR / "fetch.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)

    set_identity(IDENTITY)
    log.info("SEC 10-K Fetcher v2  |  %d tickers  |  %d sections  |  "
             "rate=%.1f rps  |  chunk=%d words",
             len(TICKERS), len(TARGET_SECTIONS), RATE_LIMIT_RPS, CHUNK_SIZE)

    summary    = Summary()
    all_metas: list[FilingMeta] = []

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(TICKERS))) as pool:
        futures = {
            pool.submit(fetch_ticker, ticker, OUTPUT_DIR): ticker
            for ticker in TICKERS
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                metas = future.result()
                for m in metas:
                    summary.record(m)
                all_metas.extend(metas)
            except Exception as exc:
                log.error("[%s] Unhandled exception: %s", ticker, exc)

    # ── Write catalog ────────────────────────────────────────
    _append_to_catalog(all_metas)

    summary.print_report()


if __name__ == "__main__":
    main()