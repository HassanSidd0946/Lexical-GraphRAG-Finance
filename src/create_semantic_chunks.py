import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# 1. Initialize the token counter
tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    return len(tokenizer.encode(text))

# 2. Configure the Semantic Splitter (800 token limit for text-embedding-3-small)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 3. Target files and metadata mapping
target_files = [
    "sec_data/JPM_Item1A.txt",
    "sec_data/JPM_Item7.txt",
    "sec_data/PYPL_Item1A.txt",
    "sec_data/PYPL_Item7.txt"
]

# Hardcoding URLs and dates for the sake of the academic dataset
# (In a production system, edgartools would pass these down dynamically)
metadata_lookup = {
    "JPM": {"date": "2026-02-15", "url": "https://www.sec.gov/edgar/browse/?CIK=19617"},
    "PYPL": {"date": "2026-02-10", "url": "https://www.sec.gov/edgar/browse/?CIK=1633918"}
}

output_file = "sec_semantic_chunks_master.jsonl"
total_chunks_saved = 0
filtered_chunks = 0

print(f"Starting semantic chunking. Outputting to {output_file}...\n")

# Open the JSONL file in append mode
with open(output_file, "w", encoding="utf-8") as out_file:
    
    for filename in target_files:
        if os.path.exists(filename):
            print(f"Processing {filename}...")
            
            with open(filename, "r", encoding="utf-8") as f:
                raw_text = f.read()
                
            chunks = text_splitter.split_text(raw_text)
            
            base_name = os.path.basename(filename)
            ticker = base_name.split("_")[0]
            section = base_name.split("_")[1].replace(".txt", "")
            
            for i, chunk_text in enumerate(chunks):
                # Calculate tokens exactly
                tokens = tiktoken_len(chunk_text)
                
                # THE FILTER: Skip chunks smaller than 50 tokens
                if tokens < 50:
                    filtered_chunks += 1
                    continue
                
                # Construct the enhanced metadata payload
                chunk_data = {
                    "chunk_id": f"{ticker}_{section}_{i:04d}",
                    "ticker": ticker,
                    "section": section,
                    "filing_date": metadata_lookup[ticker]["date"],
                    "source_url": metadata_lookup[ticker]["url"],
                    "chunk_index": i,
                    "token_count": tokens,
                    "text": chunk_text
                }
                
                # Write as a single line in the JSONL file
                out_file.write(json.dumps(chunk_data) + "\n")
                total_chunks_saved += 1
                
        else:
            print(f"  -> Warning: {filename} not found.")

print("\n--- Chunking Summary ---")
print(f"Total valid chunks saved: {total_chunks_saved}")
print(f"Small chunks filtered out (< 50 tokens): {filtered_chunks}")
print(f"Data is securely stored in {output_file} and ready for Phase 2!")