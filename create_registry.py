import os
from pathlib import Path
import json
from datetime import datetime
import hashlib

def create_initial_registry(docs_dir: str = "docs", registry_file: str = "faiss_index_docs_registry.json"):
    """Create initial registry for existing markdown files"""
    base_path = Path(docs_dir).resolve()
    registry = {}
    
    # Recursively find all markdown files
    for md_file in base_path.rglob("*.md"):
        # Get relative path from docs directory
        relative_path = str(md_file.relative_to(base_path)).replace('/', '\\')  # Use Windows-style paths
        
        # Calculate file hash using SHA-256
        mtime = os.path.getmtime(md_file)
        with open(md_file, 'rb') as f:
            content = f.read()
        content_hash = hashlib.sha256(content).hexdigest()
        doc_hash = f"{content_hash}_{mtime}"
        
        # Add to registry
        registry[relative_path] = {
            'hash': doc_hash,
            'last_processed': datetime.now().isoformat()
        }
        print(f"Added {relative_path} to registry")
    
    # Save registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"\nRegistry saved with {len(registry)} entries to {registry_file}")

if __name__ == "__main__":
    create_initial_registry() 