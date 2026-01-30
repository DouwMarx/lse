**What we're building:**
A `uv` Python script that shows emojis next to filenames based on semantic similarity.

**Two modes:**

1. **`lse --index <path>`** (one-time setup)
   - Recursively scan directory tree
   - Read all files, fit TF-IDF vectorizer on entire corpus
   - Load emoji dataset (HuggingFace), compute emoji TF-IDF vectors using same vectorizer
   - Save: `vectorizer.pkl`, `emoji_vectors.npy`, `emoji_list.json`

2. **`lse [path]`** (daily use)
   - Load saved vectorizer + emoji vectors
   - List files in current/specified directory
   - Transform each file using saved vectorizer
   - Cosine similarity against emoji vectors
   - Print: `{emoji} {filename}`


**Output format:**
```
🐍 train.py
📊 results.csv
📝 README.md
```

**Dependencies:**
sklearn, numpy


## Detailed Implementation Instructions

### Step 1: Set Up Project Structure

```bash
mkdir lse
cd lse
touch lse.py
chmod +x lse.py
```

### Step 2: Hardcode a starter set

Create `emoji_data.json`:
```json
[
  {"emoji": "🐍", "description": "python programming language code script interpreter"},
  {"emoji": "📊", "description": "data visualization chart graph statistics analytics"},
  {"emoji": "🧠", "description": "neural network machine learning artificial intelligence model"},
  {"emoji": "📝", "description": "text document markdown notes writing readme"},
  {"emoji": "🎨", "description": "design graphics css style image visual art"},
  {"emoji": "⚙️", "description": "configuration settings yaml json toml config"},
  {"emoji": "🔧", "description": "build tool makefile cmake compiler build system"},
  {"emoji": "📦", "description": "package dependency library module npm pip"},
  {"emoji": "🗄️", "description": "database sql storage data persistence"},
  {"emoji": "🌐", "description": "web html http api server network"},
  {"emoji": "🔐", "description": "security encryption key certificate authentication"},
  {"emoji": "📋", "description": "log output trace debug information"},
  {"emoji": "🧪", "description": "test testing unit integration pytest"},
  {"emoji": "📚", "description": "documentation guide tutorial reference manual"},
  {"emoji": "🚀", "description": "deployment docker kubernetes container orchestration"},
  {"emoji": "💾", "description": "binary executable compiled application program"},
  {"emoji": "🔍", "description": "search query index elasticsearch"},
  {"emoji": "⚡", "description": "performance optimization fast speed efficient"},
  {"emoji": "🐛", "description": "bug error exception debugging fix"},
  {"emoji": "📁", "description": "directory folder"}
]
```

### Step 3: Write the Main Script

Create `lse.py`:

```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "scikit-learn>=1.3.0",
#   "numpy>=1.24.0",
# ]
# ///

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# Configuration
# ============================================================================

MODEL_DIR = Path.home() / ".lse"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
EMOJI_VECS_PATH = MODEL_DIR / "emoji_vectors.npy"
EMOJI_LIST_PATH = MODEL_DIR / "emoji_list.json"
EMOJI_DATA_PATH = Path(__file__).parent / "emoji_data.json"

# ============================================================================
# File Reading Utilities
# ============================================================================

def read_file_safe(filepath: Path, max_bytes: int = 8192) -> str:
    """
    Safely read file content, handling binary files and errors.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_bytes)
        return content
    except (PermissionError, IsADirectoryError, FileNotFoundError):
        return ""

def collect_files_recursive(root_path: Path) -> List[Tuple[Path, str]]:
    """
    Recursively collect all files and their contents from root_path.
    Returns list of (filepath, content) tuples.
    """
    files = []
    for entry in root_path.rglob('*'):
        if entry.is_file():
            content = read_file_safe(entry)
            if content.strip():  # Only include non-empty files
                files.append((entry, content))
    return files

# ============================================================================
# Emoji Loading
# ============================================================================

def load_emoji_data() -> List[Dict[str, str]]:
    """
    Load emoji dataset from JSON file.
    """
    if not EMOJI_DATA_PATH.exists():
        print(f"Error: Emoji data not found at {EMOJI_DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    
    with open(EMOJI_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# ============================================================================
# Training Mode
# ============================================================================

def train_mode(index_path: Path):
    """
    Index a directory tree and train the TF-IDF model.
    """
    print(f"🔍 Indexing files in: {index_path}")
    
    # Collect all files
    files = collect_files_recursive(index_path)
    if not files:
        print("No files found to index.", file=sys.stderr)
        sys.exit(1)
    
    print(f"📚 Found {len(files)} files")
    
    # Extract file contents
    file_contents = [content for _, content in files]
    
    # Load emoji data
    emoji_data = load_emoji_data()
    emoji_descriptions = [e['description'] for e in emoji_data]
    emoji_symbols = [e['emoji'] for e in emoji_data]
    
    print(f"😀 Loaded {len(emoji_data)} emojis")
    
    # Fit TF-IDF on combined corpus
    print("🔨 Training TF-IDF vectorizer...")
    all_texts = file_contents + emoji_descriptions
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    all_vectors = vectorizer.fit_transform(all_texts)
    
    # Split vectors
    emoji_vectors = all_vectors[len(file_contents):].toarray()
    
    # Save model artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving model to {MODEL_DIR}")
    
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    np.save(EMOJI_VECS_PATH, emoji_vectors)
    
    with open(EMOJI_LIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(emoji_symbols, f, ensure_ascii=False)
    
    print("✅ Training complete!")

# ============================================================================
# Inference Mode
# ============================================================================

def load_model() -> Tuple[TfidfVectorizer, np.ndarray, List[str]]:
    """
    Load trained model artifacts.
    """
    if not VECTORIZER_PATH.exists():
        print("❌ Model not found. Run 'lse --index <path>' first.", file=sys.stderr)
        sys.exit(1)
    
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    emoji_vectors = np.load(EMOJI_VECS_PATH)
    
    with open(EMOJI_LIST_PATH, 'r', encoding='utf-8') as f:
        emoji_list = json.load(f)
    
    return vectorizer, emoji_vectors, emoji_list

def get_emoji_for_file(
    content: str,
    filename: str,
    vectorizer: TfidfVectorizer,
    emoji_vectors: np.ndarray,
    emoji_list: List[str]
) -> str:
    """
    Get the best matching emoji for a file.
    """
    # If file is too small/empty, use filename
    text = content if len(content) > 50 else filename * 5
    
    # Transform using trained vectorizer
    file_vector = vectorizer.transform([text]).toarray()
    
    # Compute cosine similarity
    similarities = cosine_similarity(file_vector, emoji_vectors)[0]
    
    # Get best match
    best_idx = similarities.argmax()
    
    return emoji_list[best_idx]

def inference_mode(target_path: Path):
    """
    List directory with emojis.
    """
    # Load model
    vectorizer, emoji_vectors, emoji_list = load_model()
    
    # List directory contents
    if not target_path.exists():
        print(f"Error: Path {target_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    entries = sorted(target_path.iterdir())
    
    for entry in entries:
        if entry.is_dir():
            print(f"📁  {entry.name}/")
        else:
            content = read_file_safe(entry)
            emoji = get_emoji_for_file(
                content,
                entry.name,
                vectorizer,
                emoji_vectors,
                emoji_list
            )
            print(f"{emoji}  {entry.name}")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="lse - ls with semantic emojis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lse --index ~/projects     Index your projects directory
  lse                        List current directory with emojis
  lse /path/to/dir           List specific directory with emojis
        """
    )
    
    parser.add_argument(
        '--index',
        type=Path,
        metavar='PATH',
        help='Index a directory tree and train the model'
    )
    
    parser.add_argument(
        'path',
        nargs='?',
        type=Path,
        default=Path.cwd(),
        help='Directory to list (default: current directory)'
    )
    
    args = parser.parse_args()
    
    if args.index:
        train_mode(args.index)
    else:
        inference_mode(args.path)

if __name__ == "__main__":
    main()
```

### Step 4: Test the Implementation

**Train the model:**
```bash
./lse.py --index ~/projects
```

Expected output:
```
🔍 Indexing files in: /home/user/projects
📚 Found 247 files
😀 Loaded 20 emojis
🔨 Training TF-IDF vectorizer...
💾 Saving model to /home/user/.lse
✅ Training complete!
```

**Run inference:**
```bash
./lse.py
```

Expected output:
```
📁  src/
🐍  train.py
📊  results.csv
📝  README.md
⚙️  config.yaml
```

### Step 5: Install Globally (Optional)

```bash
# Create symlink in PATH
mkdir -p ~/.local/bin
ln -s $(pwd)/lse.py ~/.local/bin/lse

# Add to PATH if not already (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Test
lse ~
```

### Step 6: Expand Emoji Dataset (Optional)

Download full dataset from HuggingFace:

```python
# If you want to programmatically download:
# /// script
# dependencies = ["datasets"]
# ///

from datasets import load_dataset

dataset = load_dataset("badrex/LLM-generated-emoji-descriptions")
emojis = []

for row in dataset['train']:
    emojis.append({
        'emoji': row['emoji'],
        'description': row['description']
    })

with open('emoji_data.json', 'w') as f:
    json.dump(emojis, f, ensure_ascii=False, indent=2)
```

Then re-run `lse --index`.

### Step 7: Handle Edge Cases

**Add these improvements if needed:**

1. **Ignore patterns:**
```python
# Add to collect_files_recursive
IGNORE_PATTERNS = {'.git', '__pycache__', 'node_modules', '.venv'}

if any(ignored in entry.parts for ignored in IGNORE_PATTERNS):
    continue
```

2. **Progress bar for indexing:**
```python
# Add dependency: tqdm
from tqdm import tqdm

for entry in tqdm(root_path.rglob('*'), desc="Scanning"):
    # ...
```

3. **Cache file hashes to detect changes:**
```python
# Save file hashes during indexing
# Only re-index if files changed
```

### Troubleshooting

**Issue: "Model not found"**
- Run `lse --index <path>` first

**Issue: All files get same emoji**
- Your corpus is too homogeneous
- Index a more diverse directory tree
- Add more varied emoji descriptions

**Issue: Wrong emojis**
- Tune `max_features`, `min_df`, `max_df` in TfidfVectorizer
- Improve emoji descriptions to be more specific
- Use larger `max_bytes` when reading files

**Issue: Slow performance**
- Reduce `max_features` to 1000
- Index smaller directory tree
- Add file type filters
