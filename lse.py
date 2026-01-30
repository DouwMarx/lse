#!/usr/bin/env -S uv run --script
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
EMOJI_DATA_PATH = MODEL_DIR / "emoji_data.json"
EMOJI_DATA_FALLBACK = Path(__file__).resolve().parent / "emoji_data.json"

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
    Checks ~/.lse/ first, falls back to script directory, and copies if needed.
    """
    import shutil

    # Try primary location first
    if EMOJI_DATA_PATH.exists():
        with open(EMOJI_DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Try fallback location (script directory)
    if EMOJI_DATA_FALLBACK.exists():
        # Copy to ~/.lse/ for next time
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(EMOJI_DATA_FALLBACK, EMOJI_DATA_PATH)
        print(f"📋 Copied emoji_data.json to {MODEL_DIR}")

        with open(EMOJI_DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Not found anywhere
    print(f"Error: emoji_data.json not found!", file=sys.stderr)
    print(f"Expected at: {EMOJI_DATA_PATH}", file=sys.stderr)
    print(f"Or at: {EMOJI_DATA_FALLBACK}", file=sys.stderr)
    sys.exit(1)

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
        min_df=2, # ignore rare terms
        max_df=0.95, # ignore very common terms
        stop_words='english',
        ngram_range=(1, 2) # unigrams and bigrams (like cat sat, and not just cat)
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
