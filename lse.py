#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "scikit-learn>=1.3.0",
#   "numpy>=1.24.0",
#   "datasets>=2.0.0",
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

def is_simple_emoji(emoji: str) -> bool:
    """
    Check if an emoji is simple (single codepoint, no modifiers/ZWJ sequences).
    Returns False for compound emojis that might render as multiple characters.
    """
    # Zero Width Joiner - used in compound emojis
    if '\u200d' in emoji:
        return False

    # Skin tone modifiers (U+1F3FB to U+1F3FF)
    skin_tones = ['\U0001F3FB', '\U0001F3FC', '\U0001F3FD', '\U0001F3FE', '\U0001F3FF']
    if any(tone in emoji for tone in skin_tones):
        return False

    # Variation selectors (U+FE00 to U+FE0F)
    if any('\ufe00' <= c <= '\ufe0f' for c in emoji):
        return False

    # Keycap combining character
    if '\u20e3' in emoji:
        return False

    # Regional indicator symbols (flag emojis)
    if any('\U0001F1E6' <= c <= '\U0001F1FF' for c in emoji):
        return False

    # Check if it's more than one actual character (excluding variation selectors)
    # Simple heuristic: if len > 2, it's likely compound
    if len(emoji) > 2:
        return False

    return True

def download_full_dataset(sample_size: int = 150) -> bool:
    """
    Download emoji dataset from HuggingFace and save a random sample.
    Filters out compound emojis that might render incorrectly.
    Returns True if successful, False otherwise.
    """
    import random
    import shutil

    print(f"🌐 Downloading emoji dataset from HuggingFace...")
    print(f"   (This may take a moment on first run)")

    try:
        from datasets import load_dataset

        # Load dataset
        dataset = load_dataset("badrex/LLM-generated-emoji-descriptions")
        total = len(dataset['train'])
        print(f"✅ Downloaded {total} emojis")

        # Convert to our format
        emojis_data = []
        seen_emojis = set()
        skipped_compound = 0

        for row in dataset['train']:
            if not isinstance(row, dict):
                continue

            emoji = row.get('character', '').strip()
            description = row.get('LLM description', '').strip()

            # Skip if empty or duplicate
            if not emoji or not description or emoji in seen_emojis:
                continue

            # Skip compound emojis
            if not is_simple_emoji(emoji):
                skipped_compound += 1
                continue

            seen_emojis.add(emoji)
            emojis_data.append({
                "emoji": emoji,
                "description": description
            })

        print(f"🔍 Filtered out {skipped_compound} compound emojis (skin tones, ZWJ sequences, flags)")
        print(f"📝 {len(emojis_data)} simple emojis available")

        # Take random sample
        if len(emojis_data) > sample_size:
            print(f"🎲 Selecting random sample of {sample_size} emojis...")
            emojis_data = random.sample(emojis_data, sample_size)

        # Backup existing emoji_data.json if it exists
        if EMOJI_DATA_PATH.exists():
            backup_path = MODEL_DIR / "emoji_data.json.backup"
            shutil.copy(EMOJI_DATA_PATH, backup_path)
            print(f"📦 Backed up existing emoji data to {backup_path.name}")

        # Save to ~/.lse/emoji_data.json
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(EMOJI_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(emojis_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(emojis_data)} emojis to {EMOJI_DATA_PATH}")
        return True

    except Exception as e:
        print(f"❌ Failed to download dataset: {e}", file=sys.stderr)
        print(f"   Falling back to default emoji set", file=sys.stderr)
        return False

def reset_emojis():
    """
    Reset emoji data to the default 20-emoji set from the repo.
    """
    import shutil

    print("🔄 Resetting to default emoji set...")

    # Check if fallback exists
    if not EMOJI_DATA_FALLBACK.exists():
        print(f"❌ Error: Default emoji_data.json not found at {EMOJI_DATA_FALLBACK}", file=sys.stderr)
        sys.exit(1)

    # Backup current if it exists
    if EMOJI_DATA_PATH.exists():
        backup_path = MODEL_DIR / "emoji_data.json.backup"
        shutil.copy(EMOJI_DATA_PATH, backup_path)
        print(f"📦 Backed up current emoji data to {backup_path.name}")

    # Copy default to ~/.lse/
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(EMOJI_DATA_FALLBACK, EMOJI_DATA_PATH)

    print(f"✅ Reset to default 20 emojis")
    print(f"💡 Run 'lse --index <path>' to retrain with the default set")

# ============================================================================
# Training Mode
# ============================================================================

def train_mode(index_path: Path, use_full_dataset: bool = False, sample_size: int = 150):
    """
    Index a directory tree and train the TF-IDF model.
    """
    # Download full dataset if requested
    if use_full_dataset:
        success = download_full_dataset(sample_size)
        if not success:
            print("⚠️  Continuing with existing emoji dataset...")

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
  lse --index ~/projects                    Index your projects directory
  lse --index ~/projects --full-dataset     Use extended emoji dataset (150 random)
  lse --index ~/projects --full-dataset --sample-size 300  Larger sample
  lse --reset-emojis                        Reset to default 20 emojis
  lse                                       List current directory with emojis
  lse /path/to/dir                          List specific directory with emojis
        """
    )

    parser.add_argument(
        '--index',
        type=Path,
        metavar='PATH',
        help='Index a directory tree and train the model'
    )

    parser.add_argument(
        '--full-dataset',
        action='store_true',
        help='Download and use extended emoji dataset from HuggingFace (use with --index)'
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        default=150,
        metavar='N',
        help='Number of random emojis to sample from full dataset (default: 150)'
    )

    parser.add_argument(
        '--reset-emojis',
        action='store_true',
        help='Reset to default emoji set (20 emojis)'
    )

    parser.add_argument(
        'path',
        nargs='?',
        type=Path,
        default=Path.cwd(),
        help='Directory to list (default: current directory)'
    )

    args = parser.parse_args()

    if args.reset_emojis:
        reset_emojis()
    elif args.index:
        train_mode(args.index, args.full_dataset, args.sample_size)
    else:
        inference_mode(args.path)

if __name__ == "__main__":
    main()
