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
CONFIG_PATH = MODEL_DIR / "config.json"
CONFIG_FALLBACK = Path(__file__).resolve().parent / "config.json"

# Default configuration
DEFAULT_CONFIG = {
    "display": {
        "top_k": 1,
        "separator": ""
    },
    "tfidf": {
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.95,
        "stop_words": "english",
        "ngram_range": [1, 2]
    },
    "dataset": {
        "default_sample_size": 150
    }
}

# ============================================================================
# Configuration Management
# ============================================================================

def load_config() -> dict:
    """
    Load configuration from ~/.lse/config.json.
    Falls back to default if not found.
    """
    import shutil

    # Try primary location
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge with defaults to handle missing keys
                for section, values in DEFAULT_CONFIG.items():
                    if section not in config:
                        config[section] = values
                    else:
                        for key, value in values.items():
                            if key not in config[section]:
                                config[section][key] = value
                return config
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}. Using defaults.", file=sys.stderr)
            return DEFAULT_CONFIG.copy()

    # Try fallback location (repo)
    if CONFIG_FALLBACK.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(CONFIG_FALLBACK, CONFIG_PATH)
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Use defaults
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    return DEFAULT_CONFIG.copy()

def save_config(config: dict):
    """Save configuration to ~/.lse/config.json"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def show_config():
    """Display current configuration"""
    config = load_config()
    print("📝 Current configuration:")
    print(f"   Location: {CONFIG_PATH}")
    print()
    print(json.dumps(config, indent=2))
    print()
    print("💡 Edit with: nano ~/.lse/config.json")
    print("💡 Reset with: lse --reset-config")

def reset_config():
    """Reset configuration to defaults"""
    import shutil

    print("🔄 Resetting configuration to defaults...")

    # Backup current if it exists
    if CONFIG_PATH.exists():
        backup_path = MODEL_DIR / "config.json.backup"
        shutil.copy(CONFIG_PATH, backup_path)
        print(f"📦 Backed up current config to {backup_path.name}")

    # Save defaults
    save_config(DEFAULT_CONFIG)
    print(f"✅ Reset to default configuration")

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

    # Load configuration
    config = load_config()
    tfidf_config = config['tfidf']

    # Fit TF-IDF on combined corpus
    print("🔨 Training TF-IDF vectorizer...")
    print(f"   Config: max_features={tfidf_config['max_features']}, "
          f"min_df={tfidf_config['min_df']}, max_df={tfidf_config['max_df']}")
    all_texts = file_contents + emoji_descriptions

    vectorizer = TfidfVectorizer(
        max_features=tfidf_config['max_features'],
        min_df=tfidf_config['min_df'],
        max_df=tfidf_config['max_df'],
        stop_words=tfidf_config['stop_words'],
        ngram_range=tuple(tfidf_config['ngram_range'])
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

def get_emojis_for_file(
    content: str,
    filename: str,
    vectorizer: TfidfVectorizer,
    emoji_vectors: np.ndarray,
    emoji_list: List[str],
    top_k: int = 1
) -> List[str]:
    """
    Get the top-k best matching emojis for a file.
    """
    # If file is too small/empty, use filename
    text = content if len(content) > 50 else filename * 5

    # Transform using trained vectorizer
    file_vector = vectorizer.transform([text]).toarray()

    # Compute cosine similarity
    similarities = cosine_similarity(file_vector, emoji_vectors)[0]

    # Get top k matches
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [emoji_list[idx] for idx in top_indices]

def inference_mode(target_path: Path, top_k: int = None):
    """
    List directory with emojis.
    """
    # Load configuration
    config = load_config()
    if top_k is None:
        top_k = config['display']['top_k']
    separator = config['display']['separator']

    # Load model
    vectorizer, emoji_vectors, emoji_list = load_model()

    # List directory contents
    if not target_path.exists():
        print(f"Error: Path {target_path} does not exist", file=sys.stderr)
        sys.exit(1)

    entries = sorted(target_path.iterdir())

    # Calculate padding: assume each emoji is ~2 chars wide
    # Add space for separator between emojis
    emoji_width = top_k * 2 + len(separator) * (top_k - 1)
    padding = emoji_width + 2  # +2 for the two spaces after emojis

    for entry in entries:
        if entry.is_dir():
            # Pad directory emoji to match file emoji width
            print(f"{'📁':<{padding}} {entry.name}/")
        else:
            content = read_file_safe(entry)
            emojis = get_emojis_for_file(
                content,
                entry.name,
                vectorizer,
                emoji_vectors,
                emoji_list,
                top_k
            )
            emoji_str = separator.join(emojis)
            # Pad emoji string to consistent width
            print(f"{emoji_str:<{padding}} {entry.name}")

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
  lse --top-k 3                             Show top 3 emojis per file
  lse --show-config                         Display current configuration
  lse --reset-config                        Reset configuration to defaults
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
        '--top-k',
        type=int,
        metavar='K',
        help='Show top K emojis per file (default: from config, usually 1)'
    )

    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Display current configuration'
    )

    parser.add_argument(
        '--reset-config',
        action='store_true',
        help='Reset configuration to defaults'
    )

    parser.add_argument(
        'path',
        nargs='?',
        type=Path,
        default=Path.cwd(),
        help='Directory to list (default: current directory)'
    )

    args = parser.parse_args()

    if args.show_config:
        show_config()
    elif args.reset_config:
        reset_config()
    elif args.reset_emojis:
        reset_emojis()
    elif args.index:
        train_mode(args.index, args.full_dataset, args.sample_size)
    else:
        inference_mode(args.path, args.top_k)

if __name__ == "__main__":
    main()
