# lse - ls with semantic emojis 🎨

A Python script that enhances directory listings by automatically adding contextual emojis to filenames based on their content using TF-IDF and cosine similarity.

## Screenshots

**Basic usage:**

![lse example](images/example.png)

**Top-k mode (showing multiple emojis per file):**

![lse top-k example](images/topk.png)

## How It Works

`lse` uses TF-IDF vectorization + cosine similarity to match file contents with emoji descriptions, giving visual cues about what each file contains at a glance.

## Prerequisites

You need [uv](https://github.com/astral-sh/uv) - a fast Python package installer and runner.

**Install uv:**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew
brew install uv
```

Verify installation:
```bash
uv --version
```

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/DouwMarx/lse.git
cd lse
./setup.sh
```

Then reload your shell or open a new terminal.

### 2. Index Your Files (Required First Step)

Train the model on your codebase to learn what kinds of files you work with:

```bash
./lse.py --index ~/projects
```

This creates a trained model in `~/.lse/` based on your actual files.

**Tips:**
- Index a diverse directory tree (like your entire projects folder) for best results
- The more varied your files, the better the emoji matching
- Re-run indexing anytime to update the model

**Want more emoji variety?** Use the extended dataset (1,381 simple emojis):

```bash
lse --index ~/projects --full-dataset
```

<details>
<summary>Extended dataset details</summary>

By default, `lse` uses 20 curated emojis optimized for code. The extended dataset downloads a random sample from HuggingFace (5,000+ emojis, filtered to 1,381 simple ones):

```bash
# Default: 150 random emojis
lse --index ~/projects --full-dataset

# Larger sample
lse --index ~/projects --full-dataset --sample-size 300

# Reset to default 20 emojis
lse --reset-emojis
lse --index ~/projects
```

**Extended dataset:** More diverse, higher entropy, less repetition
**Default dataset:** Predictable, semantic, focused on code

</details>

## Usage

```bash
# List current directory with emojis
lse

# List specific directory
lse /path/to/directory

# Show top 3 emojis per file
lse --top-k 3

# Re-index/train on different directory
lse --index ~/new-projects

# Use extended emoji dataset (150 random emojis from 5000+)
lse --index ~/projects --full-dataset

# Customize sample size
lse --index ~/projects --full-dataset --sample-size 300

# Reset to default 20 emojis
lse --reset-emojis

# Configuration management
lse --show-config      # View current config
lse --reset-config     # Reset config to defaults
```

All data is stored in `~/.lse/` (config, emoji descriptions, trained model)

## Configuration

`lse` stores settings in `~/.lse/config.json`:

```bash
lse --show-config      # View config
nano ~/.lse/config.json # Edit config
lse --reset-config     # Reset to defaults
```

**Common settings:**

```bash
# Show multiple emojis per file
lse --top-k 3

# Or set permanently in config
nano ~/.lse/config.json  # Change "top_k": 3, "separator": "|"
```

<details>
<summary>All configuration options</summary>

```json
{
  "display": {
    "top_k": 1,           // Number of emojis per file
    "separator": ""       // String between emojis (try "|" or " ")
  },
  "tfidf": {
    "max_features": 5000, // TF-IDF vocabulary size
    "min_df": 2,          // Ignore rare terms (< N docs)
    "max_df": 0.95,       // Ignore common terms (> N% docs)
    "stop_words": "english",
    "ngram_range": [1, 2] // Unigrams and bigrams
  },
  "dataset": {
    "default_sample_size": 150 // Default for --full-dataset
  }
}
```

**Examples:**
- `top_k=1`: `🐍    main.py`
- `top_k=3`: `🐍📝🔧      main.py`
- `top_k=2, separator="|"`: `🐍|📝     main.py`

</details>

**Shell alias (optional):**

```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.config/fish/config.fish
alias ls='lse'
```

**Customize emojis:**

```bash
# Edit emoji descriptions
nano ~/.lse/emoji_data.json

# Add your own (e.g., {"emoji": "🦀", "description": "rust..."})
# Then re-index
lse --index ~/projects
```

## Troubleshooting

**"Model not found" error**
- Run `lse --index <path>` first to train the model

**All files get the same emoji**
- Index a more diverse directory with varied file types
- Add more specific emoji descriptions to `~/.lse/emoji_data.json`

**Wrong emoji assignments**
- Index a larger corpus: `lse --index ~/projects`
- Improve emoji descriptions to be more distinctive
- The model learns from your actual files, so index representative code

**Slow performance**
- Reduce `max_features` in [lse.py:107](lse.py#L107) from 5000 to 1000
- Index a smaller directory tree

## How the ML Works

1. **Training (`--index`):**
   - Recursively reads all files in the indexed directory
   - Fits a TF-IDF vectorizer on the combined corpus (files + emoji descriptions)
   - Computes emoji vectors using the same vectorizer
   - Saves model artifacts to `~/.lse/`

2. **Inference (default):**
   - Loads trained vectorizer and emoji vectors
   - Transforms each file's content into a TF-IDF vector
   - Computes cosine similarity between file and all emoji vectors
   - Assigns the emoji with highest similarity score

**Key parameters:**
- `max_features=5000`: Vocabulary size
- `min_df=2`: Ignore rare terms (appear in < 2 docs)
- `max_df=0.95`: Ignore very common terms (appear in > 95% of docs)
- `ngram_range=(1,2)`: Use both unigrams and bigrams

## License

MIT
