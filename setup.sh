#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Setting up lse..."

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LSE_PATH="$SCRIPT_DIR/lse.py"

# Check if lse.py exists
if [ ! -f "$LSE_PATH" ]; then
    echo "❌ Error: lse.py not found in $SCRIPT_DIR"
    exit 1
fi

# Make executable
chmod +x "$LSE_PATH"
echo "✅ Made lse.py executable"

# Create ~/.local/bin if it doesn't exist
mkdir -p "$HOME/.local/bin"
echo "✅ Created ~/.local/bin directory"

# Create symlink
ln -sf "$LSE_PATH" "$HOME/.local/bin/lse"
echo "✅ Created symlink: ~/.local/bin/lse -> $LSE_PATH"

# Copy emoji_data.json to ~/.lse/
EMOJI_DATA="$SCRIPT_DIR/emoji_data.json"
if [ -f "$EMOJI_DATA" ]; then
    mkdir -p "$HOME/.lse"
    cp "$EMOJI_DATA" "$HOME/.lse/emoji_data.json"
    echo "✅ Copied emoji_data.json to ~/.lse/"
else
    echo -e "${YELLOW}⚠️  emoji_data.json not found in $SCRIPT_DIR${NC}"
fi

# Copy config.json to ~/.lse/
CONFIG_FILE="$SCRIPT_DIR/config.json"
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$HOME/.lse/config.json"
    echo "✅ Copied config.json to ~/.lse/"
else
    echo -e "${YELLOW}⚠️  config.json not found in $SCRIPT_DIR${NC}"
fi

# Detect shell and add to PATH if needed
add_to_path() {
    local config_file=$1
    local path_line=$2

    if [ -f "$config_file" ]; then
        if ! grep -q '.local/bin' "$config_file" 2>/dev/null; then
            echo "" >> "$config_file"
            echo "# Added by lse setup" >> "$config_file"
            echo "$path_line" >> "$config_file"
            echo -e "${GREEN}✅ Added to PATH in $config_file${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  PATH already configured in $config_file${NC}"
            return 1
        fi
    fi
    return 1
}

# Track if we added to any config
ADDED_TO_CONFIG=false

# Try to add to shell configs
if [ -n "$BASH_VERSION" ] || [ -n "$ZSH_VERSION" ]; then
    # Bash or Zsh
    for config in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.zprofile"; do
        if add_to_path "$config" 'export PATH="$HOME/.local/bin:$PATH"'; then
            ADDED_TO_CONFIG=true
        fi
    done
fi

# Check for fish shell config
if [ -d "$HOME/.config/fish" ]; then
    FISH_CONFIG="$HOME/.config/fish/config.fish"
    if add_to_path "$FISH_CONFIG" 'set -gx PATH $HOME/.local/bin $PATH'; then
        ADDED_TO_CONFIG=true
    fi
fi

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""

if [ "$ADDED_TO_CONFIG" = true ]; then
    echo "To use lse immediately in this shell, run:"
    echo "  source ~/.bashrc    # or ~/.zshrc or ~/.config/fish/config.fish"
    echo ""
    echo "Or simply open a new terminal."
else
    echo "⚠️  Could not automatically add to PATH."
    echo "Please manually add this line to your shell config:"
    echo ""
    echo "  For bash/zsh (~/.bashrc or ~/.zshrc):"
    echo '    export PATH="$HOME/.local/bin:$PATH"'
    echo ""
    echo "  For fish (~/.config/fish/config.fish):"
    echo '    set -gx PATH $HOME/.local/bin $PATH'
fi

echo ""
echo "Next steps:"
echo "  1. Reload your shell (or open a new terminal)"
echo "  2. Index your files: lse --index ~/projects"
echo "  3. Try it out: lse"
