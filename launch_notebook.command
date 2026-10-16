#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec uv run --locked jupyter lab tumor_growth_explorer.ipynb
