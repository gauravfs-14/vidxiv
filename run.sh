#!/bin/sh
# Start vidxiv and open it in the browser.
cd "$(dirname "$0")" && exec uv run vidxiv serve "$@"
