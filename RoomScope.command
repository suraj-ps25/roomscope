#!/bin/bash
# Double-click in Finder: sets roomscope up the first time, then opens its page in the browser.
cd "$(dirname "$0")" || exit 1
if [[ ! -x .venv/bin/roomscope ]]; then
  echo "First run: installing roomscope (about 5 minutes, once)."
  scripts/setup.sh || { echo "Setup failed; see the messages above."; read -r -p "Press Return to close."; exit 1; }
fi
exec .venv/bin/roomscope serve --open
