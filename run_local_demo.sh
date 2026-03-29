#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}"
  echo "Create it with:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements-dev.txt"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

export LOCALIZATION_WEIGHTS_PATH="${ROOT_DIR}/localization_dpn_unet_dpn92_0_best_dice"
export DAMAGE_WEIGHTS_PATH="${ROOT_DIR}/pseudo_dpn_seamese_unet_shared_dpn92_0_best_xview"

cd "${ROOT_DIR}"
exec streamlit run app.py
