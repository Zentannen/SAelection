#!/usr/bin/env bash
# Push kaggle/ notebook as a Kaggle kernel. Reads KAGGLE_ACCOUNT / KAGGLE_API_TOKEN (environment
# secrets) and maps them to the names the Kaggle CLI expects.
set -euo pipefail
cd "$(dirname "$0")/.."
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-${KAGGLE_ACCOUNT:?set KAGGLE_ACCOUNT}}"
export KAGGLE_KEY="${KAGGLE_KEY:-${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}}"
pip install -q kaggle
sed -i "s#\"id\": \"[^/]*/#\"id\": \"${KAGGLE_USERNAME}/#" kaggle/kernel-metadata.json
kaggle kernels push -p kaggle
echo "status: kaggle kernels status ${KAGGLE_USERNAME}/saelection-gauteng-2026"
