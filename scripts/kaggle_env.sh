# Source before using the Kaggle CLI:  source scripts/kaggle_env.sh
# The cloud environment stores the account as KAGGLE_ACCOUNT and the (classic) API key as
# KAGGLE_API_TOKEN; the Kaggle CLI reads them as KAGGLE_USERNAME / KAGGLE_KEY.
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-$KAGGLE_ACCOUNT}"
export KAGGLE_KEY="${KAGGLE_KEY:-$KAGGLE_API_TOKEN}"
