# Fraud Detection marimo notebook

This repository contains a marimo notebook (`notebooks/fraud_detection.py`) for exploring the Credit Card Fraud Detection dataset and experimenting with models. The instructions below explain how to set up dependencies with [`uv`](https://docs.astral.sh/uv/), add the dataset locally, and launch the notebook.

## Prerequisites
- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

If you do not have `uv` installed, follow the official installation guide linked above.

## Install dependencies
`uv` uses the `pyproject.toml` in this repository to resolve and install dependencies into a virtual environment. Run:

```bash
uv sync
```

`uv sync` will create a virtual environment (in `.venv` by default) and install all required packages, including `marimo` and the scientific Python stack.

## Add the dataset
The notebook expects a file named `creditcard.csv` at `data/creditcard.csv`.

1. Create the data directory:
   ```bash
   mkdir -p data
   ```
2. Download `creditcard.csv` from the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (or your preferred mirror).
3. Place the downloaded `creditcard.csv` inside the `data` directory so the path is `data/creditcard.csv`.

### Why the dataset is not in the repo
The dataset is ~150 MB and subject to distribution terms on Kaggle. Committing it would bloat the repository, slow down clones, and potentially violate licensing/terms of use. Keeping it out of version control avoids these issues while letting you supply the file locally.

## Run the marimo notebook
After dependencies are installed and the dataset is in place, start the marimo app with `uv run`:

```bash
uv run marimo edit notebooks/fraud_detection.py
```

`marimo edit` starts an interactive UI in your browser. The app will load `data/creditcard.csv` via `fraud_detection.data.loader.read_data`; if the file is missing, you will see a file-not-found error. You can also run the notebook non-interactively (e.g., to render outputs) with:

```bash
uv run marimo run notebooks/fraud_detection.py
```

Stop the server with `Ctrl+C` when finished.
