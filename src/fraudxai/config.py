"""Central configuration so the notebook, tests and Streamlit app share one set of settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
OPENML_ID = 1597
OPENML_PARQUET_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
DATA_FILE = DATA_DIR / "creditcard.parquet"

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time"] + V_COLS + ["Amount", "Class"]
TARGET = "Class"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
MAX_TIME_S = 172_800          # the dataset covers two days (48 h) of September 2013

SEEDS = [0, 1, 2, 3, 4]       # >= 5 seeds (UREC2)
WARMUP_FRACTION = 0.20        # early, time-ordered slice used only for tuning
THRESHOLD = 0.5               # decision threshold used for G-mean
PK_BLOCK = 10_000             # alert budget: k alerts per block of transactions
PK_K = 10                     # k = 10 per 10,000 (0.1 % budget, below the 0.172 % fraud rate)
