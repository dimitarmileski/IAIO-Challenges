import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
