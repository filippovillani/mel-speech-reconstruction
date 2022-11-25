from pathlib import Path
import os

MAIN_DIR = Path(__file__).parent
DATA_DIR = MAIN_DIR / "data"
RESULTS_DIR = MAIN_DIR / "results"
AUDIO_IN_PATH = DATA_DIR / 'in.wav'

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)