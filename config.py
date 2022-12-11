from pathlib import Path
import os

MAIN_DIR = Path(__file__).parent
DATA_DIR = MAIN_DIR / "data"
RESULTS_DIR = MAIN_DIR / "results"
GLA_RESULTS_DIR = RESULTS_DIR / "gla"
WINDOWS_IMG_DIR = RESULTS_DIR / "windows"
MELSPEC2SPEC_DIR = RESULTS_DIR / 'melspec2spec'

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
    
if not os.path.exists(WINDOWS_IMG_DIR):
    os.mkdir(WINDOWS_IMG_DIR)  
    
if not os.path.exists(GLA_RESULTS_DIR):
    os.mkdir(GLA_RESULTS_DIR)
    
if not os.path.exists(MELSPEC2SPEC_DIR):
    os.mkdir(MELSPEC2SPEC_DIR)