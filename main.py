import argparse
from pathlib import Path
import scipy.signal
import soundfile as sf

import config
from griffinlim import griffin_lim_base

def main(args):
    out_path = config.DATA_DIR / (args.out_name + '.wav')
    with open(config.AUDIO_IN_PATH, 'rb') as f:
        audio, _ = sf.read(f)
    
    _, _, spectrogram = scipy.signal.stft(audio)
    griffin_lim_base(spectrogram, out_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_name',
                        type=str,
                        help='name to give to the output',
                        default='out')
    args = parser.parse_args()
    main(args)