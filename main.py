import argparse
from pathlib import Path
import scipy.signal
import soundfile as sf
import json

import config
from griffinlim import griffin_lim_base
from plots import plot_reconstructed_audio
from metrics import ssnr
def main(args):
    snr_path = config.RESULTS_DIR / (args.out_name + '.json')
    with open(config.AUDIO_IN_PATH, 'rb') as f:
        audio, sr = sf.read(f)
    
    _, _, spectrogram = scipy.signal.stft(audio)
    x_glb = griffin_lim_base(spectrogram, args.out_name, sr=sr)
    plot_reconstructed_audio(audio, x_glb, args.out_name)
    snr = {"snr": ssnr(x_glb, audio)}
    
    with open(snr_path, "w") as fp:
        json.dump(snr, fp)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_name',
                        type=str,
                        help='name to give to the outputs',
                        default='out')
    args = parser.parse_args()
    main(args)
