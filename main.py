import argparse
from pathlib import Path
import soundfile as sf
import json
import librosa
import numpy as np

import config
from griffinlim import griffin_lim_base, griffin_lim_librosa, fast_griffin_lim
from plots import plot_reconstructed_audio
from metrics import si_snr

def main(args):
    snr_path = config.RESULTS_DIR / (args.out_name + '.json')
    gla_path = args.out_name + 'glb'
    fgl_path = args.out_name + 'fgla'
    librosa_path = args.out_name +'librosa'
    with open(config.AUDIO_IN_PATH, 'rb') as f:
        audio, sr = sf.read(f)
    
    # audio = (audio - audio.mean()) / audio.std()
    spectrogram = np.abs(librosa.stft(audio, n_fft=args.n_fft))
    # x_gla = griffin_lim_base(spectrogram, gla_path, args.num_iter, sr=sr, n_fft=args.n_fft)
    # x_libr = griffin_lim_librosa(spectrogram, librosa_path, args.num_iter, sr=sr, n_fft=args.n_fft)
    x_fgla = fast_griffin_lim(spectrogram, fgl_path, args.num_iter, sr=sr, n_fft=args.n_fft)
    
    # plot_reconstructed_audio(audio, x_gla, gla_path)
    # plot_reconstructed_audio(audio, x_libr, librosa_path)
    plot_reconstructed_audio(audio, x_fgla, fgl_path)

    # snr = {"snr_gla": si_snr(audio, x_gla),
    #        "snr_fgla": si_snr(audio, x_libr),
    #        "snr_librosa": si_snr(audio, x_libr)}
    
    # with open(snr_path, "w") as fp:
    #     json.dump(snr, fp)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_name',
                        type=str,
                        help='name to give to the outputs',
                        default='out')
    parser.add_argument('--num_iter',
                        type=int,
                        help='number of iterations of Griffin Lim Algorithm',
                        default='512')
    parser.add_argument('--n_fft',
                        type=int,
                        help='number of points for FFT',
                        default='1024')
    args = parser.parse_args()
    main(args)
