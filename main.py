import argparse
from pathlib import Path
import soundfile as sf
import json
import librosa
import numpy as np

import config
from griffinlim import griffin_lim_base, griffin_lim_librosa, fast_griffin_lim
from plots import plot_reconstructed_audio, plot_metric_numiter
from metrics import si_ssnr

def main(args):
    results_path = config.RESULTS_DIR / (args.out_name + '.json')
    gla_path = args.out_name + '_gla'
    fgla_path = args.out_name + '_fgla'
    librosa_path = args.out_name +'_librosa'
    with open(config.AUDIO_IN_PATH, 'rb') as f:
        audio, sr = sf.read(f)
    
    spectrogram = np.abs(librosa.stft(audio, n_fft=args.n_fft))
    x_gla, snr_hist_gla = griffin_lim_base(spectrogram, gla_path, args.num_iter, sr=sr, n_fft=args.n_fft)
    x_fgla, snr_hist_fgla = fast_griffin_lim(spectrogram, fgla_path, args.num_iter, sr=sr, n_fft=args.n_fft)
    x_libr = griffin_lim_librosa(spectrogram, librosa_path, args.num_iter, sr=sr)
    
    plot_metric_numiter(snr_hist_gla, gla_path, "SI-SNR")
    plot_metric_numiter(snr_hist_fgla, fgla_path, "SI-SNR")
    
    plot_reconstructed_audio(audio, x_gla, gla_path)
    plot_reconstructed_audio(audio, x_libr, librosa_path)
    plot_reconstructed_audio(audio, x_fgla, fgla_path)

    results = {"snr_gla": si_ssnr(spectrogram, np.abs(librosa.stft(x_gla, n_fft=args.n_fft))),
               "snr_fgla": si_ssnr(spectrogram, np.abs(librosa.stft(x_fgla, n_fft=args.n_fft))),
               "snr_librosa": si_ssnr(spectrogram, np.abs(librosa.stft(x_libr, n_fft=args.n_fft))), 
               "snr_hist_gla": snr_hist_gla,
               "snr_hist_fgla": snr_hist_fgla}
    
    with open(results_path, "w") as fp:
        json.dump(results, fp)
    
    
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
