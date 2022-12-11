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
    results_path = config.GLA_RESULTS_DIR / (args.gla_type + '_' + args.phase_init + '.json')
    in_audio_path = config.DATA_DIR / args.in_audio_path
    gla_path = args.gla_type + '_' + args.phase_init
    out_audio_path = config.GLA_RESULTS_DIR / (gla_path + '.wav')

    with open(in_audio_path, 'rb') as f:
        audio, sr = sf.read(f)
    
    spectrogram = np.abs(librosa.stft(audio, n_fft=args.n_fft))
    
    if args.gla_type == 'librosa':
        x_gla = griffin_lim_librosa(spectrogram, 
                                    args.n_iter, 
                                    sr=sr, 
                                    init=args.phase_init) 
        plot_reconstructed_audio(audio, x_gla, gla_path)
        sf.write(out_audio_path, x_gla, samplerate=sr)

        results = {f"snr_{args.gla_type}": si_ssnr(spectrogram, np.abs(librosa.stft(x_gla, n_fft=args.n_fft)))}
        with open(results_path, "w") as fp:
            json.dump(results, fp)
        
        return
    
    elif args.gla_type == 'gla':
        x_gla, snr_hist = griffin_lim_base(spectrogram, 
                                           args.n_iter, 
                                           n_fft=args.n_fft, 
                                           init=args.phase_init,
                                           eval=args.eval)
    elif args.gla_type == 'fgla':
        x_gla, snr_hist = fast_griffin_lim(spectrogram, 
                                           args.n_iter, 
                                           n_fft=args.n_fft,
                                           init=args.phase_init,
                                           eval=args.eval)
    else:
        raise ValueError("gla_type must be one in ['gla', 'fgla', 'librosa']")
    
    sf.write(out_audio_path, x_gla, samplerate=sr)
    if args.eval:
        plot_reconstructed_audio(audio, x_gla, gla_path)
        plot_metric_numiter(snr_hist, gla_path)      

        results = {f"snr_{args.gla_type}": si_ssnr(spectrogram, np.abs(librosa.stft(x_gla, n_fft=args.n_fft))),
                f"snr_hist_{args.gla_type}": snr_hist}
        
        with open(results_path, "w") as fp:
            json.dump(results, fp)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gla_type',
                        type=str,
                        choices=['gla', 'fgla', 'librosa'],
                        help='Type of Griffin-Algorithm',
                        default='fgla')
    parser.add_argument('--in_audio_path',
                        type=str,
                        help='Path to the audio.wav file in data directory',
                        default='in.wav')
    parser.add_argument('--n_iter',
                        type=int,
                        help='Number of iterations of Griffin Lim Algorithm',
                        default='10000')
    parser.add_argument('--n_fft',
                        type=int,
                        help='Number of FFT points',
                        default='1024')
    parser.add_argument('--phase_init',
                        type=str,
                        choices=['zeros', 'random'],
                        help='Type of initialization for the phase',
                        default='random')
    parser.add_argument('--eval',
                        type=bool,
                        default=True)
    args = parser.parse_args()
    main(args)
