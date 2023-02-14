import argparse
import os

import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

import config
from metrics import SI_SDR
from griffinlim import fast_griffin_lim, griffin_lim
from networks.build_model import build_model
from utils.audioutils import (denormalize_db_spectr, normalize_db_spectr, set_mean_std,
                              segment_audio, to_db, to_linear, save_audio, open_audio, 
                              compute_wav, initialize_random_phase)
from utils.plots import plot_melspec_prediction
from utils.utils import save_to_json, load_config


    
def predict(args):  
    
    # Paths
    audio_path = config.DATA_DIR / args.audio_path
    melspec2spec_dir = config.MELSPEC2SPEC_DIR / args.melspec2spec_weights_dir
    experiment_dir = config.MELSPEC2WAV_DIR / (args.melspec2spec_model_name + "_" + args.spec2wav_model_name)
    config_path = melspec2spec_dir / "config.json"
    if args.spec2wav_weights_dir is not None:
        degli_config_path = config.SPEC2WAV_DIR / args.spec2wav_weights_dir / "config.json"
    x_wav_hat_path = experiment_dir / 'gla_from_melspec.wav'
    metrics_path = experiment_dir / 'prediction_metrics.json'    
    prediction_img_path = experiment_dir / 'prediction.png'  

    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    
    # Load hparams and build melspec2spec_model
    if args.melspec2spec_model_name != 'pinv':
        hparams = load_config(config_path)
        melspec2spec_weights_dir = config.WEIGHTS_DIR / args.melspec2spec_weights_dir
        melspec2spec_model = build_model(hparams, args.melspec2spec_model_name, melspec2spec_weights_dir, best_weights=True)
        melspec2spec_model.eval()
    else:
        hparams = config.create_hparams()
    
    # Load hparams and build melspec2spec_model
    if args.spec2wav_model_name == "degli":
        spec2wav_weights_dir = config.WEIGHTS_DIR / args.spec2wav_weights_dir
        degli_hparams = load_config(degli_config_path)
        spec2wav_model = build_model(degli_hparams, args.spec2wav_model_name, spec2wav_weights_dir, best_weights=True)
        spec2wav_model.repetitions = args.degli_blocks
        spec2wav_model.eval()
        
    # Metrics
    pesq = PerceptualEvaluationSpeechQuality(fs=hparams.sr, mode="wb")
    stoi = ShortTimeObjectiveIntelligibility(fs=hparams.sr)
    sisdr = SI_SDR()
    
    x_wav = open_audio(audio_path, hparams.sr, hparams.audio_len).to(hparams.device)
    x_stftspec = torch.abs(torch.stft(x_wav, 
                                      n_fft=hparams.n_fft,
                                      hop_length=hparams.hop_len,
                                      window = torch.hann_window(hparams.n_fft).to(x_wav.device),
                                      return_complex=True)).to(hparams.device)  
    audio_seg, mean_seg, std_seg = segment_audio(audio_path, hparams.sr, hparams.audio_len)
    x_wav_hat = []
    with torch.no_grad():
        for n in range(audio_seg.shape[0]):
            x_stftspec_ = torch.abs(torch.stft(audio_seg[n], 
                                               n_fft=hparams.n_fft,
                                               hop_length=hparams.hop_len,
                                               window = torch.hann_window(hparams.n_fft).to(audio_seg[n].device),
                                               return_complex=True)).to(hparams.device)
            x_stftspec_ = normalize_db_spectr(to_db(x_stftspec_))
            x_melspec_db_norm = torch.matmul(melspec2spec_model.pinvblock.melfb, x_stftspec_).unsqueeze(0).unsqueeze(0)
            
            # MELSPEC2SPEC
            pred = melspec2spec_model(x_melspec_db_norm).squeeze()
            pred = to_linear(denormalize_db_spectr(pred))
            
            # STFT2WAV
            if args.spec2wav_model_name == "degli":
                pred_init = initialize_random_phase(pred.unsqueeze(0))
                x_stft_hat = spec2wav_model(pred_init, pred.unsqueeze(0)).squeeze()
                x_wav_hat_ = compute_wav(x_stft_hat, n_fft=hparams.n_fft)
            elif args.spec2wav_model_name == "fgla":
                x_wav_hat_ = fast_griffin_lim(pred)
            elif args.spec2wav_model_name == "gla":
                x_wav_hat_ = griffin_lim(pred)
                
            x_wav_hat_ = set_mean_std(x_wav_hat_, mean_seg[n], std_seg[n])
            x_wav_hat.append(x_wav_hat_)
            
        x_wav_hat = torch.stack(x_wav_hat, dim=0).reshape(-1)[:len(x_wav)]
        x_stftspec_hat = torch.abs(torch.stft(x_wav_hat, 
                                              n_fft=hparams.n_fft,
                                              hop_length=hparams.hop_len,
                                              window = torch.hann_window(hparams.n_fft).to(x_wav_hat.device),
                                              return_complex=True))
    
    save_audio(x_wav_hat, x_wav_hat_path, sr=hparams.sr)
    
    # Compute out_hat 's spectrogram and compare it to the original spectrogram

    plot_melspec_prediction(to_db(x_stftspec).cpu().numpy().squeeze(), 
                            to_db(x_stftspec_hat).cpu().numpy().squeeze(), 
                            sr = hparams.sr,
                            n_fft = hparams.n_fft,
                            hop_len = hparams.hop_len,
                            save_path = prediction_img_path)
    

    metrics = {"si-sdr": float(sisdr(x_stftspec_hat, x_stftspec)),
               "stoi": float(stoi(x_wav_hat, x_wav)),
               "pesq": float(pesq(x_wav_hat, x_wav))}
    save_to_json(metrics, metrics_path)
  
    
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--melspec2spec_model_name', 
                        choices = ["unet", "pinvconv", "pinv", "pinvconvres"],
                        type=str,
                        default = 'pinvconvres')
    
    parser.add_argument('--spec2wav_model_name', 
                        choices = ["degli", "fgla", "gla"],
                        type=str,
                        default = 'degli')
    
    parser.add_argument('--melspec2spec_weights_dir',
                        type=str,
                        default='pinvconvres04')
    
    parser.add_argument('--spec2wav_weights_dir',
                        type=str,
                        default='degli_B1_noiseM6P12')
    
    parser.add_argument('--degli_blocks',
                        type=int,
                        default=80)
    
    parser.add_argument('--audio_path',
                        type=str,
                        default='in.wav')
    
    args = parser.parse_args()
    predict(args)