import argparse

import librosa
import soundfile as sf
import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics import ScaleInvariantSignalDistortionRatio

import config
from griffinlim import fast_griffin_lim
from networks.build_model import build_model
from utils.audioutils import (denormalize_db_spectr, normalize_db_spectr, standardization, set_mean_std,
                              segment_audio, to_db, to_linear, save_audio, open_audio)
from utils.plots import plot_melspec_prediction
from utils.utils import save_to_json

# TODO: fix everything, add DeGLI

    
def predict(args):
    
    # Paths
    audio_path = config.DATA_DIR / args.audio_path
    experiment_dir = config.MELSPEC2SPEC_DIR / args.weights_dir
    config_path = experiment_dir / "config.json"
    x_wav_hat_path = experiment_dir / 'gla_from_melspec.wav'
    metrics_path = experiment_dir / 'prediction_metrics.json'    
    prediction_img_path = experiment_dir / 'prediction.png'  

    # Load hparams and build model
    if args.model_name != 'pinv':
        hparams = config.load_config(config_path)
        weights_dir = config.WEIGHTS_DIR / args.weights_dir
        model = build_model(hparams, args.model_name, weights_dir, best_weights = True)
        model.eval()
    else:
        hparams = config.create_hparams()
    
    # Metrics
    pesq = PerceptualEvaluationSpeechQuality(fs=hparams.sr, mode="wb")
    stoi = ShortTimeObjectiveIntelligibility(fs=hparams.sr)
    sisdr = ScaleInvariantSignalDistortionRatio().to(hparams.device)
    
    x_wav = open_audio(audio_path, hparams.sr, hparams.audio_len).to(hparams.device)
    x_stftspec = torch.abs(torch.stft(x_wav, 
                                    n_fft=hparams.n_fft,
                                    hop_length=hparams.hop_len,
                                    return_complex=True)).to(hparams.device)  
    audio_seg, mean_seg, std_seg = segment_audio(audio_path, hparams.sr, hparams.audio_len)
    # x_stftspec_hat = []
    x_wav_hat = []
    with torch.no_grad():
        for n in range(audio_seg.shape[0]):
            x_stftspec_ = torch.abs(torch.stft(audio_seg[n], 
                                            n_fft=hparams.n_fft,
                                            hop_length=hparams.hop_len,
                                            return_complex=True)).to(hparams.device)

            x_melspec = torch.matmul(model.pinvblock.melfb, x_stftspec_**2)
            x_melspec_db_norm = normalize_db_spectr(to_db(x_melspec, power_spectr=True)).unsqueeze(0).unsqueeze(0)
            pred = model(x_melspec_db_norm).squeeze()
            pred = to_linear(denormalize_db_spectr(pred))
            
            # x_stftspec_hat.append(pred)
            x_wav_hat_ = fast_griffin_lim(torch.abs(pred).squeeze())
            x_wav_hat_ = set_mean_std(x_wav_hat_, mean_seg[n], std_seg[n])
            x_wav_hat.append(x_wav_hat_)
            
        # x_stftspec_hat = torch.stack(x_stftspec_hat, dim=0).reshape(x_stftspec.shape[0], x_stftspec.shape[1][:n*hparams.n_frames]) #TODO: fix reshape
        x_wav_hat = torch.stack(x_wav_hat, dim=0).reshape(-1)[:len(x_wav)]
        x_stftspec_hat = torch.abs(torch.stft(x_wav_hat, 
                                            n_fft=hparams.n_fft,
                                            hop_length=hparams.hop_len,
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
    parser.add_argument('--model_name', 
                        choices = ["unet", "pinvconv", "pinv"],
                        type=str,
                        default = 'pinvconv')
    parser.add_argument('--weights_dir',
                        type=str,
                        default='pinvconv02')
    parser.add_argument('--audio_path',
                        type=str,
                        default='in.wav')
    
    args = parser.parse_args()
    predict(args)