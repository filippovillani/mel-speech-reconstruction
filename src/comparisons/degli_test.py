import os
from time import time

import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

import config
from networks.build_model import build_model
from utils.audioutils import compute_wav
from utils.utils import save_to_json, load_config
from utils.plots import plot_degli_metrics, plot_degli_time

class DeGLITester:
    def __init__(self, args):
        
        self.model_name = args.degli_name
        self.repetitions = args.n_blocks
        self._set_paths()
        self.hprms = load_config(self.config_path)
        
        degli = build_model(self.hprms, "degli", self.weights_dir, best_weights=True)
        degli.eval()
        self.degliblock = degli.degliblock
        
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.hprms.sr, mode="wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.hprms.sr)
    
    def test_degli(self, test_dl):
        
        metrics_hist, metrics = self._test_degli_metrics(test_dl)
        time_hist = self._test_degli_time(test_dl)
        metrics["time"] = time_hist[-1]
        
        _data = (metrics, metrics_hist, time_hist)
        _paths = (self.metrics_path, self.metrics_hist_path, self.time_hist_path)
        save_to_json(_data, _paths)
        plot_degli_metrics(metrics_hist, self.metric_plot_path)
        plot_degli_time(time_hist, self.time_plot_path)
            
        return metrics_hist, time_hist
    
    def _test_degli_metrics(self,
                            test_dl):
        
        metrics_hist = {"stoi_hist": [0] * self.repetitions,
                        "pesq_hist": [0] * self.repetitions}
        pbar = tqdm(test_dl, desc="DeGLI metrics test: ")
        with torch.no_grad():
            for n, batch in enumerate(pbar):
                x_stft = batch["stft"].to(self.hprms.device)
                x_stft_mag = torch.abs(x_stft)
                x_stft_noisy = self._initialize_random_phase(x_stft_mag)
                
                x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
                batch_metrics = self._degli_for_metrics_test(x_stft_noisy, x_stft_mag, x_wav)
                # online update of metrics history
                metrics_hist = {k: [(v[m]+((1./(n+1))*(batch_metrics[k][m]-v[m]))).item()
                                    for m in range(self.repetitions)] for k, v in metrics_hist.items()}
                
                scores_to_print = str({k.replace("_hist", ""): round(float(max(v)), 4) for k, v in metrics_hist.items() if v != 0.})
                pbar.set_postfix_str(scores_to_print)
                if n == 10:
                    break
            
        metrics = {"stoi": max(metrics_hist["stoi_hist"]),
                   "pesq": max(metrics_hist["pesq_hist"])}
        
        return metrics_hist, metrics
    
    def _degli_for_metrics_test(self, 
                                x_stft_noisy, 
                                x_stft_mag,
                                x_wav):
        
        metrics = {"stoi_hist": [],
                   "pesq_hist": []}
        
        for _ in range(self.repetitions):
            x_stft_noisy = self.degliblock(x_stft_noisy, x_stft_mag)
            x_wav_hat = compute_wav(self.degliblock.magnitude_projection(x_stft_noisy, x_stft_mag),
                                    n_fft = self.hprms.n_fft).squeeze()
            metrics["stoi_hist"].append(self.stoi(x_wav, x_wav_hat))
            metrics["pesq_hist"].append(self.pesq(x_wav, x_wav_hat))

        return metrics
    
    def _test_degli_time(self,
                         test_dl):
        
        time_hist = [0] * self.repetitions
        with torch.no_grad():
            for n, batch in enumerate(tqdm(test_dl, desc="DeGLI time test: ")):
                x_stft = batch["stft"].to(self.hprms.device)
                x_stft_mag = torch.abs(x_stft)
                x_stft_noisy = self._initialize_random_phase(x_stft_mag)
                batch_times = self._degli_for_time_test(x_stft_noisy, x_stft_mag)
                time_hist = [(time_hist[m]+((1./(n+1))*(batch_times[m]-time_hist[m])))
                            for m in range(self.repetitions)]
            
                if n == 50:
                    break
        return time_hist
    
    def _degli_for_time_test(self,
                             x_stft_noisy,
                             x_stft_mag):
        
        blocks_time = []
        start_time = time()
        for _ in range(self.repetitions):
            x_stft_noisy = self.degliblock(x_stft_noisy, x_stft_mag)
            blocks_time.append(time() - start_time)
        return blocks_time

    def _initialize_random_phase(self, x_stft_mag):
        
        phase = torch.zeros_like(x_stft_mag)
        x_stft_noisy = x_stft_mag * torch.exp(1j * phase)
        return x_stft_noisy
    
    def _set_paths(self):
        
        self.weights_dir = config.WEIGHTS_DIR / self.model_name
        self.config_path = config.SPEC2WAV_DIR / self.model_name / "config.json"
        
        degli_results_dir = config.COMPARISONS_DIR / 'degli'
        self.metrics_path = degli_results_dir / f'n{self.repetitions}_metrics.json'
        self.metrics_hist_path = degli_results_dir / f'n{self.repetitions}_metrics_hist.json'
        self.time_hist_path = degli_results_dir / f'n{self.repetitions}_time_hist.json'
        self.metric_plot_path = degli_results_dir / f'n{self.repetitions}_metrics_plot.png'
        self.time_plot_path = degli_results_dir / f'n{self.repetitions}_time_plot.png'
        
        if not os.path.exists(degli_results_dir):
            os.mkdir(degli_results_dir)
