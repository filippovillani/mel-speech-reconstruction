import os
from time import time

import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from tqdm import tqdm

import config
from griffinlim import initialize_phase
from utils.audioutils import compute_wav, min_max_normalization
from utils.plots import plot_gla_metrics, plot_gla_time
from utils.utils import save_to_json


class GLATester:
    def __init__(self, args):
        self.hprms = config.create_hparams()
        self.n_iter = args.n_iter
        self._set_paths()
        self.window = torch.hann_window(self.hprms.n_fft).to(self.hprms.device)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.hprms.sr, mode="wb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.hprms.sr)
        
    def test_gla(self, test_dl):
        
        gla_metrics_hist, gla_metrics = self._test_gla_metrics(test_dl, "gla")
        gla_time_hist = self._test_gla_time(test_dl, "gla")
        gla_metrics["time"] = gla_time_hist[-1]
        
        fgla_metrics_hist, fgla_metrics = self._test_gla_metrics(test_dl, "fgla")
        fgla_time_hist = self._test_gla_time(test_dl, "fgla")
        fgla_metrics["time"] = fgla_time_hist[-1]

        _data = (gla_metrics, gla_metrics_hist, gla_time_hist,
                 fgla_metrics, fgla_metrics_hist, fgla_time_hist)
        _paths = (self.gla_metrics_path, self.gla_metrics_hist_path, self.gla_time_hist_path,
                  self.fgla_metrics_path, self.fgla_metrics_hist_path, self.fgla_time_hist_path)
        save_to_json(_data, _paths)
        
        plot_gla_metrics(gla_metrics_hist, self.gla_metric_plot_path)
        plot_gla_time(gla_time_hist, self.gla_time_plot_path)
        plot_gla_metrics(fgla_metrics_hist, self.fgla_metric_plot_path)
        plot_gla_time(fgla_time_hist, self.fgla_time_plot_path)

        return gla_metrics_hist, gla_time_hist, fgla_metrics_hist, fgla_time_hist

    def _test_gla_metrics(self,
                          test_dl,
                          algorithm = "gla"):

        metrics_hist = {"stoi_hist": [0] * (self.n_iter//10),
                        "pesq_hist": [0] * (self.n_iter//10)}
        pbar = tqdm(test_dl, desc="GLA metrics test: ")
        for n, batch in enumerate(pbar):
            x_stft = batch["stft"].to(self.hprms.device)
            x_stft_mag = torch.abs(x_stft)
            x_wav = compute_wav(x_stft, n_fft=self.hprms.n_fft).squeeze()
            if algorithm == "gla":
                batch_metrics = self._gla_for_metrics_test(x_stft_mag, x_wav)
            else:
                batch_metrics = self._fgla_for_metrics_test(x_stft_mag, x_wav)
            # online update of metrics history
            metrics_hist = {k: [(v[m]+((1./(n+1))*(batch_metrics[k][m]-v[m])))
                                for m in range(self.n_iter//10)] for k, v in metrics_hist.items()}
            
            scores_to_print = str({k.replace("_hist", ""): round(float(max(v)), 4) for k, v in metrics_hist.items() if v != 0.})
            pbar.set_postfix_str(scores_to_print)
            if n == 50:
                break

        metrics = {"stoi": max(metrics_hist["stoi_hist"]),
                   "pesq": max(metrics_hist["pesq_hist"])}

        return metrics_hist, metrics

    def _gla_for_metrics_test(self,
                              spectrogram: torch.Tensor,
                              x_wav: torch.Tensor,
                              alpha: float = 0.99,
                              n_fft: int = 1024,
                              init: str = "zeros"):

        metrics = {"stoi_hist": [],
                   "pesq_hist": []}

        X_init_phase = initialize_phase(spectrogram, init)

        # Initialize the algorithm
        x_stft_hat = spectrogram * torch.exp(1j * X_init_phase)

        for n in range(self.n_iter):
            X_hat = torch.istft(x_stft_hat, 
                                window = self.window,
                                n_fft=n_fft).squeeze()    # G+ cn
            
            if n % 10 == 0: # TODO: try to remove min_max_norm
                metrics["pesq_hist"].append(self.pesq(X_hat, x_wav).item())
                metrics["stoi_hist"].append(self.stoi(X_hat, x_wav).item())

            X_hat = torch.stft(X_hat, 
                               n_fft=n_fft, 
                               window = self.window,
                               return_complex=True)  # G G+ cn

            X_phase = torch.angle(X_hat)
            x_stft_hat = spectrogram * torch.exp(1j * X_phase)   # Pc1(Pc2(cn-1))
            
        return metrics
    
    
    def _fgla_for_metrics_test(self,
                               spectrogram: torch.Tensor,
                               x_wav: torch.Tensor,
                               alpha: float = 0.99,
                               n_fft: int = 1024,
                               init: str = "zeros"):

        metrics = {"stoi_hist": [],
                   "pesq_hist": []}

        X_init_phase = initialize_phase(spectrogram, init)

        # Initialize the algorithm
        x_stft_hat = spectrogram * torch.exp(1j * X_init_phase)
        prev_proj = torch.istft(x_stft_hat, 
                                window = self.window,
                                n_fft=n_fft)
        prev_proj = torch.stft(prev_proj, 
                               n_fft=n_fft, 
                               window = self.window,
                               return_complex=True)
        prev_proj_phase = torch.angle(prev_proj)
        prev_proj = spectrogram * torch.exp(1j * prev_proj_phase)

        for n in range(self.n_iter):
            curr_proj = torch.istft(x_stft_hat, 
                                    window = self.window,
                                    n_fft=n_fft).squeeze()    # G+ cn
            
            if n % 10 == 0: # TODO: try to remove min_max_norm
                metrics["pesq_hist"].append(self.pesq(curr_proj, x_wav).item())
                metrics["stoi_hist"].append(self.stoi(curr_proj, x_wav).item())

            curr_proj = torch.stft(curr_proj, 
                                   n_fft=n_fft, 
                                   window = self.window,
                                   return_complex=True)  # G G+ cn

            curr_proj_phase = torch.angle(curr_proj)
            curr_proj = spectrogram * torch.exp(1j * curr_proj_phase)   # Pc1(Pc2(cn-1))

            x_stft_hat = curr_proj + alpha * (curr_proj - prev_proj)
            prev_proj = curr_proj

        return metrics

    def _test_gla_time(self,
                       test_dl,
                       algorithm = "gla"):

        time_hist = [0] * self.n_iter
        for n, batch in enumerate(tqdm(test_dl, desc="GLA time test: ")):
            x_stft = batch["stft"].to(self.hprms.device)
            x_stft_mag = torch.abs(x_stft)
            if algorithm == "gla":
                batch_times = self._gla_for_time_test(x_stft_mag)
            else:
                batch_times = self._fgla_for_time_test(x_stft_mag)
            time_hist = [(time_hist[m]+((1./(n+1))*(batch_times[m]-time_hist[m])))
                         for m in range(self.n_iter)]

        return time_hist

    def _gla_for_time_test(self,
                           spectrogram: torch.Tensor,
                           n_fft: int = 1024,
                           init: str = "zeros"):
        batch_times = []
        start_time = time()
        # Initialize the algorithm
        X_init_phase = initialize_phase(spectrogram, init)
        x_stft_hat = spectrogram * torch.exp(1j * X_init_phase)
        for _ in range(self.n_iter):
            X_hat = torch.istft(x_stft_hat, 
                                window = self.window,
                                n_fft=n_fft).squeeze()    # G+ cn
            X_hat = torch.stft(X_hat, 
                               n_fft=n_fft,
                               window = self.window,
                               return_complex=True)  # G G+ cn

            X_phase = torch.angle(X_hat)
            x_stft_hat = spectrogram * torch.exp(1j * X_phase)   # Pc1(Pc2(cn-1))

            batch_times.append(time() - start_time)

        return batch_times
    
    
    def _fgla_for_time_test(self,
                            spectrogram: torch.Tensor,
                            alpha: float = 0.99,
                            n_fft: int = 1024,
                            init: str = "zeros"):
        batch_times = []
        start_time = time()
        # Initialize the algorithm
        X_init_phase = initialize_phase(spectrogram, init)
        x_stft_hat = spectrogram * torch.exp(1j * X_init_phase)
        prev_proj = torch.istft(x_stft_hat, 
                                window = self.window,
                                n_fft=n_fft)
        prev_proj = torch.stft(prev_proj, 
                               n_fft=n_fft, 
                               window = self.window,
                               return_complex=True)
        prev_proj_phase = torch.angle(prev_proj)
        prev_proj = spectrogram * torch.exp(1j * prev_proj_phase)

        for _ in range(self.n_iter):
            curr_proj = torch.istft(x_stft_hat, 
                                    window = self.window,
                                    n_fft=n_fft).squeeze()    # G+ cn
            curr_proj = torch.stft(curr_proj, 
                                   n_fft=n_fft,
                                   window = self.window,
                                   return_complex=True)  # G G+ cn

            curr_proj_phase = torch.angle(curr_proj)
            curr_proj = spectrogram * torch.exp(1j * curr_proj_phase)   # Pc1(Pc2(cn-1))

            x_stft_hat = curr_proj + alpha * (curr_proj - prev_proj)
            prev_proj = curr_proj
            batch_times.append(time() - start_time)

        return batch_times

    def _set_paths(self):

        gla_results_dir = config.COMPARISONS_DIR / 'gla_baseline'
        fgla_results_dir = config.COMPARISONS_DIR / 'fgla_baseline'
        
        self.gla_metrics_path = gla_results_dir / f'gla_n{self.n_iter}_metrics.json'
        self.fgla_metrics_path = fgla_results_dir / f'fgla_n{self.n_iter}_metrics.json'
        self.gla_metrics_hist_path = gla_results_dir / f'gla_n{self.n_iter}_metrics_hist.json'
        self.fgla_metrics_hist_path = fgla_results_dir / f'fgla_n{self.n_iter}_metrics_hist.json'
        self.gla_time_hist_path = gla_results_dir / f'gla_n{self.n_iter}_time_hist.json'
        self.fgla_time_hist_path = fgla_results_dir / f'fgla_n{self.n_iter}_time_hist.json'
        self.gla_metric_plot_path = gla_results_dir / f'gla_n{self.n_iter}_metrics_plot.png'
        self.fgla_metric_plot_path = fgla_results_dir / f'fgla_n{self.n_iter}_metrics_plot.png'
        self.gla_time_plot_path = gla_results_dir / f'gla_n{self.n_iter}_time_plot.png'
        self.fgla_time_plot_path = fgla_results_dir / f'fgla_n{self.n_iter}_time_plot.png'

        if not os.path.exists(gla_results_dir):
            os.mkdir(gla_results_dir)
        if not os.path.exists(fgla_results_dir):
            os.mkdir(fgla_results_dir)