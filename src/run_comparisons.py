import argparse

import config
from dataset import build_dataloader
from comparisons.gla_test import GLATester
from comparisons.degli_test import DeGLITester
from utils.plots import plot_degli_gla_metrics_time

def main(args):
    
    hparams = config.create_hparams()
    test_dl = build_dataloader(hparams, config.DATA_DIR, "spec2wav", "test")
    gla_tester = GLATester(args)
    degli_tester = DeGLITester(args)
    gla_metrics_hist, gla_time_hist = gla_tester.test_gla(test_dl)
    degli_metrics_hist, degli_time_hist = degli_tester.test_degli(test_dl)
    plot_degli_gla_metrics_time(config.COMPARISONS_DIR,
                                gla_metrics_hist, 
                                gla_time_hist, 
                                degli_metrics_hist, 
                                degli_time_hist)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter',
                        type = int,
                        default = 1000,
                        help = "GLA number of iterations")
    parser.add_argument('--n_blocks',
                        type = int,
                        default = 100,
                        help = "Number of blocks of DeGLI")
    parser.add_argument('--degli_name',
                        type = str,
                        default = "degli_def00")
    args = parser.parse_args()
    main(args)