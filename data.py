'''
Generate and save melspectrograms (input) and spectrograms (target)
'''
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from argparse import Namespace

import config
from audioutils import melspectrogram, stft

def build_timit_df():
    timit_metadata_path = config.DATA_DIR / 'train_data.csv'
    timit_audio_dir = config.DATA_DIR / 'timit'
    timit_df = pd.read_csv(timit_metadata_path)
    timit_df = timit_df.loc[timit_df['is_audio']==True].loc[timit_df['is_converted_audio']==True]
    # Dropping all the columns but speech_path
    timit_df['speech_path'] = timit_audio_dir / timit_df['path_from_data_dir'].astype(str)
    timit_df = timit_df['speech_path'].astype(str)
    timit_df = timit_df.sample(frac=1., random_state=config.SEED)
    timit_df = timit_df.reset_index(drop=True)
    
    return timit_df

def build_data(df: pd.DataFrame,
               hparams: Namespace):
    for path in df:
        # extract file name and use it to save the spectrograms
        audio_name = path.split("\\")[-1]
        spectr_path = config.SPECTR_DIR / audio_name
        melspectr_path = config.MELSPECTR_DIR / audio_name
        # open audio and normalize it
        audio, _ = librosa.load(path, sr=hparams.sr)
        audio = (audio - audio.mean()) / (audio.std() + 1e-12)
        # compute and save spectrogram
        spectr = stft(audio, 
                      hparams.n_fft,
                      hparams.hop_len)
        np.save(spectr_path, spectr, allow_pickle=False)
        # compute and save melspectrogram
        melspecstr = melspectrogram(audio, 
                                    hparams.sr, 
                                    hparams.n_mels, 
                                    hparams.n_fft, 
                                    hparams.hop_len)
        np.save(melspectr_path, melspecstr, allow_pickle=False)

def main():
    hparams = config.create_hparams()
    df = build_timit_df()
    build_data(df, hparams)


if __name__ == "__main__":
    main()