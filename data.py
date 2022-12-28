'''
Generate and save melspectrograms, stftspectrograms and wav as .npy files
'''
import numpy as np
import pandas as pd
import librosa
from argparse import Namespace
import os
import torch

import config


def build_timit_df():
    
    timit_dir = config.DATA_DIR / 'timit'
    timit_metadata_path = timit_dir / 'train_data.csv'
    timit_df = pd.read_csv(timit_metadata_path)
    timit_df = timit_df.loc[timit_df['is_audio']==True].loc[timit_df['is_converted_audio']==True]
    # Dropping all the columns but speech_path
    timit_df['speech_path'] = timit_dir / timit_df['path_from_data_dir'].astype(str)
    timit_df = timit_df['speech_path']
    timit_df = timit_df.sample(frac=1.)
    timit_df = timit_df.reset_index(drop=True)
    
    return timit_df

def split_dataframes(df,
                     split_ratio: list = [0.6, 0.2, 0.2]):
    
    df_len = len(df)
    train_len = int(df_len * split_ratio[0])
    val_len = int(df_len * split_ratio[1])
    train_df = df[:train_len]
    val_df = df[train_len:train_len+val_len]
    test_df = df[train_len+val_len:]
    
    return train_df, val_df, test_df
    

def build_data(hparams: Namespace,
               df: pd.DataFrame,
               type: str = "train"):
    
    out_dir = config.STFT_DIR / type
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
        
    for path in df:
        # extract file name and use it to save the spectrograms
        audio_name = path.name
        spectr_path = out_dir / (audio_name + '.pt')
        # Open audio  
        audio, _ = librosa.load(path, sr=hparams.sr)
        
        # pad or trunc all vectors to the same size
        if len(audio) < hparams.audio_len:
            pad_begin_len = np.random.randint(0, hparams.audio_len - len(audio))
            pad_end_len = hparams.audio_len - len(audio) - pad_begin_len
            
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)
        
            audio = np.concatenate((pad_begin, audio, pad_end))  
            
        else:
            start_position = np.random.randint(0, len(audio) - hparams.audio_len)
            end_position = start_position + hparams.audio_len
            audio = audio[start_position:end_position]
        
        # Normalize audio
        audio = (audio - audio.mean()) / (audio.std() + 1e-12)
        
        spectr = librosa.stft(y=audio, 
                              n_fft=hparams.n_fft,
                              hop_length=hparams.hop_len)
        spectr = torch.as_tensor(spectr)
        torch.save(spectr, spectr_path)

def main():

    hparams = config.create_hparams()
    df = build_timit_df()
    train_df, val_df, test_df = split_dataframes(df)
    build_data(hparams, train_df, "train")
    build_data(hparams, val_df, "validation")
    build_data(hparams, test_df, "test")


if __name__ == "__main__":
    main()