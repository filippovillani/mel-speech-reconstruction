'''
Generate and save stft as .pt files
'''
import os
from argparse import Namespace

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
from utils.audioutils import standardization


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
                     split_ratio: list = [0.9, 0.05, 0.05]):
    
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
        
    for n, path in enumerate(tqdm(df)):
        audio_name = f'ex_{n}'
        # Open audio  
        audio, _ = librosa.load(path, sr=hparams.sr)
        audio = audio[int(0.1 * hparams.sr):int(len(audio) - 0.5 * hparams.sr)] # remove samps from start and end
        audio_power = np.mean(audio ** 2)
        # pad or trunc all vectors to the same size
        if len(audio) < hparams.audio_len:
            pad_begin_len = np.random.randint(0, hparams.audio_len - len(audio))
            pad_end_len = hparams.audio_len - len(audio) - pad_begin_len
            
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)
        
            audio_out = np.expand_dims(np.concatenate((pad_begin, audio, pad_end)), 0)
            
        else:
            audio_out = []
            n = 0
            while len(audio[n * hparams.audio_len : (n+1) * hparams.audio_len]) == hparams.audio_len:
                audio_seg = audio[n * hparams.audio_len : (n+1) * hparams.audio_len]
                seg_power = np.mean(audio_seg ** 2)
                n += 1
                # Compare audio to a threshold to detect if there is no utterance in it
                if seg_power > hparams.audio_thresh * audio_power:
                    audio_out.append(audio_seg)
                else:
                    continue
                
            if len(audio_out) != 0:
                audio_out = np.stack(audio_out, axis=0)
            else: 
                audio_out = np.empty(0)
        
        for n in range(audio_out.shape[0]):
            # Normalize audio
            out = torch.as_tensor(standardization(audio_out[n]))
            spectr_path = out_dir / (audio_name + f'_seg{n}.pt')
            spectr = torch.stft(input=out, 
                                n_fft=hparams.n_fft,
                                hop_length=hparams.hop_len,
                                return_complex=True)
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