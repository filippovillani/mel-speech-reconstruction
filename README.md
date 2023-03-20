# Installation and Prediction
Run from the project's root directory
```
pip install virtualenv
py -m venv env_name
env_name\Scripts\activate  
pip install -r requirements.txt
```
To test the models on an  ```audio.wav``` in ```/data``` with Mel Spectrogram Inversion model's weights in ```melspec2spec_weights_dir``` and Phase Reconstruction model's weights in ```spec2wav_weights_dir```:
```
predict.py --audio_path audio.wav --spec2wav_weights_dir  --melspec2spec_weights_dir
```
The generated spectrogram and the speech audio file are saved in ```/results/melspec2wav/spec2wav_weights_dir```.

# Speech Reconstruction from Mel Spectrograms
Speech reconstruction from mel spectrograms, a crucial task in achieving high-quality data after processing speech signals in the time-frequency domain through techniques such as speech enhancement, source separation, and text-to-speech.

The framework is composed of two stages: the first is a novel model that performs mel spectrogram inversion to STFT spectrograms by refining through an autoencoder of the estimate given by the pseudo-inverse matrix of the mel filterbank; for the second task, we tested the Griffin-Lim algorithm [[1]](#1), the Fast Griffin-Lim algorithm [[2]](#2), and the Deep Griffin-Lim iteration [[3]](#3).

# Examples
## Mel Spectrogram Inversion

Mel Spectrogram inversion from 80 mel coefficients:

<img src="https://user-images.githubusercontent.com/93431189/226337465-5d9500ca-6501-4513-b7e5-4c7c7acf81dd.png" width="600" height="450">

Mel Spectrogram inversion from 250 mel coefficients:

<img src="https://user-images.githubusercontent.com/93431189/226339261-5f0bdada-e86c-463c-8178-8af128576f8b.png" width="600" height="450">

## Speech Reconstruction

https://user-images.githubusercontent.com/93431189/226341060-cc109820-9628-4fa6-a2c1-b3f3b63f6dc2.mp4

Original audio.

https://user-images.githubusercontent.com/93431189/226341362-497f0c45-451f-4772-9a61-df45b043e4db.mp4

Prediction from 80 mels with DeGLI.

https://user-images.githubusercontent.com/93431189/226341572-304fac89-08d6-41e8-905b-26de12dcfa2c.mp4

Prediction from 250 mels with FGLA.

## References
<a id="1">[1]</a> 
D. Griffin and J. Lim, “Signal estimation from modified short-time fourier
transform,” IEEE Transactions on acoustics, speech, and signal processing,
vol. 32, no. 2, pp. 236–243, 1984.

<a id="2">[2]</a>
N. Perraudin, P. Balazs, and P. L. Søndergaard, “A fast griffin-lim algorithm,”
in 2013 IEEE Workshop on Applications of Signal Processing to Audio and
Acoustics, pp. 1–4, IEEE, 2013.

<a id="3">[3]</a>
Y. Masuyama, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada, “Deep griffin–
lim iteration: Trainable iterative phase reconstruction using neural network,”
IEEE Journal of Selected Topics in Signal Processing, vol. 15, no. 1, pp. 37–50,
2020.

