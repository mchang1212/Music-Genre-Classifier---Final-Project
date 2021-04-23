# Music Genre Classifier - Software Carpentry Final Project

Here, we have created a model to predict and classify an audio file's music genre. To train and make predictions using our model(s), we utilized the GTZAN database of 1000 audio files each 30 seconds long in duration, used initially in the paper “Musical genre classification of audio signals " by G. Tzanetakis and P. Cook. There are 10 genres defined for that audio dataset: blues, rock, pop, hip hop, jazz, classical, country, reggae, metal, and disco.

## Downloading audio file dataset

## Extracting audio features and prepping data for model
To extract features information from each audio file, we used the audio processing package Librosa. Be sure to install the package before running our code. 
```bash
pip install librosa
```
Features extracted are Mel frequency cepstral coefficients (MFCC), zero-crossing rate, spectral centroid, spectral contrast, and tempo.

## Training model and making predictions

## KNN vs. Keras method

## Other notes
Please make sure that you have defined your file path (to where the downloaded audio files are stored) correctly! This is specifically
for the string variable data_path in either KNN_model.py or Keras_model.py, depending on which model you wish to use.
