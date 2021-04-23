# Music Genre Classifier - Software Carpentry Final Project

Here, we have created a model to predict and classify an audio file's music genre. To train and make predictions using our model(s), we utilized the GTZAN database of 1000 audio files each 30 seconds long in duration, used initially in the paper “Musical genre classification of audio signals " by G. Tzanetakis and P. Cook. There are 10 genres defined for that audio dataset: blues, rock, pop, hip hop, jazz, classical, country, reggae, metal, and disco.

## Downloading audio file dataset
The dataset of audio files to train and test our model for classifying music genres can be found in the 10 folders labeled blues, rock, pop, hip hop, jazz, classical, country, reggae, metal, and disco. There are 50 audio files in each folder, totaling to 500 files. We decided to cut the dataset in half from the original TZAN database set to limit the storage space needed. Be sure to download all 10 folders and verify that all audio files are of the WAV format. You will need to specify the exact file path to the folders of the audio files that you downloaded in order to perform the model training and predictions in our code.

## Extracting audio features and prepping data for model
To extract features information from each audio file, we used the audio processing package Librosa. Be sure to install the package before running our code. 
```bash
pip install librosa
```
In the file process_audio.py, each audio file will be initialized as a class object and then features extracted are Mel frequency cepstral coefficients (MFCC), zero-crossing rate, spectral centroid, spectral contrast, and tempo. Each audio file's features can be saved as a pickled file if wished.
In the file dataset_prep.py, we walk through all the audio files in the dataset, performing feature extraction and then storing all that information in a dictionary, which is saved as a pickled file.

## Training model and making predictions
We decided to make a model using two different methods - KNN algorithm (utilizing scikit-learn) vs. Keras. Be sure to also install these two packages before running either code.
```bash
pip install -U scikit-learn
pip install keras
```
K nearest neighbors algorithm is performed in the file KNN_model.py. We also utilized train_test_split, StandardScaler, and classification_report from sklearn to split the audio files into a training set and a testing set and also to further process the features data before training the model. KNN utilizes a distance metric to find K nearest neighbors to a particular data point (audio file) of interest. We have written functions for both Euclidean distance and Minkowski distance, and the user can choose which one of the two to use while running the code. Once the model is trained by the training set and predictions of the genre of audio files in the testing set are made, we analyzed the precision, recall, and accuracy of the KNN model in predicting music genres. Our KNN model gives us a max accuracy of around 0.45.

Keras model algorithm is performed in the file Keras_model.py. We again utilized train_test_split from sklearn to split the audio files into a training set and a testing set. SOME STUFF HERE ABOUT HERE IT WORKS. This model can then be analyzed by the summary function to find the accuracy of the model in predicting music genres. Our Keras model gives us a max accuracy of around 0.60.

## Other notes
*IMPORTANT:* Please make sure that you have defined your file path (to where the downloaded audio files are stored) correctly! This is specifically for the string variable data_path in either KNN_model.py or Keras_model.py, depending on which model you wish to use.
