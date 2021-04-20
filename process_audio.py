'''
Michelle Chang and Hyunwoo (Michael) Cho
Software Carpentry - Final Project
Music Genre Classifier
04/23/2021

Author(s): Michelle Chang 
'''

import librosa
import pickle
import numpy as np
import matplotlib.pyplot as plt
class Extract_Features():
    def __init__(self, audio_file, genre, duration=30, offset=None):
        '''
        Initializes audio file as a floating point time series using
        librosa.
        **Parameters**
            audio_file: *str*
                name and file path of desired audio file to analyze
            genre: *str*
                the designated genre of audio file in training data
            duration: *int*
                we only load up to this much audio (in seconds)
            offset: *int*
                we start reading audio file after this time (in seconds)
        **Returns**
            None
        '''
        self.name = audio_file
        self.genre = genre
        self.y, self.sr = librosa.load(self.name,
                                       duration=duration, offset=offset)
        self.features = []
        # features is a list of lists of means and standard deviations of mfcc,
        # zero_crossing, centroid, and contrast, and then the tempo value.

    def extract_mfcc(self):
        '''
        This function analyzes audio file and determines the Mel 
        frequency cepstral coefficient (MFCC) mean and standard 
        deviation values. We append this to the features variable.
        **Parameters**
            None
        **Returns**
            None
        '''
        mfcc = librosa.feature.mfcc(self.y,
                                    sr=self.sr, n_mfcc=20)
        mfcc_mean = []
        mfcc_sd = []
        for i in range(len(mfcc)):
            mfcc_mean.append(np.mean(mfcc[i]))
            mfcc_sd.append(np.std(mfcc[i]))
        # mfcc_mean and mfcc_sd are both lists
        self.features.append(mfcc_mean)
        self.features.append(mfcc_sd)
        return (mfcc_mean, mfcc_sd)
        # returns a tuple of two lists

    def extract_zero_crossing(self):
        '''
        This function analyzes audio file and determines the zero crossing
        rate values. We append this to the features variable.
        **Parameters**
            None
        **Returns**
            None
        '''
        zero_crossing = librosa.feature.zero_crossing_rate(self.y)
        # zero_crossing is a list
        return zero_crossing[0].tolist()

    def extract_spectral_centroid(self):
        '''
        This function analyzes audio file and determines the spectral
        centroid values. We append this to the features variable.
        **Parameters**
            None
        **Returns**
            None
        '''
        centroid = librosa.feature.spectral_centroid(self.y, sr=self.sr)
        # centroid is a list
        return centroid[0].tolist()

    def extract_spectral_contrast(self):
        '''
        This function analyzes audio file and determines the spectral
        contrast mean and standard deviation values. We append this 
        to the features variable.
        **Parameters**
            None
        **Returns**
            None
        '''
        contrast = librosa.feature.spectral_contrast(self.y, sr=self.sr)
        contrast_mean = []
        contrast_sd = []
        for i in range(len(contrast)):
            contrast_mean.append(np.mean(contrast[i]))
            contrast_sd.append(np.std(contrast[i]))
        # contrast_mean and contrast_sd are both lists
        self.features.append(contrast_mean)
        self.features.append(contrast_sd)
        return (contrast_mean, contrast_sd)
        # returns a tuple of two lists

    def extract_tempo(self):
        '''
        This function analyzes audio file and determines the Mel 
        frequency cepstral coefficient (MFCC) mean and standard 
        deviation values. We append this to the features variable.
        **Parameters**
            audio_file: *str*
                name and file path of desired audio file to analyze
            genre: *str*
                the designated genre of audio file in training data
            duration: *int*
            offset: *int*
        **Returns**
            None
        '''
        onset_env = librosa.onset.onset_strength(self.y, sr=self.sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)
        self.features.append(float(list(tempo)[0]))
        return float(list(tempo)[0])
        # returns a single float value

    def save_features(self):
        '''
        This function will save all the information and features about our 
        audio file 
        **Parameters**
            none
        **Returns**
            none (creates and writes pickle file)
        '''
        base_name = self.name.split(".wav")[0]
        file_path = base_name + "_features.txt"
        file = open(file_path, "wb")
        pickle.dump(self.features, file)
        file.close()


if __name__ == '__main__':
    audio_file = "/Users/michellechang/Desktop/genres/blues/blues.00000.wav"
    genre = "blues"
    processed_file = Extract_Features(
        audio_file, genre, duration=30, offset=None)
    print(processed_file.extract_mfcc())
    print(processed_file.extract_zero_crossing())
    print(processed_file.extract_spectral_centroid())
    print(processed_file.extract_spectral_contrast())
    print(processed_file.extract_tempo())
    processed_file.save_features()
