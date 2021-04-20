'''
Michelle Chang and Hyunwoo (Michael) Cho
Software Carpentry - Final Project
Music Genre Classifier
04/23/2021
Author(s): Michelle Chang
'''

import pickle
from math import *
from decimal import Decimal
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import dataset_prep

# functions below are for setting up K nearest neighbors (KNN)
# algorithm for classification


def euclidean_dist(data1, data2):
    '''
    This function calculates the Euclidean distance between
    two data points (of vector form)
    **Parameters**
        data1: *list* of floats
            the features info of first audio file
        data2: *list* of floats
            the features info of second audio file
    **Returns**
        euc_dist: *float*
            the Euclidean distance between two data points
    '''
    distance = 0.0
    for i in range(len(data1)):
        distance = distance + (data1[i] - data2[i])**2
    euc_dist = sqrt(distance)
    return euc_dist


def minkowski_dist(data1, data2, pval):
    '''
    This function calculates the Minkowski distance between
    two data points (of vector form)
    **Parameters**
        data1: *list* of floats
            the features info of first audio file
        data2: *list* of floats
            the features info of second audio file
        pval: *int*
            the order p between two data points used in
            Minkowski distance equation
    **Returns**
        mink_dist: *float*
            the Minkowski distance between two data points
    '''
    distance = 0.0
    for i in range(len(data1)):
        distance = distance + abs((data1[i] - data2[i]))**pval
    mink_dist = distance**(1/float(pval))
    return mink_dist


def find_Kneighbors(data_train, indiv_test, num_neighbors):
    '''
    This function finds the most similar/nearest neighbors
    with the specified number of neighbors wanted
    **Parameters**
        data_train: *list* of lists
            the features info of all the audio files in training set
        indiv_test: *list* of floats
            the features info of one particular audio file of interest
            in our test set
        num_neighbors: *int*
            the number of closest neighbors (audio files) we wish to
            find for one particular audio file
    **Returns**
        neighbors: *list* of lists
            the closest neighbors (audio files) to the audio file
            of interest
    '''
    distances = []
    neighbors = []
    data_train = data_train.tolist()
    for indiv_train in data_train:
        dist = euclidean_dist(indiv_test, indiv_train)
        distances.append(dist)
    for i in range(num_neighbors):
        min_dist = distances.index(min(distances))
        neigh = data_train[min_dist]
        neighbors.append(neigh)
        distances.remove(min(distances))
    return neighbors


def predict_type(data_train, indiv_test, num_neighbors):
    '''
    This function makes a prediction of the genre of a test file
    **Parameters**
        data_train: *list* of lists
            the features info of all the audio files in training set
        indiv_test: *list* of floats
            the features info of one particular audio file of interest
            in our test set
        num_neighbors: *int*
            the number of closest neighbors (audio files) we wish to
            find for one particular audio file
    **Returns**
        prediction: *str*
            the predicted genre of the particular audio file of interest
            in our test set
    '''
    neighbors = find_Kneighbors(data_train, indiv_test, num_neighbors)
    data_train = data_train.tolist()
    genre = []
    for i in neighbors:
        neigh = i
        index = data_train.index(neigh)
        genre.append(genre_train[index])
    prediction = max(set(genre), key=genre.count)
    return prediction


if __name__ == '__main__':
    # opening and storing features data for all audio files
    data_path = '/Users/michellechang/Desktop/genres/'
    # dataset_prep.dataset(data_path)
    file_path = data_path + "data_features.txt"
    file = open(file_path, "rb")
    audio_dataset = pickle.load(file)
    file.close()

    # preprocessing features data and selecting just one feature to
    # use to create model. This is a limitation of the KNN algorithm,
    # as we can only use one dimension of the data at a time to analyze it
    mfcc_vals = audio_dataset['mfcc']
    mfcc_mean = []
    for i in range(len(mfcc_vals)):
        mfcc_mean.append(mfcc_vals[i][0])
    correct_genre = audio_dataset['mapping']

    # splitting dataset into train and test sets 
    # (70% in train and 30% in test)
    data_train, data_test, genre_train, genre_test = train_test_split(
        mfcc_mean, correct_genre, test_size=0.30)

    # feature scaling
    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # making predictions on the genre of each audio file in our test set
    # using KNN algorithm
    num_neighbors = 5
    genre_pred = []
    for i in range(len(data_test)):
        indiv_test = data_test[i]
        prediction = predict_type(data_train, indiv_test, num_neighbors)
        genre_pred.append(prediction)

    # evaluating model and its predictions in terms precision,
    # recall, and accuracy
    print(classification_report(genre_test, genre_pred))

    '''
    NOTEs: 
    - Using mfcc_means, both euc_dist and mink_dist around 0.45 max accuracy
    - Using mfcc_sd, euc around 0.33, mink around 0.25
    - Using spec_contrast_means, both euc and mink around 0.35-0.4
    - Using spec_contrast_sd, euc around 0.3, mink around 0.25
    '''
