'''
Michelle Chang and Hyunwoo (Michael) Cho
Software Carpentry - Final Project
Music Genre Classifier
04/23/2021
Author(s): Hyunwoo (Michael) Cho
'''

from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import keras

# Identify dataset and data_path
dataset = 'data_features.pickle'
data_path = '/Users/hwcho/OneDrive/JHU/21-22/Software Carpentry/final project/'


def load_dataset(dataset, data_path):
    '''
    Loads desired dataset at the designated datapath
    and returns numpy arrays of numbered labels, which correspond
    to genres and of mean values of mfccs.
    **Parameters**
        dataset: *str*
            Name of dataset. Has to be a .pickle file
        data_path: *str*
            location of audio files
    **Returns**
        mfcc_m: *numpy array*
            Array of means of mfccs of each audio file
        correct_genre: *numpy array*
            Array of numbered lables corresponding to each genre
    '''
    filepath = data_path + dataset
    file = open(filepath, 'rb')
    data = pickle.load(file)
    mfcc_vals = data['mfcc']
    mfcc_mean = []
    for i in range(len(mfcc_vals)):
        mfcc_mean.append(mfcc_vals[i][0])
    mfcc_m = np.array(mfcc_mean)
    correct_genre = np.array(data['numbered_label'])

    return mfcc_m, correct_genre


if __name__ == '__main__':
    mfcc_m, correct_genre = load_dataset(dataset, data_path)
    # 30% of input data is used to train model. 70% will be tested
    data_train, data_test, genre_train, genre_test = train_test_split(
        mfcc_m, correct_genre, test_size=0.3)

    model = keras.Sequential([
        # 1st hidden layer, Rectified Linear Unit (ReLU )
        keras.layers.Dense(512, activation='relu'),
        # 2nd
        keras.layers.Dense(256, activation='relu'),
        # 3rd
        keras.layers.Dense(128, activation='relu'),
        # 4th
        keras.layers.Dense(64, activation='relu'),
        # output layer, 10 bc we have 10 genres
        keras.layers.Dense(10, activation='softmax')
    ])
    # Compile the prepared neural network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the network
    model.fit(data_train, genre_train, validation_data=(
        data_test, genre_test), epochs=50, batch_size=32)
    
    # Summary of the model
    model.summary()
