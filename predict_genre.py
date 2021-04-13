'''
Michelle Chang and Hyunwoo (Michael) Cho
Software Carpentry - Final Project
Music Genre Classifier
04/23/2021

Author(s): Michelle Chang 
'''

import math
from process_audio import Extract_Features
# from model import make_model, predction


def audio_list(txt_file, path):
    '''
    This function will create a list of all the audio files we will
    use to train model.
    **Parameters**
        file: *str*
            name of txt file with list of all training audio files
    **Returns**
        audio_list: *list* of strings
            a list of names/file paths of the training audio files
    '''
    file_path = txt_file
    file = open(file_path, 'r')
    audio_list = []

    r_lines = file.readlines()
    for i in r_lines:
        # need to process audio file name and path to get the correct wav file
        file_name = i.split()[0]
        base_name = file_name.split(".au")[0]
        audio_name = base_name.split("/Users/gtzan/data/sound/genres/")[1]
        processed_name = path + audio_name + ".wav"
        audio_list.append(i)
    file.close()

    return audio_list


def load_features(file_path):
    '''
    This function opens and loads the pickled feature information for
    an audio file
    **Parameters**
        file_path: *str*
            name and path of desired audio file's features to load
    **Returns**
        features: *list* of lists
            the features information of an audio file
    '''
    file = open(file_path, "rb")
    features = pickle.load(file)
    return features
    # might not need this function, depending on if we keep the code
    # from the process_audio file pickling the features data


if __name__ == "__main__":

    file_path = "/Users/michellechang/Desktop/genres/"
    training_data = file_path + "bextract_single.mf"
    genres = ["classical", "country", "disco", "hiphop", "jazz", "rock",
              "blues", "reggae", "pop", "metal"]
    audio_list = audio_list(training_data, file_path)
    feature_matrix = []
    genre_list = []

    index = 0
    for i in audio_list:
        audio_file = i
        num = math.floor(index/50)
        # since there are 50 audio files of each genre, num can be
        # to determine the genre type that audio file is of.
        audio_genre = genres[num]
        genre_list.append(genre)

        processed_file = Extract_Features(
            audio_file, audio_genre, duration=30, offset=None)
        processed_file.extract_mfcc()
        processed_file.extract_zero_crossing()
        processed_file.extract_spectral_centroid()
        processed_file.extract_spectral_contrast()
        processed_file.extract_tempo()
        processed_file.save_features_info()

        base_name = audio_file.split(".wav")[0]
        file_path = base_name + "_features.txt"
        features = load_features(file_path)
        feature_matrix.append(features)
        index = index + 1

    '''
    My thoughts are that we would run through all the training set files
    and extract the features. Also at this point, it might seem useful 
    to create a pickled file and then just reopen it later... Well,
    we would create then a matrix containing each audio file's features (so
    500 elements of features which each themselves are a list of lists) 
    and then input that into the model, along with a list of the genre
    corresponding to each audio file. From there I'm not too sure lol
    but we would take some sample song and input the model and the song
    into a function that outputs the predicted genre.
    '''

    model = make_model(feature_matrix, genre_list)
    query_file = "/Users/michellechang/Desktop/song_name.mp3" 
    # can we use an mp3 file?
    predicted_genre = prediction(model, query_file)
    print(predicted_genre)
