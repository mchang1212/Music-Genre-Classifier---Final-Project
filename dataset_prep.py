'''
Michelle Chang and Hyunwoo (Michael) Cho
Software Carpentry - Final Project
Music Genre Classifier
04/23/2021
Author(s): Hyunwoo (Michael) Cho
'''
# Data set preparation
import os
import pickle
import process_audio
from process_audio import Extract_Features


data = {
    'mapping': [],
    'numbered_label': [],
    'mfcc': [],
    'zero_crossing': [],
    'spectral_centroid': [],
    'spectral_contrast': [],
    'tempo': []
}
genres = ["blues", "classical", "country", "disco",
          "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def dataset(data_path):
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):
        # make sure we are dealing with genre folders
        if dirpath is not data_path:
            dirpath_comp = dirpath.split('/')
            genre_name = dirpath_comp[-1]
            data['mapping'].append(genre_name)
            print('\nProcessing {}'.format(genre_name))

            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                output = Extract_Features(file_path, genres[i - 1],
                                          duration=30, offset=None)
                mfcc = output.extract_mfcc()
                zero_cross = output.extract_zero_crossing()
                spect_cent = output.extract_spectral_centroid()
                spect_cont = output.extract_spectral_contrast()
                temp = output.extract_tempo()
                data['mfcc'].append(mfcc)
                data['zero_crossing'].append(zero_cross)
                data['spectral_centroid'].append(spect_cent)
                data['spectral_contrast'].append(spect_cont)
                data['tempo'].append(temp)
                data['numbered_label'].append(i - 1)
    pickle.dump(data, open('data_features.txt', 'wb'))


if __name__ == '__main__':
    data_path = '/Users/hwcho/Desktop/genres/genres/'
    dataset(data_path)
