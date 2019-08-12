import os, sys, wave, re
import copy
import math
import numpy as np
import pickle, string, csv

from emotion_inferring.dataset.iemocap_utils import *
from gensim.models.keyedvectors import KeyedVectors

emotions_used = np.array(['ang', 'exc', 'hap', 'neu', 'sad'])
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

def read_iemocap_mocap(data_path, word2vec_path, renew = False):
    file_path = data_path + '/../' + 'data_collected.pickle'
    fea_folder_path = data_path + '/../' + 'audio_features_ComParE2016/'
    if not os.path.isfile(file_path) or renew:
        Word2Vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        data = []
        ids = {}
        for session in sessions:
            path_to_wav = data_path + session + '/dialog/wav/'
            path_to_emotions = data_path + session + '/dialog/EmoEvaluation/'
            path_to_transcriptions = data_path + session + '/dialog/transcriptions/'
            path_to_features = fea_folder_path + session +'/'
            files2 = os.listdir(path_to_wav)
            files = []
            for f in files2:
                if f.endswith(".wav"):
                    if f[0] == '.':
                        files.append(f[2:-4])
                    else:
                        files.append(f[:-4])

            for f in files:
                print('Processing' + f + ' ...')
                wav = get_audio(path_to_wav, f + '.wav')
                with open(path_to_features + f +'.csv', newline='') as fea_file:
                    reader = csv.reader(fea_file, delimiter=';')
                    first_line = True
                    features = []
                    for row in reader:
                        if first_line :
                            first_line = False
                            continue
                        features.append(np.array(row[1:], dtype=np.float))

                transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
                emotions = get_emotions(path_to_emotions, f + '.txt')
                sample   = split_wav(wav, features, emotions)

                for ie, e in enumerate(emotions):
                    e['signal'] = sample[ie]['left']
                    e['acoustic_features'] = sample[ie]['acoustic_features']
                    e.pop("left", None)
                    e.pop("right", None)
                    transcriptions_list = re.split(r' ', transcriptions[e['id']])
                    transcriptions_emb = []
                    for word in transcriptions_list:
                        word = ''.join(filter(str.isalpha, word))
                        if len(word) < 1:
                            continue
                        try:
                            transcriptions_emb.append(np.array(Word2Vec[word]))
                        except:
                            continue
                    transcriptions_emb = np.asarray(transcriptions_emb)
                    e['transcription'] = transcriptions_emb
                    if e['emotion'] in emotions_used:
                        if e['emotion'] == 'exc':
                            e['emotion'] = 'hap'
                        if e['id'] not in ids:
                            data.append(e)
                            ids[e['id']] = 1

        sort_key = get_field(data, "id")
        data_pac = np.array(data)[np.argsort(sort_key)]
        with open(file_path, 'wb') as handle:
            pickle.dump(data_pac, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(file_path, 'rb') as handle:
            data_pac = pickle.load(handle)
    return data_pac
