import fnmatch, random
import os, io, re
import threading, librosa, time

import soundfile as sf
import numpy as np
import tensorflow as tf

from multiprocessing.dummy import Pool as ThreadPool
from emotion_inferring.dataset.iemocap_reader import *

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'

def audio_to_float(audio, bits =16):
    audio = np.array(audio) / (2**(bits-1))
    return audio

def normalization_initial(data, hparams, log_dir):
    if hparams.input_type == 'raw':
        min_max_save_dir = hparams.mel_min_max_save_dir
    elif hparams.input_type == 'ComParE':
        min_max_save_dir = hparams.ComParE_min_max_save_dir
    else:
        raise ('The input type is wrong !!')
    if not os.path.exists(min_max_save_dir):
        print('Computing the min_max value for normalization ...')
        mel_min, mel_max = _min_max_value_compute(hparams, data)
        np.save(min_max_save_dir,[mel_min, mel_max])
    else:
        mel_min, mel_max = np.load(min_max_save_dir)
    print('Normalization parameters initialized ...')
    np.save(log_dir + '/mel_min_max_var', [mel_min, mel_max])

def aoustic_features_generator(data, hparams):
    if hparams.input_type == 'raw':
        mel_min, mel_max = np.load(hparams.mel_min_max_save_dir)
    elif hparams.input_type == 'ComParE':
        mel_min, mel_max = np.load(hparams.ComParE_min_max_save_dir)
    else:
        raise ('The input type is wrong !!')
    while 1:
        randomized_files = randomize_files(data)
        lock_length_rec = 0
        stack_lock = 100
        for filename in randomized_files:
            audio = filename['signal']
            audio = audio_to_float(audio)
            if lock_length_rec == 0:
                reg_length = len(audio)
                lock_length_rec += 1
                stack_lock = 1000
            elif lock_length_rec == hparams.batch_size:
                lock_length_rec = 0
            text = filename['transcription']
            if stack_lock != 0:
                if not (reg_length // 2  < len(audio) or len(audio) < reg_length * 1.25):
                    stack_lock -= 1
                    continue
            if len(audio)==0:
                continue
            if text.shape[0]<1:
                continue
            emotion = filename['emotion']
            if emotion in hparams.emotion_used:
                emotion_class = np.argwhere(hparams.emotion_used==emotion)
            else:
                continue
            if hparams.input_type == 'raw':
                features = acoustic_gen(hparams, audio, mel_max=mel_max, mel_min=mel_min)
            elif hparams.input_type == 'ComParE':
                features = np.array(filename['acoustic_features'])
                features = features[:,:hparams.condition_num]
                # features = _normalize_min_max(features, mel_max[:hparams.condition_num], mel_min[:hparams.condition_num],
                #                               max_value = 1.0, min_value = 0.0)
            else:
                raise('The input type is wrong !!')
            if len(features) > 3200:
                features = features[-3200:]
            if hparams.padding_len != None:
                features_pad = np.zeros((hparams.padding_len, hparams.num_mels))
                if len(features)> hparams.padding_len:
                    features_pad = features[:hparams.padding_len]
                else:
                    features_pad[:len(features)] = features
                features = features_pad
            yield features, text, emotion_class

def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]

def load_metadata(hparams, renew = False):
    data = read_iemocap_mocap(hparams.data_dir, hparams.word2vec_path, renew= renew)
    data, weight = class_balance(data, hparams.emotion_used)
    print('ALL:', weight)
    np.random.shuffle(data)
    data_train = data[:int(len(data) * 0.9)]
    data_valid = data[int(len(data) * 0.9):]
    np.random.shuffle(data_train)
    np.random.shuffle(data_valid)
    data_train, weight = class_balance(data_train, hparams.emotion_used)
    print('Train:', weight)
    data_valid, weight = class_balance(data_valid, hparams.emotion_used)
    print('Valid:', weight)
    return data_train, data_valid

def class_balance(data, emotion_used):
    c_a,c_e,c_n,c_s = 0,0,0,0
    for sample in data:
        emo_class = sample['emotion']
        if sample['transcription'].shape[0]<1:
            continue
        if emotion_used[0] == emo_class:
            c_a += 1
        elif emotion_used[1] == emo_class:
            c_e += 1
        elif emotion_used[2] == emo_class:
            c_n += 1
        elif emotion_used[3] == emo_class:
            c_s += 1
    sum = c_a + c_e + c_n + c_s
    return data,(np.array([sum,c_a,c_e,c_n,c_s]),np.array([c_a/sum,c_e/sum,c_n/sum,c_s/sum]))

def acoustic_gen(hparams, x, mel_max, mel_min):
    if hparams.trim_silence:
        x = trim_silence(x, hparams)
    condition = melspectrogram(x, hparams).astype(np.float32)
    condition = _normalize_min_max(condition.T, mel_max, mel_min, max_value = 1.0, min_value = 0.0)
    return condition

def trim_silence(wav, hparams):
    return librosa.effects.trim(wav,
                                top_db= hparams.trim_top_db,
                                frame_length=hparams.trim_fft_size,
                                hop_length=hparams.trim_hop_size)[0]

def melspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size)

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
							   fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
			 -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)
    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_mels:
        return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
    else:
        return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))


def _min_max_value_compute(hparams, data):
    if hparams.input_type == 'raw':
        all_mels_min = []
        all_mels_max = []
        mel_com_par = mel_compute_par(hparams=hparams)
        pool = ThreadPool(32)
        for y in pool.imap_unordered(mel_com_par.mel_compute_min_max, data):
            all_mels_min.append(y[0])
            all_mels_max.append(y[1])
        pool.close()
        pool.join()
        return np.amin(np.array(all_mels_min), axis=0), np.amax(np.array(all_mels_max), axis=0)
    elif hparams.input_type == 'ComParE':
        randomized_files = randomize_files(data)
        all_features = []
        for filename in randomized_files:
            all_features.append(np.array(filename['acoustic_features']))
        all_features = np.concatenate(all_features, axis=0)
        print(all_features.shape)
        return np.amin(all_features, axis=0), np.amax(all_features, axis=0)
    else:
        return False

class mel_compute_par(object):
    def __init__(self,
                 hparams):
        self.hparams = hparams
    def mel_compute_min_max(self, filename):
        audio = filename['signal']
        audio = audio_to_float(audio)
        mels = melspectrogram(audio, self.hparams).astype(np.float32)
        mels = mels.T
        return np.amin(mels, axis=0), np.amax(mels, axis=0)

def _normalize_min_max(spec, maxs, mins, max_value=1.0, min_value=0.0):
    spec_dim  = spec.shape[-1]
    num_frame = spec.shape[0]

    max_min = maxs - mins
    max_min = np.reshape(max_min, (1, spec_dim))
    max_min[max_min <= 0.0] = 1.0

    target_max_min = np.zeros((1, spec_dim))
    target_max_min.fill(max_value - min_value)
    target_max_min[max_min <= 0.0] = 1.0

    spec_min   = np.tile(mins, (num_frame, 1))
    target_min = np.tile(min_value, (num_frame, spec_dim))
    spec_range = np.tile(max_min, (num_frame, 1))
    norm_spec  = np.tile(target_max_min, (num_frame, 1)) / spec_range
    norm_spec  = norm_spec * (spec - spec_min) + target_min
    return norm_spec

def _denormalize_min_max(spec, maxs, mins, max_value = 1.0, min_value = 0.0):
    spec_dim = len(spec.T)
    num_frame = len(spec)

    max_min = maxs - mins
    max_min = np.reshape(max_min, (1, spec_dim))
    max_min[max_min <= 0.0] = 1.0

    target_max_min = np.zeros((1, spec_dim))
    target_max_min.fill(max_value - min_value)
    target_max_min[max_min <= 0.0] = 1.0

    spec_min = np.tile(mins, (num_frame, 1))
    target_min = np.tile(min_value, (num_frame, spec_dim))
    spec_range = np.tile(max_min, (num_frame, 1))
    denorm_spec = spec_range / np.tile(target_max_min, (num_frame, 1))
    denorm_spec = denorm_spec * (spec - target_min) + spec_min
    return denorm_spec
