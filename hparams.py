import numpy as np
import tensorflow as tf

# Default hyperparameters

hparams = tf.contrib.training.HParams(
	#Audio
	num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
	trim_silence = False, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)

	#Mel spectrogram
	n_fft = 2048, #Extra window size is filled with 0 paddings to match this parameter
	hop_size =  160,  #For 16K Hz 160 ~= 10ms
	win_size =  640,  #For 16K Hz 400 ~= 40 ms
	sample_rate = 16000, # Targeted speech sampling rate
	frame_shift_ms = None,

	#M-AILABS (and other datasets) trim params
	trim_fft_size = 512,
	trim_hop_size = 128,
	trim_top_db = 23,

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True,
	allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max]
	normalize_for = True, #whether to rescale to [0, 1]

	#Limits
	min_level_db = -100,
	ref_level_db = 20,
	fmin = 125, #Set this to 75 if your speaker is male! if female, 125 should help taking off noise. (To test depending on dataset)
	fmax = 7600,
	mel_min_max_dir = None,

	#Word2vec
	word2vec_path = '/mnt/runnan/word2vec_models/GoogleNews-vectors-negative300.bin.gz',

	##########################################################################################################################################
	# Training settings
    database = 'iemocap',  # can be 'iemocap' or 'xiaoice'
    input_type = 'raw',  # 'raw' or 'ComParE'
	condition_num = 80,  # raw= num_mels , ComParE =  28 or 55(with de)

	mel_min_max_save_dir = '/mnt/runnan/ER-system/min_max_var.npy',
	data_dir = '/mnt/runnan/ER-system/IEMOCAP_full_release/',
	ComParE_min_max_save_dir = '/mnt/runnan/ER-system/ComParE_min_max_var.npy',
    ComParE_features_dir = '/mnt/runnan/ER-system/audio_features_ComParE2016/',

	emotion_used = np.array(['ang', 'hap', 'neu', 'sad']),
	padding_len  = None,
	log_dir_root='logdir/',
	class_dim= 4,

	# Acoustic features extractor
	CNN_extractor   = True,
	self_attention  = True,

	# Acoustic representation generator
	# these two kinds of attention are mutually-exclusive
	global_attention= True,
	GCA_iterations  = 3,
	mixture_attention  = False,
    MA_iterations  = 3,

	# Input features pre-fix
	acoustic_enable = True,
	text_enable = True,

	units= 256,  # units used in the RNNcells
	gradients_limit = 1,
	momentum = 0.9,

	warm_up_step = 5*1e3,
	decay_step   = 1*1e4,
	mini_lr      = 5*1e-6,

	num_steps = int(1e7), # training steps
	checkpoint_every = 100,
	batch_size = 32,
	max_to_keep = 3,

	training_samples = 1000,
	warm_up_field = 200, #warm up field used in training to avoid cold start

	learning_rate = 1e-4,
	adam_beta1 = 0.9,
	adam_beta2 = 0.999,
	adam_epsilon = 1e-8,
	ema_decay = 0.9999, #decay rate of exponential moving average

	L2_reg = None,
	is_overwritten_training = False,
	###########################################################################################################################################
	#Testing
	test_wave_dir = './Test_samples',
	threads_num = 16,
	wavernn_seed = -1e-5
	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
