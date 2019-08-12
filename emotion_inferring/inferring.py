from __future__ import print_function

from datetime import datetime
import os
import time
import numpy as np
import tensorflow as tf

from .predictor import emotion_predictor
from .utils import *

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


def listdir(path):
  list_name = []
  for file in os.listdir(path):
    file_path = os.path.join(path, file)
    if os.path.splitext(file_path)[-1] == '.wav':
      list_name.append(file)
  return list_name


def inferring(args, hparams):
  started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
  logdir = os.path.join(args.logdir, 'logdir', 'generate', started_datestring)
  os.makedirs(logdir)
  sample_dir = hparams.test_wave_dir

  ## Load Model
  ER_predictor = emotion_predictor(hparams=hparams,
                                   logdir=logdir,
                                   sample_dir=sample_dir,
                                   checkpoint=args.checkpoint,
                                   is_training=False,
                                   presentation_output=args.presentation_output,
                                   convert_to_pb=args.convert_to_pb,
                                   pb_save_dir=args.target_dir)

  list_file = listdir(sample_dir)
  start_time = time.time()

  ## Pooling
  pool = ThreadPool(hparams.threads_num)
  outputs = pool.map(ER_predictor.inferring, list_file)
  pool.close()
  pool.join()
  end_time = time.time()
  print('Used time : ', end_time - start_time)
  print(outputs)


def er_inferring(args, hparams):
  return inferring(args, hparams)
