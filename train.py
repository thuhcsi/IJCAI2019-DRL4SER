from __future__ import print_function

import argparse, os, sys, time
import numpy as np
from hparams import hparams
from emotion_inferring.train import er_train

HPARAMETER_FILE= ''

def get_arguments(hparams):
    parser = argparse.ArgumentParser(description = 'Train The Emotion Recognition System')
    parser.add_argument('--output-model-path', dest='outputdir', default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--log-dir', dest='philly_log_dir', default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--batch_size', type=int, default=hparams.batch_size,
                        help='Wav files numbers to process simultaneously')
    parser.add_argument('--data_dir', type=str, default=hparams.data_dir,
                        help='Speech data directory.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=hparams.log_dir_root,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int, default=hparams.checkpoint_every,
                        help='How many steps to save each checkpoint after. Default: '
                             + str(hparams.checkpoint_every) + '.')
    parser.add_argument('--num_steps', type=int, default=hparams.num_steps,
                        help='Number of training steps. Default: ' + str(hparams.num_steps) + '.')
    parser.add_argument('--learning_rate', type=float, default=hparams.learning_rate,
                        help='Learning rate for training. Default: ' + str(hparams.learning_rate) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float, default= hparams.momentum)
    parser.add_argument('--device', type=str, default=-1,
                        help='The employed GPU ID for model training.')
    parser.add_argument('--database',  type=str, default= hparams.database)
    return parser.parse_args()


def main():
    hparam = hparams.parse(HPARAMETER_FILE)
    args = get_arguments(hparam)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    er_train(args, hparam)

if __name__ == '__main__':
    main()