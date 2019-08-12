from __future__ import print_function

import argparse, os
from hparams import hparams
from emotion_inferring.inferring import er_inferring

HPARAMETER_FILE = ''


def get_arguments():

  def _str_to_bool(s):
    if s.lower() not in ['true', 'false']:
      raise ValueError('Argument needs to be a ' 'boolean, got {}'.format(s))
    return {'true': True, 'false': False}[s.lower()]

  parser = argparse.ArgumentParser(description='WaveRNN baseline network')
  parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
  parser.add_argument('--logdir',
                      type=str,
                      default=os.path.dirname(os.path.realpath(__file__)))
  parser.add_argument('--device',
                      type=str,
                      default=str(-1),
                      help='The employed GPU ID.')
  parser.add_argument('--target_dir',
                      type=str,
                      default=os.path.dirname(os.path.realpath(__file__)))
  parser.add_argument('--presentation_output', type=_str_to_bool, default=False)
  parser.add_argument('--convert_to_pb', type=_str_to_bool, default=False)
  return parser.parse_args()


def main():
  hparam = hparams.parse(HPARAMETER_FILE)
  args = get_arguments()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.device
  er_inferring(args, hparam)


if __name__ == '__main__':
  main()
