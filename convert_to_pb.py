# -*- coding:utf-8 -*-

import tensorflow as tf
import argparse, os


def get_arguments():
  parser = argparse.ArgumentParser(description='WaveRNN baseline network')
  parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
  parser.add_argument('--target_dir',
                      type=str,
                      default=os.path.dirname(os.path.realpath(__file__)))
  return parser.parse_args()


def main():
  args = get_arguments()
  print(os.path.dirname(os.path.realpath(__file__)) + args.checkpoint)
  checkpoint = tf.train.get_checkpoint_state(
      os.path.dirname(os.path.realpath(__file__)) + '/' + args.checkpoint)
  input_checkpoint = checkpoint.model_checkpoint_path
  print('Loading checkpoint from: ' + input_checkpoint)

  target_save_dir = os.path.dirname(os.path.realpath(__file__)) + '/' + args.target_dir + \
                    "/" + input_checkpoint.split('/')[-2]
  os.makedirs(target_save_dir, exist_ok=True)
  output_grap = target_save_dir + "/frozen_model.pb"
  print(output_grap)

  with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)

    saver.restore(sess, input_checkpoint)
    for op in tf.get_default_graph().get_operations():
      print(op.name, op.values())

    output_grap_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names=['output'])
    with tf.gfile.GFile(output_grap, 'wb') as f:
      f.write(output_grap_def.SerializeToString())
    print("%d ops in the final graph." % len(output_grap_def.node))


if __name__ == '__main__':
  main()
