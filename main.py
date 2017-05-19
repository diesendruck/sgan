import logging
import numpy as np
import os
import sys
import scipy.misc

from model import SGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate of for adam.")
flags.DEFINE_float("g_learning_rate", 0.0002, "Learning rate of for adam.")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam.")
flags.DEFINE_integer("epoch", 51, "Epoch to train.")
flags.DEFINE_integer("train_size", np.inf, "The size of train images.")
flags.DEFINE_integer("batch_size", 50, "The size of batch images.")
flags.DEFINE_integer("input_dim", 2, "The size of input datapoint.")
flags.DEFINE_integer("output_dim", 2, "The size of output datapoint.")
flags.DEFINE_integer("z_dim", 1, "Dimension of random input to generator.")
flags.DEFINE_integer("d_per_iter", 1, "Number of D updates per optim run.")
flags.DEFINE_integer("g_per_iter", 1, "Number of G updates per optim run.")
flags.DEFINE_string("dataset", "Gaussian", "The name of dataset [Gaussian, ConcentricCircles]")
flags.DEFINE_string("d_spec", "8,4,2", "Spec for fully-connected D architecture.")
flags.DEFINE_string("g_spec", "2,4,8", "Spec for fully-connected G architecture.")
flags.DEFINE_string("z_distr", "Uniform", "Z distribution [Gaussian, Uniform]")
flags.DEFINE_string("checkpoint_dir", "checkpoints/checkpoints_test", "Directory name to save the checkpoints.")
flags.DEFINE_string("sample_dir", "samples/samples_test", "Directory name to save the image samples.")
flags.DEFINE_string("log_dir", "logs/logs_test", "Directory name to save the logs.")
flags.DEFINE_string("expt_name", "test", "Experiment name, for naming logs, samples, and checkpoint dirs.")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing.")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing.")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

  logging.basicConfig(filename="{}/run_main.log".format(FLAGS.log_dir),
          filemode="w", level=logging.INFO)    
  logging.info("Started")
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    sgan = SGAN(
        sess,
        input_dim=FLAGS.input_dim,
        output_dim=FLAGS.output_dim,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        z_dim=FLAGS.z_dim,
        dataset_name=FLAGS.dataset,
        d_spec=FLAGS.d_spec,
        g_spec=FLAGS.g_spec,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        log_dir=FLAGS.log_dir,
        expt_name=FLAGS.expt_name)

    show_all_variables()

    if FLAGS.is_train:
      sgan.train(FLAGS)
    else:
      if not sgan.load(FLAGS.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")
      
    # Below is codes for visualization
    OPTION = 1
    #visualize(sess, sgan, FLAGS, OPTION)


  logging.info("Finished")

if __name__ == '__main__':
  tf.app.run()
