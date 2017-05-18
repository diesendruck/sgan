from __future__ import division
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import random
import sys
import tensorflow as tf
import time

from glob import glob
from six.moves import xrange

from ops import *
from utils import *

plt.style.use("ggplot")


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class SGAN(object):
  def __init__(self, sess, input_dim=2, output_dim=2, batch_size=50,
          sample_num=50, z_dim=1, dataset_name="default", d_spec="8,4,2",
          g_spec="2,4,8", checkpoint_dir=None, sample_dir=None, log_dir=None,
          expt_name=None):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: (optional) Dimension of dim for Z. [100]
    """
    self.sess = sess
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.sample_num = sample_num
    self.z_dim = z_dim
    self.dataset_name = dataset_name
    self.d_spec = d_spec
    self.g_spec = g_spec
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.log_dir = log_dir
    self.expt_name = expt_name

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn0 = batch_norm(name="discriminator_bn0")
    self.d_bn1 = batch_norm(name="discriminator_bn1")
    self.d_bn2 = batch_norm(name="discriminator_bn2")
    self.d_bn3 = batch_norm(name="discriminator_bn3")
    self.d_bn_last = batch_norm(name="discriminator_bn_last")
    self.g_bn0 = batch_norm(name="generator_bn0")
    self.g_bn1 = batch_norm(name="generator_bn1")
    self.g_bn2 = batch_norm(name="generator_bn2")
    self.g_bn3 = batch_norm(name="generator_bn3")

    self.build_model()


  def build_model(self):
    input_dims = [self.input_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + input_dims, name="real_inputs")
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + input_dims, name="sample_inputs")
    self.heldout_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + input_dims, name="heldout_inputs")

    inputs = self.inputs
    sample_inputs = self.sample_inputs
    heldout_inputs = self.heldout_inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name="z")
    self.grid = tf.placeholder(
      tf.float32, [None, 2], name="grid")  # This will be a grid of 2-dim pts.
    self.z_sum = histogram_summary("z", self.z)

    # Compute D values for in-sample, generated, and heldout inputs.
    self.G = self.generator(self.z)
    self.D, self.D_logits = self.discriminator(inputs)

    self.sampler = self.sampler(self.z)  # Samples that appear in /samples.
    self.d_grid = self.discriminator(self.grid, reuse=True)  # Eval d_loss on grid.

    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.D_heldout, self.D_logits_heldout = self.discriminator(heldout_inputs, reuse=True)

    # Make histogram summaries of D values.
    self.d_heldout_sum = histogram_summary("d_heldout", self.D_heldout)
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)

    # Make summaries of G.
    self.G_sum = image_summary("G", self.G)


    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)


    # Compute D loss for heldout, in-sample, and generated inputs.
    self.d_loss_real_heldout = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_heldout, tf.ones_like(self.D_heldout)))
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_gen = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

    # TODO: Test d on its own, by defining d_loss as only related to real data.
    # Then check heatmaps to see if high prob areas are over true data.
    self.d_loss = self.d_loss_real + self.d_loss_gen
    self.d_loss_heldout = self.d_loss_real_heldout + self.d_loss_gen

    # Compute G loss.
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    # Make scalar summaries of components of D losses.
    self.d_loss_real_heldout_sum = scalar_summary("d_loss_real_heldout", self.d_loss_real_heldout)
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_gen_sum = scalar_summary("d_loss_gen", self.d_loss_gen)

    # Make scalar summaries of total D and G losses.
    self.d_loss_heldout_sum =  scalar_summary("d_loss_heldout", self.d_loss_heldout)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if "discriminator" in var.name]
    self.g_vars = [var for var in t_vars if "generator" in var.name]

    self.saver = tf.train.Saver()


  def train(self, config):
    """Train SGAN"""
    # Organize outputs according to experiment name.
    self.sample_dir = "./samples/samples_"+self.expt_name
    self.checkpoint_dir = "./checkpoints/checkpoints_"+self.expt_name
    self.log_dir = "./logs/logs_"+self.expt_name

    if config.dataset == "Gaussian":
      data = self.load_gaussian()
    elif config.dataset == "ConcentricCircles":
      data = self.load_concentric_circles()
    elif config.dataset == "SwissRoll":
      pass
    else:
      raise ValueError("Choose dataset in ['ConcentricCircles', 'SwissRoll'].")


    # Split full data into training and heldout sets.
    heldout = np.asarray(data[:self.sample_num])
    training = np.asarray(data[self.sample_num:])
    in_sample = np.asarray(training[:self.sample_num])

    d_optim = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    # Removed self.G_sum to avoid error of not having 4-D tensor.
    #self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum,
    #    self.d_loss_gen_sum, self.g_loss_sum])
    self.g_sum = merge_summary([self.z_sum, self.d__sum,
        self.d_loss_gen_sum, self.g_loss_sum])
    self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum,
        self.d_loss_sum, self.d_loss_real_heldout_sum, self.d_loss_heldout_sum])
    self.writer = SummaryWriter(self.log_dir, self.sess.graph)

    # Generate z.
    if config.z_distr== "Uniform":
      sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    else:
      sample_z = np.random.normal(0, 1, size=(self.sample_num, self.z_dim))
    print("Made random z, size: {}".format(sample_z.shape))

    # Generate grid for evaluating discriminator loss.
    nx, ny = (20, 20)
    x_grid = np.linspace(-1, 1, nx)
    y_grid = np.linspace(-1, 1, ny)
    grid = np.asarray([[i, j] for i in x_grid for j in y_grid]) 

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      batch_idxs = min(len(training), config.train_size) // config.batch_size

      for idx in xrange(batch_idxs):
        batch_inputs = training[idx*config.batch_size:(idx+1)*config.batch_size]

        if config.z_distr== "Uniform":
          batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                .astype(np.float32)
        else:
          batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim])

        # Update D network
        for _ in range(config.d_per_iter):
            _, summary_str = self.sess.run([d_optim, self.d_sum],
              feed_dict={self.inputs: batch_inputs, self.z: batch_z,
                  self.heldout_inputs: heldout})
            self.writer.add_summary(summary_str, counter)

        # Update G network.
        for _ in range(config.g_per_iter):
            _, summary_str = self.sess.run([g_optim, self.g_sum],
              feed_dict={self.z: batch_z})
            self.writer.add_summary(summary_str, counter)
          
        errD_gen = self.d_loss_gen.eval({self.z: batch_z})
        errD_real = self.d_loss_real.eval({self.inputs: batch_inputs})
        errD_real_heldout = self.d_loss_real_heldout.eval(
                {self.inputs: heldout,
                 self.heldout_inputs: heldout})
        errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %.1f, d_loss(gen, insample, heldout): (%.2f, %.2f, %.2f), g_loss: %.2f" \
          % (epoch, idx, batch_idxs, time.time() - start_time, errD_gen,
             errD_real, errD_real_heldout, errG))

        # Make plots from certain epochs.
        if np.mod(epoch, 10) == 0:
          try:
            # Run sampler and losses.
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: in_sample,
              },
            )

            # Run evaluation of discriminator on grid.
            d_grid = self.sess.run(
              [self.d_grid],
              feed_dict={
                  self.grid: grid,
              },
            )

            # Plot heatmap of discriminator on grid.
            fig, ax = plt.subplots()
            d_grid = np.reshape(d_grid[0][0], [nx, ny])
            im = ax.pcolormesh(x_grid, y_grid, d_grid)
            fig.colorbar(im)

            # Plot generated samples against true training data.
            ax.scatter([x[0] for x in training], [x[1] for x in training],
                    c="cornflowerblue", alpha=0.1, marker="+")
            ax.scatter([x[0] for x in samples], [x[1] for x in samples],
                    c="r", alpha=0.1)
            fig.savefig("sgan{}.png".format(epoch))
            plt.close(fig)
            
          #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 

          except:
            print("one pic error!...")

        # Save checkpoint for certain epochs.
        if np.mod(epoch, 100) == 0:
          self.save(self.checkpoint_dir, counter)

    # Email results from all runs.
    outputs = natural_sort(glob("./sgan*.png")) 
    attachments = " "
    for o in outputs:
        attachments += " -a {}".format(o)
    
    os.system(('echo $PWD | mutt -s "sgan: {}" momod@utexas.edu {}').format(
        self.get_config_summary(config), attachments))

  def discriminator(self, candidates, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      
      spec = self.d_spec
      hidden_dims = [d.strip() for d in spec.split(",")]
      assert len(hidden_dims) > 0, "Spec for architecture not properly defined, e.g. '8,4,2'."
      assert all([d.isdigit() for d in hidden_dims]), "Dims split on ',' and must be ints. Incorrect d_spec: '{}'".format(spec)

      # Define batch norm objects.
      bn_objects = []
      for i, dim in enumerate(hidden_dims):
        bn_objects.append(batch_norm(name="discriminator_bn{}".format(i)))

      # Define graph according to spec.
      current_layer = candidates
      for i, dim in enumerate(hidden_dims):
        bn = bn_objects[i]
        next_layer = bn(tf.layers.dense(inputs=current_layer, units=dim,
            activation=tf.nn.relu))
        current_layer = next_layer
      h_last = self.d_bn_last(tf.layers.dense(inputs=current_layer, units=1,
          activation=tf.nn.relu))
      return tf.nn.sigmoid(h_last), h_last


  def generator(self, z):
    with tf.variable_scope("generator") as scope:
      h0 = self.g_bn0(tf.layers.dense(inputs=z, units=2, activation=tf.nn.relu))
      h1 = self.g_bn1(tf.layers.dense(inputs=h0, units=4, activation=tf.nn.relu))
      h2 = self.g_bn2(tf.layers.dense(inputs=h1, units=8, activation=tf.nn.relu))
      h3 = self.g_bn3(tf.layers.dense(inputs=h2, units=2, activation=tf.nn.relu))
      return tf.nn.tanh(h3)
      #return tf.nn.sigmoid()
      #return h3


  def sampler(self, z):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      h0 = self.g_bn0(tf.layers.dense(inputs=z, units=2, activation=tf.nn.relu),
              train=False)
      h1 = self.g_bn1(tf.layers.dense(inputs=h0, units=4, activation=tf.nn.relu),
              train=False)
      h2 = self.g_bn2(tf.layers.dense(inputs=h1, units=8, activation=tf.nn.relu),
              train=False)
      h3 = self.g_bn3(tf.layers.dense(inputs=h2, units=2, activation=tf.nn.relu),
              train=False)
      return tf.nn.tanh(h3)
      #return h3


  def get_config_summary(self, config):
    summary_items = [
      ["dataset", config.dataset],
      ["d_learning_rate", config.d_learning_rate],
      ["g_learning_rate", config.g_learning_rate],
      ["d_spec", config.d_spec],
      ["g_spec", config.g_spec],
      ["epoch", config.epoch],
      ["batch_size", config.batch_size],
      ["z_dim", config.z_dim],
      ["z_distr", config.z_distr],
      ["expt_name", config.expt_name],
    ]
    summary_string = ""
    for item in summary_items:
      summary_string += "{}: {}, ".format(item[0], item[1])
    return summary_string


  def load_gaussian(self):
    """Sample from a Gaussian."""
    n = 500
    variance = 0.01

    points = np.random.multivariate_normal([0, 0],
            [[variance, 0], [0, variance]], n)

    points = np.asarray(points)
    return points 


  def load_concentric_circles(self):
    """Sample from two concentric circles."""
    n = 500
    r1 = 0.2
    r2 = 0.7


    def sample_n_from_circle_radius_r(n, r):
      sample = []
      for _ in xrange(n):
        angle = 2 * math.pi * random.random()
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        sample.append([x,y])
      return sample


    points = (sample_n_from_circle_radius_r(n, r1) +
              sample_n_from_circle_radius_r(n, r2))
    points = np.asarray(points)
    return points 


  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.dataset_name, self.batch_size, self.z_dim)
      

  def save(self, checkpoint_dir, step):
    model_name = "SGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)


  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer(r"(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
