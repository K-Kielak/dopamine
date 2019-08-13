# coding=utf-8
# Copyright 2019 Kacper Kielak
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of a vanilla Generative Adversarial Net as introduced in
Goodfellow et al. (2014)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os



from dopamine.generators.abstract_generator import AbstractGenerator
from dopamine.generative_tasks import gen_lib
import tensorflow as tf

import gin.tf


GENERATOR_SCOPE = 'generator'
DISCRIMINATOR_SCOPE = 'discriminator'


@gin.configurable
class VanillaGAN(AbstractGenerator):

  def __init__(self,
               sess,
               output_shape,
               processing_dtype=tf.float32,
               conditional_input_shape=None,
               noise_shape=(100,),
               generator_network_fn=gen_lib.mnist_generator_gan,
               discriminator_network_fn=gen_lib.mnist_discriminator_gan,
               tf_device='/cpu:*',
               max_tf_checkpoints_to_keep=4,
               g_optimizer=tf.train.AdamOptimizer(),
               d_optimizer=tf.train.AdamOptimizer(),
               k=1,
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):
    """Initializes GANs and constructs the components of their graph.

    Args:
      sess: `tf.Session`, for executing ops.
      output_shape: tuple of ints describing the output shape.
      processing_dtype: tf.DType, specifies the type used to processing data.
        Note that it should be some type of float (e.g. tf.float32 or tf.float64).
      conditional_input_shape: tuple of ints describing the conditional input
        shape. If None, no conditional input will be provided.
      generator_network_fn: function expecting three parameters:
        (noise, conditional_input, output_shape).
        This function will returnv the object containing the tensors output by
        the generator network.
      discriminator_network_fn: function expecting three parameters:
        (conditional_input, output). This function will return
        the object containing the tensor output by the discriminator network,
        and the tensor containing its logit.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow
        checkpoints to keep.
      g_optimizer: `tf.train.Optimizer`, for training the generator.
      d_optimizer: `tf.train.Optimizer`, for training the discriminator.
      k: int, number of iterations of the discriminator per generator iteration.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    tf.logging.info('Creating %s generator with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t tf_device: %s', tf_device)
    tf.logging.info('\t generator_optimizer: %s', g_optimizer)
    tf.logging.info('\t discriminator_optimizer: %s', d_optimizer)
    tf.logging.info('\t max_tf_checkpoints_to_keep: %d',
                    max_tf_checkpoints_to_keep)

    self.output_shape = output_shape
    self.processing_dtype = processing_dtype
    self.conditional_input_shape = conditional_input_shape
    self.noise_shape = noise_shape
    self.generator_network_fn = generator_network_fn
    self.discriminator_network_fn = discriminator_network_fn
    self.training_steps = 0
    self.g_optimizer = g_optimizer
    self.d_optimizer = d_optimizer
    self.k = k
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    with tf.device(tf_device):
      self._build_networks()
      self._generator_loss = self._define_generator_loss()
      self._discriminator_loss = self._define_discriminator_loss()
      self._build_generator_train_op()
      self._build_discriminator_train_op()
      self._create_summaries()

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.summary.merge_all()
    self._sess = sess
    self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)

  def _build_networks(self):
    # Define inputs
    if self.conditional_input_shape is None:
      self._input_ph = tf.placeholder(tf.int32, (), name='batch_size_ph')
      noise = tf.random.normal((self._input_ph, *self.noise_shape),
                               name='noise', dtype=self.processing_dtype)
      self._conditional_input = None
    else:
      self._input_ph = tf.placeholder(self.processing_dtype,
                                      (None, *self.conditional_input_shape),
                                      name='input_ph')
      noise = tf.random.normal((tf.shape(self._input_ph)[0], *self.noise_shape),
                               name='noise', dtype=self.processing_dtype)
      self._conditional_input = self._input_ph

    self._real_output_ph = tf.placeholder(self.processing_dtype,
                                          (None, *self.output_shape),
                                          name='output_ph')

    # Build networks
    with tf.variable_scope(GENERATOR_SCOPE):
      self._generator_outputs = self.generator_network_fn(
        noise, self._conditional_input, self.output_shape
      )
    with tf.variable_scope(DISCRIMINATOR_SCOPE):
      self._gen_discriminator_out = self.discriminator_network_fn(
        self._conditional_input, self._generator_outputs
      )
    with tf.variable_scope(DISCRIMINATOR_SCOPE, reuse=True):
      self._real_discriminator_out = self.discriminator_network_fn(
        self._conditional_input, self._real_output_ph
      )

  def _define_generator_loss(self):
    """Defines loss for the generator network.

    For vanilla GAN, generator loss is defined by:
      min log(1 - gen_discrimination) =~ (to improve gradient signal at the start)
      max log(gen_discrimination) = min -log(gen_discrimination)

    Returns: Tensor containing generator loss value.
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(self._gen_discriminator_out),
      logits=self._gen_discriminator_out,
    )
    return tf.reduce_mean(loss, name='generator_loss')

  def _define_discriminator_loss(self):
    """Defines loss for the discriminator network.

    For vanilla GAN, discriminator loss is defined by:
      max log(real_discrimination) + log(1 - gen_discrimination) =
      min -log(real_discrimination) - log(1 - gen_discrimination)

    Returns: Tensor containing discriminator loss value.
    """
    real_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(self._real_discriminator_out),
      logits=self._real_discriminator_out
    )
    real_d_loss = tf.reduce_mean(real_d_loss, name='real_discriminator_loss')
    gen_d_loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(self._gen_discriminator_out),
      logits=self._gen_discriminator_out
    )
    gen_d_loss = tf.reduce_mean(gen_d_loss, name='gen_discriminator_loss')
    return tf.add(real_d_loss, gen_d_loss, name='discrminator_loss')

  def _build_generator_train_op(self):
    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=GENERATOR_SCOPE)
    self._g_grads = self.g_optimizer.compute_gradients(
      self._generator_loss,
      var_list=generator_vars + self.g_optimizer.variables()
    )
    self._g_train_op = self.g_optimizer.apply_gradients(self._g_grads)

  def _build_discriminator_train_op(self):
    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=DISCRIMINATOR_SCOPE)
    self._d_grads = self.d_optimizer.compute_gradients(
      self._discriminator_loss,
      var_list=discriminator_vars + self.d_optimizer.variables()
    )
    self._d_train_op = self.d_optimizer.apply_gradients(self._d_grads)

  def _create_summaries(self):
    self._l1_loss = tf.abs(self._real_output_ph - self._generator_outputs)
    self._l1_loss = tf.reduce_mean(self._l1_loss)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('GeneratorLoss', self._generator_loss)
        tf.summary.scalar('DiscriminatorLoss', self._discriminator_loss)
        tf.summary.scalar('L1Loss', self._l1_loss)
      with tf.variable_scope('Discriminations'):
        real_discrimination = tf.nn.sigmoid(self._real_discriminator_out)
        real_discrimination = tf.reduce_mean(real_discrimination)
        tf.summary.scalar('RealDiscrimination', real_discrimination)
        gen_discrimination = tf.nn.sigmoid(self._gen_discriminator_out)
        gen_discrimination = tf.reduce_mean(gen_discrimination)
        tf.summary.scalar('GeneratedDiscrimination', gen_discrimination)
      with tf.variable_scope('Gradients'):
        [tf.summary.scalar(f'{var.name}_std', tf.math.reduce_std(grad))
         for grad, var in self._g_grads]
        [tf.summary.scalar(f'{var.name}_std', tf.math.reduce_std(grad))
         for grad, var in self._d_grads]

  def generate(self, input):
    """Generates data based on the received input.

    Args:
      input: numpy array, input based on which generator should generate
        output. Should be an integer specifying the batch size if the generator
        doesn't accept conditional inputs.

    Returns:
      numpy array, generated output.
    """
    assert isinstance(input, int) or \
           input.shape[1:] == self.conditional_input_shape
    return self._sess.run(self._generator_outputs, feed_dict={
      self._input_ph: input
    })

  def train(self, input, expected_output):
    """Performs one training step based on the received training batch.

    Args:
      input: numpy array, input to the generator's network. Should be an
        integer specifying the batch size if the generator doesn't accept
        conditional inputs.
      expected_output: numpy array, output that should be produced by the
        generator given input.

    Returns:
      dict, train statistics consisting of generator, discriminator, and L1
        loss.
    """
    assert (isinstance(input, int) and input == expected_output.shape[0]) or \
           (input.shape[1:] == self.conditional_input_shape and
            input.shape[0] == expected_output.shape[0])

    # Train discriminator
    _, g_loss, d_loss, l1_loss = self._sess.run(
      [self._d_train_op, self._generator_loss,
       self._discriminator_loss, self._l1_loss],
      feed_dict={self._input_ph: input, self._real_output_ph: expected_output})

    if self.training_steps % self.k == 0:
      self._sess.run(self._g_train_op, feed_dict={
        self._input_ph: input,
        self._real_output_ph: expected_output
      })

    if (self.summary_writer is not None and self.training_steps > 0 and
        self.training_steps % self.summary_writing_frequency == 0):
      summary = self._sess.run(self._merged_summaries, feed_dict={
        self._input_ph: input,
        self._real_output_ph: expected_output
      })
      self.summary_writer.add_summary(summary, self.training_steps)

    self.training_steps += 1
    return {
      'generator_loss': g_loss,
      'discriminator_loss': d_loss,
      'l1_loss': l1_loss
    }

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the generator's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.gfile.Exists(checkpoint_dir):
      return None
    # Call the Tensorflow saver to checkpoint the graph.
    self._saver.save(
        self._sess,
        os.path.join(checkpoint_dir, 'tf_ckpt'),
        global_step=iteration_number)
    return {'training_steps': self.training_steps}

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved by tf.Save.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.

    Returns:
      bool, True if unbundling was successful.
    """
    if bundle_dictionary is not None:
      for key in self.__dict__:
        if key in bundle_dictionary:
          self.__dict__[key] = bundle_dictionary[key]
    elif not self.allow_partial_reload:
      return False
    else:
      tf.logging.warning("Unable to reload the generator's parameters!")
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
