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
"""Compact implementation of a DQN agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os



from dopamine.generators.abstract_generator import AbstractGenerator
from dopamine.generative_tasks import gen_lib
import tensorflow as tf

import gin.tf


@gin.configurable
class Regressor(AbstractGenerator):

  def __init__(self,
               sess,
               input_shape,
               output_shape,
               data_dtype,
               network_fn=gen_lib.mnist_regressor_mlp,
               tf_device='/cpu:*',
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.train.AdamOptimizer(),
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):
    """Initializes the generator and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      input_shape: tuple of ints describing the input shape
      output_shape: tuple of ints describing the output shape.
      data_dtype: tf.DType, specifies the type of the data. Note that
        if your inputs are continuous, you should set this to tf.float32.
      network_fn: function expecting three parameters:
        (input_shape, network_type, output_shape). This function will return
        the object containing the tensors output by the network.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow
        checkpoints to keep.
      optimizer: `tf.train.Optimizer`, for training the generator.
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
    tf.logging.info('\t optimizer: %s', optimizer)
    tf.logging.info('\t max_tf_checkpoints_to_keep: %d',
                    max_tf_checkpoints_to_keep)

    self.input_shape = input_shape
    self.output_shape = output_shape
    self.data_dtype = data_dtype
    self.network_fn = network_fn
    self.training_steps = 0
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    with tf.device(tf_device):
      # Build network
      self._input_ph = tf.placeholder(self.data_dtype,
                                      (None, *self.input_shape),
                                      name='input_ph')
      self._expected_output_ph = tf.placeholder(self.data_dtype,
                                                (None, *self.output_shape),
                                                name='input_ph')
      self._net_outputs = self.network_fn(self._input_ph, self.output_shape)

      # Build train op
      self._loss = tf.losses.absolute_difference(self._expected_output_ph,
                                                 self._net_outputs,
                                                 reduction=tf.losses.Reduction.NONE)
      self._loss = tf.reduce_mean(self._loss)
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('L1 Loss', self._loss)
      self._train_op = self.optimizer.minimize(self._loss)

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.summary.merge_all()
    self._sess = sess
    self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)

  def generate(self, input):
    """Generates data based on the received input

    Args:
      input: numpy array, input based on which generator should generate output.

    Returns:
      numpy array, generated output.
    """
    assert input.shape[1:] == self.input_shape, \
      f'Expected ({self.input_shape}) and received ' \
      f'({input.shape[1:]}), data shapes do not match'

    return self._sess.run(self._net_outputs, feed_dict={
      self._input_ph: input
    })

  def train(self, input, expected_output):
    """Performs one training step based on the received training batch.

    Args:
      input: numpy array, input to the generator's network.
      expected_output: numpy array, output that should be produced by the
        generator given input.

    Returns:
      dict, train statistics consisting of loss only.
    """
    assert input.shape[0] == expected_output.shape[0]
    loss, _ = self._sess.run([self._loss, self._train_op], feed_dict={
      self._input_ph: input,
      self._expected_output_ph: expected_output
    })

    if (self.summary_writer is not None and self.training_steps > 0 and
       self.training_steps % self.summary_writing_frequency == 0):
      summary = self._sess.run(self._merged_summaries, feed_dict={
        self._input_ph: input,
        self._expected_output_ph: expected_output
      })
      self.summary_writer.add_summary(summary, self.training_steps)

    self.training_steps += 1
    return {'loss': loss}

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
      tf.logging.warning("Unable to reload the agent's parameters!")
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
