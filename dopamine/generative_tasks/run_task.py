# coding=utf-8
# Copyright 2018 The Dopamine Authors.
# Modifications copyright 2019 Kacper Kielak
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
"""Module defining classes and helper methods for general agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import io
import os
import time

from PIL import Image
import gin.tf
import numpy as np
import tensorflow as tf

from dopamine.generative_tasks.gen_lib import load_data
from dopamine.generators import dummy_generator
from dopamine.generators.gan import gan
from dopamine.generators.regressor import regressor
from dopamine.utils import checkpointer
from dopamine.utils import iteration_statistics
from dopamine.utils import logger


def dict_to_str(d):
  return ', '.join([f'{k}: {v}' for k, v in d.items()])


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_generator(sess, data_to_generate, inputs, generator_name=None,
                     summary_writer=None, debug_mode=False):
  """Creates a generator.

  Args:
    sess: A `tf.Session` object for running associated ops.
    data_to_generate: numpy array, data that the generator should learn to
        reproduce.
    inputs: numpy array, conditional input that the generator should receive.
      None if generator should not be conditioned on anything.
    generator_name: str, name of the generator to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    generator: A generator.

  Raises:
    ValueError: If `generator_name` is not in supported list.
  """
  assert generator_name is not None
  if not debug_mode:
    summary_writer = None
  if generator_name == 'dummy':
    return dummy_generator.DummyGenerator(data_to_generate.shape[1:])
  elif generator_name == 'regressor':
    assert inputs is not None
    return regressor.Regressor(sess,
                               inputs.shape[1:],
                               data_to_generate.shape[1:],
                               summary_writer=summary_writer)
  elif generator_name == 'vgan':
    input_shape = None if inputs is None else inputs.shape[1:]
    return gan.VanillaGAN(sess,
                          data_to_generate.shape[1:],
                          conditional_input_shape=input_shape,
                          summary_writer=summary_writer)
  else:
    raise ValueError('Unknown generator: {}'.format(generator_name))


@gin.configurable
class Runner(object):
  """Object that handles running generative tasks."""

  def __init__(self,
               base_dir,
               data_load_fn=load_data,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250,
               batch_size=100,
               evaluation_inputs=None,
               evaluation_size=None):
    """Initialize the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      data_load_fn: function that returns data as a tuple (inputs, outputs).
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      logging_file_prefix: str, prefix to use for the log files.
      log_every_n: int, the frequency for writing logs.
      num_iterations: int, the iteration number threshold (must be greater than
        start_iteration).
      training_steps: int, the number of training steps to perform.
      batch_size: int, batch size used for the training.
      evaluation_inputs: tuple of inputs to the generator that can be used
        during qualitative evaluation. If None, inputs set passed above will
        be used.
      evaluation_size: int, the number of images that should be generated
        randomly sampling from the data specified in evaluation_inputs. If
        None, all evaluation_inputs are generated.

    This constructor will take the following actions:
    - Initialize a `tf.Session`.
    - Initialize a logger.
    - Initialize a generator.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    """
    assert base_dir is not None
    inputs, data_to_generate = data_load_fn()
    assert inputs is None or inputs.shape[0] == data_to_generate.shape[0]
    assert evaluation_inputs is None or \
           evaluation_inputs.shape[1:] == inputs.shape[1:]
    assert evaluation_inputs is not None or evaluation_size is not None, \
           'Either evaluation_inputs or evaluation_size has to be initialised.'

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._data_to_generate = data_to_generate
    self._inputs = inputs
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._batch_size = batch_size
    self._evaluation_inputs = evaluation_inputs
    if self._evaluation_inputs is None:
      self._evaluation_inputs = inputs
    self._evaluation_size = evaluation_size
    self._base_dir = base_dir
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple workers on the same GPU.
    config.gpu_options.allow_growth = True
    # Set up a session and initialize variables.
    self._sess = tf.Session('', config=config)
    self._generator = create_generator(self._sess, data_to_generate, inputs,
                                       summary_writer=self._summary_writer)
    self._summary_writer.add_graph(graph=tf.get_default_graph())
    self._sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

  def _create_directories(self):
    """Create necessary sub-directories."""
    self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
    self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'))

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the generator is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    self._start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = \
      checkpointer.get_latest_checkpoint_number(self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = \
        self._checkpointer.load_checkpoint(latest_checkpoint_version)
      if self._generator.unbundle(self._checkpoint_dir,
                                  latest_checkpoint_version,
                                  experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        tf.logging.info('Reloaded checkpoint and will start from iteration %d',
                        self._start_iteration)

  def _run_train_phase(self, statistics):
    """Run training phase.

    Args:
      statistics: `IterationStatistics` object which records the experimental
        results. Note - This object is modified by this method.
    """
    start_time = time.time()
    mean_statistics = collections.defaultdict(int)
    for i in range(self._training_steps):
      batch_indices = np.random.randint(self._data_to_generate.shape[0],
                                        size=self._batch_size)
      batch_data = self._data_to_generate[batch_indices, :]
      batch_inputs = self._batch_size
      if self._inputs is not None:
        batch_inputs = self._inputs[batch_indices, :]

      batch_statistics = self._generator.train(batch_inputs, batch_data)
      for k, v in batch_statistics.items():
        mean_statistics[f'mean_{k}'] += v / self._training_steps

    statistics.append(mean_statistics)
    time_delta = time.time() - start_time
    tf.logging.info('Average statistics per training: %s',
                    dict_to_str(mean_statistics))
    tf.logging.info('Average training steps per second: %.2f',
                    self._training_steps / time_delta)

  def _run_eval_phase(self):
    """Run evaluation phase.

    Returns:
      Evaluation data generated by the generator.
    """
    # Perform the evaluation phase -- no learning.
    if self._evaluation_inputs is None:
      return self._generator.generate(self._evaluation_size)
    if self._evaluation_size is None:
      return self._generator.generate(self._evaluation_inputs)

    indices = np.random.randint(self._evaluation_inputs.shape[0],
                                size=self._evaluation_size)
    inputs = self._evaluation_inputs[indices, :]
    return self._generator.generate(inputs)

  def _run_one_iteration(self, iteration):
    """Runs one iteration of training/testing.

    Args:
      iteration: int, current iteration number, used as a global_step for saving
        Tensorboard summaries.

    Returns:
      A dict containing summary statistics for this iteration.
    """
    statistics = iteration_statistics.IterationStatistics()
    tf.logging.info('Starting iteration %d', iteration)
    self._run_train_phase(statistics)
    generated_data = self._run_eval_phase()
    self._save_tensorboard_summaries(iteration, generated_data)
    return statistics.data_lists

  def _save_tensorboard_summaries(self, iteration, generated_data):
    """Save evaluation summaries.

    Args:
      iteration: int, The current iteration number.
      generated_data: numpy array representing data saved during evaluation
        phase.
    """
    summaries = []
    for i, d in enumerate(generated_data):
      height, width, channel = d.shape
      if channel == 1:
        d = np.reshape(d, d.shape[0:2])

      d = (d + 1) / 2  # Move pixel values from [-1, 1] to [0, 1] range
      d = np.clip(d * 255., 0., 255.)  # Scale image to the 0-255 range
      data_image = Image.fromarray(np.uint8(d))
      output = io.BytesIO()
      data_image.save(output, format='PNG')
      data_string = output.getvalue()
      output.close()
      img_sum = tf.Summary.Image(height=height,
                                 width=width,
                                 colorspace=channel,
                                 encoded_image_string=data_string)
      summaries.append(tf.Summary.Value(tag=f'Eval/GeneratedData/{i}',
                                        image=img_sum))

    self._summary_writer.add_summary(tf.Summary(value=summaries), iteration)


  def _log_experiment(self, iteration, statistics):
    """Records the results of the current iteration.

    Args:
      iteration: int, iteration number.
      statistics: `IterationStatistics` object containing statistics to log.
    """
    self._logger['iteration_{:d}'.format(iteration)] = statistics
    if iteration % self._log_every_n == 0:
      self._logger.log_to_file(self._logging_file_prefix, iteration)

  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = \
      self._generator.bundle_and_checkpoint(self._checkpoint_dir, iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def run_task(self):
    """Runs a full task, spread over multiple iterations."""
    tf.logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      tf.logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, self._start_iteration)
      return

    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = self._run_one_iteration(iteration)
      self._log_experiment(iteration, statistics)
      self._checkpoint_experiment(iteration)
