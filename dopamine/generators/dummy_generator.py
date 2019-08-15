# coding=utf-8
# Copyright 2019 Kacper Kielak.
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
"""Dummy generator used for test purposes"""



from dopamine.generators.abstract_generator import AbstractGenerator
import numpy as np
import tensorflow as tf


class DummyGenerator(AbstractGenerator):
  """An implementation of the dummy generator"""

  def __init__(self, output_shape):
    """Initializes the generator

    Args:
      output_shape: tuple of ints specifying expected output shape.
    """
    tf.logging.info('Creating %s', self.__class__.__name__)
    self._output_shape = output_shape

  def generate(self, inputs):
    """Generates data based on the received input.

    Args:
      inputs: tuple of numpy arrays, input based on which generator should
        generate output.

    Returns:
      numpy array, randomly initialised values in the appropriate shape.
    """
    return np.random.rand(*self._output_shape)

  def train(self, inputs, expected_output):
    """Pretends to train the generator.

    Args:
      inputs: tuple of numpy arrays, input to the generator's network.
      expected_output: numpy array, output that should be produced by the
        generator given input.

    Returns:
      dict, empty train statistics.
    """
    return {}

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Pretends to checkpoint the generator.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects would be saved.
      iteration_number: int, iteration number that would be used for naming
      the checkpoint file.

    Returns:
      An empty dict. If the checkpoint directory does not exist, returns None.
    """
    if not tf.gfile.Exists(checkpoint_dir):
      return None

    return {}

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Pretends to restore the generator from a checkpoint.

    Args:
      checkpoint_dir: str, path to the checkpoint.
      iteration_number: int, checkpoint version.
      bundle_dictionary: dict, containing additional Python objects owned by
      the agent.

    Returns:
      bool, True (dummy unbundling is always successful).
    """
    return True
