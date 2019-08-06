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
"""Generation-specific utilities.

Some network specifications specific to certain generative tasks are provided
here.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os



from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_labels
import numpy as np
import tensorflow as tf

import gin.tf


MNIST_RANGE = np.array([-1, 1])
gin.constant('gen_lib.MNIST_EVALUATION_INPUTS', np.eye(10))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
MNIST_IMGS_PATH = os.path.join(DATASETS_DIR, 'mnist_imgs.gz')
MNIST_LABELS_PATH = os.path.join(DATASETS_DIR, 'mnist_labels.gz')


@gin.configurable
def load_data(task_name=None):
  """Loads data associated with given task

  Args:
    task_name: str, the name of the task to load.

  Returns:
    Tuple of numpy arrays (inputs, data_to_generate) normalized between 0 and 1.
  """
  assert task_name is not None

  if task_name == 'mnist':
    with open(MNIST_IMGS_PATH, 'rb') as imgs_file:
      data = extract_images(imgs_file)
    inputs = None
  elif task_name == 'cmnist':
    with open(MNIST_IMGS_PATH, 'rb') as imgs_file:
      data = extract_images(imgs_file)
    with open(MNIST_LABELS_PATH, 'rb') as labels_file:
      inputs = extract_labels(labels_file)
    inputs = np.eye(10)[inputs]
  else:
    raise ValueError('Unknown task: {}'.format(task_name))

  data = (data - data.min()) / (data.max() - data.min())  # Normalize
  data = (data * 2) - 1  # Move to range [-1, 1]
  if inputs is not None:
    inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())  # Normalize
    inputs = (inputs * 2) - 1  # Move to range [-1, 1]

  return inputs, data


@gin.configurable
def mnist_regressor_mlp(inputs, output_shape, network_size=None):
  """Builds a basic network for generation tasks, rescaling inputs to [-1, 1].

  Args:
    inputs: `tf.Tensor`, the network input.
    output_shape: tuple of ints representing dimensions of the output
    network_size: tuple of ints representing dimensions of the network.

  Returns:
    The tensor containing generated data.
  """
  if network_size is None:
    network_size = (256, 512, 1024)

  net = tf.cast(inputs, tf.float32)
  net = tf.contrib.slim.flatten(net)
  for layer in network_size:
    net = tf.contrib.slim.fully_connected(net, layer)
  output_size = np.prod(output_shape).item()
  net = tf.contrib.slim.fully_connected(net, output_size,
                                        activation_fn=tf.nn.tanh)
  return tf.reshape(net, [-1, *output_shape])
