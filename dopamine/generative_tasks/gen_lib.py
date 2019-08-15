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

  net = tf.contrib.slim.flatten(inputs[0])
  net = tf.contrib.slim.fully_connected(net, network_size[0], activation_fn=None)
  for inp in inputs[1:]:
    cond_net = tf.contrib.slim.flatten(inp)
    cond_net = tf.contrib.slim.fully_connected(cond_net, network_size[0],
                                               activation_fn=None)
    net = net + cond_net

  net = tf.nn.relu(net)
  for layer in network_size[1:]:
    net = tf.contrib.slim.fully_connected(net, layer)

  output_size = np.prod(output_shape).item()
  net = tf.contrib.slim.fully_connected(net, output_size,
                                        activation_fn=tf.nn.tanh)
  return tf.reshape(net, [-1, *output_shape])


@gin.configurable
def mnist_generator_gan(noise, conditional_inputs, output_shape,
                        network_size=None, batch_norm=False):
  """Builds a basic network for generation tasks, rescaling inputs to [-1, 1].

  Args:
    noise: `tf.Tensor`, the network noise input.
    conditional_inputs: tuple of tf.Tensors, the network conditional input.
    output_shape: tuple of ints representing dimensions of the output
    network_size: tuple of ints representing dimensions of the network.
    batch_norm: boolean, specifies if batch normalization should be used.

  Returns:
    The tensor containing generated data.
  """
  if network_size is None:
    network_size = (256, 512, 1024)

  assert len(network_size) > 0

  initializer = tf.initializers.truncated_normal(mean=0, stddev=1e-3)
  normalizer_fn = tf.layers.batch_normalization if batch_norm else None
  net = tf.contrib.slim.flatten(noise)
  net = tf.contrib.slim.fully_connected(net, network_size[0],
                                        activation_fn=None,
                                        weights_initializer=initializer,
                                        biases_initializer=initializer)
  for inp in conditional_inputs:
    cond_net = tf.contrib.slim.flatten(inp)
    cond_net = tf.contrib.slim.fully_connected(cond_net, network_size[0],
                                               activation_fn=None,
                                               weights_initializer=initializer,
                                               biases_initializer=initializer)
    net = net + cond_net

  if batch_norm:
    net = normalizer_fn(net)
  net = tf.nn.leaky_relu(net)
  for layer in network_size[1:]:
    net = tf.contrib.slim.fully_connected(net, layer,
                                          activation_fn=tf.nn.leaky_relu,
                                          normalizer_fn=normalizer_fn,
                                          weights_initializer=initializer,
                                          biases_initializer=initializer)
  output_size = np.prod(output_shape).item()
  net = tf.contrib.slim.fully_connected(net, output_size,
                                        activation_fn=tf.nn.tanh,
                                        weights_initializer=initializer,
                                        biases_initializer=initializer)
  return tf.reshape(net, [-1, *output_shape])


@gin.configurable
def mnist_discriminator_gan(conditional_inputs, output, network_size=None,
                            dropout_keep_prob=0.8, batch_norm=False):
  """Builds a basic network for generation tasks, rescaling inputs to [-1, 1].

  Args:
    conditional_inputs: tuple of tf.Tensors, the network conditional input.
    output: `tf.Tensor`, tensor containing data to discriminate.
    network_size: tuple of ints representing dimensions of the network.
    dropout_keep_prob: float, probability of keeping the element in dropout
      layers.
    batch_norm: boolean, specifies if batch normalization should be used.

  Returns:
    The tensor containing logit of the discrimination (before sigmoid).
  """
  if network_size is None:
    network_size = (1024, 512, 256)

  assert len(network_size) > 0
  assert 0. < dropout_keep_prob <= 1

  initializer = tf.initializers.truncated_normal(mean=0, stddev=1e-3)
  normalizer_fn = tf.layers.batch_normalization if batch_norm else None
  net = tf.contrib.slim.flatten(output)
  net = tf.contrib.slim.fully_connected(net, network_size[0],
                                        activation_fn=None,
                                        weights_initializer=initializer,
                                        biases_initializer=initializer)
  for inp in conditional_inputs:
    cond_net = tf.contrib.slim.flatten(inp)
    cond_net = tf.contrib.slim.fully_connected(cond_net, network_size[0],
                                               activation_fn=None,
                                               weights_initializer=initializer,
                                               biases_initializer=initializer)
    net = net + cond_net

  net = tf.nn.leaky_relu(net)
  for layer in network_size[1:]:
    net = tf.contrib.slim.dropout(net, keep_prob=dropout_keep_prob)
    net = tf.contrib.slim.fully_connected(net, layer,
                                          activation_fn=tf.nn.leaky_relu,
                                          normalizer_fn=normalizer_fn,
                                          weights_initializer=initializer,
                                          biases_initializer=initializer)

  net = tf.contrib.slim.dropout(net, keep_prob=dropout_keep_prob)
  output_size = 1
  logit = tf.contrib.slim.fully_connected(net, output_size,
                                          activation_fn=None,
                                          weights_initializer=initializer,
                                          biases_initializer=initializer)
  return logit
