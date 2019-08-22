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
"""Implementation of a Wasserstein Generative Adversarial Net as introduced in
Arjovsky et al. (2017)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


from dopamine.generators.gan import gan
from dopamine.generators.gan.gan import DISCRIMINATOR_SCOPE
from dopamine.generative_tasks import gen_lib
import tensorflow as tf

import gin.tf


@gin.configurable
class WassersteinGAN(gan.VanillaGAN):

  def __init__(self,
               sess,
               output_shape,
               processing_dtype=tf.float32,
               conditional_input_shapes=None,
               noise_shape=(100,),
               generator_network_fn=gen_lib.mnist_generator_gan,
               discriminator_network_fn=gen_lib.mnist_discriminator_gan,
               tf_device='/cpu:*',
               max_tf_checkpoints_to_keep=4,
               g_optimizer=tf.train.AdamOptimizer(),
               d_optimizer=tf.train.AdamOptimizer(),
               k=1,
               weights_clip=0.01,
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):
    """Initializes GANs and constructs the components of their graph.

    Args:
      sess: `tf.Session`, for executing ops.
      output_shape: tuple of ints describing the output shape.
      processing_dtype: tf.DType, specifies the type used to processing data.
        Note that it should be some type of float (e.g. tf.float32 or tf.float64).
      conditional_input_shapes: tuple of tuples of ints describing the
        conditional input shapes (there may be more than one input). None in
        case of no conditional inputs.
      generator_network_fn: function expecting three parameters:
        (noise, conditional_inputs, output_shape). This function will return
        the object containing the tensors output by the generator network.
      discriminator_network_fn: function expecting three parameters:
        (conditional_inputs, output). This function will return
        the object containing the tensor output by the discriminator network,
        and the tensor containing its logit.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow
        checkpoints to keep.
      g_optimizer: `tf.train.Optimizer`, for training the generator.
      d_optimizer: `tf.train.Optimizer`, for training the discriminator.
      k: int, number of iterations of the discriminator per generator iteration.
      weights_clip: float, maximum absolute value that any weight in the
        discriminator network can reach.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    assert weights_clip > 0
    self.weights_clip = weights_clip
    gan.VanillaGAN.__init__(self,
                            sess,
                            output_shape,
                            processing_dtype=processing_dtype,
                            conditional_input_shapes=conditional_input_shapes,
                            noise_shape=noise_shape,
                            generator_network_fn=generator_network_fn,
                            discriminator_network_fn=discriminator_network_fn,
                            tf_device=tf_device,
                            max_tf_checkpoints_to_keep=max_tf_checkpoints_to_keep,
                            g_optimizer=g_optimizer,
                            d_optimizer=d_optimizer,
                            k=k,
                            summary_writer=summary_writer,
                            summary_writing_frequency=summary_writing_frequency,
                            allow_partial_reload=allow_partial_reload)
    tf.logging.info('\t weights_clip: %d', weights_clip)

  def _define_generator_loss(self):
    """Defines loss for the generator network.

    For wassserstein GAN, generator loss is defined by:
      max gen_discrimination = min -gen_discrimination

    Returns: Tensor containing generator loss value.
    """
    loss = tf.reduce_mean(self._gen_discriminator_out)
    return tf.negative(loss, name='generator_loss')

  def _define_discriminator_loss(self):
    """Defines loss for the discriminator network.

    For wasserstein GAN, discriminator loss is defined by:
      max (real_discrimination - gen_discrimination) =
      = min (gen_discrimination - real_discrimination)

    Returns: Tensor containing discriminator loss value.
    """
    real_d_loss = tf.reduce_mean(self._real_discriminator_out)
    real_d_loss = tf.negative(real_d_loss, name='real_discriminator_loss')
    gen_d_loss = tf.reduce_mean(self._gen_discriminator_out,
                                name='gen_discriminator_loss')
    return tf.add(real_d_loss, gen_d_loss, name='discrminator_loss')

  def _build_discriminator_train_op(self):
    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=DISCRIMINATOR_SCOPE)
    self._d_grads = self.d_optimizer.compute_gradients(
      self._discriminator_loss,
      var_list=discriminator_vars + self.d_optimizer.variables()
    )
    loss_minimization = self.d_optimizer.apply_gradients(self._d_grads)
    with tf.get_default_graph().control_dependencies([loss_minimization]):
      clip_ops = []
      for param in discriminator_vars:
        clip = tf.clip_by_value(param,
                                clip_value_min=-self.weights_clip,
                                clip_value_max=self.weights_clip)
        clip_ops.append(tf.assign(param, clip))

    # Group training step ops (minimization + clipping)
    self._d_train_op = tf.group(loss_minimization, *clip_ops)

  def _create_summaries(self):
    self._l1_loss = tf.abs(self._real_output_ph - self._generator_outputs)
    self._l1_loss = tf.reduce_mean(self._l1_loss)
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        self._summaries += [
          tf.summary.scalar('GeneratorLoss', self._generator_loss),
          tf.summary.scalar('DiscriminatorLoss', self._discriminator_loss),
          tf.summary.scalar('L1Loss', self._l1_loss)
        ]
      with tf.variable_scope('Gradients'):
        self._summaries += [tf.summary.scalar(f'{var.name}_std',
                                              tf.math.reduce_std(grad))
                            for grad, var in self._g_grads]
        self._summaries += [tf.summary.scalar(f'{var.name}_std',
                                              tf.math.reduce_std(grad))
                            for grad, var in self._d_grads]
