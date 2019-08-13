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
"""Implementation of a Wasserstein Generative Adversarial Net with Gradient
penalty as introduced in Gulrajani et al. (2017)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from dopamine.generators.wgan import wgan
from dopamine.generators.gan.gan import DISCRIMINATOR_SCOPE
from dopamine.generative_tasks import gen_lib
import tensorflow as tf

import gin.tf


@gin.configurable
class WassersteinGANGP(wgan.WassersteinGAN):

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
               penalty_coeff=10,
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
        (noise, conditional_input, output_shape). This function will return
        the object containing the tensors output by the generator network.
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
      penalty_coeff: float, coefficient specifying the importance of gradient
        penalty in the overall discriminator loss function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      allow_partial_reload: bool, whether we allow reloading a partial agent
        (for instance, only the network parameters).
    """
    self.penalty_coeff = penalty_coeff
    wgan.WassersteinGAN.__init__(self,
                                 sess,
                                 output_shape,
                                 processing_dtype=processing_dtype,
                                 conditional_input_shape=conditional_input_shape,
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
    tf.logging.info('\t penalty_coeff: %d', penalty_coeff)

  def _define_discriminator_loss(self):
    """Defines loss for the discriminator network.

    For wasserstein GAN with Gradient Penalty, discriminator loss is defined by:
      max (real_discrimination - gen_discrimination - gradient_penalty) =
      = min (gen_discrimination - real_discrimination + gradient_penalty)

    Returns: Tensor containing discriminator loss value.
    """
    # Calculate standard loss
    real_d_loss = tf.reduce_mean(self._real_discriminator_out)
    real_d_loss = tf.negative(real_d_loss, name='real_discriminator_loss')
    gen_d_loss = tf.reduce_mean(self._gen_discriminator_out,
                                name='gen_discriminator_loss')
    non_penalized_loss = tf.add(real_d_loss, gen_d_loss,
                                name='non_penalized_discrminator_loss')

    # Calculate gradient penalty
    differences = tf.subtract(self._generator_outputs,
                              self._real_output_ph,
                              name='differences')
    random_scaling = tf.random_uniform(
      shape=[tf.shape(self._real_output_ph)[0], *([1]*len(self.output_shape))],
      dtype=self.processing_dtype, minval=0, maxval=1
    )
    interpolates = tf.add(self._real_output_ph,
                          differences * random_scaling,
                          name='interpolates')

    with tf.variable_scope(DISCRIMINATOR_SCOPE, reuse=True):
      interpolates_discriminator_out = self.discriminator_network_fn(
        self._conditional_input, interpolates
      )

    interp_grads = tf.gradients(interpolates_discriminator_out, [interpolates],
                                name='interpolates_gradients')[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(interp_grads),
                                   reduction_indices=[1]), name='slopes')
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2,
                                      name='gradient_penalty')
    gradient_penalty = tf.scalar_mul(self.penalty_coeff, gradient_penalty)

    return tf.add(non_penalized_loss, gradient_penalty,
                  name='discriminator_loss')

  def _build_discriminator_train_op(self):
    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=DISCRIMINATOR_SCOPE)
    self._d_grads = self.d_optimizer.compute_gradients(
      self._discriminator_loss,
      var_list=discriminator_vars + self.d_optimizer.variables()
    )
    self._d_train_op = self.d_optimizer.apply_gradients(self._d_grads)
