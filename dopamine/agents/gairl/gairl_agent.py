# coding=utf-8
# Copyright 2018 Kacper Kielak.
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
"""Implementation of the GAIRL framework"""

import collections
import os
import random
import time



from dopamine.agents.abstract_agent import AbstractAgent
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
from dopamine.generators import dummy_generator
from dopamine.generators.gan import gan
from dopamine.generators.regressor import regressor
from dopamine.generators.wgan import wgan
from dopamine.generators.wgan_gp import wgan_gp
from dopamine.replay_memory import circular_replay_buffer
import numpy as np
import tensorflow as tf

import gin.tf


AGENT_APPENDIX = '@a'
STATE_APPENDIX = '@s'
REWTERM_APPENDIX = '@r'
AGENT_SUBDIR = 'agent'
STATE_SUBDIR = 'state'
REWTERM_SUBDIR = 'rewterm'
TRAIN_MEM_SUBDIR = 'train_mem'
TEST_MEM_SUBDIR = 'test_mem'


def dict_to_str(d):
  return ', '.join([f'{k}: {v}' for k, v in d.items()])


def _calculate_classification_statistics(output, target):
  output = np.round(np.clip(output, 0, 1))
  target = np.round(np.clip(target, 0, 1))

  true_positives = np.sum(output * target)
  if true_positives == 0:
    return 0., 0., 0.

  precision = true_positives / np.sum(output)
  recall = true_positives / np.sum(target)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall


@gin.configurable
def create_agent(sess, agent_name, num_actions,
                 observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=atari_lib.NATURE_DQN_DTYPE,
                 stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
                 summary_writer=None):
  """Creates an agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    agent_name: str, name of the agent to create.
    num_actions: int, number of actions the agent can take at any state.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    observation_shape: tuple of ints describing the observation shape.
    observation_dtype: tf.DType, specifies the type of the observations. Note
      that if your inputs are continuous, you should set this to tf.float32.
    stack_size: int, number of frames to use in state stack.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list or one of the
      GAIRL submodules is not in supported list when the chosen agent is GAIRL.
  """
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(
      sess, num_actions, observation_shape=observation_shape,
      observation_dtype=observation_dtype, stack_size=stack_size,
      summary_writer=summary_writer
    )
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
      sess, num_actions, observation_shape=observation_shape,
      observation_dtype=observation_dtype, stack_size=stack_size,
      summary_writer=summary_writer
    )
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
      sess, num_actions, summary_writer=summary_writer
    )
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


@gin.configurable
def create_generator(sess, generator_name, output_shape,
                     input_shapes=None, summary_writer=None):
  """Creates a generator.

  Args:
    sess: A `tf.Session` object for running associated ops.
    generator_name: str, name of the generator to create.
    output_shape: tuple of ints describing the output shape.
    input_shapes: tuple of tuples of ints describing input shapes (there may
      be more than one input).
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    generator: A generator.

  Raises:
    ValueError: If `generator_name` is not in supported list.
  """
  assert generator_name is not None
  if generator_name == 'dummy':
    return dummy_generator.DummyGenerator(output_shape)
  elif generator_name == 'regressor':
    return regressor.Regressor(sess, input_shapes, output_shape,
                               summary_writer=summary_writer)
  elif generator_name == 'vgan':
    return gan.VanillaGAN(sess, output_shape,
                          conditional_input_shapes=input_shapes,
                          summary_writer=summary_writer)
  elif generator_name == 'wgan':
    return wgan.WassersteinGAN(sess, output_shape,
                               conditional_input_shapes=input_shapes,
                               summary_writer=summary_writer)
  elif generator_name == 'wgan_gp':
    return wgan_gp.WassersteinGANGP(sess, output_shape,
                                    conditional_input_shapes=input_shapes,
                                    summary_writer=summary_writer)
  else:
    raise ValueError('Unknown generator: {}'.format(generator_name))


@gin.configurable
class GAIRLAgent(AbstractAgent):
  """An implementation of the GAIRL agent."""

  def __init__(self,
               sess,
               num_actions,
               rl_agent_name='dqn',
               state_gen_name='wgan_gp',
               rewterm_gen_name='regressor',
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               model_free_length=10000,
               model_learning_length=50000,
               model_learning_logging_frequency=100,
               model_based_length=50000,
               train_memory_capacity=40000,
               test_memory_capacity=10000,
               memory_batch_size=256,
               summary_writer=None,
               eval_mode=False):
    """Initializes the agent by combining generative models with the
    reinforcement learning agent.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      rl_agent_name: agent, the main decision making agent behind the GAIRL
        framework.
      state_gen_name: generative model, generative model that will be used
        to learn and simulate (state1, action) -> state2 transition.
      rewterm_gen_name: generative model, generative model that will be used
        to learn and simulate (state1, action) -> (reward, is_terminal)
        transition.
      observation_shape: tuple of ints describing the observation shape.
      stack_size: int, number of frames to use in state stack.
      model_free_length: int, how many model-free steps are performed in
        a single GAIRL iteration.
      model_learning_length: int, how many model learning iterations are
        performed in a single GAIRL iteration.
      model_based_length: int, how many reinforcement learning steps will be
        performed in a simulated environment in a single GAIRL iteration.
      train_memory_capacity: int, capacity of the memory used for training
        the generative models.
      test_memory_capacity: int, capacity of the memory used for testing
        the generative models.
      memory_batch_size: int, batch size used when replaying transitions from
        the memories.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      eval_mode: bool, True for evaluation and False for training.
    """
    tf.logging.info('Creating %s agent with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t Model free agent: %s', rl_agent_name)
    tf.logging.info('\t State generator: %s', state_gen_name)
    tf.logging.info('\t Rewterm generator: %s', rewterm_gen_name)
    tf.logging.info('\t Model free length: %d', model_free_length)
    tf.logging.info('\t Model learning length: %d', model_learning_length)
    tf.logging.info('\t Model based length: %d', model_based_length)
    tf.logging.info('\t Train memory capacity: %d', train_memory_capacity)
    tf.logging.info('\t Test memory capacity: %d', test_memory_capacity)
    tf.logging.info('\t Memory batch size: %d', memory_batch_size)

    AbstractAgent.__init__(self,
                           num_actions,
                           observation_shape=observation_shape,
                           stack_size=stack_size)
    self.observation_dtype = observation_dtype
    self.model_free_steps = 0
    self.model_free_length = model_free_length
    self.model_learning_steps = 0
    self.model_learning_length = model_learning_length
    self.model_learning_logging_frequency = model_learning_logging_frequency
    self.model_based_steps = 0
    self.model_based_length = model_based_length
    self.eval_mode = eval_mode
    self.summary_writer = summary_writer
    self.action_onehot_template = np.eye(num_actions,
                                         dtype=observation_dtype.as_numpy_dtype)

    # Initialising submodels
    state_shape = self.observation_shape + (stack_size,)
    input_shapes = (state_shape, (num_actions,))
    with tf.variable_scope('agent'), gin.config_scope('agent'):
      self.rl_agent = create_agent(sess, rl_agent_name, num_actions,
                                   observation_shape=observation_shape,
                                   observation_dtype=observation_dtype,
                                   stack_size=stack_size,
                                   summary_writer=summary_writer)
    with tf.variable_scope('state_gen'), gin.config_scope('state_gen'):
      self.state_gen = create_generator(sess, state_gen_name, state_shape,
                                        input_shapes=input_shapes,
                                        summary_writer=summary_writer)
    with tf.variable_scope('rewterm_gen'), gin.config_scope('rewterm_gen'):
      self.rewterm_gen = create_generator(sess, rewterm_gen_name, (2,),
                                          input_shapes=input_shapes,
                                          summary_writer=summary_writer)

    # Each episode goes either to train or to test memory
    total_memory = (train_memory_capacity + test_memory_capacity)
    self._test_episode_prob = test_memory_capacity / total_memory
    self._train_memory = self._build_memory(train_memory_capacity,
                                            memory_batch_size)
    self._test_memory = self._build_memory(test_memory_capacity,
                                           memory_batch_size)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._is_test_episode = False
    self._train_observation = None
    self._last_train_observation = None

  def _build_memory(self, capacity, batch_size):
    """Creates the replay buffer used by the generators.

    Args:
      capacity: int, maximum capacity of the memory unit.
      batch_size int, batch size of the batch produced during memory replay.

    Returns:
      A OutOfGraphReplayBuffer object.
    """
    return circular_replay_buffer.OutOfGraphReplayBuffer(
      self.observation_shape,
      self.stack_size,
      capacity,
      batch_size,
      observation_dtype=self.observation_dtype.as_numpy_dtype,
    )

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._is_test_episode = random.random() < self._test_episode_prob

    if not self.eval_mode:
      self._train_observation = np.reshape(observation, self.observation_shape)
      self._train_step()

    self.rl_agent.eval_mode = self.eval_mode
    self.action = self.rl_agent.begin_episode(observation)
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    if not self.eval_mode:
      self._last_train_observation = self._train_observation
      self._train_observation = np.reshape(observation, self.observation_shape)
      self._store_transition(self._last_train_observation, self.action,
                             reward, False)
      self._train_step()

    self.rl_agent.eval_mode = self.eval_mode
    self.action = self.rl_agent.step(reward, observation)
    return self.action

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(self._train_observation, self.action, reward, True)

    self.rl_agent.eval_mode = self.eval_mode
    self.rl_agent.end_episode(reward)

  def _store_transition(self, last_observation, action, reward, is_terminal):
    """Stores an experienced transition consisting of the tuple
    (last_observation, action, reward, is_terminal) in an appropriate
    memory unit (train or test).

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
    """
    if self._is_test_episode:
      self._test_memory.add(last_observation, action, reward, is_terminal)
    else:
      self._train_memory.add(last_observation, action, reward, is_terminal)

  def _train_step(self):
    """Increment model free steps count.

    Run model training followed by model based training if model free phase
    finished.
    """
    self.model_free_steps += 1
    if self.model_free_steps % self.model_free_length == 0:
      self._train_generators()
      self._train_model_based()

  def _train_generators(self):
    """Run model learning phase - train generative models"""
    tf.logging.info('***Starting model learning phase.***')
    start_time = time.time()
    mean_statistics = collections.defaultdict(int)
    while True:
      # Prepare data
      batch_data = self._train_memory.sample_transition_batch()
      batch_inputs, batch_next_state, batch_rewterm = \
        self._prepare_transitions_batch(batch_data)
      # Train models
      state_statistics = self.state_gen.train(batch_inputs, batch_next_state)
      rewterm_statistics = self.rewterm_gen.train(batch_inputs, batch_rewterm)
      for k, v in state_statistics.items():
        weighted_value = v / self.model_learning_logging_frequency
        mean_statistics[f'mean_state_{k}'] += weighted_value
      for k, v in rewterm_statistics.items():
        weighted_value = v / self.model_learning_logging_frequency
        mean_statistics[f'mean_rewterm_{k}'] += weighted_value

      self.model_learning_steps += 1

      # Log
      if self.model_learning_steps % self.model_learning_logging_frequency == 0:
        time_delta = time.time() - start_time
        tf.logging.info('Step: %d', self.model_learning_steps)
        tf.logging.info('Average statistics per training: %s',
                        dict_to_str(mean_statistics))
        tf.logging.info('Average training steps per second: %.2f',
                        self.model_learning_logging_frequency / time_delta)
        start_time = time.time()
        mean_statistics = collections.defaultdict(int)
        self._save_model_learning_summaries()

      # Stop training after specified
      if self.model_learning_steps % self.model_learning_length == 0:
        break

    tf.logging.info('***Finished model learning phase.***')

  def _save_model_learning_summaries(self):
    train_data = self._train_memory.sample_transition_batch()
    train_summs = self._prepare_model_learning_summaries(train_data, 'Train')
    test_data = self._test_memory.sample_transition_batch()
    test_summs = self._prepare_model_learning_summaries(test_data, 'Test')
    summaries = train_summs + test_summs
    self.summary_writer.add_summary(tf.Summary(value=summaries),
                                    self.model_learning_steps)

  def _prepare_model_learning_summaries(self, batch_data, test_or_train):
    batch_inputs, batch_next_state, batch_rewterm = \
      self._prepare_transitions_batch(batch_data)
    batch_reward = batch_rewterm[:, 0]
    batch_terminal = batch_rewterm[:, 1]

    gen_next_state = self.state_gen.generate(batch_inputs)
    state_l1 = np.mean(np.abs(gen_next_state - batch_next_state))

    gen_rewterm = self.rewterm_gen.generate(batch_inputs)
    gen_reward = gen_rewterm[:, 0]
    gen_terminal = gen_rewterm[:, 1]
    rewterm_l1 = np.mean(np.abs(gen_rewterm - batch_rewterm))
    reward_l2 = np.mean(np.square(gen_reward - batch_reward))
    term_precision, term_recall, term_f1 = \
      _calculate_classification_statistics(gen_terminal, batch_terminal)
    return [
      tf.Summary.Value(tag=f'State/{test_or_train}L1Loss',
                       simple_value=state_l1),
      tf.Summary.Value(tag=f'Rewterm/{test_or_train}L1Loss',
                       simple_value=rewterm_l1),
      tf.Summary.Value(tag=f'Rewterm/{test_or_train}RewardL2Loss',
                       simple_value=reward_l2),
      tf.Summary.Value(tag=f'Rewterm/{test_or_train}TerminalPrecision',
                       simple_value=term_precision),
      tf.Summary.Value(tag=f'Rewterm/{test_or_train}TerminalRecall',
                       simple_value=term_recall),
      tf.Summary.Value(tag=f'Rewterm/{test_or_train}TerminalF1',
                       simple_value=term_f1)
    ]

  def _prepare_transitions_batch(self, batch_data):
    """Transforms batch data from memory into separate batches usable by
    generative models.

    Args:
      batch_data: tuple of numpy arrays, tuple returned by the memory
        consisting of all important information about sampled transitions.

    Returns:
      tuple of numpy arrays, (batch_inputs, batch_next_state, batch_rewterm),
        all necessary and prepared pieces of transition data.
    """
    batch_states = batch_data[0]
    batch_actions_onehot = self.action_onehot_template[batch_data[1]]
    batch_inputs = (batch_states, batch_actions_onehot)
    batch_next_state = batch_data[3]
    batch_reward = batch_data[2]
    batch_terminal = batch_data[6]
    batch_rewterm = np.column_stack((batch_reward, batch_terminal))
    return batch_inputs, batch_next_state, batch_rewterm

  def _train_model_based(self):
    pass

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

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

    agent_path = os.path.join(checkpoint_dir, AGENT_SUBDIR)
    if not os.path.exists(agent_path):
      os.mkdir(agent_path)
    agent_bundle = self.rl_agent.bundle_and_checkpoint(agent_path,
                                                       iteration_number)
    agent_bundle = {k + AGENT_APPENDIX: v
                    for (k, v) in agent_bundle.items()}

    state_path = os.path.join(checkpoint_dir, STATE_SUBDIR)
    if not os.path.exists(state_path):
      os.mkdir(state_path)
    state_bundle = self.state_gen.bundle_and_checkpoint(state_path,
                                                        iteration_number)
    state_bundle = {k + STATE_APPENDIX: v
                    for (k, v) in state_bundle.items()}

    rewterm_path = os.path.join(checkpoint_dir, REWTERM_SUBDIR)
    if not os.path.exists(rewterm_path):
      os.mkdir(rewterm_path)
    rewterm_bundle = self.rewterm_gen.bundle_and_checkpoint(rewterm_path,
                                                            iteration_number)
    rewterm_bundle = {k + REWTERM_APPENDIX: v
                      for (k, v) in rewterm_bundle.items()}

    train_mem_path = os.path.join(checkpoint_dir, TRAIN_MEM_SUBDIR)
    if not os.path.exists(train_mem_path):
      os.mkdir(train_mem_path)
    self._train_memory.save(train_mem_path, iteration_number)

    test_mem_path = os.path.join(checkpoint_dir, TEST_MEM_SUBDIR)
    if not os.path.exists(test_mem_path):
      os.mkdir(test_mem_path)
    self._test_memory.save(test_mem_path, iteration_number)

    return {**agent_bundle, **state_bundle, **rewterm_bundle}

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
    agent_path = os.path.join(checkpoint_dir, AGENT_SUBDIR)
    agent_bundle = {k[:-2]: v for k, v in bundle_dictionary.items()
                    if k[-2:] == AGENT_APPENDIX}
    if not self.rl_agent.unbundle(agent_path, iteration_number,
                                  agent_bundle):
      return False

    state_path = os.path.join(checkpoint_dir, STATE_SUBDIR)
    state_bundle = {k[:-2]: v for k, v in bundle_dictionary.items()
                    if k[-2:] == STATE_APPENDIX}
    if not self.state_gen.unbundle(state_path, iteration_number,
                                   state_bundle):
      return False

    rewterm_path = os.path.join(checkpoint_dir, REWTERM_SUBDIR)
    rewterm_bundle = {k[:-2]: v for k, v in bundle_dictionary.items()
                      if k[-2:] == REWTERM_APPENDIX}
    if not self.rewterm_gen.unbundle(rewterm_path, iteration_number,
                                     rewterm_bundle):
      return False

    train_mem_path = os.path.join(checkpoint_dir, TRAIN_MEM_SUBDIR)
    self._train_memory.load(train_mem_path, iteration_number)

    test_mem_path = os.path.join(checkpoint_dir, TEST_MEM_SUBDIR)
    self._test_memory.load(test_mem_path, iteration_number)

    return True
