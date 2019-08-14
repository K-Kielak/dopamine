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

import os
import random



from dopamine.agents.abstract_agent import AbstractAgent
from dopamine.discrete_domains import atari_lib
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


@gin.configurable
class GAIRLAgent(AbstractAgent):
  """An implementation of the GAIRL agent."""

  def __init__(self,
               rl_agent,
               state_gen,
               rewterm_gen,
               num_actions,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               model_free_length=10000,
               model_learning_length=50000,
               model_based_length=50000,
               train_memory_capacity=40000,
               test_memory_capacity=10000,
               memory_batch_size=256,
               eval_mode=False):
    """Initializes the agent by combining generative models with the
    reinforcement learning agent.

    Args:
      rl_agent: agent, the main decision making agent behind the GAIRL
      framework.
      state_gen: generative model, generative model that will be used
      to learn and simulate (state1, action) -> state2 transition.
      rewterm_gen: generative model, generative model that will be used
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
      eval_mode: bool, True for evaluation and False for training.
    """
    assert rl_agent.num_actions == num_actions
    assert rl_agent.observation_shape == observation_shape
    assert rl_agent.stack_size == stack_size

    tf.logging.info('Creating %s agent with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t Model free agent: %s', rl_agent.__class__.__name__)
    tf.logging.info('\t State generator: %s', state_gen.__class__.__name__)
    tf.logging.info('\t Rewterm generator: %s',
                    rewterm_gen.__class__.__name__)

    AbstractAgent.__init__(self,
                           num_actions,
                           observation_shape=observation_shape,
                           stack_size=stack_size)
    self.rl_agent = rl_agent
    self.state_gen = state_gen
    self.rewterm_gen = rewterm_gen
    self.observation_dtype = observation_dtype
    self.model_free_steps = 0
    self.model_free_length = model_free_length
    self.model_learning_steps = 0
    self.model_learning_length = model_learning_length
    self.model_based_steps = 0
    self.model_based_length = model_based_length
    state_shape = (1,) + self.observation_shape + (stack_size,)
    self.train_state = np.zeros(state_shape)
    self.eval_mode = eval_mode

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
      self._reset_train_state()
      self._record_train_observation(observation)
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
      self._record_train_observation(observation)
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

  def _reset_train_state(self):
    """Resets the agent state by filling it with zeros."""
    self.train_state.fill(0)

  def _record_train_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._train_observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.train_state = np.roll(self.train_state, -1, axis=-1)
    self.train_state[0, ..., -1] = self._train_observation

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
    pass

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
