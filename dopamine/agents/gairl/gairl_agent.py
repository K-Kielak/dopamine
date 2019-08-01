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



from dopamine.agents.abstract_agent import AbstractAgent
from dopamine.discrete_domains import atari_lib
import numpy as np
import tensorflow as tf

import gin.tf


AGENT_APPENDIX = '@a'
STATE_APPENDIX = '@s'
REWTERM_APPENDIX = '@r'
AGENT_SUBDIR = 'agent'
STATE_SUBDIR = 'state'
REWTERM_SUBDIR = 'rewterm'


@gin.configurable
class GAIRLAgent(AbstractAgent):
  """An implementation of the GAIRL agent."""

  def __init__(self,
               rl_agent,
               state_gen,
               rewterm_gen,
               num_actions,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               model_free_length=1000,
               model_learning_length=10000,
               model_based_length=10000):
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
    self._rl_agent = rl_agent
    self._state_gen = state_gen
    self._rewterm_gen = rewterm_gen
    self._model_free_steps = 0
    self._model_free_length = model_free_length
    self._model_learning_steps = 0
    self._model_learning_length = model_learning_length
    self._model_based_steps = 0
    self._model_based_length = model_based_length
    state_shape = (1,) + self.observation_shape + (stack_size,)
    self.train_state = np.zeros(state_shape)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._train_observation = None
    self._last_train_observation = None

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    action = self._rl_agent.begin_episode(observation)
    if not self.eval_mode:
      self._reset_train_state()
      self._record_train_observation(observation)
      self._train_step()

    return action

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
    action = self._rl_agent.step(reward, observation)

    if not self.eval_mode:
      self._last_train_observation = self._train_observation
      self._record_train_observation(observation)
      self._train_step()

    return action

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    self._rl_agent.end_episode(reward)

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

  def _train_step(self):
    """Increment model free steps count

    Run model training followed by model based training if model free phase
    finished
    """
    self._model_free_steps += 1
    if self._model_free_steps % self._model_free_length == 0:
      for _ in range(self._model_free_length):
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
    agent_bundle = self._rl_agent.bundle_and_checkpoint(agent_path,
                                                        iteration_number)
    agent_bundle = {k + AGENT_APPENDIX: v
                    for (k, v) in agent_bundle.items()}

    state_path = os.path.join(checkpoint_dir, STATE_SUBDIR)
    if not os.path.exists(state_path):
      os.mkdir(state_path)
    state_bundle = self._state_gen.bundle_and_checkpoint(state_path,
                                                         iteration_number)
    state_bundle = {k + STATE_APPENDIX: v
                    for (k, v) in state_bundle.items()}

    rewterm_path = os.path.join(checkpoint_dir, REWTERM_SUBDIR)
    if not os.path.exists(rewterm_path):
      os.mkdir(rewterm_path)
    rewterm_bundle = self._rewterm_gen.bundle_and_checkpoint(rewterm_path,
                                                             iteration_number)
    rewterm_bundle = {k + REWTERM_APPENDIX: v
                      for (k, v) in rewterm_bundle.items()}

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
    if not self._rl_agent.unbundle(agent_path, iteration_number,
                                   agent_bundle):
      return False

    state_path = os.path.join(checkpoint_dir, STATE_SUBDIR)
    state_bundle = {k[:-2]: v for k, v in bundle_dictionary.items()
                    if k[-2:] == STATE_APPENDIX}
    if not self._state_gen.unbundle(state_path, iteration_number,
                                    state_bundle):
      return False

    rewterm_path = os.path.join(checkpoint_dir, REWTERM_SUBDIR)
    rewterm_bundle = {k[:-2]: v for k, v in bundle_dictionary.items()
                      if k[-2:] == REWTERM_APPENDIX}
    if not self._rewterm_gen.unbundle(rewterm_path, iteration_number,
                                      rewterm_bundle):
      return False

    return True
