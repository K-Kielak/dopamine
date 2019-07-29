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
"""Abstract agent specifying required contract"""

from abc import abstractmethod



from dopamine.abstract_ml_model import AbstractMLModel


class AbstractAgent(AbstractMLModel):

  def __init__(self, num_actions, observation_shape=None, stack_size=1):
    """Initializes the agent.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints describing the observation shape.
      stack_size: int, number of frames to use in state stack.
    """
    assert isinstance(observation_shape, tuple)
    self.num_actions = num_actions
    self.observation_shape = tuple(observation_shape)
    self.stack_size = stack_size

  @abstractmethod
  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    pass

  @abstractmethod
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
    pass

  @abstractmethod
  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    pass