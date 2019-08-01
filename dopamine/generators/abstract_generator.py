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
"""Abstract generator specifying required contract"""

from abc import abstractmethod



from dopamine.abstract_ml_model import AbstractMLModel


class AbstractGenerator(AbstractMLModel):

  @abstractmethod
  def generate(self, input):
    """Generates data based on the received input

    Args:
      input: numpy array, input based on which generator should generate output.

    Returns:
      numpy array, generated output.
    """
    pass

  @abstractmethod
  def train(self, input, expected_output):
    """Performs one training step based on the received training batch.

    Args:
      input: numpy array, input to the generator's network.
      expected_output: numpy array, output that should be produced by the
        generator given input.

    Returns:
      dict, train statistics.
    """
    pass
