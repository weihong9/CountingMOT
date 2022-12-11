from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mot import MotTrainer, CMotTrainer


train_factory = {
  'mot': MotTrainer,
  'cmot': CMotTrainer
}
