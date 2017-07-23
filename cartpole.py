#   By: Nicholas Vadivelu
#   Created On: 23 July 2017
#   Last Updated: 23 July 2017

#   Train Neural Network with Open AI

import gym #open ai gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median 
