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

LR = 1e-3 #learning rate
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500 #how many frames it can stay up
score_requirement = 50 #learn from games that get a score of this or above
initial_games = 10000 

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render() #to see what's going on, will go faster if you disable
            action = env.action_space.sample() #takes a random action in your environment
            #observation is info from game (lot of time can be pixel data, for this game it's physics-based data)
            #reward is 1 or 0, so balanced or not
            #done is whether game is done or not
            observation, reward, done, info = env.step(action)
            if done:
                break

some_random_games_first() #see if things are working as you expect