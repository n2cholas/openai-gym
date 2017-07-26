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
from collections import Counter

LR = 1e-3 #learning rate
env = gym.make('BipedalWalkerHardcore-v2')
env.reset()
goal_steps = 500 #how many frames it can stay up
score_requirement = 50 #learn from games that get a score of this or above
initial_games = 10000

savename = 'x.model' #control save name
loadFile = false #adjust if you load a model or save it

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

def initial_population():
    training_data = [] #observation and move made (all random), but only use if score above 50
    scores = []
    accepted_scores = []
    for _ in range(initial_games): #underscore says variables doesn't matter, just tryna iterate
        score = 0
        game_memory = [] #store movements
        prev_observation = []
        for _ in range (goal_steps):
            action = action_space.sample()  #env.action_space.sample() might be better, but this corresponds to left and right movement
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action]) #because this action caused current observation, so previous observation was what lead to current action
            
            prev_observation = observation
            score += reward #reward is one or zero
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)

            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                
                training_data.append([data[0], output])
        
        env.reset()
        scores.append(scores)
    
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation = 'relu') #input layer
    network = dropout(network, 0.8) #0.8 is keep rate

    network = fully_connected(network, 256, activation = 'relu') #input layer
    network = dropout(network, 0.8) #0.8 is keep rate

    network = fully_connected(network, 512, activation = 'relu') #input layer
    network = dropout(network, 0.8) #0.8 is keep rate

    network = fully_connected(network, 256, activation = 'relu') #input layer
    network = dropout(network, 0.8) #0.8 is keep rate

    network = fully_connected(network, 128, activation = 'relu') #input layer
    network = dropout(network, 0.8) #0.8 is keep rate

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1) #gets observations
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input':X},{'targets':y}, n_epoch=5,snapshot_step=500, show_metric=True, run_id='openaistuff') 
    #too many epochs, it's going to overfit
    #in fact, 95% or greater accuract might mean overfitting --> trouble

    return model


#--------------------------------Main Program

if not loadFile:
    #some_random_games_first() #see if things are working as you expect
    training_data = initial_population()
    model = train_model(training_data)
    model.save(savename) #could save the  model and import for longer ones
else:
    training_data = initial_population()
    model = train_model(training_data)
    model.load('cartpole_model.model')

scores = []
choices = []

for each_game in range(10): #very similar to above function, could probably combine
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs), 1))[0])

        choices.append(action) #want to know all actions so we can look at it and analyze later, just in case it is predicting one option too often

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action]) #don't need unless you want to retrain, can keep saving game and retraining
        score+=reward
        if done:
            break

    scores.append(score)

print('Average score:', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))