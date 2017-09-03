#   By: Nicholas Vadivelu
#   Created On: 2 September 2017
#   Last Updated: 2 September 2017

#   Train Neural Network with Open AI

import gym #open ai gym
import random #initially random moves
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median 
from collections import Counter

LR = 1e-3 #learning rate
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500 #how many frames it can stay up
score_requirement = 50 #learn from games that get a score of this or above
initial_games = 5000
n_games = 20
keep_rate = 0.8

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

def reinforced_learning(model, n_games=10):
    training_data = []
    scores = []
    accepted_scores = [0]
    game_num = 1
    total_games = initial_games
    while game_num <= n_games:
        score = 0
        game_memory = [] #store movements
        prev_obs = []
        done = False
        while not done:
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs), 1))[0])
            
            observation, reward, done, info = env.step(action)

            if len(prev_obs) > 0:
                game_memory.append([prev_obs, action]) #because this action caused current observation, so previous observation was what lead to current action
            
            prev_obs = observation
            score += reward #reward is one or zero (zero if lost)

            if score > 1000:
                break

        if score >= accepted_scores[-1]: #it's better than the last game
            accepted_scores.append(score)
            game_num = game_num + 1
            for data in game_memory: 
                if data[1] == 1:
                    output = [0,1] 
                elif data[1] == 0:
                    output = [1,0]
                
                training_data.append([data[0], output])
            model = train_model(training_data, model)
            print('Game Number:', game_num, '. Last Score:', accepted_scores[-1], '. Total Games:', total_games)
        
        training_data = []
        env.reset()
        
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores)) #counts how many of each score appear

    return model

def initial_population():
    training_data = [] #observation and move made (all random), but only use if score above 50
    scores = []
    accepted_scores = []
    for _ in range(initial_games): #underscore says variables doesn't matter, just tryna iterate
        score = 0
        game_memory = [] #store movements
        prev_observation = []
        for _ in range (goal_steps):
            action = env.action_space.sample()
            #()random.randrange(0, 2) #env.action_space.sample() might be better, but this corresponds to left and right movement
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action]) #because this action caused current observation, so previous observation was what lead to current action
            
            prev_observation = observation
            score += reward #reward is one or zero (zero if lost)

            if done:
                break
            
        if score >= score_requirement: #if it's a winning game
            accepted_scores.append(score)

            for data in game_memory: 
                if data[1] == 1:
                    output = [0,1] 
                elif data[1] == 0:
                    output = [1,0]
                
                training_data.append([data[0], output])
        
        env.reset()
        scores.append(scores)
    
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores)) #counts how many of each score appear
    return training_data

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation = 'relu') #input layer
    network = dropout(network, keep_rate) #0.8 is keep rate

    network = fully_connected(network, 256, activation = 'relu') #input layer
    network = dropout(network, keep_rate) #0.8 is keep rate

    network = fully_connected(network, 512, activation = 'relu') #input layer
    network = dropout(network, keep_rate) #0.8 is keep rate

    network = fully_connected(network, 256, activation = 'relu') #input layer
    network = dropout(network, keep_rate) #0.8 is keep rate

    network = fully_connected(network, 128, activation = 'relu') #input layer
    network = dropout(network, keep_rate) #0.8 is keep rate

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model

def train_model(training_data, model=False, n_epochs=1):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1) #gets observations
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input':X},{'targets':y}, n_epoch=n_epochs, snapshot_step=5000) 
    #too many epochs, it's going to overfit
    #in fact, 95% or greater accuract might mean overfitting --> trouble

    return model

#some_random_games_first() #see if things are working as you expect

training_data = initial_population()
model = train_model(training_data, n_epochs=3)
model = reinforced_learning(model, n_games)
model.save('cartpole_model-improved.model') #could save the  model and import for longer ones
#model.load('cartpole_model.model')

scores = []
choices = []

for each_game in range(100): #very similar to above function, could probably combine
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()

    for _ in range(goal_steps):
        #env.render()
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
print()
print('Average score :', sum(scores)/len(scores))
print('Choice 1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))