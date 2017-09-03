#   By: Nicholas Vadivelu
#   Created On: 2 September 2017
#   Last Updated: 2 September 2017

#   Solve the MountainCar environment 

import gym #open ai gym
import random #initially random moves
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median 

LR = 1e-3 #learning rate
env = gym.make('MountainCar-v0')
env.reset()
score_requirement = -150 #learn from games that get a score of this or above
pos_requirement = -0.2
initial_games = 5000
n_games = 20
num_actions = 3

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(100):
            env.render() #to see what's going on, will go faster if you disable
            action = env.action_space.sample() #takes a random action in your environment
            #observation is info from game (lot of time can be pixel data, for this game it's physics-based data)
            #reward is 1 or 0, so balanced or not
            #done is whether game is done or not
            observation, reward, done, info = env.step(action)
            print(observation)
            if done:
                break

def reinforced_learning(model, n_games=10):
    accepted_scores = [-200]
    prev_max_pos = [-1.2]
    game_num = 1
    total_games = initial_games

    while game_num <= n_games:
        score = 0
        game_memory = [] #store movements
        prev_obs = []
        training_data = []
        max_pos = -1.2
        done = False
        while not done:
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs), 1))[0])
            
            observation, reward, done, info = env.step(action)
            
            if observation[0] > max_pos:
                max_pos = observation[0]

            if len(prev_obs) > 0:
                game_memory.append([prev_obs, action]) #because this action caused current observation, so previous observation was what lead to current action
            
            prev_obs = observation
            score = score - 1

        if score > accepted_scores[-1] or max_pos >= prev_max_pos[-1]: #it's better than the last game
            prev_max_pos.append(max_pos)
            accepted_scores.append(score)
            game_num = game_num + 1
            for data in game_memory: 
                output = [0]*num_actions
                output[data[1]] = 1
                
                training_data.append([data[0], output])
                
            model = train_model(training_data, model)
            print('Game Number:', game_num, '. Last Score:', accepted_scores[-1], 'Last Max Pos:', prev_max_pos[-1], '. Total Games:', total_games)
        
        total_games = total_games + 1
        
        if total_games%50 == 0:
            print('Game Number:', game_num, '. Last Score:', accepted_scores[-1], 'Last Max Pos:', prev_max_pos[-1], '. Total Games:', total_games)

        env.reset()
        
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))

    return model

def initial_population():
    training_data = [] #observation and move made (all random), but only use if score above 50
    scores = []
    accepted_scores = []
    done = False
    for _ in range(initial_games): #underscore says variables doesn't matter, just tryna iterate
        score = 0
        game_memory = [] #store movements
        prev_observation = []
        max_pos = -1.2
        done = False
        while not done:
            action = env.action_space.sample()
            #()random.randrange(0, 2) #env.action_space.sample() might be better, but this corresponds to left and right movement
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action]) #because this action caused current observation, so previous observation was what lead to current action
            
            if observation[0] > max_pos:
                max_pos = observation[0]

            prev_observation = observation
            
            score = score - 1

        if score >= score_requirement or max_pos > pos_requirement: #if it's a winning game
            print(score, max_pos)
            accepted_scores.append(score)

            for data in game_memory: 
                output = [0]*num_actions
                output[data[1]] = 1

                training_data.append([data[0], output])
        
        env.reset()
    
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print('Number of examples: ', len(accepted_scores))
    return training_data

def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation = 'relu') #input layer

    network = fully_connected(network, 256, activation = 'relu') #input layer

    network = fully_connected(network, 128, activation = 'relu') #input layer

    network = fully_connected(network, num_actions, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model

def train_model(training_data, model=False, n_epochs=1):
    #print([i[0] for i in training_data])
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
print('Going to train')
model = train_model(training_data, n_epochs=3)
print('Done training')
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
    done = False
    while not done:
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