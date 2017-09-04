'''
Solving the Cartpole Environment with Deep Q-learning

Nicholas Vadivelu - based on https://keon.io/deep-q-learning/
3 September 2017
'''

import gym
import numpy as np 
from collections import deque
import random 
import tflearn as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

episodes = 1000
learning_rate = 0.001
sample_size = 32
max_time = 500

class DeepQLearning:
    def __init__ (self, n_actions, model):
        self.n_actions = n_actions #number of potential actions
        self.model = model #deep neural network model
        self.memory = deque(maxlen=2000) #double ended queue to remember
        self.gamma = 0.95 #discount rate, because future rewards are worth less than immediate awards
        self.epsilon = 1.0 #exploration rate, at first this will be high to try out new things, overtime decays because the network knows better
        self.epsilon_min = 0.01 #stops decaying at this number
        self.epsilon_decay = 0.99 #exploration rate decreases by this every iteration
    
    '''
    The neural network will forget older experiences (overwrites with newer ones).
    Need to store old states to retrain model with these states.
    '''
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        #generate a random number and check if it's lower than exploration rate (if so return random action)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        else:
            return np.argmax(self.model.predict(state)[0]) #otherwise return prediction
    
    '''
    The model needs to look for future reward as well. This will replay
    some old experiences in order to make the model perform better in 
    the long term. But, these old experiences that are replayed need to 
    have some discount value gamma, because future reward is worth less 
    than an immediate reward.
    '''
    def replay(self, sample_size):
        sample = random.sample(self.memory, sample_size)
        for state, action, reward, next_state, done in sample:
            target = reward
            if not done: #predict future discounted reward
                target = reward+self.gamma*np.amax(self.model.predict(next_state)[0])
            
            # make agent map current state to future discounted awards
            target_f = self.model.predict(state)
            target_f[0][action] = target

            #train network with target_f
            #self.model.fit({'input':state},{'targets':target_f}, n_epoch=1) #tflearn
            self.model.fit(state, target_f, epochs=1, verbose=0) #keras
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#       number of states per move, number of possible actions
def deepNeuralNet(n_states, n_actions, learning_rate):
    #Keras
    model = Sequential()
    model.add(Dense(64, input_dim=n_states, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(loss='mse',optimizer=Adam(lr=learning_rate)) #try loss mse
    
    # TFLearn
    '''
    network = tf.input_data(shape=[None, n_states], name='input') #input layer
    network = tf.fully_connected(network, 128, activation = 'relu')
    network = tf.fully_connected(network, 128, activation = 'relu') #input layer
    network = tf.fully_connected(network, n_actions, activation='softmax')
    network = tf.regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')
    model = tf.DNN(network)
    '''
    
    return model

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = deepNeuralNet(n_states, n_actions, learning_rate)
    #can load and save model
    done = False
    dqn = DeepQLearning(n_actions, model)

    for episode in range(episodes):
        state = np.reshape(env.reset(), [1, n_states])
        for time in range(max_time):
            #env.render()
            action = dqn.act(state) 
            next_state, reward, done, info = env.step(action) #get information from environment
            reward = reward if not done else -10 #lowers score if game is over (pole falls)
            next_state = np.reshape(next_state, [1, n_states]) #reshape to right size for netowrk
            dqn.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(episode, episodes, time, dqn.epsilon))
                break

        if len(dqn.memory) > sample_size: #makes it relearn old moves with rewards
            dqn.replay(sample_size)