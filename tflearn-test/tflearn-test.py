import tflearn
#these functions don't exist in tensorflow, but are implemented here in this high level implementation
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X, Y, test_x, test_y = mnist.load_data(one_hot = True)

X = X.reshape([-1, 28, 28, 1]) 
test_x = test_x.reshape([-1, 28, 28, 1])

convnet = input_data(shape=[None, 28, 28, 1], name='input'); #input layer

convnet = conv_2d(convnet, 32, 2, activation='relu') #size, window size, activation function
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu') #32*32 = 1024 for size
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.fit({'input':X},{'targets':Y}, n_epoch=10, 
    validation_set=({'input':test_x},{'targets':test_y}),
    snapshot_step=500, show_metric=True, run_id='mnist')

model.save('tflearn_cnn.model') #saves your weights, not the rest of the network, so need that stuff

#model.load('tflearn_cnn.model')
#print(model.predict([test_x[1]]))