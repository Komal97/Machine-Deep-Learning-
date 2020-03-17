'''
1) input with weights -> hidden layer 1 with weights (activation function) ->
   hidden layer 2 with weights (activation function) -> ouput
2) Compare output with intended output -> cost function(cross entropy)
3) Optimization function (optimizer) -> minimize cost(AdamOptimizer, Stochastic Gradient Descent(SGD),
   AdaGrad,..)
4) Go backward and manipulate weights (Backpropagation)
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#no. of nodes in each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#no. of classes
n_classes = 10

#batch size which is no. of neurons feed at a time to the network
batch_size = 100

x = tf.placeholder('float', [None, 784])   #in array(which is shape), height = None, width = 28x28
y = tf.placeholder('float')

def neural_network_model(data):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    #(input_data * weights) + biases
    #after summation, pass it through 'RELU' activation function
    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3)
    
    output = tf.add(tf.matmul(layer_3, output_layer['weights']), output_layer['biases'])
    
    return output


#reduce_mean = mean 
#softmax_cross_entropy_with_logits = apply softmax then entropy
def train_neural_network(x):
    
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    
    #Default learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #cycles feed forward + backpropagation
    no_of_epochs = 10
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        
        #training the data 
        for epoch in range(no_of_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = session.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', no_of_epochs, ', loss=', epoch_loss)
        
        #create an array across axis-1(row) which contain largest value of each row and 
        #then finally create a boolean array using tf.equal 
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        
        #tensor.eval converts a tensor object into a form that can be printed
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            
train_neural_network(x)