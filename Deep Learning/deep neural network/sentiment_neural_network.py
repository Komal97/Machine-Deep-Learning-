import tensorflow as tf
import numpy as np
from create_sentiment_features import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 2
batch_size = 100
no_of_epochs = 10

x = tf.placeholder('float', [None, len(train_x[0])])  
y = tf.placeholder('float')

def neural_network_model(data):
    
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3)
    
    output = tf.add(tf.matmul(layer_3, output_layer['weights']), output_layer['biases'])
    
    return output


def train_neural_network(x):
    
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        
        for epoch in range(no_of_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = session.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch+1, 'completed out of', no_of_epochs, ', loss = ', epoch_loss)
            
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

if __name__ == '__main__':
    train_neural_network(x)
