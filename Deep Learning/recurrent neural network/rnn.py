import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10
n_epochs = 3
n_batches = 128
chunk_size = 28
n_chunk = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunk, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network_model(x):
    
    layer = { "weights": tf.Variable(tf.random_normal([rnn_size, n_classes])),
              "biases" : tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunk, 0)
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
    output = tf.add(tf.matmul(outputs[-1], layer["weights"]), layer["biases"]) 
    return output

def train_neural_network(x):
    
    prediction = recurrent_neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            for _ in range(int(mnist.train.num_examples/n_batches)):
                epoch_x, epoch_y = mnist.train.next_batch(n_batches)
                epoch_x = epoch_x.reshape((n_batches, n_chunk, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epochs, ', loss=', epoch_loss)
        
        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape(-1, n_chunk, chunk_size), y: mnist.test.labels}))
    
train_neural_network(x)

