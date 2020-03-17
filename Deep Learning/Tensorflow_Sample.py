import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1,x2)   #x1*x2
#print("result: ", result)

'''
session = tf.Session()
print(session.run(result))
session.close()
'''

#here no need to close session explicitly and output is also accessible outside
with tf.Session() as sess:
    output = sess.run(result)
    print("result: ", output)
    v = tf.Variable(1.)
    v.assign(2.)
    print(tf.print(v))