import tensorflow as tf
import __init__ as tb
import pandas as pd
import numpy as np


#Data
data = pd.read_csv("/data/ex2/ex2data2.txt")

x_data = data.ix[:,"university1":"university2"].as_matrix()
y_data = np.expand_dims(data.ix[:,"admitted"].as_matrix(), 1)

# Variables
x = tf.placeholder(tf.float32, [None, 2], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name="y")


# x
# h = sigmoid(x * w + b)
# Build network
[h, trainer] = (
  x.builder()
  .connect_layer(10, fn=tf.nn.tanh)
  .connect_layer(5, fn=tb.nn.polynomic)
  .connect_layer(1, weights_name="w", bias_name="b")
  .branch(lambda z:
  [
  	z
    .map(tf.nn.sigmoid)
  	,
  	z
  	.map(lambda logit: tf.nn.sigmoid_cross_entropy_with_logits(logit, y))
	.map(lambda loss: tf.train.AdamOptimizer(0.01).minimize(loss))
  ])
)



# Measurements
accuracy = tf.reduce_mean(
	tf.cast(
		tf.equal(
	    tf.cast(h.tensor > 0.5, tf.float32),
	    tf.cast(y > 0.5, tf.float32)
		),
		tf.float32
	)
)

# Training
feed = dict(feed_dict={
    x: x_data,
    y: y_data
})


with tf.Session() as s:
    s.run(tf.initialize_all_variables())

    for i in range(4000):
        s.run(trainer.tensor, **feed)
        if i % 100 == 0:
            print(s.run(accuracy, **feed))
