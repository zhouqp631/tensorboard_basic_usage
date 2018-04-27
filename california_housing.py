import numpy as np
import os
import shutil
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
# normalization
m,n = housing.data.shape
x,y = housing.data,housing.target
x,y = (x-x.mean())/x.std(), (y-y.mean())/y.std()
x = np.c_[np.ones((m,1)),x]

# build the model
# X*theta = y   ----------> min  1/m*(X*theta-y)'*(X*theta-y)
learning_rate = 0.01
with tf.name_scope('input'):
      x = tf.constant(x,dtype=tf.float32,name='X')
      y = tf.constant(y.reshape(-1,1),dtype=tf.float32,name='y')

with tf.name_scope('theta'):
     theta = tf.Variable(tf.random_uniform([n+1,1]),name='theta')

with tf.name_scope('prediction'):
     y_pred = tf.matmul(x,theta,name='prediction')

with tf.name_scope('mse'):
     mse = tf.reduce_mean(tf.square(y_pred-y),name='mse')
     mse_summary = tf.summary.scalar('MSE', mse)

with tf.name_scope('training'):
      gradients = 2/m*tf.matmul(tf.transpose(x),y_pred-y)
      train_op = tf.assign(theta,theta-learning_rate*gradients)
      # train_op = tf.train.AdamOptimizer().minimize(mse)

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "housing_log"
logdir = "{}/run-{}/".format(root_logdir, now)
if os.path.exists(logdir):
    shutil.rmtree(logdir)

init = tf.global_variables_initializer()
writer = tf.summary.FileWriter(logdir,graph=tf.get_default_graph())

sess = tf.Session()
sess.run(init)
epoches = 1000
for i in range(epoches):
  if i%10 == 0:
      [msevalue,mse_sum] = sess.run([mse,mse_summary])
      writer.add_summary(mse_sum, global_step=i)

      print('Epoch',i,'  MSE=',msevalue)
  sess.run(train_op)

writer.close()
sess.close()