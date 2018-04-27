Tensorboard基本使用方法
-------------------------
1. 一个简单的例子
2. 模型的参数寻优

软件要求
-----------------------
[Tensorflow 1.7](https://www.tensorflow.org/install/?hl=zh-cn) tensorboard在安装tensorflow时，会自动安装。使用1.7的目的是：可视化的时候可以避免一些错误，同时，最新版可视化效果更佳。
tensorboard<1.7会显示没有统计的tensor![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/tfless17.png),而1.7只会显示程序中已统计的tensor(没有统计的在INACTIVE里面)
![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/tf17.png)



### 1. 一个简单的例子
-------------------------
* 只统计0-dim tensor
```python
import tensorflow as tf
x = tf.Variable(1.0)
xnew = tf.assign(x,x+0.2)
x_summary = tf.summary.scalar('x_trace',xnew)

writer= tf.summary.FileWriter('basic_tf',graph=tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _,x_smy = sess.run([xnew,x_summary])
        writer.add_summary(x_smy,global_step=i)
writer.close()
```
通过`tf.summary.scalar`定义统计的变量，通过`tf.summary.FileWriter`制定日志文件的目录，然后使用`add_summary`添加每一步的变量值。
![结果为](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/basic_tf.gif)

* 统计多维tensor
```python
import tensorflow as tf

x = tf.Variable(1.0)
xnew = tf.assign(x,x+0.2)
x_summary = tf.summary.scalar('x_trace',xnew)

y = tf.Variable(tf.ones([28,28]),name='2d_value')
ynew = tf.assign(y,tf.add(y,tf.random_normal([28,28])))
y_summary = tf.summary.histogram('2d_value',ynew)

merged_summary = tf.summary.merge_all()

writer= tf.summary.FileWriter('basic_tf2',graph=tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _,merged_smy = sess.run([xnew,merged_summary])
        writer.add_summary(merged_smy,global_step=i)
writer.close()
```
通过`tf.summary.histogram`统计`ynew`的值，然后用`merged_summary = tf.summary.merge_all()`把·tf.summary.scalar`和`tf.summary.histogram`统计的tensor放在一起，最后在迭代中写入。

![结果为](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/tf_basic2.gif)