Tensorboard基本使用方法
-------------------------
1. 一个简单的例子
2. 模型调参

软件要求
-----------------------
- **TensorFlow1.7: ** 安装[Tensorflow 1.7](https://www.tensorflow.org/install/?hl=zh-cn)时，会自动安装tensorboard。使用最新版本可以避免tensorboard不正常显示的错误。
tensorboard<1.7会显示没有统计的tensor![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/tfless17.png),而1.7只会显示已统计的tensor(没有统计的tensor在INACTIVE里面)
![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/tf17.png)
- **chrome插件: ** [GitHub with MathJax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima/related) 为了正常查看公式

### 1. 一个简单的例子
-------------------------
* 统计0-dim tensor (文件：`summaryUsage.py`)
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
运行程序
```python
python summaryUsage.py
```
打开tensorboard
```python
tensorboard  --logdir=basic_tf

```
TensorBoard结果为![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/basic_tf.gif)

* 统计n-d(多维)tensor
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
通过`tf.summary.histogram`统计`ynew`的值，然后用`merged_summary = tf.summary.merge_all()`把`tf.summary.scalar`和`tf.summary.histogram`放在一起，最后在迭代中写入。

结果为![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/tf_basic2.gif)

### 2. 模型调参
-------------------------
* 数据使用的[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html). 为了更快看到计算结果，只选择了其中3类:**cat,dog,horse**.数据集的个数为：

|training data | validataion data|
|------------ | -------------|
|     15000    | 3000|
      
* 卷积网络结构参考[ConvNetJS CIFAR-10 demo](https://cs.stanford.edu/~karpathy/convnetjs/demo/cifar10.html).前面的卷积层一样，但最后多了一个全连接层.

### 模型展示
![模型的graph](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/modelgraph.png)

#### 2.1. $L_2$正则化
机器学习中，用来减小测试误差的策略统称为正则化。深度学习常用的正则化方法有参数范数惩罚，数据集增强(data augmentation), Dropout。

此处测试参数范数惩罚的作用。
* 下面结果分别为train和test过程中的`accuracy and loss`.
可以看出，使用$L_2$以后，训练`accuracy`降低，但是测试`accuray`确实增加了。

Training accuracy![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/reg_train.png)

Test accuracy![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/reg_test.png)





* 添加$L_2$范数的一种方式(在`cifar3_model.py`中). 
   * 其中0.1用来权衡`cross_entropy_mean`和`l2_loss`，其值越大，对参数的惩罚就越大。
   * 通常只对权重做惩罚，而不惩罚bias

```python
with tf.name_scope("loss"):
    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="cross_entropy_mean")
    loss = cross_entropy_mean
    if use_regu:
        trainable_vars = tf.trainable_variables()
        l2_loss = 0.1 * tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if not 'b' in v.name])
        loss = cross_entropy_mean+l2_loss
    tf.summary.scalar("loss", loss)

```
    

#### 2.2. 学习率的选择
学习率对训练精度也有很大的影响.本例中不同学习率的训练和测试`accuracy`如下. 可以看出,`learning rate=3E-04`太高，`learning rate=1E-05`太低。
![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/lr_vs.png)


- 可用来调节学习率的参考图![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/lr.png)







### Main References
- [Deep Learning](http://www.deeplearningbook.org/)
- [understanding-tensorboard](https://github.com/secsilm/understanding-tensorboard)
- [Hands-on TensorBoard (TensorFlow Dev Summit 2017)](https://www.youtube.com/watch?v=eBbEDRsCmv4&t=1105s)



### Apendix
* 代码主要函数
![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/codeflow.png)




