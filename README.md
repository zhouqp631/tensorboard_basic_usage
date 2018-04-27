Tensorboard基本使用方法
-------------------------
1. 一个简单的例子
2. 模型调参

软件要求
-----------------------
[Tensorflow 1.7](https://www.tensorflow.org/install/?hl=zh-cn)安装tensorflow时，会自动安装tensorboard。使用最新版本可以避免tensorboard不正常显示的错误。
tensorboard<1.7会显示没有统计的tensor![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/tfless17.png),而1.7只会显示已统计的tensor(没有统计的tensor在INACTIVE里面)
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

### 1. 模型调参
-------------------------
* 数据使用的[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html).作为练习，为了更快看到计算结果，只选择了3类:cat,dog,horse.所以数据集的个数为：
training data | validataion data
--------------|-----------------
\[5000*3\]    |\[1000*3]
* 卷积网络参考[ConvNetJS CIFAR-10 demo](https://cs.stanford.edu/~karpathy/convnetjs/demo/cifar10.html).前面的卷积层一样，但最后多了一个全连接层.

### 模型展示
![模型的graph](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/modelgraph.png)


#### 1.1. 分析数据增广的作用
data augmentation通过增加数据扰动提高模型的泛化能力。一个图示(左边为原始图像，右边为augmentation后图像)
![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/data_aug.png)



* 在`cifar3_model.py`中`cifar3_model`中使用方式如下：
> `keep_prob`是`dropout`层的保留概率，当其值小于1是，为训练阶段，所以可以曾广。而其值等于1时，为测试阶段，输入数据不需要处理。
> `tf.cond`是tensorflow中的判断语句。如果使用下面方式，会出错：

```python
is_training = tf.placeholder(tf.bool)
if is_training:
   x_image = data_augmentation(x_image)

```

```python
if use_data_aug :
    x_image = tf.cond(keep_prob<1,lambda:data_augmentation(x_image),lambda:x_image)
    tf.summary.image('train',x_image, max_outputs=5)
else:
    tf.summary.image('test', x_image, max_outputs=5)

```
在`cifar10_read.py`中，代码为(对于每一个batch单独处理):
```python
def data_augmentation(images):
# images: 4-D tensor of [batch_size,height,width,channesl]
    with tf.name_scope('data_augmentation'):
        distorted_image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),images)
        distorted_image = tf.map_fn(lambda img: tf.image.random_flip_up_down(img),distorted_image)

        distorted_image = tf.map_fn(lambda img: tf.image.random_hue(img,max_delta=0.05),distorted_image) #色调
        distorted_image = tf.map_fn(lambda img: tf.image.random_saturation(img,lower=0.0, upper=2.0),distorted_image)#饱和

        distorted_image = tf.map_fn(lambda img: tf.image.random_brightness(img,max_delta=0.2),distorted_image)#亮度
        distorted_image = tf.map_fn(lambda img: tf.image.random_contrast(img,lower=0.2,upper=1.0),distorted_image)#对比度
        distorted_image = tf.map_fn(lambda img:tf.image.per_image_standardization(img),distorted_image)
        
        distorted_image = tf.map_fn(lambda img:tf.maximum(img,0.0),distorted_image)
        imgs = tf.map_fn(lambda img:tf.minimum(img,1.0),distorted_image)
    return imgs
```

#### 1.2. 学习率的选择



### References
- [understanding-tensorboard](https://github.com/secsilm/understanding-tensorboard)
- [Hands-on TensorBoard (TensorFlow Dev Summit 2017)](https://www.youtube.com/watch?v=eBbEDRsCmv4&t=1105s)

### Apendix
* 代码主要函数
![](https://github.com/zhouqp631/tensorboard_basic_usage/blob/master/files/codeflow.png)




