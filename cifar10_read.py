import os
import tensorflow as tf
import numpy as np
import _pickle as pickle

# 训练数据文件
TRAIN_FILE = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]
# 验证数据文件
EVAL_FILE = ['test_batch']

def unpickle(filename):
    '''Decode the dataset files.'''
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        return d

def onehot(labels, cls=None):
    ''' One-hot encoding, zero-based'''
    n_sample = len(labels)
    if not cls:
        n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample,n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

def merge_data(dataset_dir, onehot_encoding=True):
    train_images = unpickle(os.path.join(dataset_dir, TRAIN_FILE[0]))['data']
    train_labels = unpickle(os.path.join(dataset_dir, TRAIN_FILE[0]))['labels']
    valid_images = unpickle(os.path.join(dataset_dir, EVAL_FILE[0]))['data']
    valid_labels = unpickle(os.path.join(dataset_dir, EVAL_FILE[0]))['labels']
    # 训练集
    for i in range(1,len(TRAIN_FILE)):
        batch = unpickle(os.path.join(dataset_dir, TRAIN_FILE[i]))
        train_images = np.concatenate((train_images, batch['data']), axis=0)
        train_labels = np.concatenate((train_labels, batch['labels']), axis=0)
    # 验证集
    for i in range(1, len(EVAL_FILE)):
        batch = unpickle(os.path.join(dataset_dir, TRAIN_FILE[i]))
        valid_images = np.concatenate((valid_images, batch['data']), axis=0)
        valid_labels = np.concatenate((valid_labels, batch['labels']), axis=0)
    if onehot_encoding:
        train_labels = onehot(train_labels)
        valid_labels = onehot(valid_labels)

        catdog_train_index = (train_labels[:, 3] == 1) | (train_labels[:, 5] == 1) | (train_labels[:, 7] == 1)
        catdog_test_index = (valid_labels[:, 3] == 1) | (valid_labels[:, 5] == 1)| (valid_labels[:, 7] == 1)
        train_images, train_labels = train_images[catdog_train_index], \
                                     train_labels[catdog_train_index,:][:,[3,5,7]]
        valid_images, valid_labels = valid_images[catdog_test_index], \
                                     valid_labels[catdog_test_index,:][:,[3,5,7]]
    return train_images, valid_images, train_labels, valid_labels


class Cifar10(object):

    def __init__(self, images, lables):
        '''dataset_dir: the dir which saves the dataset files.
           onehot: if ont-hot encoding or not'''
        self._num_exzamples = len(lables)
        self._images = images
        self._labels = lables
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_exzamples(self):
        return self._num_exzamples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_exzamples:
            # 重新开始一个新的 epoch
            self._epochs_completed += 1
            # 重新打乱数据集
            idx = np.arange(self._num_exzamples)
            np.random.shuffle(idx)
            self._images = self._images[idx, :]
            self._labels = self._labels[idx, :]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._images[start:end, :], self._labels[start:end, :]

def read_dataset(dataset_dir, onehot_encoding=True):
    class Datasets(object):
        pass
    dataset = Datasets()
    train_images, valid_images, train_labels, valid_labels = merge_data(dataset_dir, onehot_encoding)
    dataset.train = Cifar10(train_images, train_labels)
    dataset.valid = Cifar10(valid_images, valid_labels)
    return dataset


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

def visual_image(imgs,labels):
    fig, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    i = 0
    for j in range(3):
        for k in range(3):
            axes1[j][k].set_axis_off()
            if labels[i, 0] == 1:
                axes1[j][k].set_title('cat')
            elif labels[i, 1] == 1:
                axes1[j][k].set_title('dog')
            else:
                axes1[j][k].set_title('horse')
            axes1[j][k].imshow(imgs[i])
            i += 1
    plt.subplots_adjust(wspace=0.05, hspace=0.5)

if __name__ == '__main__':
   datadir = r'cifar-10-batches-py'
   cifar10 = read_dataset(datadir, onehot_encoding=True)
   #
   print('train_image:',cifar10.train.images.shape)
   print('train_labels:', cifar10.train.labels.shape)
   print('valid_image:',cifar10.valid.images.shape)
   print('valid_labels:', cifar10.valid.labels.shape)

   # visual 25 images
   import  matplotlib.pyplot as plt
   X = cifar10.train.images
   X = X.reshape(15000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

   fig, axes1 = plt.subplots(5, 5, figsize=(8, 8))
   index = [np.random.choice(range(len(X))) for _ in range(9)]
   imgs = X[index]
   Y = cifar10.train.labels[index]
   visual_image(imgs, Y)


   # visual argumented data
   data_aug = True
   if data_aug:
       sess = tf.Session()
       float_imgs = tf.image.convert_image_dtype(imgs, tf.float32)
       aug_imgs = data_augmentation(float_imgs)
       aug_imgs = tf.image.convert_image_dtype(aug_imgs,tf.uint8)
       aug_imgs =  sess.run(aug_imgs)

       visual_image(aug_imgs, Y)

       sess.close()







