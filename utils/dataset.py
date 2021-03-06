# -*- coding: UTF-8 -*-
import os
import math
import random
import cv2
from scipy import misc
import skimage.transform
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from utils import tools


DEBUG = False
MULTI_OUTPUT = False


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    print(path_exp, path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def load_dataset(data_path, shuffle=True, validation_ratio=0.0, min_nrof_cls=1, max_nrof_cls=999999999, filter_cb=None):
    '''
data_path dir style:
data
├── folder1
│   ├── 00063.jpg
│   ├── 00068.jpg
└── folder2
    ├── 00070.jpg
    ├── 00072.jpg

    :param data_path:
    :param shuffle:
    :param validation_ratio: if > 0: will split train-set and validation-set,
           其中， validation-set accounts for is validation_ratio.
    :param min_nrof_cls: min number samples of each class
    :param max_nrof_cls: max number samples of each class
    :return:
    '''
    label_count = 0
    images_path = []
    images_label = []
    images = os.listdir(data_path)
    images.sort()
    if DEBUG:
        images = images[0:50]
    for i, image in enumerate(images):
        cls_path = os.path.join(data_path, image)
        if os.path.isfile(cls_path):
            print('[load_dataset]:: {} is not dir!'.format(cls_path))
            continue

        imgs = os.listdir(cls_path)
        if len(imgs) < min_nrof_cls:
            continue
        if len(imgs) > max_nrof_cls:
            np.random.shuffle(imgs)
            imgs = imgs[0:max_nrof_cls]

        imgs = [os.path.join(data_path, image, img) for img in imgs]
        if filter_cb is not None:
            imgs = filter_cb(imgs)
        images_path.extend(imgs)
        images_label.extend([label_count] * len(imgs))
        label_count += 1
        tools.view_bar('loading: ', i + 1, len(images))
    print('')

    images = np.array([images_path, images_label]).transpose()

    if shuffle:
        np.random.shuffle(images)

    images_path = images[:, 0]
    images_label = images[:, 1].astype(np.int32)

    if DEBUG and False:
        images_path = images_path[0:500]
        images_label = images_label[0:500]

    if validation_ratio > 0.0:
        if not shuffle:
            raise Exception('When there is a validation set split requirement, shuffle must be True.')
        validation_size = int(len(images_path) * validation_ratio)
        validation_images_path = images_path[0:validation_size]
        validation_images_label = images_label[0:validation_size]

        train_images_path = images_path[validation_size:]
        train_images_label = images_label[validation_size:]

        print('\n********************************样 本 总 量*************************************')
        print('len(train_images_path)={}, len(train_images_label)={}'.format(len(train_images_path), len(train_images_label)))
        print('len(validation_images_path)={}, len(validation_images_label)={}'.format(len(validation_images_path), len(validation_images_label)))
        print('num_train_class={}, num_validation_class={}'.format(len(set(train_images_label)), len(set(validation_images_label))))
        print('*******************************************************************************\n')

        return train_images_path, train_images_label, validation_images_path, validation_images_label
    else:
        print('\n********************************样 本 总 量*************************************')
        print('len(images_path)={}, len(images_label)={}'.format(len(images_path), len(images_label)))
        print('num_class={}'.format(len(set(images_label))))
        print('*******************************************************************************\n')

        return images_path, images_label


def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*(1-split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
            if split==nrof_images_in_class:
                split = nrof_images_in_class-1
            if split>=min_nrof_images_per_class and nrof_images_in_class-split >= 1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
            else:
                raise ValueError('TODO: what happened!' % mode)
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set


def get_image_paths_and_labels(dataset, shuffle=False):
    raise Exception('废弃，可以使用dataset.py下的load_dataset或者utils/tools.py下的load_images')

    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)

    if shuffle:
        image_labels = np.vstack((image_paths_flat, labels_flat)).transpose()
        np.random.shuffle(image_labels)
        image_paths_flat = image_labels[:, 0].tolist()
        labels_flat = image_labels[:, 1].astype(np.int32).tolist()

    return image_paths_flat, labels_flat


def recurse_load(datas_path):
    '''
    递归加载数据集
    :param datas_path:
    :return:
    '''
    def _recurse_load(datas_path):
        '''
        这个嵌套函数和外层函数是一样的，之所以用嵌套写法只是为了能在最外层打印总体进度而已。
        '''
        recv_datas = []
        data_list = os.listdir(datas_path)
        for i, dat in enumerate(data_list):
            datpath = os.path.join(datas_path, dat)
            if dat == '22482232@N00':
                print('find {}'.format(datpath))
            if os.path.isdir(datpath):
                datas = _recurse_load(datpath)
                recv_datas.extend(datas)
            else:
                recv_datas.append(datpath)
        return recv_datas

    recv_datas = []
    data_list = os.listdir(datas_path)
    if DEBUG:
        data_list = data_list[0:5]

    for i, dat in enumerate(data_list):
        if dat == '224':
            print('find {}'.format(dat))
        datpath = os.path.join(datas_path, dat)
        if os.path.isdir(datpath):
            datas = _recurse_load(datpath)
            recv_datas.extend(datas)
        else:
            recv_datas.append(datpath)
        tools.view_bar('Recursive loading: ', i + 1, len(data_list))
    # print('')
    return recv_datas


def load_data(data_dir, validation_set_split_ratio=0.05, min_nrof_val_images_per_class=0):
    raise Exception('废弃，可以使用dataset.py下的load_dataset或者utils/tools.py下的load_images')

    seed = 666
    np.random.seed(seed=seed)
    random.seed(seed)
    dataset = get_dataset(data_dir)
    #if filter_filename is not None:
    #    dataset = filter_dataset(dataset, os.path.expanduser(filter_filename),
    #                             filter_percentile, filter_min_nrof_images_per_class)

    if validation_set_split_ratio > 0.0:
        train_set, val_set = split_dataset(dataset, validation_set_split_ratio,
                                                   min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []

    return train_set, val_set


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    try:
        return misc.imrotate(image, angle, 'bicubic')  # scipy.misc 可能已经被弃用，在部分系统中已经无法使用，可使用skimage代替
    except Exception as e:
        # TODO： 最好还是要解决一下使用scipy.misc调用rotate的问题。 因为skimage不支持输入为tfTensor
        return skimage.transform.rotate(image.numpy(), angle, order=3, preserve_range=True)


def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)


def _tfparse_function_train(filename, label):
    raise Exception('已经废弃，迁移至ImageParse')
    # shape = [28, 28, 1]
    shape = [160, 160, 3]
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, shape[2])
    # image = tf.convert_to_tensor(image, dtype=tf.float32)

    # image = tf.image.resize_image_with_crop_or_pad(image, shape[0], shape[1])
    image = tf.image.random_crop(image, shape)
    image = tf.py_function(random_rotate_image, [image], tf.uint8)

    # image = tf.image.random_brightness(image, 0.4)
    # image = tf.image.random_contrast(image, 0.8, 2)

    image = tf.image.random_flip_left_right(image)

    # image = tf.image.per_image_standardization(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 128.0
    return image, (label, label)


def _tfparse_function_validate(filename, label):
    raise Exception('已经废弃，迁移至ImageParse')
    # shape = [28, 28, 1]
    shape = [160, 160, 3]
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, shape[2])
    # image = tf.convert_to_tensor(image, dtype=tf.float32)

    image = tf.image.resize_with_crop_or_pad(image, shape[0], shape[1])

    # image = tf.image.per_image_standardization(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 128.0
    return image, (label, label)


class ImageParse(object):
    RANDOM_ROTATE = 1
    RANDOM_CROP = 2
    RANDOM_LEFT_RIGHT_FLIP = 4
    STANDARDIZATION = 8
    FLIP = 16
    RANDOM_GLASS = 32
    RANDOM_COLOR = 64
    FIXED_CONTRACT = 128

    def __init__(self, imshape=(28, 28, 1), n_classes=None):
        '''
        :param imshape:
        :param n_classes: when n_classes is None, not using one-hot encode;
        when n_classes > 0, using one-hot encode.
        '''
        self.imshape = imshape
        self.n_classes = n_classes

        self.set_train_augment()
        self.set_validation_augment()

    def set_train_augment(self, random_crop=1, random_rotate=True, random_left_right_flip=True, standardization=1):
        """
        :param random_crop: 图像随机裁剪，目前支持3种模式，分别是：0表示不裁剪，1表示使用随机裁剪，2表示使用中心裁剪或者padding
        :param random_rotate:
        :param random_left_right_flip:
        :param standardization: 图像标准化，目前支持3种模式，分别是：0表示不使用标准化，1表示使用固定标准化，2表示使用实例标准化
        :return:
        """
        self.train_augment = {ImageParse.RANDOM_CROP: random_crop,
                              ImageParse.RANDOM_ROTATE: random_rotate,
                              ImageParse.RANDOM_LEFT_RIGHT_FLIP: random_left_right_flip,
                              ImageParse.STANDARDIZATION: standardization
                              }

    def set_validation_augment(self, random_crop=2, random_rotate=False, random_left_right_flip=False, standardization=1):
        self.validation_augment = {ImageParse.RANDOM_CROP: random_crop,
                                   ImageParse.RANDOM_ROTATE: random_rotate,
                                   ImageParse.RANDOM_LEFT_RIGHT_FLIP: random_left_right_flip,
                                   ImageParse.STANDARDIZATION: standardization
                                   }

    def _imdecode(self, filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, self.imshape[2])
        # image = tf.image.decode_png(image, 3)
        # image = tf.convert_to_tensor(image, dtype=tf.float32)
        return image

    def _image_augment(self, image, augments):
        if augments[ImageParse.RANDOM_CROP] == 0:
            pass
        elif augments[ImageParse.RANDOM_CROP] == 1:
            image = tf.image.random_crop(image, self.imshape)
        elif augments[ImageParse.RANDOM_CROP] == 2:
            # image = tf.image.resize_image_with_crop_or_pad(image, self.imshape[0], self.imshape[1])
            image = tf.image.resize_with_crop_or_pad(image, self.imshape[0], self.imshape[1])
        else:
            raise Exception('not supported random_crop parameter.')

        # image = (tf.cast(image, tf.float32) - 127.5) / 128.0
        # return image

        if augments[ImageParse.RANDOM_ROTATE]:
            # image = tf.py_function(random_rotate_image, [image], tf.uint8)
            image = tf.py_function(random_rotate_image, [image], tf.float32)

        # image = tf.image.random_brightness(image, 0.4)
        # image = tf.image.random_contrast(image, 0.8, 2)
        # image = tf.image.resize_images(image, (self.imshape[0]+10, self.imshape[1]-10), align_corners=True, preserve_aspect_ratio=True)

        # rand_w = tf.random_uniform([], 172, 192, dtype=tf.int32)
        # rand_h = tf.random_uniform([], 172, 192, dtype=tf.int32)
        #
        # image = tf.image.resize_images(image, (rand_w, rand_h), align_corners=False,
        #                                        preserve_aspect_ratio=False)  # 如果preserve_aspect_ratio为True，则保持宽高比对原图进行缩放，缩放后的图像宽或高等于image_size中的最小值
        # image = tf.cast(image, dtype=tf.uint8)
        #
        # return image

        if augments[ImageParse.RANDOM_LEFT_RIGHT_FLIP]:
            image = tf.image.random_flip_left_right(image)

        if augments[ImageParse.STANDARDIZATION] == 0:  # 不使用标准化
            pass
        elif augments[ImageParse.STANDARDIZATION] == 1:  # 使用固定标准化
            image = (tf.cast(image, tf.float32) - 127.5) / 128.0
        elif augments[ImageParse.STANDARDIZATION] == 2:  # 使用实例标准化
            image = tf.image.per_image_standardization(image)
        else:
            raise Exception('not supported standardization parameter.')

        return image

    def train_parse_func(self, filename, label):
        image = self._imdecode(filename)
        image = self._image_augment(image, self.train_augment)

        if self.n_classes is not None:
            label = tf.one_hot(label, depth=self.n_classes)

        if MULTI_OUTPUT:
            return image, (label, label)
        else:
            return image, label

    def validation_parse_func(self, filename, label):
        image = self._imdecode(filename)
        image = self._image_augment(image, self.validation_augment)

        if self.n_classes is not None:
            label = tf.one_hot(label, depth=self.n_classes)

        if MULTI_OUTPUT:
            return image, (label, label)
        else:
            return image, label


class TFDataGenerator(ImageParse):
    def __init__(self, sess, images_path, images_label, imshape, batch_size=16, phase_train=True, repeat=-1):
        """
        :param sess:
        :param images_path:
        :param images_label:
        :param imshape:
        :param batch_size:
        :param phase_train:
        """
        super(TFDataGenerator, self).__init__(imshape=imshape)
        self._started = False
        self._epoch = 0

        self.sess = sess
        self.batch_size = batch_size
        self.repeat = repeat
        # if config.debug == 1:
        #     end_idx = 10000
        # else:
        #     end_idx = None
        # image_list, label_list, val_image_list, val_label_list = Datset.load_feedata(config.train_data_path, end_idx=end_idx)

        # imparse = Datset.ImageParse(imshape=imshape)
        if phase_train:
            # parse_func = imparse.train_parse_func
            self.parse_func = self.train_parse_func
        else:
            # parse_func = imparse.validation_parse_func
            self.parse_func = self.validation_parse_func

        self.images_info = np.stack([images_path, images_label], axis=1)

    def _start_discard(self):
        # filenames = tf.constant(images_path)
        # filelabels = tf.constant(images_label, dtype=tf.int64)
        # self.dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        # self._start()
        raise Exception('废弃，因为tf.data.Dataset.shuffle的reshuffle_each_iteration操作只能在buffer_size中reshuffle，而不能在整个数据集上reshuffle!')

    def _start(self):
        """
        因为tf.data.Dataset.shuffle的reshuffle_each_iteration操作只能在buffer_size中reshuffle，而不能在整个数据集上reshuffle,
        故这里使用np.random.shuffle来进行全局的shuffle.
        :return:
        """
        # np.random.shuffle(self.images_info)
        filenames = tf.constant(self.images_info[:, 0])
        filelabels = tf.constant(self.images_info[:, 1], dtype=tf.int64)
        self.dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        self.dataset = self.dataset.map(self.parse_func, num_parallel_calls=4)  # tf.data.experimental.AUTOTUNE
        self.dataset = self.dataset.shuffle(buffer_size=10,
                                  # seed=tf.compat.v1.set_random_seed(666),
                                  # reshuffle_each_iteration=True
                                  ).batch(self.batch_size).prefetch(5)  # repeat 不指定参数表示允许无穷迭代
        # self.dataset = self.dataset.repeat(self.repeat)

        self.iterator = self.dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

        self.sess.run(self.iterator.initializer)

    def next_batch(self):
        if not self._started:
            self._start()
            self._started = True

        try:
            images, labels = self.sess.run(self.next_element)
        except tf.errors.OutOfRangeError:
            if self.repeat == 1:
                raise Exception('iterate finish!')
            elif self.repeat == -1:
                # self.sess.run(self.iterator.initializer)
                self._start()
                images, labels = self.sess.run(self.next_element)
            else:
                raise Exception('iterate finish! On the other hand, repeat parameter is not supported!')

        return images, labels


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'pixels':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64)
        })
    decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [182, 182, 3])

    images = tf.image.resize_with_crop_or_pad(images, 160, 160)

    images = (images - 127.5) / 128.0

    labels = tf.cast(features['label'], tf.int32)
    #pixels = tf.cast(features['pixels'],tf.int32)
    return images, labels


class TFRecordDataGenerator(ImageParse):
    def __init__(self, sess, tfrecords, imshape, batch_size=16, phase_train=True, repeat=-1):
        """
        :param sess:
        :param images_path:
        :param images_label:
        :param imshape:
        :param batch_size:
        :param phase_train:
        """
        super(TFRecordDataGenerator, self).__init__(imshape=imshape)
        self._started = False
        self._epoch = 0

        self.sess = sess
        self.tfrecords = tfrecords
        self.batch_size = batch_size
        self.repeat = repeat

        if phase_train:
            self.parse_func = self.train_parse_func
        else:
            self.parse_func = self.validation_parse_func

    def _start_discard(self):
        # filenames = tf.constant(images_path)
        # filelabels = tf.constant(images_label, dtype=tf.int64)
        # self.dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        # self._start()
        raise Exception('废弃，因为tf.data.Dataset.shuffle的reshuffle_each_iteration操作只能在buffer_size中reshuffle，而不能在整个数据集上reshuffle!')

    def _start(self):
        train_files = tf.train.match_filenames_once(self.tfrecords, name='tfrecords')
        dataset = tf.data.TFRecordDataset(train_files)
        dataset = dataset.map(parser, num_parallel_calls=8)
        dataset = dataset.shuffle(buffer_size=5000,
                                  # seed=tf.compat.v1.set_random_seed(666),
                                  reshuffle_each_iteration=True
                                  ).batch(self.batch_size).prefetch(buffer_size=1000)  # tf.data.experimental.AUTOTUNE

        self.iterator = dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

        # self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.variables_initializer([train_files]))

        self.sess.run(self.iterator.initializer)

    def next_batch(self):
        if not self._started:
            self._start()
            self._started = True

        try:
            images, labels = self.sess.run(self.next_element)
        except tf.errors.OutOfRangeError:
            if self.repeat == 1:
                raise Exception('iterate finish!')
            elif self.repeat == -1:
                self.sess.run(self.iterator.initializer)
                # self._start()
                images, labels = self.sess.run(self.next_element)
            else:
                raise Exception('iterate finish! On the other hand, repeat parameter is not supported!')

        return images, labels


class DataSet(object):
    def __init__(self, data_path, batch_size=1, repeat=1, buffer_size=10000):
        super(DataSet, self).__init__()

        train_set, val_set = load_data(data_path)
        self.n_classes = len(train_set)
        train_image_list, train_label_list = get_image_paths_and_labels(train_set, shuffle=True)
        self.epoch_size = math.ceil(len(train_image_list) / batch_size)  # 迭代一轮所需要的训练次数

        filenames = tf.constant(train_image_list)
        labels = tf.constant(train_label_list)

        # 此时dataset中的一个元素是(filename, label)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 此时dataset中的一个元素是(image_resized, label)
        dataset = dataset.map(_parse_function)

        # 此时dataset中的一个元素是(image_resized_batch, label_batch)
        # dataset = dataset.shuffle(buffer_size=1000).batch(32).repeat(10)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=tf.set_random_seed(666), reshuffle_each_iteration=True).batch(batch_size).repeat(repeat)

        iterator = dataset.make_one_shot_iterator()
        self.get_next = iterator.get_next()


class TFDataset(object):
    def __init__(self, datas, labels, is_train=True):
        super(TFDataset, self).__init__()
        batch_size = 100
        buffer_size = 1000
        repeat = 1
        self.epoch_size = math.ceil(len(datas) / batch_size)  # 迭代一轮所需要的训练次数

        filenames = tf.constant(datas)
        labels = tf.constant(labels)

        # 此时dataset中的一个元素是(filename, label)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        # 此时dataset中的一个元素是(image_resized, label)
        dataset = dataset.map(_parse_function)

        # 此时dataset中的一个元素是(image_resized_batch, label_batch)
        # dataset = dataset.shuffle(buffer_size=1000).batch(32).repeat(10)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=tf.set_random_seed(666),
                                  reshuffle_each_iteration=True).batch(batch_size).repeat(repeat)

        iterator = dataset.make_one_shot_iterator()
        self.get_next = iterator.get_next()

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        self.count += 1
        return self.get_next

        if self.count == len(self.dataset):
            raise StopIteration

        images = []
        labels = []
        start_index = min(len(self.dataset), self.count)
        end_index = min(len(self.dataset), self.count + self.batch_size)

        # datas = self.dataset[start_index:end_index]  # 如果是SiameseDataset则无法使用start:end这种索引方式，不过下面这个方式是通用的。

        for i, index in enumerate(range(start_index, end_index)):
            data = self.dataset[index]
            (img1, img2), label = data

            image_pair = []
            channel = 1
            for img in [img1, img2]:
                if channel == 1:
                    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                    image = np.expand_dims(image, axis=2)
                elif channel == 3:
                    image = cv2.imread(data)
                else:
                    raise Exception("just support 1 and 3 channel!")
                image = image.astype(np.float32)
                image = image / 255.0
                image_pair.append(image)

            labels.append(label)
            images.append(image_pair)
            self.count += 1
        images = np.array(images)
        images1 = images[:, 0]
        images2 = images[:, 1]
        labels = np.array(labels)

        return (images1, images2), labels


class FaceDataset:
    def __init__(self, path, train=True, transform=None):
        train_set, valid_set = load_data(path)

        self.train = train
        self.transform = transform
        self.train_data, self.train_labels = get_image_paths_and_labels(train_set)
        self.test_data, self.test_labels = get_image_paths_and_labels(valid_set)

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels)
        self.test_data = np.array(self.test_data)
        self.test_labels = np.array(self.test_labels)

    def __len__(self):
        return len(self.train_data)


class DataIterator:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.count = 0
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.count = 0
        return self

    def __next__ok(self):
        if self.count == len(self.dataset):
            raise StopIteration

        datas = []
        start_index = min(len(self.dataset), self.count)
        end_index = min(len(self.dataset), self.count + self.batch_size)

        # datas = self.dataset[start_index:end_index]  # 如果是SiameseDataset则无法使用start:end这种索引方式，不过下面这个方式是通用的。

        for i, index in enumerate(range(start_index, end_index)):
            data = self.dataset[index]
            datas.append(data)
            self.count += 1
        datas = np.array(datas)

        return datas

    def __next__(self):
        if self.count == len(self.dataset):
            raise StopIteration

        images = []
        labels = []
        start_index = min(len(self.dataset), self.count)
        end_index = min(len(self.dataset), self.count + self.batch_size)

        # datas = self.dataset[start_index:end_index]  # 如果是SiameseDataset则无法使用start:end这种索引方式，不过下面这个方式是通用的。

        for i, index in enumerate(range(start_index, end_index)):
            data = self.dataset[index]
            (img1, img2), label = data

            image_pair = []
            channel = 1
            for img in [img1, img2]:
                if channel == 1:
                    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                    image = np.expand_dims(image, axis=2)
                elif channel == 3:
                    image = cv2.imread(data)
                else:
                    raise Exception("just support 1 and 3 channel!")
                image = image.astype(np.float32)
                image = image / 255.0
                image_pair.append(image)

            labels.append(label)
            images.append(image_pair)
            self.count += 1
        images = np.array(images)
        images1 = images[:, 0]
        images2 = images[:, 1]
        labels = np.array(labels)

        return (images1, images2), labels


class Dataset(object):
    """
    普通数据集，具有迭代功能，能够顺序产生数据
    """

    def __init__(self, datas, labels, shape=(), batch_size=1, is_train=True, one_hot=True):
        super(Dataset, self).__init__()
        self.count = 0
        self.datas = datas
        self.labels = labels
        self.shape = shape
        self.batch_size = batch_size
        self.num_class = len(set(labels))
        self.one_hot = one_hot

        self.train = is_train
        # self.transform = self.mnist_dataset.transform

        """
        if self.train:
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
        """

    """
    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.datas[index], self.labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.datas[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        '''
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        '''
        return img1, img2, target
    """

    def __iter__(self):
        """这个函数并不会被调用
        :return:
        """
        self.count = 0
        return self

    def __next__(self):
        if self.count == len(self.datas):
            self.count = 0
            # raise StopIteration

        datas = []
        labels = []
        start_index = min(len(self.datas), self.count)
        end_index = min(len(self.datas), self.count + self.batch_size)

        channel = self.shape[-1]
        for i, index in enumerate(range(start_index, end_index)):
            data = self.datas[index]
            label = self.labels[index]
            image = []
            if channel == 1:
                image = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
                image = np.expand_dims(image, axis=2)
            elif channel == 3:
                image = cv2.imread(data)
            image = image.astype(np.float32)
            image = image / 255.0
            # image /= 127.5
            datas.append(image)
            labels.append(label)
            self.count += 1
        datas = np.array(datas)
        if self.one_hot:
            labels = to_categorical(labels, self.num_class)
        else:
            labels = np.array(labels)

        return datas, labels

    def __len__(self):
        """这个函数并不会被调用
        :return:
        """
        return len(self.datas)


# TODO: 试试继承tf.data.Dataset能不能实现迭代。
class TFSiameseDataset(tf.data.Dataset):
    def __init__(self):
        super(TFSiameseDataset, self).__init__()


def tmp_sample_balance(positive_pairs, negative_pairs):
    np.random.shuffle(positive_pairs)
    np.random.shuffle(negative_pairs)
    offset = len(positive_pairs) - len(negative_pairs)
    if offset < 0:
        negative_pairs = negative_pairs[-offset:]
    else:
        positive_pairs = positive_pairs[offset:]

    return positive_pairs, negative_pairs


class SiameseDataset(object):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, datas, labels, is_train=True):
        super(SiameseDataset, self).__init__()
        self.datas = datas
        self.labels = labels

        self.train = is_train
        # self.transform = self.mnist_dataset.transform

        print('\n***************************SiameseDataset 总 样 本 量***************************')
        print('len(self.labels)={}, len(self.datas)={}'.format(len(self.labels), len(self.datas)))
        print('num_class={}'.format(len(set(self.labels))))
        print('*******************************************************************************\n')

        if self.train:
            self.labels_set = set(self.labels)
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.labels
            self.test_data = self.datas
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = []
            for i in range(0, len(self.test_data), 1):
                label_indices = self.label_to_indices[self.test_labels[i].item()]
                if len(label_indices) <= 1:
                    continue
                siamese_index = random_state.choice(label_indices)
                while siamese_index == i:
                    siamese_index = np.random.choice(label_indices)
                if [i, siamese_index, 1] in positive_pairs or [siamese_index, i, 1] in positive_pairs:
                    continue
                positive_pairs.append([i, siamese_index, 1])

            negative_pairs = []
            for i in range(1, len(self.test_data), 1):
                label1 = self.test_labels[i].item()
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = random_state.choice(self.label_to_indices[siamese_label])
                if [i, siamese_index, 0] in negative_pairs or [siamese_index, i, 0] in negative_pairs:
                    continue
                negative_pairs.append([i, siamese_index, 0])

            positive_pairs, negative_pairs = tmp_sample_balance(positive_pairs, negative_pairs)

            self.test_pairs = positive_pairs + negative_pairs
            np.random.shuffle(self.test_pairs)

            print('\n**************************Siamese pairs 样 本 量********************************')
            print('len(self.labels)={}, len(self.datas)={}'.format(len(self.labels), len(self.datas)))
            print('len(positive_pairs)={}, len(negative_pairs)={}, total_pairs={}'.format(len(positive_pairs), len(negative_pairs), len(self.test_pairs)))
            print('*******************************************************************************\n')

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.datas[index], self.labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.datas[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        """
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        """
        return (img1, img2), target

    def __len__(self):
        return len(self.datas)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, datas, labels, shape=(), batch_size=1, one_hot=True, shuffle=True):
        super(DataGenerator, self).__init__()
        self.count = 0
        self.datas = datas
        self.labels = labels
        self.shape = shape
        self.batch_size = batch_size
        self.num_class = len(set(labels))
        self.shuffle = shuffle
        self.one_hot = one_hot

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.datas) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = self.datas[indexes]
        batch_label = self.labels[indexes]

        X, y = self.data_generation(batch_data, batch_label)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.datas))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_data, batch_label):
        # X = np.empty((self.batch_size, *self.shape), dtype=np.float32)  # python2不支持
        X = np.empty((self.batch_size, self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        channel = self.shape[-1]
        for i, (data, label) in enumerate(zip(batch_data, batch_label)):
            image = []
            if channel == 1:
                image = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
                image = np.expand_dims(image, axis=2)
            elif channel == 3:
                image = cv2.imread(data)
            image = image / 255
            X[i, ] = image  # .astype(np.float32)
            y[i] = label

        return X, keras.utils.to_categorical(y, num_classes=self.num_class)


def show_dataset(dataset):
    for data in dataset.take(1):
        (images, proccessed_images), (labels1, labels2) = data
        for i, (raw_image, preccessed_image) in enumerate(zip(images, proccessed_images)):
            cv2.imwrite(str(i)+'_tmp_raw_image.jpg', raw_image.numpy())
            cv2.imwrite(str(i)+'_tmp_proccessed.jpg', preccessed_image.numpy())
            # cv2.imshow('raw_image', raw_image.numpy())
            # cv2.imshow('preccessed_image', preccessed_image.numpy())
            # cv2.waitKey(0)
            print('cv2.imwrite: {}/{}'.format(i, len(images)))
        print('epoch end!')


def test_parse_function(data):
    for i in range(1000):
        data = data / 10
        data = data * 10

    return data


def config_feedata(images_info, label_axis=0, dup_base=(), validation_ratio=0.0, filter_cb=None):
    """
    有的时候为了样本均衡，需要对样本量较少的类别进行重复采样。
    注意：按照本函数思路，仅仅是对原样本进行copy来重复采样。所以在训练之前还应当做数据增强。
    :param images_info:
    :param label_axis: 指定images_info中的哪一列是类别标签信息。
    :param dup_base: 该元组长度只能是1或2，当len(dup_base)=1时，dup2size=dup_base[0];
        当len(dup_base)=2时，dup2size根据每类样本的数量线性计算得到，计算公式详见代码.
    :param validation_ratio: 0.0, 不拆分数据集；大于0小于1时，表示按该比例拆分；-1，按代码中的公式线性拆分
    :param filter_cb: 自定义对每个类别样本的回调函数
        最终每个类别的数据均衡应该同时满足dup2range和filter_cb这两个限定条件。
    :return:
    """
    VAL_STATE = 1
    TRAIN_STATE = 0
    if len(dup_base) == 0:
        dup2size_func = None
    elif len(dup_base) == 1:
        dup2size_func = lambda x: dup_base[0]
    elif len(dup_base) == 2:
        # np.random.seed(666)
        # dup2size_func = lambda x: np.random.randint(dup_base[0], dup_base[1])  # 随机重复抽样到dup2range范围内

        base_x1 = 100
        base_x2 = 200
        assert (base_x1 > dup_base[0]) and (base_x2 > dup_base[1])
        dupk = (base_x2-dup_base[1])/(base_x1-dup_base[0])
        dupb = base_x2-base_x1*dupk
        dup2size_func = lambda x: min(base_x2, int(dupk*x+dupb))
    else:
        raise Exception('len(dup_base) just in range [0, 2]!')

    if validation_ratio == 0:
        val_ratio_func = None
    elif validation_ratio == -1:
        valk = (10 - 3) / (350 - 20)
        valb = 3 - 20 * valk
        val_ratio_func = lambda x: min(10, int(valk * x + valb))
    elif (validation_ratio > 0) and (validation_ratio < 1):
        val_ratio_func = lambda x: min(10, int(x*validation_ratio))
    else:
        raise Exception('validation_ratio just equal to -1 or [0, 1)!')

    extend_images_info = []
    cls_names = set(images_info[:, label_axis])
    for i, cls in enumerate(cls_names):
        info = images_info[np.where(images_info[:, label_axis] == cls)]
        np.random.shuffle(info)

        if len(info) < 5:
            print('debug')

        val_info = []
        if val_ratio_func is not None:
            val_size = val_ratio_func(len(info))
            val_info = info[0:val_size]
            info = info[val_size:]

        if dup2size_func is not None:
            if filter_cb is None or (filter_cb is not None and filter_cb(info)):
                dup2size = dup2size_func(len(info))
                if len(info) < dup2size:
                    extend_info = []

                    if len(info) == 0:
                        print('debug')

                    multiple = dup2size // len(info) - 1
                    for m in range(multiple):
                        extend_info.extend(info)

                    extend_info.extend(info[0:dup2size % len(info)])
                    extend_info = np.array(extend_info)
                    info = np.concatenate((info, extend_info))

        if len(val_info) > 0:
            info = np.concatenate((val_info, info))

        mark = np.zeros((len(info), 1), dtype=np.int32)
        mark[0:len(val_info)] = VAL_STATE
        label = np.ones([len(info), 1], dtype=np.int32) * i
        info = np.hstack((mark, label, info))
        extend_images_info.extend(info)
        tools.view_bar('config feed data:', i + 1, len(cls_names))
    print('')
    images_info = np.array(extend_images_info)

    return images_info


# 制作、加载用于feed到网络的数据.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def make_feedata(datasets, save_file, title_line='\n', filter_cb=None):
    """
    制作用于feed到网络的数据.
    :param datasets: datasets must be a list, and every element in the list must be dict.
    :param save_file: save the resulting data to save_file.
    :param title_line: first line write to save_file, just remark.
    :return:
    """
    dataset_keys = ['root_path', 'csv_file']
    for dataset in datasets:
        if type(dataset) != dict:
            raise Exception('datasets must be a list, and every element in the list must be dict.')
        for key in dataset.keys():
            if key not in dataset_keys:
                raise Exception('dataset dict expect {}, but received {}'.format(dataset_keys, key))

    images_info = []
    for dataset in datasets:
        imgs_info = tools.load_csv(dataset['csv_file'], start_idx=1)
        imgs_path = [os.path.join(info[0], info[2]) for info in imgs_info]
        imgs_info = tools.concat_dataset(dataset['root_path'], imgs_path, imgs_info[:, 1])
        images_info.extend(imgs_info)
    images_info = np.array(images_info)

    # images_info = config_feedata(images_info, validation_ratio=-1, dup_base=(20, 80), filter_cb=filter_cb)
    images_info = config_feedata(images_info, validation_ratio=-1, dup_base=(80, 180), filter_cb=filter_cb)

    tools.save_csv(images_info, save_file, title_line=title_line)


def load_feedata(feed_file, shuffle=True, start_idx=1, end_idx=None, all4train=False):
    imgs_info = tools.load_csv(feed_file, start_idx=start_idx, end_idx=end_idx)  # , end_idx=10000
    if shuffle:
        np.random.shuffle(imgs_info)

    if False:
        # 如果feed_file中的数据是按照 'cls_name,image_path'这样的格式排列的，则需要通过下面的方法赋予 label.
        # ************************************************************
        images_path = imgs_info[:, 1]
        labels_name = imgs_info[:, 0]
        cls_name = set(labels_name)
        images_label = np.zeros_like(labels_name, dtype=np.int32)
        for i, cls in enumerate(cls_name):
            images_label[np.where(labels_name == cls)] = i

            tools.view_bar('load feed data:', i + 1, len(cls_name))
        print('')
        # ************************************************************
    else:
        train_info = imgs_info[np.where(imgs_info[:, 0] == '0')]
        validation_info = imgs_info[np.where(imgs_info[:, 0] == '1')]

        train_images_label = train_info[:, 1].astype(np.int32).tolist()
        train_images_path = train_info[:, 3].tolist()

        validation_images_label = validation_info[:, 1].astype(np.int32).tolist()
        validation_images_path = validation_info[:, 3].tolist()

        if all4train:
            train_images_path = train_images_path + validation_images_path
            train_images_label = train_images_label + validation_images_label
            validation_images_path = []
            validation_images_label = []

            if shuffle:
                image_label = np.array([train_images_label, train_images_path]).transpose()
                np.random.shuffle(image_label)
                train_images_label = image_label[:, 0].astype(np.int32).tolist()
                train_images_path = image_label[:, 1].tolist()

    return train_images_path, train_images_label, validation_images_path, validation_images_label
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# 直接调用DataIterator的话，跟for循环迭代列表没什么两样
# 本示例只是验证一下DataIterator代码是否能用。
if __name__ == '__main__1':
    from utils import util
    data_path = '/path/to/mnist'
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 100
    dataiter = DataIterator(images_path, batch_size=10)

    for data in dataiter:
        print(data)

    for data in dataiter:
        print(data)


# 使用自定义的Dataset数据集
if __name__ == '__main__2':
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 100

    dataset = Dataset(images_path, images_label, shape=(28, 28, 1), batch_size=batch_size)

    for data in dataset:
        print(data)


if __name__ == '__main__3':
    '''
    SiameseDataset产生的是siamese图片路径和标签，而DataIterator是迭代加载SiameseDataset的图片和标签。
    使用自定义的SiameseDataset和DataIterator配合产生siamese数据
    '''
    images_path, images_label = util.get_dataset(data_path)
    num_class = len(set(images_label))
    batch_size = 100

    dataset = SiameseDataset(images_path, images_label)
    dataset = DataIterator(dataset, batch_size=10)

    for images1, images2, label in dataset:
        print(images1.shape, images2.shape, label.shape)


if __name__ == '__main__test':
    raise Exception('废弃')
    data_path = '/home/xiaj/res/mnist'
    train_set, val_set = load_data(data_path)
    train_image_list, train_label_list = get_image_paths_and_labels(train_set)

    filenames = tf.constant(train_image_list)
    labels = tf.constant(train_label_list)

    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    # dataset = dataset.shuffle(buffer_size=1000).batch(32).repeat(10)
    dataset = dataset.shuffle(buffer_size=10, seed=666).batch(5).repeat(5)

    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            count = 0
            while True:
                count += 1
                image, label = sess.run(one_element)
                print(count, image.shape, label)
        except tf.errors.OutOfRangeError:
            print('end!')


if __name__ == '__main__4':  # 检查显示ImageParse增强后的图片
    tf.enable_eager_execution()
    print('is eager executing: ', tf.executing_eagerly())

    # data_path = '/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44'
    # data_path = '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44'
    data_path = '/disk2/res/VGGFace2/Experiment/mtcnn_align182x182_margin44'
    train_images_path, train_images_label, validation_images_path, validation_images_label = load_dataset(data_path, min_nrof_cls=10, max_nrof_cls=40000, validation_ratio=0.2)

    train_count = len(train_images_path)
    validation_count = len(validation_images_path)
    n_classes = len(set(train_images_label))
    batch_size = 64
    buffer_size = min(train_count, 1000)
    repeat = 100
    # initial_epochs = min(repeat, 300)
    initial_epochs = repeat + 1
    imparse = ImageParse(imshape=(160, 160, 3))

    filenames = tf.constant(train_images_path)
    labels = tf.constant(train_images_label)
    train_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    train_dataset = train_dataset.map(imparse.train_parse_func)
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=tf.set_random_seed(666),
                              reshuffle_each_iteration=True).batch(batch_size).repeat(repeat)

    show_dataset(train_dataset)


if __name__ == '__main__5':  # 检验一下tf.data.Dataset.shuffle的reshuffle_each_iteration参数效果
    import time

    tf.enable_eager_execution()
    print('is eager executing: ', tf.executing_eagerly())

    num_train_samples = 100
    train_dataset = np.arange(num_train_samples)
    batch_size = 5
    buffer_size = min(num_train_samples, 20)
    train_steps_per_epoch = num_train_samples // batch_size

    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.map(test_parse_function, 4)

    train_dataset = train_dataset.shuffle(buffer_size=buffer_size,
                                          seed=tf.compat.v1.set_random_seed(666),
                                          reshuffle_each_iteration=True).batch(batch_size).repeat()

    while True:
        start_time = time.time()
        for i, data in enumerate(train_dataset):
            print(data.numpy(), end=' ')
            if (i+1) * batch_size >= num_train_samples:
                print('debug')
            # if (i != 0) and ((i+1) % train_steps_per_epoch == 0):
            #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            #     print('time: ', time.time() - start_time)
            #     start_time = time.time()
            #     print('\n')


def dup_sampling_check_happyjuzi(info):
    if len(info) > 200:
        return False

    marks = ['happyjuzi_mainland', 'happyjuzi_HongkongTaiwan', 'happyjuzi_JapanKorea']
    # samples_info = []
    for dat in info:
        if 'VGGFace2' in dat[-1]:
            # 不对VGGFace2数据集进行重复抽样
            if len(info) < 60:
                print('include VGGFace2: but len(info):{} < 60\n'.format(len(info)))
                print('************************************************************')
                print(info)
                print('************************************************************')
                return True
            else:
                return False

        findit = False
        for mark in marks:
            if mark in dat[-1]:
                findit = True
                # samples_info.append(dat)
                break
        if not findit:
            raise Exception('Only support dataset: {}'.format(marks))

    # samples_info = np.array(samples_info)
    # return samples_info
    return True


def dup_sampling_check_gcface(info):
    if len(info) > 200:
        return False

    marks = []
    # samples_info = []
    for dat in info:
        if 'VGGFace2' in dat[-1]:
            # 不对VGGFace2数据集进行重复抽样
            if len(info) < 60:
                print('include VGGFace2: but len(info):{} < 60\n'.format(len(info)))
                print('************************************************************')
                print(info)
                print('************************************************************')
                return True
            else:
                return False

        findit = False
        for mark in marks:
            if mark in dat[-1]:
                findit = True
                # samples_info.append(dat)
                break
        if not findit:
            raise Exception('Only support dataset: {}'.format(marks))

    # samples_info = np.array(samples_info)
    # return samples_info
    return True


def make_feedata_from_gcface(data_path, save_file, title_line='\n'):
    images_list = tools.load_image(data_path, subdir='', min_num_image_per_class=5, del_under_min_num_class=False, min_area4del=0)

    images_info = []
    for info in images_list:
        label_name = 'gcface_' + info[0]
        image_path = os.path.join(data_path, info[0], info[1])
        images_info.append([label_name, image_path])
    images_info = np.array(images_info)

    images_info = config_feedata(images_info, validation_ratio=0, dup_base=(80, 180), filter_cb=None)

    tools.save_csv(images_info, save_file, title_line=title_line)


def make_feedata_from_CASIAFaceV5(data_path, save_file, title_line='\n'):
    images_list = tools.load_image(data_path, subdir='', min_num_image_per_class=4, del_under_min_num_class=False, min_area4del=0)

    images_info = []
    for info in images_list:
        label_name = 'CASIAFaceV5_' + info[0]
        image_path = os.path.join(data_path, info[0], info[1])
        images_info.append([label_name, image_path])
    images_info = np.array(images_info)

    images_info = config_feedata(images_info, validation_ratio=0, dup_base=(80, 180), filter_cb=None)

    tools.save_csv(images_info, save_file, title_line=title_line)


if __name__ == '__main__':
    data_path = '/disk1/home/xiaj/res/face/gcface/duopler/clean/20190912-cleaned-mtcnn_align182x182_margin44'
    make_feedata_from_gcface(data_path, 'gcface.csv', title_line='train_val,label,person_name,image_path\n')

    # data_path = '/disk2/res/CASIA-FaceV5/CASIA-FaceV5-000-499-mtcnn_align182x182_margin44'
    # make_feedata_from_CASIAFaceV5(data_path, 'CASIA-FaceV5.csv', title_line='train_val,label,person_name,image_path\n')


if __name__ == '__main__7':  # 加载facenet训练所需要的数据集
    # train_dataset = [{'root_path': '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44',
    #                   'csv_file': '/disk1/home/xiaj/res/face/VGGFace2/Experiment/VGGFace2_cleaned_with_happyjuzi_mainland.csv'},
    #                  {'root_path': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_mainland_cleaning',
    #                   'csv_file': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_mainland_cleaned.csv'},
    #                  ]

    # train_dataset = [{# 'root_path': '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44',
    #                   'root_path': '/disk2/res/VGGFace2/Experiment/mtcnn_align182x182_margin44',
    #                   'csv_file': '/disk1/home/xiaj/res/face/VGGFace2/Experiment/VGGFace2_cleaned_with_happyjuzi_mainland_HongkongTaiwan.csv'},
    #
    #                  {'root_path': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_mainland_cleaning',
    #                   'csv_file': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_mainland_cleaned-while_include_HkTw.csv'},
    #
    #                  {'root_path': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_HongkongTaiwan_cleaning',
    #                   'csv_file': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_HongkongTaiwan_cleaned.csv'},
    #                  ]

    train_dataset = [
                     # {  # 'root_path': '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44',
                     # 'root_path': '/disk2/res/VGGFace2/Experiment/mtcnn_align182x182_margin44',
                     # 'csv_file': '/disk1/home/xiaj/res/face/VGGFace2/Experiment/VGGFace2_cleaned_with_happyjuzi_mainland_HongkongTaiwan_JapanKorea.csv'},
                     {'root_path': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_mainland_cleaning',
                      'csv_file': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_mainland_cleaned_after_clean_HKTwJaKo.csv'},
                     {'root_path': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_HongkongTaiwan_cleaning',
                      'csv_file': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_HongkongTaiwan_cleaned_after_clean_HkTwJaKo.csv'},
                     {'root_path': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/mtcnn_align182x182_margin44_happyjuzi_JapanKorea_cleaning',
                      'csv_file': '/disk1/home/xiaj/res/face/GC-WebFace/Experiment/happyjuzi_JapanKorea_cleaned_after_clean_HkTwJaKo.csv'},
    ]

    make_feedata(train_dataset, 'tmp3_vggfromdisk2.csv', title_line='train_val,label,person_name,image_path\n', filter_cb=dup_sampling_check_happyjuzi)
    # datset.load_feedata('tmp2.csv')


