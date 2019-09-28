# -*- coding: UTF-8 -*-
import os
import math
import random
import cv2
# from scipy import misc
import skimage.transform
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from utils import tools


DEBUG = False


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


def load_dataset(data_path, shuffle=True, validation_ratio=0.0, min_nrof_cls=1, max_nrof_cls=999999999):
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
    images_path = []
    images_label = []
    images = os.listdir(data_path)
    images.sort()
    if DEBUG:
        images = images[0:5]
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

        images_path.extend([os.path.join(data_path, image, img) for img in imgs])
        images_label.extend([i] * len(imgs))
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
    # return misc.imrotate(image, angle, 'bicubic')  # scipy.misc 可能已经被弃用，在部分系统中已经无法使用，可使用skimage代替
    # TODO： 最好还是要解决一下使用scipy.misc调用rotate的问题。 因为skimage不支持输入为tfTensor
    return skimage.transform.rotate(image.numpy(), angle, order=3, preserve_range=True)


def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)


def _tfparse_function_train(filename, label):
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
    FIXED_STANDARDIZATION = 8
    FLIP = 16
    RANDOM_GLASS = 32
    RANDOM_COLOR = 64
    FIXED_CONTRACT = 128

    def __init__(self, imshape=(28, 28, 1)):
        self.imshape = imshape

        self.set_train_augment(random_crop=False)
        self.set_validation_augment(random_crop=False)

    def set_train_augment(self, random_crop=True, random_rotate=True, random_left_right_flip=True, fixed_standardization=True):
        self.train_augment = {ImageParse.RANDOM_CROP: random_crop,
                              ImageParse.RANDOM_ROTATE: random_rotate,
                              ImageParse.RANDOM_LEFT_RIGHT_FLIP: random_left_right_flip,
                              ImageParse.FIXED_STANDARDIZATION: fixed_standardization
                              }

    def set_validation_augment(self, random_crop=False, random_rotate=False, random_left_right_flip=False, fixed_standardization=True):
        self.validation_augment = {ImageParse.RANDOM_CROP: random_crop,
                                   ImageParse.RANDOM_ROTATE: random_rotate,
                                   ImageParse.RANDOM_LEFT_RIGHT_FLIP: random_left_right_flip,
                                   ImageParse.FIXED_STANDARDIZATION: fixed_standardization
                                   }

    def _imdecode(self, filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, self.imshape[2])
        # image = tf.convert_to_tensor(image, dtype=tf.float32)
        return image

    def _image_augment(self, image, augments):
        if augments[ImageParse.RANDOM_CROP]:
            image = tf.image.random_crop(image, self.imshape)
        else:
            image = tf.image.resize_image_with_crop_or_pad(image, self.imshape[0], self.imshape[1])
            # image = tf.image.resize_with_crop_or_pad(image, self.imshape[0], self.imshape[1])

        if augments[ImageParse.RANDOM_ROTATE]:
            image = tf.py_function(random_rotate_image, [image], tf.uint8)

        # image = tf.image.random_brightness(image, 0.4)
        # image = tf.image.random_contrast(image, 0.8, 2)

        if augments[ImageParse.RANDOM_LEFT_RIGHT_FLIP]:
            image = tf.image.random_flip_left_right(image)

        if augments[ImageParse.FIXED_STANDARDIZATION]:
            image = (tf.cast(image, tf.float32) - 127.5) / 128.0
        else:
            image = tf.image.per_image_standardization(image)

        return image

    def train_parse_func(self, filename, label):
        image = self._imdecode(filename)
        image = self._image_augment(image, self.train_augment)

        return image, (label, label)

    def validation_parse_func(self, filename, label):
        image = self._imdecode(filename)
        image = self._image_augment(image, self.validation_augment)

        return image, (label, label)


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
            for i in range(0, len(self.test_data), 8):
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
            for i in range(1, len(self.test_data), 8):
                label1 = self.test_labels[i].item()
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = random_state.choice(self.label_to_indices[siamese_label])
                if [i, siamese_index, 0] in negative_pairs or [siamese_index, i, 0] in negative_pairs:
                    continue
                negative_pairs.append([i, siamese_index, 0])

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
        images, (labels1, labels2) = data
        for image in images:
            cv2.imshow('augment', image.numpy())
            cv2.waitKey(0)


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


if __name__ == '__main__':
    tf.enable_eager_execution()
    print('is eager executing: ', tf.executing_eagerly())

    data_path = '/home/xiajun/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44'
    train_images_path, train_images_label, validation_images_path, validation_images_label = load_dataset(data_path, min_nrof_cls=10, max_nrof_cls=40000, validation_ratio=0.2)

    train_count = len(train_images_path)
    validation_count = len(validation_images_path)
    n_classes = len(set(train_images_label))
    batch_size = 64
    buffer_size = train_count
    repeat = 100
    # initial_epochs = min(repeat, 300)
    initial_epochs = repeat + 1

    filenames = tf.constant(train_images_path)
    labels = tf.constant(train_images_label)
    train_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    train_dataset = train_dataset.map(_tfparse_function_train)
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=tf.set_random_seed(666),
                              reshuffle_each_iteration=True).batch(batch_size).repeat(repeat)

    show_dataset(train_dataset)