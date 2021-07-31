import random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DATA_SIZE = 5000
random.seed(4)

def get_data_set(name, subset_num):
    """
    Get dataset by his name and his subset number, the option are:
    cifar_10, [0,1]
    fashion_mnist, [0,1]
    kmnist, [0,1,2]
    emnist_digits, [0,1,2]
    emnist_letters, [0,1,2]
    cifar_100, [0-9]
    @param name: data set name: cifar_10, fashion_mnist, kmnist, emnist_digits, emnist_letters, cifar_100
    @param subset_num: subset number
    @return: tuple (name_batch_num, data, num_class)
    """
    if name == 'cifar10':
        data = get_cifar_10(subset_num)
    elif name == 'cifar100':
        data = get_cifar_100(subset_num)
    elif name == 'beans':
        data = get_beans(subset_num)
    elif name == 'caltech101':
        data = get_caltech_101(subset_num)
    elif name == 'fashion_mnist':
        data = get_fashion_mnist(subset_num)
    elif name == 'mnist':
        data = get_mnist(subset_num)
    elif name == 'food101':
        data = get_food_101(subset_num)
    elif name == 'kmnist':
        data = get_kmnist(subset_num)
    else:
        raise ValueError('Data set name not exist, try: cifar10, cifar100, beans, caltech101, fashion_mnist, mnist, '
                         'food101 or kmnist')

    return (f'{name}_{subset_num}', data)


def get_cifar_10(subset_num):
    """
    cifar10 dataset
    :param subset_num: number of the wanted subset
    :return: tuple (subset, image shape, number of unique labels in the subset)
    """
    assert (subset_num <= 1), "cifar10 dataset don't have this subset, try 0 or 1"
    labels = [i for i in range(0, 10)]
    random.shuffle(labels)
    labels = np.reshape(labels, (2, 5))
    num_each_label = int(DATA_SIZE / 5)

    dataset, info = tfds.load(name='cifar10', with_info=True)
    train = dataset['train']
    test = dataset['test']
    data = train.concatenate(test)

    curr_labels = labels[subset_num]
    ds_all = filter_dataset_by_labels(curr_labels, data, num_each_label)
    print(f'** cifar10, {subset_num}, {curr_labels}, {len([x for x in ds_all])}')

    return (ds_all, ds_all.element_spec[0].shape, len(curr_labels))


def get_beans(subset_num):
    """
    Beans dataset
    :param subset_num: number of the wanted subset
    :return: tuple (subeset, image shape, number of unique labels in the subset)
    """
    assert (subset_num <= 0), "beans dataset don't have this subset, try 0"
    labels = [i for i in range(0, 3)]
    num_each_label = int(DATA_SIZE / 3)

    dataset, info = tfds.load(name='beans', with_info=True)
    train = dataset['train']
    test = dataset['test']
    data = train.concatenate(test)

    ds_all = filter_dataset_by_labels(labels, data, num_each_label)
    ds_all = normalize_data(ds_all)
    print(f'** beans, {subset_num}, {labels}, {len([x for x in ds_all])}')

    return (ds_all, ds_all.element_spec[0].shape, len(labels))


def get_caltech_101(subset_num):
    """
    caltech101 dataset
    :param subset_num: number of the wanted subset
    :return: tuple (subset, image shape, number of unique labels in the subset)
    """
    assert (subset_num <= 1), "caltech101 dataset don't have this subset, try 0 or 1"
    labels = [i for i in range(0, 100)]
    random.shuffle(labels)
    labels = np.reshape(labels, (2, 50))
    num_each_label = int(DATA_SIZE / 50)

    dataset, info = tfds.load(name='caltech101', with_info=True)
    train = dataset['train']
    test = dataset['test']
    data = train.concatenate(test)

    curr_labels = labels[subset_num]
    ds_all = filter_dataset_by_labels(curr_labels, data, num_each_label)
    ds_all = normalize_data(ds_all)
    return (ds_all, ds_all.element_spec[0].shape, len(curr_labels))


def get_fashion_mnist(subset_num):
    """
    fashion_mnist dataset
    :param subset_num: number of the wanted subset
    :return: tuple (subset, image shape, number of unique labels in the subset)
    """
    assert (subset_num <= 2), "fashion_mnist dataset don't have this subset, try 0, 1, 2"
    labels = [i for i in range(0, 9)]
    random.shuffle(labels)
    labels = np.reshape(labels, (3, 3))
    num_each_label = int(DATA_SIZE / 3)

    dataset, info = tfds.load(name='fashion_mnist', with_info=True)
    train = dataset['train']
    test = dataset['test']
    data = train.concatenate(test)

    curr_labels = labels[subset_num]
    ds_all = filter_dataset_by_labels(curr_labels, data, num_each_label)
    ds_all = normalize_data(ds_all)
    return (ds_all, ds_all.element_spec[0].shape, len(curr_labels))


def get_mnist(subset_num):
    """
    mnist dataset
    :param subset_num: number of the wanted subset
    :return: tuple (subset, image shape, number of unique labels in the subset)
    """
    assert (subset_num <= 2), "mnist dataset don't have this subset, try 0, 1, 2"
    labels = [i for i in range(0, 9)]
    random.shuffle(labels)
    labels = np.reshape(labels, (3, 3))
    num_each_label = int(DATA_SIZE / 3)

    dataset, info = tfds.load(name='mnist', with_info=True)
    train = dataset['train']
    test = dataset['test']
    data = train.concatenate(test)

    curr_labels = labels[subset_num]
    ds_all = filter_dataset_by_labels(curr_labels, data, num_each_label)
    ds_all = normalize_data(ds_all)
    return (ds_all, ds_all.element_spec[0].shape, len(curr_labels))


def get_kmnist(subset_num):
    """
     kmnist dataset
     :param subset_num: number of the wanted subset
     :return: tuple (subset, image shape, number of unique labels in the subset)
     """
    assert (subset_num <= 2), "kmnist dataset don't have this subset, try 0, 1, 2"
    labels = [i for i in range(0, 9)]
    random.shuffle(labels)
    labels = np.reshape(labels, (3, 3))
    num_each_label = int(DATA_SIZE / 3)

    dataset, info = tfds.load(name='kmnist', with_info=True)
    train = dataset['train']
    test = dataset['test']
    data = train.concatenate(test)

    curr_labels = labels[subset_num]
    ds_all = filter_dataset_by_labels(curr_labels, data, num_each_label)
    ds_all = normalize_data(ds_all)
    return (ds_all, ds_all.element_spec[0].shape, len(curr_labels))


def get_food_101(subset_num):
    """
    food101 dataset
    :param subset_num: number of the wanted subset
    :return: tuple (subset, image shape, number of unique labels in the subset)
    """
    assert (subset_num <= 1), "food101 dataset don't have this subset, try 0, 1"
    labels = [i for i in range(0, 100)]
    random.shuffle(labels)
    labels = np.reshape(labels, (2, 50))
    num_each_label = int(DATA_SIZE / 50)

    dataset, info = tfds.load(name='food101', with_info=True)
    train = dataset['train']
    validation = dataset['validation']
    data = train.concatenate(validation)

    curr_labels = labels[subset_num]
    ds_all = filter_dataset_by_labels(curr_labels, data, num_each_label)
    ds_all = normalize_data(ds_all)
    return (ds_all, ds_all.element_spec[0].shape, len(curr_labels))


def get_cifar_100(subset_num):
    """
    cifar100 dataset
    :param subset_num: number of the wanted subset
    :return: tuple (subset, image shape, number of unique labels in the subset)
    """
    assert (subset_num <= 4), "cifar100 dataset don't have this subset, try 0, 1, 2, 3, 4"
    labels = [i for i in range(0, 100)]
    random.shuffle(labels)
    labels = np.reshape(labels, (5, 20))
    num_each_label = int(DATA_SIZE / 20)

    dataset, info = tfds.load(name='cifar100', with_info=True)
    train = dataset['train']
    test = dataset['test']
    data = train.concatenate(test)

    curr_labels = labels[subset_num]
    ds_all = filter_dataset_by_labels(curr_labels, data, num_each_label)
    ds_all = normalize_data(ds_all)
    return (ds_all, ds_all.element_spec[0].shape, len(curr_labels))


def normalize_data(data):
    """
    Rescale an input in the [0, 255] range to be in the [0, 1] range
    :param data:
    :return:
    """
    if data.element_spec[0].shape[2] == 1:
        data = data.map(lambda x, y: (tf.image.grayscale_to_rgb(
            tf.image.resize(x, [32, 32])), y))
    else:
        data = data.map(lambda x, y: (tf.image.resize(x, [32, 32]), y))
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    normalized_ds = data.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds


def filter_dataset_by_labels(labels, data, num_each_label):
    """
    Create subset of the given data by selecting only the indices of the given labels
    @param labels: list with the labels the subset datset should be created
    :param data: dataset
    :param num_each_label: number that should be taken from each label to create balance dataset
    :return: no
    """
    first = True
    all_ds = None
    for label in labels:
        ds_l = data.filter(lambda d: d.get("label") == label).take(num_each_label)
        if first:
            all_ds = ds_l
            first = False
        else:
            all_ds = all_ds.concatenate(ds_l)
    y_ds = tf.cast(list(labels), tf.int64)
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(y_ds),
            values=tf.constant(list(range(len(y_ds)))),
        ),
        default_value=tf.constant(-1),
        name="class"
    )
    all_ds = all_ds.map(lambda d: (d.get('image'), table.lookup(d.get('label')))).cache()
    all_ds = all_ds.shuffle(DATA_SIZE, reshuffle_each_iteration=True).cache()
    return all_ds
