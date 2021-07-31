import tensorflow as tf


def list_to_tf_dataset_string(dataset, input_shape):
    """
    Create tf.Dataset with tuple of all images and all corresponding labels
    @param dataset: tuple of list of images and list of corresponding labels
    @param input_shape: shape og images
    @return: tf.Dataset with tuple of all images and all corresponding labels
    """
    def _dataset_gen():
        for e in dataset:
            yield e[0], e[1]

    ds = tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types=(tf.uint8, tf.int64),
        output_shapes=(input_shape, ())
    )
    return ds


def get_train_test_split_from_indices(input_shape, test_indices, train_val_indices,
                                      x_ds):
    """
    split data into train/val and test set by the given indices
    @param input_shape: images shape
    @param test_indices: indices for test set
    @param train_val_indices:  indices for train/validation set
    @param x_ds: data the need to be split
    @return: train ds and test ds
    """
    train_val_samples = []
    test_samples = []
    for samples_idx, samples in enumerate(iter(x_ds)):
        if samples_idx in test_indices:
            test_samples.append(samples)
        elif samples_idx in train_val_indices:
            train_val_samples.append(samples)
    test_ds = list_to_tf_dataset_string(test_samples, input_shape)  # maybe change to tf.string
    train_val_ds = list_to_tf_dataset_string(train_val_samples, input_shape)
    return train_val_ds, test_ds
