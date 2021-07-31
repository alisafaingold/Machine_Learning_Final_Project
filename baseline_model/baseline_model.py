import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.applications import resnet
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.metrics import Precision, AUC, TrueNegatives, TruePositives, FalsePositives, FalseNegatives
from tensorflow.keras.optimizers import Adam

from common.metrics import calculate_metrics


@tf.autograph.experimental.do_not_convert
def my_loss_fn(y_true, y_pred):
    ohe_labels = tf.one_hot(y_true, y_pred.shape[1])
    ohe_labels = tf.reshape(ohe_labels, [-1, y_pred.shape[1]])
    return tf.keras.losses.categorical_crossentropy(ohe_labels, y_pred)


class BaselineModel:
    def __init__(self, image_shape, lr, dropout_rate, num_classes):
        self.image_shape = image_shape
        self.model = self.create_baseline(lr, dropout_rate, num_classes)

    def create_baseline(self, lr, dropout_rate, num_classes):
        base_model = resnet.ResNet101(include_top=False,
                                      input_shape=self.image_shape,
                                      weights='imagenet')

        # freeze all the weights except for the last
        for layer in base_model.layers[:-1]:
            layer.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=lr), loss=my_loss_fn,
                      metrics='accuracy')
        return model

    @tf.autograph.experimental.do_not_convert
    def train(self, x_train, epochs):
        self.model.fit(x_train, epochs=epochs, verbose=0)

    def evaluate(self, test_ds):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        len_dataset = 0
        y_pred_value, y_pred_score, y = [], [], []
        for input_batch, label_batch in test_ds:
            len_dataset += input_batch.shape[0]
            probabilities = self.model.predict(input_batch)
            y_pred_score.extend(np.array(probabilities))
            y_pred_value.append(np.argmax(probabilities, axis=1).squeeze().tolist())
            y.extend(np.array(label_batch))

        accuracy, all_metric = calculate_metrics(y, y_pred_score,
                                                 [item for sublist in y_pred_value for item in sublist])
        return accuracy, all_metric
