import glob
import os
import shutil
import sys
import time

import tensorflow as tf

from common.metrics import *


class BANN_paper:
    def __init__(self, dataset_name, teacher_model, student_model, train_loader, val_loader, optimizer_student,
                 num_gen, num_classes,
                 loss_fn=tf.keras.losses.KLDivergence(), temp=20.0, distil_weight=0.5, log=False, batch_size_train=512,
                 batch_size_valid=256, improved=False):
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_student = optimizer_student
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.ce_fn = tf.keras.losses.CategoricalCrossentropy()
        self.gen = 0
        self.num_gen = num_gen
        self.init_weights = student_model.get_weights()
        self.init_optim = student_model.get_weights()
        self.num_classes = num_classes
        self.best_model_gen = 0
        self.improved = improved
        self.base_dir = f"./improved_models_{self.dataset_name}" if improved else f"./paper_models_{self.dataset_name}"
        self.models_dir = self.base_dir + "/student-{}.h5"

    def train(self, epochs, n_batch):
        self.best_acc = 0
        for k in range(self.num_gen):
            print("Born Again : Gen {}/{}".format(k + 1, self.num_gen))
            print(f'Curr best acc {self.best_acc} achieved by {self.best_model_gen} gen')

            student_acc = self._train(
                epochs, n_batch
            )
            print(f'{k + 1} model achieved {student_acc} acc')
            save_dir = os.path.dirname(self.models_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.student_model.save_weights(self.models_dir.format(k + 1))
            if self.improved:
                if student_acc > self.best_acc:
                    self.best_acc = student_acc
                    self.best_model_gen = k + 1
                    print(f'Best Model changed to {self.best_model_gen}')
                self.teacher_model.load_weights(self.models_dir.format(self.best_model_gen))
            else:
                self.teacher_model.set_weights(self.student_model.get_weights())
            self.student_model.set_weights(self.init_weights)
            self.gen += 1

    # train the generator and discriminator
    def _train(self, n_epochs=20, n_batch=256):
        # manually enumerate epochs
        for i in range(n_epochs):
            len_dataset = 0
            # show the current epoch number
            print("[INFO] starting epoch {}/{}...".format(
                i + 1, n_epochs), end="")
            sys.stdout.flush()
            epochStart = time.time()
            epoch_loss = 0.0
            correct = 0
            for input_batch, label_batch in self.train_loader:
                label_batch_np = label_batch.numpy()
                ohe_labels = tf.one_hot(label_batch, self.num_classes)
                len_dataset += input_batch.shape[0]
                with tf.GradientTape(persistent=True) as tape:
                    student_pred = self.student_model(input_batch, training=True)
                    pred = np.argmax(student_pred, axis=1)
                    correct += np.sum(pred == label_batch_np)
                    teacher_pred = self.teacher_model(input_batch, training=False)
                    curr_loss = self.calculate_kd_loss(student_pred, teacher_pred, ohe_labels)
                    epoch_loss += curr_loss.numpy()
                gradient = tape.gradient(curr_loss, self.student_model.trainable_variables)
                self.optimizer_student.apply_gradients(zip(gradient, self.student_model.trainable_variables))
            # d_loss_epoch.append(mean(d_loss_batch))
            epoch_acc = correct / len_dataset
            _, epoch_val_acc = self._evaluate_model(self.student_model)
            # show timing information for the epoch
            epochEnd = time.time()
            elapsed = (epochEnd - epochStart) / 60.0
            # print("took {:.4} minutes".format(elapsed))
            # print(f'Epoch train acc - {epoch_acc}; Epoch loss - {epoch_loss}; Epoch val acc - {epoch_val_acc}')
        return epoch_val_acc

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """
        if self.gen == 0:
            return self.loss_fn(y_true, y_pred_student)
        s_i = tf.nn.log_softmax(y_pred_student / self.temp)
        t_i = tf.nn.softmax(y_pred_teacher / self.temp)
        KD_loss = tf.keras.losses.KLDivergence()(s_i, t_i) * (
                self.distil_weight * self.temp * self.temp
        )
        KD_loss += tf.keras.losses.CategoricalCrossentropy()(y_pred_student, y_true) * (1.0 - self.distil_weight)
        return KD_loss

    def evaluate(self):
        print("Evaluating Model Ensemble")
        models_dir = glob.glob(os.path.join(self.base_dir, "*.h5"))
        len_models = len(models_dir)
        outputs = []
        model = self.student_model
        for model_weight in models_dir:
            model.load_weights(model_weight)
            output, _ = self._evaluate_model(model, verbose=False)
            outputs.append(output)
        print("Total Models: ", len(outputs))
        total = 0
        correct = 0
        y_pred_value, y_pred_score, y = [], [], []
        for idx, (data_batch, target_batch) in enumerate(self.val_loader):
            output = outputs[0][idx] / len_models
            for k in range(1, len_models):
                output += outputs[k][idx] / len_models

            pred = np.argmax(output, axis=1)
            correct += np.sum(pred == np.array(target_batch))
            y_pred_value.extend(pred.squeeze().tolist())
            y_pred_score.extend(np.array(output))
            y.extend(np.array(target_batch))

        accuracy, all_metric = calculate_metrics(y, y_pred_score, y_pred_value)
        return accuracy, all_metric

    def _evaluate_model(self, model, n_batch=128, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        len_dataset = 0
        correct = 0
        outputs = []
        for input_batch, label_batch in self.val_loader:
            len_dataset += input_batch.shape[0]
            student_pred = model(input_batch, training=False)
            outputs.append(student_pred)
            pred = np.argmax(student_pred, axis=1)
            correct += np.sum(pred == label_batch.numpy())

        accuracy = correct / len_dataset

        if verbose:
            print("-" * 80)
            print("Validation Accuracy: {}".format(accuracy))
        return outputs, accuracy

    def clear_folder(self):
        folder = self.base_dir
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
