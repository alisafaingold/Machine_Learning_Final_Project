import numpy as np
from bayes_opt import BayesianOptimization
from keras import optimizers
from sklearn.model_selection import StratifiedKFold

from common import model
from common.utils import *
from paper import paper_model


def test_paper_model(dataset_name, train_data, valid_data, num_classes, input_shape, epochs=2, num_gen=2,
                     learning=0.001, batch_size_train=512, batch_size_valid=256, improved=False):
    student_model = model.get_model(input_shape, num_classes)
    teacher_model = model.get_model(input_shape, num_classes)
    student_model.set_weights(teacher_model.get_weights())
    optimizer = optimizers.SGD(lr=learning, momentum=0.9)
    train_data = train_data.batch(batch_size_train)
    valid_data_training = valid_data.batch(batch_size_valid)
    BANN = paper_model.BANN_paper(dataset_name, teacher_model, student_model, train_data, valid_data_training, optimizer, num_gen,
                                  num_classes,
                                  batch_size_train=512, batch_size_valid=256, improved=improved)
    BANN.train(epochs, 128)
    accuracy, all_metrics = BANN.evaluate()
    return BANN, accuracy, all_metrics


def model_cv(dataset_name, epochs, num_gen, learning, dataset, num_classes, improved=False):
    """Random Forest cross validation.
    This function will instantiate a random forest classifier with parameters
    n_estimators, min_samples_split, and max_features. Combined with data and
    targets this will in turn be used to perform cross validation. The result
    of cross validation is returned.
    Our goal is to find combinations of n_estimators, min_samples_split, and
    max_features that minimzes the log loss.
    """
    x_ds = dataset[0]
    input_shape = dataset[1]
    y_ds = [int(item[1]) for item in iter(x_ds)]
    x_placeholder = np.zeros(len(y_ds))
    cv_outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    scores = []
    best_score = 0
    for fold_idx, (train_val_indices, test_indices) in enumerate(cv_outer.split(x_placeholder, y_ds)):
        train_val_ds, test_ds = get_train_test_split_from_indices(input_shape,
                                                                  test_indices,
                                                                  train_val_indices, x_ds)
        model, accuracy, _ = test_paper_model(dataset_name, train_val_ds, test_ds, num_classes, input_shape, epochs=epochs,
                                              num_gen=num_gen,
                                              learning=learning, improved=improved)
        model.clear_folder()
        scores.append(accuracy)
        if accuracy > best_score:
            best_score = accuracy
    print(f'best_score - {best_score}')
    print(f'average score k cross - {np.mean(scores)}')
    return np.mean(scores)


def optimize_paper_model(dataset_name, train_dataset, num_classes, improved=False):
    """Apply Bayesian Optimization to Random Forest parameters."""

    def model_crossval(epochs, num_gen, learning):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return model_cv(
            dataset_name,
            epochs=int(epochs),
            num_gen=int(num_gen),
            learning=learning,
            dataset=train_dataset,
            num_classes=num_classes,
            improved=improved
        )

    optimizer = BayesianOptimization(
        f=model_crossval,
        pbounds={
            "epochs": (30, 40),
            "num_gen": (4, 7),
            "learning": (0.00001, 0.0001),
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(n_iter=25)

    print("Final result:", optimizer.max)
    return optimizer.max
