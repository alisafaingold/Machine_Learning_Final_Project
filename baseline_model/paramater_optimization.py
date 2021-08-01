import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold

from baseline_model import baseline_model
from common.utils import *


def test_baseline(train_data, valid_data, num_classes, input_shape, epochs=2,
                  learning=0.001, dropout=0.3, batch_size_train=256, batch_size_valid=128):
    train_data = train_data.batch(batch_size_train)
    valid_data_training = valid_data.batch(batch_size_valid)
    baseline = baseline_model.BaselineModel(image_shape=input_shape, lr=learning, dropout_rate=dropout,
                                            num_classes=num_classes)
    baseline.train(train_data, epochs=epochs)
    accuracy, all_metrics = baseline.evaluate(valid_data_training)
    return baseline, accuracy, all_metrics


def model_cv(epochs, learning, dataset, num_classes, dropout):
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
        model, accuracy, _ = test_baseline(train_val_ds, test_ds, num_classes, input_shape, epochs=epochs,
                                           learning=learning, dropout=dropout)
        scores.append(accuracy)
        if accuracy > best_score:
            best_score = accuracy
    print(f'best_score - {best_score}')
    print(f'average score k cross - {np.mean(scores)}')
    return np.mean(scores)


def optimize_baseline_model(train_dataset, num_classes):
    """Apply Bayesian Optimization to Random Forest parameters."""

    def model_crossval(learning, dropout, epochs=2):
        """Wrapper of RandomForest cross validation.
        Notice how we ensure n_estimators and min_samples_split are casted
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return model_cv(
            learning=learning,
            dropout=dropout,
            dataset=train_dataset,
            num_classes=num_classes,
            epochs=int(epochs)
        )

    optimizer = BayesianOptimization(
        f=model_crossval,
        pbounds={
            "learning": (0.00001, 0.0001),
            "dropout": (0.1, 0.3),
            "epochs": (30, 40),
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(n_iter=25)

    print("Final result:", optimizer.max)
    return optimizer.max
