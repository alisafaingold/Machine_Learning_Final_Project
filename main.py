import sys
import time as t

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from baseline_model import paramater_optimization
from common.utils import *
from data_loader import get_data_set
from paper import parameter_optimizer


def run_experiment(algo_name, dataset_name, data, batch_size_train=512, batch_size_valid=256):
    if algo_name not in ['paper', 'improved', 'baseline']:
        raise ValueError('Algorithm name not exist, try: paper, improved or baseline')

    x_ds = data[0]
    input_shape = data[1]
    num_classes = data[2]

    metric_df = pd.DataFrame(columns=['AlgoName', 'Dataset Name', 'Cross Validation', 'Hyper Parameters Values',
                                      'Accuracy', 'TPR', 'FPR',
                                      'Precision', 'AUC', 'PR_Curve',
                                      'Training Time (s)', 'Inference Time (s)'])
    # Start kfold
    y_ds = [int(item[1]) for item in iter(x_ds)]
    x_placeholder = np.zeros(len(y_ds))
    cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for fold_idx, (train_val_indices, test_indices) in enumerate(cv_outer.split(x_placeholder, y_ds)):
        train_val_ds, test_ds = get_train_test_split_from_indices(input_shape,
                                                                  test_indices,
                                                                  train_val_indices, x_ds, batch_size_train,
                                                                  batch_size_valid)
        # Train on the 9 folds
        st_train = t.time()
        if algo_name == 'paper' or algo_name == 'improved':
            improved = True if algo_name == 'improved' else False
            opt_values = parameter_optimizer.optimize_paper_model(dataset_name, [train_val_ds, input_shape],
                                                                  num_classes, improved=improved)
        else:
            opt_values = paramater_optimization.optimize_baseline_model([train_val_ds, input_shape],
                                                                        num_classes)


        # Get optimization parameters
        if algo_name == 'paper' or algo_name == 'improved':
            num_gen = opt_values.get('params').get('num_gen')
            epochs = opt_values.get('params').get('epochs')
            learning = opt_values.get('params').get('learning')
            print(f"Fold : {fold_idx}, accuracy: {opt_values.get('target')}, epochs: {round(epochs, 2)},"
                  f" num_gen: {round(num_gen, 2)}, learning: {round(learning, 2)}")

        else:
            dropout = opt_values.get('params').get('dropout')
            learning = opt_values.get('params').get('learning')
            epochs = opt_values.get('params').get('epochs')
            print(f"Fold : {fold_idx}, accuracy: {opt_values.get('target')}, epochs: {round(epochs, 2)},"
                  f" dropout: {round(dropout, 2)}, learning: {round(learning, 2)}")

        if algo_name == 'paper' or algo_name == 'improved':
            improved = True if algo_name == 'improved' else False
            model, _, _ = parameter_optimizer.test_paper_model(dataset_name, train_val_ds, train_val_ds, num_classes,
                                                               input_shape,
                                                               learning=round(learning, 4),
                                                               epochs=int(round(epochs)),
                                                               num_gen=int(round(num_gen)),
                                                               improved=improved)

        else:
            model, _, _ = paramater_optimization.test_baseline(train_val_ds, train_val_ds, num_classes, input_shape,
                                                               learning=round(learning, 4),
                                                               epochs=int(round(epochs)),
                                                               dropout=dropout)
        et_train = t.time()

        # Test on the 1 fold
        st_test = t.time()
        if algo_name == 'paper' or algo_name == 'improved':
            test_ds = test_ds.batch(batch_size_valid)
            model.val_loader = test_ds
            test_accuracy, metric_test = model.evaluate()
            model.clear_folder()
        else:
            test_ds = test_ds.batch(batch_size_valid)
            test_accuracy, metric_test = model.evaluate(test_ds)
        et_test = t.time()
        # Save results
        row = [algo_name, dataset_name, fold_idx+1, opt_values.get('params')] + metric_test + [(et_train - st_train), (et_test - st_test)]
        metric_df.loc[fold_idx] = row
        metric_df.to_csv(f"./{dataset_name}_{algo_name}_model.csv")


if __name__ == '__main__':
    parameters = sys.argv[1:]
    print(parameters[0])
    data = get_data_set(parameters[0], int(parameters[1]))
    run_experiment(parameters[2], *data)

