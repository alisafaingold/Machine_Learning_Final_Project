# Machine Learning Final Project
The final project in Ben-gurion University Machine Learning course. 
The task was to implement a ML solution published in recent years in top-ranking conventions, suggest and implemwnt an improve and 
then to evaluate it, analyze the results and comapre to well-known algorithm.

In this project, we choose to implement [Born Again Neural Networks](http://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf).



### Documentation
#### Python Files 

1. main -The main file of the project - responsible of connecting and running all parts of the project.

2. statistics - The File contains the code for stage 5 of the project - the Friedman's Test and the post-hoc tests we did on the data. 

3. data_loader - The file contains the code for part of stage 4 of the project - load one from the 20 data sets that have been choose by the user.

4. paper - The folder contation our implementation for the paper and our improved (stage 1 and 2)
  - paper_model - model implementation
  - paramater_optimizar - preforam 3-fold Cross Validation for hyperparameter optimization

5. baseline_model - The folder contation implementation for ResNet101 (stage 3)
  - paper_model - model implementation
  - paramater_optimizar - preforam 3-fold Cross Validation for hyperparameter optimization

6. common - The folder contation cmponents that are shared across applications (calaculate matrics, implementation of ResNet121 for the paper mdel and utils)


### Project Arguments

In order to run the project, the following arguments must be provided:

- Date set name - the wanted data set, could be:
  - cifar10
  - beans
  - caltech101
  - kmnist
  - mnist
  - fashion_mnist
  - food101
  - cifar100
- Subset data set number - the number of the wanted data set subset, could be:
  - cifar10 - 0 or 1
  - beans - 0
  - caltech101 - 0 or 1
  - kmnist - 0, 1 or 2
  - mnist - 0, 1 or 2
  - fashion_mnist - 0, 1 or 2
  - food101 - 0 or 1
  - cifar100 - 0, 1, 2, 3 or 4
- Name of the algorithem that should be appleid, couled be:
  - paper - for the algorithmen that suggest in the paper
  - improved - for our umpeoved algorithem based on the paper algorithem
  - baseline - for ResNet121



### Quick Start

In order to run the project there a few neccesary steps: 

1. Clone This repository

2. Clone infinteboost repo using the following command. Make sure it's in the project folder. 

`$ git clone https://github.com/alisafaingold/machine_learning_project.git` 

3. Install the required packages (according to the requirments file)

`$ pip install -r requirments.txt` 

4. Create a data folder and put all datasets in the folder. The folder should be at the root directory of this git repository. 

5. run `python main.py [dataset_name] [subset_num] [algorithem name]`

