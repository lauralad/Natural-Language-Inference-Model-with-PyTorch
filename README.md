# Natural Language Inference Model with PyTorch
 Comp 545 - Natural Language Processing

## Introduction:
This project focuses on building and training machine learning models to solve the Natural Language Inference (NLI) task, where the goal is to determine whether a 'hypothesis' sentence logically follows from a 'premise' sentence. The models are implemented using PyTorch and trained on a dataset with binary labels indicating entailment.

## Objectives:
* Implement data preprocessing functions including batching, shuffling, and tensor conversions.
* Design and train a baseline logistic regression model using PyTorch.
* Experiment with more sophisticated neural network architectures such as pooled logistic regression and multi-layer neural networks.
* Evaluate model performance using the F1 score on a validation dataset.

## Technologies Used:
* Programming Language: Python
* Frameworks/Libraries: PyTorch
* Development Tools: Kaggle (for GPU utilization), GitHub

## Features:
* Data Preprocessing: Custom functions to transform text data into vectors and batch loaders for model training.
* Model Architecture: Implementation of different models starting from a simple logistic regression to more complex structures.
* Training and Validation: Setup of a complete training loop with forward and backward propagation, including hyperparameter tuning and loss computation.
* Performance Evaluation: Calculation of F1 scores to assess model accuracy on unseen data.

## How to Use:
Clone the repository.

Install dependencies: ``` pip install torch ```

Run the provided Jupyter notebook to train the model and evaluate its performance.
