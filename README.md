# Power Outage Detection

This GitHub repository contains code for power outage detection using transfer learning and language models. The code explores the application of pretraining and fine-tuning language models to detect power outages, even with limited labeled data. It also includes baselines using classical models for comparison.

## Data
The repository provides code to load and preprocess the data. It uses a CSV file (`data.csv`) containing social media tweets associated with power outages. The data is split into training and testing sets, ensuring a balanced distribution of samples for each class ('outage' and 'no_outage').

## Baselines
The repository includes implementation of baseline models using classical machine learning algorithms. Specifically, it demonstrates the use of TF-IDF vectorization and classification models such as XGBoost, SVM, and Logistic Regression. The evaluation metrics, including accuracy, precision, recall, and F1-score, are calculated for each baseline model.

## Transfer Learning with Language Models (LLMs)
The code showcases the application of transfer learning with language models for power outage detection. It includes zero-shot learning and few-shot learning approaches using popular LLMs like BERT and GPT. The evaluation metrics are calculated for each LLM and finetuning percentage.

### Zero-Shot Learning
The repository provides code to perform zero-shot classification using the BERT and GPT language models. It uses a pipeline approach to classify the testing data into 'outage' and 'no_outage' categories. Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated.

### Few-Shot Learning
The code demonstrates few-shot learning by fine-tuning the language models with a limited percentage of training data. It randomly samples a balanced subset of the training data and evaluates the performance on the testing data. The evaluation metrics, including accuracy, precision, recall, and F1-score, are calculated for each finetuning percentage and model type (BERT and GPT).

## Prerequisites
To run the code in this repository, ensure that you have the following dependencies installed:

- pandas
- torch
- scikit-learn
- transformers
- xgboost
