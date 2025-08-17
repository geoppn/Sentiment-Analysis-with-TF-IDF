# Sentiment Analysis with Pretrained Transformers

**AI II – Deep Learning for Natural Language Processing, Spring 2024-2025**

## Overview

This project implements sentiment analysis using a classic TF-IDF + Logistic Regression pipeline, designed for social media text classification. The core methods and insights described here are suitable for rapid prototyping and benchmarking, but can be extended to use modern pretrained transformers.

## 1. Exploratory Data Analysis (EDA)

- Loads train, validation, and test datasets.
- Inspects missing data and dataset structure.
- Visualizes sentiment distribution using `seaborn` countplots.

## 2. Data Preprocessing

- Converts text to lowercase and demojizes emojis.
- Expands contractions (e.g., "can't" → "can not").
- Handles negations by prefixing affected words (e.g., "not good" → "not NEG_good").
- Removes URLs, user mentions, and hashtag symbols.
- Tokenizes text and removes stopwords, but preserves key sentiment words.
- Adds special tokens for exclamation/question marks and emojis (e.g., `EXCLAM`, `HAPPY_EMOJI`).
- Outputs a cleaned text column for model input.

## 3. Feature Engineering & Model Pipeline

- Uses `TfidfVectorizer` for n-gram extraction (unigrams, bigrams, trigrams).
- Configures vectorizer for feature selection and normalization.
- Trains a `LogisticRegression` classifier with L1 regularization for sparsity and interpretability.
- Hyperparameter tuning section is included but commented out for speed.

## 4. Model Training & Evaluation

- Trains model on the training set, evaluates on validation set.
- Reports standard metrics: accuracy, precision, recall, F1-score.
- Plots confusion matrix for visual analysis.
- Checks for overfitting/underfitting by comparing train and validation accuracy.

## 5. Error Analysis & Feature Importance

- Samples and prints misclassified validation examples to understand model limitations.
- Extracts and displays top features (words/phrases) driving positive and negative sentiment predictions.


## Key Insights

- **Negation Handling:** Prefixing negated words improves sentiment signal.
- **Emoji & Punctuation Features:** Emojis and exclamation/question marks are important sentiment cues.
- **TF-IDF + Logistic Regression:** Provides strong baseline performance and interpretability.
- **Feature Importance:** Top positive/negative words are intuitive (e.g., "happy", "good" vs "sad", "bad").
- **Error Analysis:** Some misclassifications arise from ambiguous or short texts.
