#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sports vs Politics Text Classifier
Roll Number: B23CM1050
Assignment 1 - Problem 4

This script implements a text classifier to distinguish between sports and politics articles.
I'm comparing three different ML algorithms: Naive Bayes, Logistic Regression, and SVM.

Author: B23CM1050
Date: February 2026
"""

import os
import re
import pickle
import numpy as np
from collections import Counter, defaultdict
import math
import random

# setting random seed so results are reproducable
random.seed(42)
np.random.seed(42)


class TextPreprocessor:
    """
    Handles text cleaning and preprocessing
    This class does all the dirty work of cleaning up the text before we feed it to models
    """
    
    def __init__(self):
        # common english stop words - these dont add much meaning
        # got this list from various NLP resources online
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'just', 'now'
        ])
    
    def clean_text(self, text):
        """
        Clean the input text - remove unwanted stuff
        """
        # converting to lowercase first
        text = text.lower()
        
        # removing URLs - dont need those for classification
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # removing email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # removing numbers but keeping words with numbers like '90s'
        text = re.sub(r'\b\d+\b', '', text)
        
        # removing special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # removing extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Split text into tokens (words)
        Also removes stop words becuase they dont help much
        """
        # simple whitespace tokenization
        tokens = text.split()
        
        # removing stop words and very short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return tokens
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return tokens


class FeatureExtractor:
    """
    Extracts features from text for ML models
    Supports: Bag of Words, TF-IDF, and n-grams
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 1), use_tfidf=False):
        """
        Initialize feature extractor
        max_features: maximum number of features to keep (vocabulary size)
        ngram_range: tuple (min_n, max_n) for n-gram range, e.g., (1,1) for unigrams, (1,2) for uni+bigrams
        use_tfidf: whether to use TF-IDF weighting or just counts
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_tfidf = use_tfidf
        self.vocabulary = {}  # word to index mapping
        self.idf = {}  # inverse document frequency for each word
        self.preprocessor = TextPreprocessor()
    
    def get_ngrams(self, tokens, n):
        """
        Generate n-grams from token list
        e.g., ['the', 'quick', 'fox'] with n=2 -> ['the quick', 'quick fox']
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def extract_ngrams(self, tokens):
        """
        Extract all n-grams in the specified range
        """
        all_ngrams = []
        min_n, max_n = self.ngram_range
        
        for n in range(min_n, max_n + 1):
            if n == 1:
                all_ngrams.extend(tokens)  # unigrams are just the tokens
            else:
                all_ngrams.extend(self.get_ngrams(tokens, n))
        
        return all_ngrams
    
    def fit(self, documents):
        """
        Build vocabulary from training documents
        Also computes IDF if using TF-IDF
        """
        print(f"Building vocabulary (max {self.max_features} features)...")
        
        # count all ngrams across all documents
        ngram_counts = Counter()
        doc_frequency = defaultdict(int)  # how many docs contain each ngram
        
        for doc in documents:
            tokens = self.preprocessor.preprocess(doc)
            ngrams = self.extract_ngrams(tokens)
            
            # counting total occurences
            ngram_counts.update(ngrams)
            
            # counting document frequency for IDF calculation
            unique_ngrams = set(ngrams)
            for ngram in unique_ngrams:
                doc_frequency[ngram] += 1
        
        # selecting top max_features most common ngrams
        most_common = ngram_counts.most_common(self.max_features)
        
        # building vocabulary (ngram -> index mapping)
        self.vocabulary = {ngram: idx for idx, (ngram, _) in enumerate(most_common)}
        
        # calculating IDF if using TF-IDF
        if self.use_tfidf:
            num_docs = len(documents)
            self.idf = {}
            for ngram in self.vocabulary:
                # IDF = log(N / df) where N is total docs and df is doc frequency
                # adding 1 to avoid division by zero
                self.idf[ngram] = math.log(num_docs / (doc_frequency[ngram] + 1))
        
        print(f"Vocabulary built with {len(self.vocabulary)} features")
        if self.ngram_range[1] > 1:
            print(f"Using n-grams: {self.ngram_range}")
        if self.use_tfidf:
            print("Using TF-IDF weighting")
        else:
            print("Using raw counts (Bag of Words)")
    
    def transform(self, documents):
        """
        Transform documents into feature vectors
        Returns: numpy array of shape (num_docs, num_features)
        """
        features = []
        
        for doc in documents:
            tokens = self.preprocessor.preprocess(doc)
            ngrams = self.extract_ngrams(tokens)
            
            # creating feature vector
            feature_vec = np.zeros(len(self.vocabulary))
            
            # counting ngrams in this document
            ngram_counts = Counter(ngrams)
            
            for ngram, count in ngram_counts.items():
                if ngram in self.vocabulary:
                    idx = self.vocabulary[ngram]
                    if self.use_tfidf:
                        # TF-IDF = TF * IDF
                        # using log normalization for TF: log(1 + count)
                        tf = math.log(1 + count)
                        feature_vec[idx] = tf * self.idf[ngram]
                    else:
                        # just use raw counts for BoW
                        feature_vec[idx] = count
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def fit_transform(self, documents):
        """
        Fit vocabulary and transform in one step
        """
        self.fit(documents)
        return self.transform(documents)


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier
    Works well for text classification with count-based features
    """
    
    def __init__(self, alpha=1.0):
        """
        alpha: Laplace smoothing parameter
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}  # P(class)
        self.feature_probs = {}  # P(feature|class)
        self.feature_counts = {}  # count of each feature per class
        self.class_feature_totals = {}  # total feature counts per class
        self.num_features = 0
    
    def fit(self, X, y):
        """
        Train the Naive Bayes model
        X: feature matrix (num_samples, num_features)
        y: labels (num_samples,)
        """
        print("Training Naive Bayes classifier...")
        
        self.classes = np.unique(y)
        self.num_features = X.shape[1]
        
        # calculating class priors
        for cls in self.classes:
            class_count = np.sum(y == cls)
            self.class_priors[cls] = class_count / len(y)
        
        # calculating feature probabilities for each class
        for cls in self.classes:
            # getting samples for this class
            X_class = X[y == cls]
            
            # sum features across all documents in this class
            # adding alpha for smoothing
            self.feature_counts[cls] = np.sum(X_class, axis=0) + self.alpha
            
            # total count for normalization
            self.class_feature_totals[cls] = np.sum(self.feature_counts[cls])
            
            # probability = count / total (with smoothing)
            self.feature_probs[cls] = self.feature_counts[cls] / self.class_feature_totals[cls]
        
        print("Naive Bayes training complete!")
    
    def predict(self, X):
        """
        Predict class labels for samples
        """
        predictions = []
        
        for x in X:
            # calculating log probability for each class
            # using log to avoid underflow with small probabilities
            log_probs = {}
            
            for cls in self.classes:
                # starting with prior
                log_prob = math.log(self.class_priors[cls])
                
                # adding log likelihood for each feature
                # only considering non-zero features for efficiency
                for idx in np.where(x > 0)[0]:
                    # P(feature|class) ^ count
                    log_prob += x[idx] * math.log(self.feature_probs[cls][idx])
                
                log_probs[cls] = log_prob
            
            # predicting class with highest probability
            pred_class = max(log_probs, key=log_probs.get)
            predictions.append(pred_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        probas = []
        
        for x in X:
            log_probs = {}
            
            for cls in self.classes:
                log_prob = math.log(self.class_priors[cls])
                for idx in np.where(x > 0)[0]:
                    log_prob += x[idx] * math.log(self.feature_probs[cls][idx])
                log_probs[cls] = log_prob
            
            # converting log probs to probabilities
            # subtracting max for numerical stability
            max_log_prob = max(log_probs.values())
            exp_probs = {cls: math.exp(log_prob - max_log_prob) 
                        for cls, log_prob in log_probs.items()}
            
            # normalizing
            total = sum(exp_probs.values())
            probs = {cls: prob / total for cls, prob in exp_probs.items()}
            
            probas.append([probs[cls] for cls in sorted(self.classes)])
        
        return np.array(probas)


class LogisticRegressionClassifier:
    """
    Logistic Regression for binary classification
    Using gradient descent for training
    """
    
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=0.01):
        """
        learning_rate: step size for gradient descent
        num_iterations: number of training iterations
        regularization: L2 regularization strength (to prevent overfitting)
        """
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.reg = regularization
        self.weights = None
        self.bias = None
        self.classes = None
    
    def sigmoid(self, z):
        """
        Sigmoid activation function: 1 / (1 + e^(-z))
        Squashes values to (0, 1) range
        """
        # clipping to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train logistic regression using gradient descent
        """
        print("Training Logistic Regression...")
        
        self.classes = np.unique(y)
        
        # converting labels to 0 and 1
        y_binary = (y == self.classes[1]).astype(int)
        
        num_samples, num_features = X.shape
        
        # initializing weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # gradient descent
        for i in range(self.num_iterations):
            # forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # computing gradients
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y_binary))
            db = (1 / num_samples) * np.sum(predictions - y_binary)
            
            # adding L2 regularization to weights (not bias)
            dw += (self.reg / num_samples) * self.weights
            
            # updating parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # printing progress every 100 iterations
            if (i + 1) % 100 == 0:
                loss = self.compute_loss(X, y_binary)
                print(f"Iteration {i+1}/{self.num_iterations}, Loss: {loss:.4f}")
        
        print("Logistic Regression training complete!")
    
    def compute_loss(self, X, y):
        """
        Compute binary cross-entropy loss with L2 regularization
        """
        num_samples = len(y)
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        
        # binary cross-entropy
        # avoiding log(0) with small epsilon
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # adding L2 regularization term
        loss += (self.reg / (2 * num_samples)) * np.sum(self.weights ** 2)
        
        return loss
    
    def predict_proba(self, X):
        """
        Predict probability of each class
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        prob_class1 = self.sigmoid(linear_pred)
        prob_class0 = 1 - prob_class1
        return np.column_stack([prob_class0, prob_class1])
    
    def predict(self, X):
        """
        Predict class labels
        """
        probas = self.predict_proba(X)
        predictions = probas[:, 1] >= 0.5
        return np.where(predictions, self.classes[1], self.classes[0])


class SVMClassifier:
    """
    Support Vector Machine using gradient descent
    This is a simplified version - not as optimized as sklearn's SVM but works!
    """
    
    def __init__(self, learning_rate=0.001, num_iterations=1000, regularization=0.01):
        """
        learning_rate: step size for gradient descent
        num_iterations: number of training iterations
        regularization: C parameter (inverse of regularization strength)
        """
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.C = 1.0 / regularization  # C is inverse of regularization
        self.weights = None
        self.bias = None
        self.classes = None
    
    def fit(self, X, y):
        """
        Train SVM using gradient descent on hinge loss
        """
        print("Training SVM...")
        
        self.classes = np.unique(y)
        
        # converting labels to -1 and +1 (standard for SVM)
        y_binary = np.where(y == self.classes[1], 1, -1)
        
        num_samples, num_features = X.shape
        
        # initializing weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # gradient descent with hinge loss
        for i in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                # condition for hinge loss
                condition = y_binary[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
                if condition:
                    # no violation, only update weights with regularization
                    self.weights -= self.lr * (2 * (1/self.C) * self.weights)
                else:
                    # violation, update with hinge loss gradient
                    self.weights -= self.lr * (2 * (1/self.C) * self.weights - np.dot(x_i, y_binary[idx]))
                    self.bias -= self.lr * y_binary[idx]
            
            # printing progress
            if (i + 1) % 100 == 0:
                loss = self.compute_loss(X, y_binary)
                print(f"Iteration {i+1}/{self.num_iterations}, Loss: {loss:.4f}")
        
        print("SVM training complete!")
    
    def compute_loss(self, X, y):
        """
        Compute hinge loss with regularization
        """
        num_samples = len(y)
        distances = 1 - y * (np.dot(X, self.weights) + self.bias)
        # hinge loss: max(0, 1 - y*f(x))
        hinge_loss = np.maximum(0, distances)
        
        # total loss = hinge loss + regularization
        loss = (1/self.C) * np.sum(self.weights ** 2) + np.mean(hinge_loss)
        return loss
    
    def predict(self, X):
        """
        Predict class labels
        """
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.sign(linear_output)
        # converting back to original class labels
        return np.where(predictions >= 0, self.classes[1], self.classes[0])


def load_dataset(data_dir='data'):
    """
    Load sports and politics articles from files
    Expected directory structure:
        data/
            sports/
                article1.txt
                article2.txt
                ...
            politics/
                article1.txt
                article2.txt
                ...
    """
    documents = []
    labels = []
    
    # loading sports articles
    sports_dir = os.path.join(data_dir, 'sports')
    if os.path.exists(sports_dir):
        for filename in os.listdir(sports_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(sports_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                    labels.append('sports')
    
    # loading politics articles
    politics_dir = os.path.join(data_dir, 'politics')
    if os.path.exists(politics_dir):
        for filename in os.listdir(politics_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(politics_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
                    labels.append('politics')
    
    return documents, labels


def split_data(documents, labels, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets
    """
    # combining and shuffling
    combined = list(zip(documents, labels))
    random.shuffle(combined)
    
    # calculating split indices
    total = len(combined)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # splitting
    train_data = combined[:train_end]
    val_data = combined[train_end:val_end]
    test_data = combined[val_end:]
    
    # unpacking
    train_docs, train_labels = zip(*train_data) if train_data else ([], [])
    val_docs, val_labels = zip(*val_data) if val_data else ([], [])
    test_docs, test_labels = zip(*test_data) if test_data else ([], [])
    
    return (list(train_docs), list(train_labels), 
            list(val_docs), list(val_labels),
            list(test_docs), list(test_labels))


def evaluate_model(y_true, y_pred):
    """
    Calculate evaluation metrics
    Returns: accuracy, precision, recall, f1_score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # for binary classification, calculate per-class metrics
    classes = np.unique(y_true)
    
    # assuming positive class is second one (usually politics or sports depending on order)
    pos_class = classes[1]
    
    # true positives, false positives, false negatives
    tp = np.sum((y_true == pos_class) & (y_pred == pos_class))
    fp = np.sum((y_true != pos_class) & (y_pred == pos_class))
    fn = np.sum((y_true == pos_class) & (y_pred != pos_class))
    
    # precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 score (harmonic mean of precision and recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1


def print_results(name, metrics, train_time):
    """
    Print evaluation results in a nice format
    """
    accuracy, precision, recall, f1 = metrics
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"Training Time: {train_time:.2f}s")
    print(f"{'='*60}")


def main():
    """
    Main function to run the complete experiment
    """
    print("="*70)
    print("SPORTS vs POLITICS TEXT CLASSIFIER")
    print("Roll Number: B23CM1050")
    print("="*70)
    print()
    
    # step 1: load dataset
    print("Step 1: Loading dataset...")
    documents, labels = load_dataset('data')
    
    # if no data directory, create sample data for demonstration
    if len(documents) == 0:
        print("No data found! Creating sample dataset for demonstration...")
        print("(In real scenario, you would have actual articles in data/sports and data/politics folders)")
        
        # sample sports articles (short examples)
        sports_samples = [
            "The cricket team won the match by 50 runs. The captain scored a century and bowled brilliantly.",
            "Football championship finals saw an exciting penalty shootout. The striker scored the winning goal.",
            "Tennis player won the grand slam tournament defeating the world number one in straight sets.",
            "Basketball game went into overtime with spectacular three pointers from the star player.",
            "The athlete broke the world record in 100 meter sprint at the olympic games.",
        ] * 20  # replicating to get more samples
        
        # sample politics articles (short examples)
        politics_samples = [
            "The parliament passed the new bill with majority votes. The opposition criticized the decision.",
            "Prime minister addressed the nation on economic policies and announced tax reforms for citizens.",
            "Election campaing intensified as candidates debated on healthcare and education reforms.",  # typo: campaing
            "The government announced new foreign policy initiatives to strengthen diplomatic relations.",
            "Political party leader resigned following controversy over corruption allegations in the cabinet.",
        ] * 20
        
        documents = sports_samples + politics_samples
        labels = ['sports'] * len(sports_samples) + ['politics'] * len(politics_samples)
        
        print(f"Created sample dataset with {len(documents)} documents")
    
    print(f"Total documents loaded: {len(documents)}")
    print(f"Sports articles: {labels.count('sports')}")
    print(f"Politics articles: {labels.count('politics')}")
    print()
    
    # step 2: split data
    print("Step 2: Splitting data (70% train, 15% validation, 15% test)...")
    train_docs, train_labels, val_docs, val_labels, test_docs, test_labels = split_data(
        documents, labels, train_ratio=0.7, val_ratio=0.15
    )
    print(f"Train: {len(train_docs)}, Validation: {len(val_docs)}, Test: {len(test_docs)}")
    print()
    
    # step 3: experiment with different feature representations
    results = {}
    
    print("="*70)
    print("EXPERIMENT 1: Naive Bayes with different features")
    print("="*70)
    print()
    
    # Experiment 1a: Naive Bayes + BoW
    print("1a. Naive Bayes + Bag of Words")
    print("-" * 40)
    import time
    start_time = time.time()
    
    feature_extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 1), use_tfidf=False)
    X_train = feature_extractor.fit_transform(train_docs)
    X_test = feature_extractor.transform(test_docs)
    
    nb_model = NaiveBayesClassifier(alpha=1.0)
    nb_model.fit(X_train, np.array(train_labels))
    
    train_time = time.time() - start_time
    
    y_pred = nb_model.predict(X_test)
    metrics = evaluate_model(test_labels, y_pred)
    results['NB_BoW'] = (metrics, train_time)
    print_results("Naive Bayes + Bag of Words", metrics, train_time)
    
    # Experiment 1b: Naive Bayes + TF-IDF
    print("\n1b. Naive Bayes + TF-IDF")
    print("-" * 40)
    start_time = time.time()
    
    feature_extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 1), use_tfidf=True)
    X_train = feature_extractor.fit_transform(train_docs)
    X_test = feature_extractor.transform(test_docs)
    
    nb_model = NaiveBayesClassifier(alpha=1.0)
    nb_model.fit(X_train, np.array(train_labels))
    
    train_time = time.time() - start_time
    
    y_pred = nb_model.predict(X_test)
    metrics = evaluate_model(test_labels, y_pred)
    results['NB_TFIDF'] = (metrics, train_time)
    print_results("Naive Bayes + TF-IDF", metrics, train_time)
    
    # Experiment 1c: Naive Bayes + TF-IDF + Bigrams
    print("\n1c. Naive Bayes + TF-IDF + Bigrams")
    print("-" * 40)
    start_time = time.time()
    
    feature_extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 2), use_tfidf=True)
    X_train = feature_extractor.fit_transform(train_docs)
    X_test = feature_extractor.transform(test_docs)
    
    nb_model = NaiveBayesClassifier(alpha=1.0)
    nb_model.fit(X_train, np.array(train_labels))
    
    train_time = time.time() - start_time
    
    y_pred = nb_model.predict(X_test)
    metrics = evaluate_model(test_labels, y_pred)
    results['NB_TFIDF_Bigrams'] = (metrics, train_time)
    print_results("Naive Bayes + TF-IDF + Bigrams", metrics, train_time)
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: Logistic Regression with different features")
    print("="*70)
    print()
    
    # Experiment 2a: Logistic Regression + BoW
    print("2a. Logistic Regression + Bag of Words")
    print("-" * 40)
    start_time = time.time()
    
    feature_extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 1), use_tfidf=False)
    X_train = feature_extractor.fit_transform(train_docs)
    X_test = feature_extractor.transform(test_docs)
    
    lr_model = LogisticRegressionClassifier(learning_rate=0.01, num_iterations=500, regularization=0.01)
    lr_model.fit(X_train, np.array(train_labels))
    
    train_time = time.time() - start_time
    
    y_pred = lr_model.predict(X_test)
    metrics = evaluate_model(test_labels, y_pred)
    results['LR_BoW'] = (metrics, train_time)
    print_results("Logistic Regression + Bag of Words", metrics, train_time)
    
    # Experiment 2b: Logistic Regression + TF-IDF
    print("\n2b. Logistic Regression + TF-IDF")
    print("-" * 40)
    start_time = time.time()
    
    feature_extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 1), use_tfidf=True)
    X_train = feature_extractor.fit_transform(train_docs)
    X_test = feature_extractor.transform(test_docs)
    
    lr_model = LogisticRegressionClassifier(learning_rate=0.01, num_iterations=500, regularization=0.01)
    lr_model.fit(X_train, np.array(train_labels))
    
    train_time = time.time() - start_time
    
    y_pred = lr_model.predict(X_test)
    metrics = evaluate_model(test_labels, y_pred)
    results['LR_TFIDF'] = (metrics, train_time)
    print_results("Logistic Regression + TF-IDF", metrics, train_time)
    
    print("\n" + "="*70)
    print("EXPERIMENT 3: SVM with different features")
    print("="*70)
    print()
    
    # Experiment 3a: SVM + BoW
    print("3a. SVM + Bag of Words")
    print("-" * 40)
    start_time = time.time()
    
    feature_extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 1), use_tfidf=False)
    X_train = feature_extractor.fit_transform(train_docs)
    X_test = feature_extractor.transform(test_docs)
    
    svm_model = SVMClassifier(learning_rate=0.001, num_iterations=500, regularization=0.01)
    svm_model.fit(X_train, np.array(train_labels))
    
    train_time = time.time() - start_time
    
    y_pred = svm_model.predict(X_test)
    metrics = evaluate_model(test_labels, y_pred)
    results['SVM_BoW'] = (metrics, train_time)
    print_results("SVM + Bag of Words", metrics, train_time)
    
    # Experiment 3b: SVM + TF-IDF (best expected performance)
    print("\n3b. SVM + TF-IDF")
    print("-" * 40)
    start_time = time.time()
    
    feature_extractor = FeatureExtractor(max_features=3000, ngram_range=(1, 1), use_tfidf=True)
    X_train = feature_extractor.fit_transform(train_docs)
    X_test = feature_extractor.transform(test_docs)
    
    svm_model = SVMClassifier(learning_rate=0.001, num_iterations=500, regularization=0.01)
    svm_model.fit(X_train, np.array(train_labels))
    
    train_time = time.time() - start_time
    
    y_pred = svm_model.predict(X_test)
    metrics = evaluate_model(test_labels, y_pred)
    results['SVM_TFIDF'] = (metrics, train_time)
    print_results("SVM + TF-IDF", metrics, train_time)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL RESULTS")
    print("="*70)
    print()
    print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Time(s)':<10}")
    print("-" * 90)
    
    for name, (metrics, train_time) in results.items():
        accuracy, precision, recall, f1 = metrics
        print(f"{name:<30} {accuracy*100:>10.2f}% {precision*100:>10.2f}% {recall*100:>8.2f}% {f1*100:>8.2f}% {train_time:>8.2f}s")
    
    print("="*70)
    print("\n✓ Experiments completed successfully!")
    print("\nNote: Results may vary with actual dataset.")
    print("For best results, use a larger dataset with real articles.")


if __name__ == "__main__":
    main()
