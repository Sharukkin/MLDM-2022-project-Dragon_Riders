# Intermediate Report

Our team worked on different approaches for current competition.

Grigory Kryukov - implemented classical ML algorithms such as Linear Regression (Ridge and Lasso versions), SVR, Decision Trees, Random Forest, XGB regression, etc.
Feature engineering for these models were conduct via nltk library by using diverse computational linguistics methods.

Mikhail Sulamanidze and Valentin Kopylov - worked with deep learning models, main focus was on Deberta. 
Main points concerning the architecture:
* We use pretrained weigts of deberta-v3-base model 
* We fine-tune the last linear layer in respect to 6 desired metrics
* Training via 4-fold cross-validation
* Averaging the score using model from each fold
