# Intermediate Report

Our team worked on different approaches for current competition.

### Grigory Kryukov:
Implemented classical ML algorithms such as Linear Regression (Ridge and Lasso versions), SVR, Decision Trees, Random Forest, XGB regression, etc.
<br />
Feature engineering for these models were conducted via nltk library by using diverse computational linguistics methods.

<br />

### Mikhail Sulamanidze and Valentin Kopylov:
Worked with deep learning models, main focus was on Deberta. 

Main points concerning the architecture:
* We use pretrained weigts of deberta-v3-base model 
* We fine-tune the last linear layer in respect to 6 desired metrics
* Training via 4-fold cross-validation
* Averaging the score using model from each fold

<br />

## Plan for further work:
We want to perform more sophisticated fine-tuning of already implemented models and also try to create some ensembles based on them.
<br />
Also we would like to try more models for both chosen approaches.
