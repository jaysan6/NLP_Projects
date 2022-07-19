CSE 143 Assignment 1: Hate Speech Detector

Members: Sanjay Shrikanth, Matthew Daxner, Collin McColl

How to run code:

No modifications were made to the initial setup.

To change which csv file to use for testing accuracies, just change 
the csv string to dev or test in the try-except block.


To run Naive Bayes
python3 main.py --model NaiveBayes

To run Logistic Regression
python3 main.py --model LogisticRegression

To see the configurations of L2 Regularization, just modify the "lam" variable
in fit() for the LogisticRegression class in classifiers.py. Same applies
for running different values for learning rate and epochs.

 