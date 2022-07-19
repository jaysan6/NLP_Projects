############################
#
#
# CSE 143 - NLP Assignment 1
#
# Members: Sanjay Shrikanth, Matthew Daxner, Collin McColl
#
############################
import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass

    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.prob_hate_words = np.array([])
        self.prob_non_hate_words = np.array([])
        self.prior_hate = 0
        self.prior_not_hate = 0

    def fit(self, X, Y):
        sum_y = np.sum(Y, axis=0)
        len_y = Y.shape[0]
        self.prior_hate = sum_y / len_y
        self.prior_not_hate =  1 - self.prior_hate
        vocabulary_size = X[0].shape[0]

        count_hate_words = np.array([1.]*vocabulary_size)  ## Laplace Smoothing
        count_non_hate_words = np.array([1.]*vocabulary_size)

        for i,y in enumerate(Y):
            if y == 1: count_hate_words += X[i]
            else: count_non_hate_words += X[i]

        self.prob_hate_words =  np.log(count_hate_words / np.sum(count_hate_words))
        self.prob_non_hate_words = np.log(count_non_hate_words / np.sum(count_non_hate_words))
        
        # code to get the highest/lowest word ratios
        # stuff = self.prob_hate_words / self.prob_non_hate_words
        # idx =np.argpartition(stuff, 10)
        # print(idx[:10])
        
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        log_prior_hate, log_prior_not_hate = np.log(self.prior_hate), np.log(self.prior_not_hate)
        for i,sentence in enumerate(X):
            tmp_pos = tmp_neg = 0
            for j,word in enumerate(sentence):
                if word > 0:
                    tmp_pos += self.prob_non_hate_words[j]
                    tmp_neg += self.prob_hate_words[j]
            predictions[i] = int(tmp_pos+log_prior_not_hate < tmp_neg+log_prior_hate)
        return predictions

# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        self.beta = None
        self.loglikelihood = 0

    def to_feature(self, X):
        X[X > 1] = 1
        return X

    def sigmoid(self, features):
        return 1. / (1 + np.exp(-1 * np.dot(self.beta, features)))
                 
    def fit(self, X, Y):
        features = self.to_feature(X)
        self.beta = np.zeros(features.shape[1])
        alpha = 0.006
        epochs = 110
        lam = 10  # set to 0 for no regularization

        while epochs: ## gradient descent
            #print(epochs)  # uncomment to make epoch progression visible
            old_beta = self.beta
            for i, y in enumerate(Y):
                gradient = -1 * (y - self.sigmoid(features[i])) * features[i]
                regularizer = 2 * lam * old_beta
                new_beta = old_beta - alpha * gradient - regularizer
                old_beta = new_beta
            self.beta = old_beta
            epochs -= 1

    def predict(self, X):
        test_feature = self.to_feature(X)
        predictions = np.zeros(test_feature.shape[0], dtype=np.int8)
        for i, sentence in enumerate(test_feature):
            predictions[i] = 1 if self.sigmoid(sentence) > 0.5 else 0
        return predictions


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
