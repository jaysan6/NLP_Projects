############################
#
#
# CSE 143 - NLP Assignment 2
#
# Members: Sanjay Shrikanth, Matthew Daxner, Collin McColl, Soren Larsen
#
############################
from pyexpat.errors import XML_ERROR_NOT_STANDALONE
import pandas as pd
from ngrams import *
import numpy as np
import argparse


def feature_extractor(filepath):
    """
    Takes in the file and parses the sentences using <space> as a demiter
    Returns a tokenized version of the inputs with the <START> <STOP> tokens appended
    """
    features = []
    with open(filepath, "r", encoding='UTF-8') as f:
            for line in f:
                splitted = line.strip().split(" ")
                words = [word for word in splitted]
                words = ["<START>"] + words + ["<STOP>"]
                features.append(words)
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram', 'interpolate'])
    parser.add_argument('--path', type=str, default = './', help='path to datasets')

    parser.add_argument('--smoothing', type=float, default = 0, help='additive smoothing')

    parser.add_argument('--set', type=str, default = 'train',
                        choices=['train', 'dev', 'test'])
    args = parser.parse_args()
    print(args)

    # Convert text into features
    if args.model == "unigram":
        model = Unigram()
    elif args.model == "bigram":
        model = Bigram()
    elif args.model == "trigram":
        model = Trigram()
    elif args.model == 'interpolate':
        model = InterpolatedNGram()
    else:
        raise Exception("Pass unigram, bigram or trigram to --feature")

    train = feature_extractor(f"./1b_benchmark.train.tokens")
    #train = train[:len(train) // 2]  -- uncomment to use half of the training set
    validate = feature_extractor(f"./1b_benchmark.{args.set}.tokens")

    print(f"EVALUATING THE {args.set} DATASET")

    if args.model != 'interpolate':

        vocab_size = model.read_ngram(train)
        print(f"There are {vocab_size} unique tokens in the {args.model} model (excluding \"<START>\")")

        vocab = model.get_vocab()
        print(f"There are {vocab} tokens in the vocabulary")

        print(f"With Additive Smoothing (alpha) = {args.smoothing}:")

        perplexity = model.model_perplexity(validate, args.smoothing)
        print(f"Perplexity of {args.model} model on the {args.set} data is {perplexity}")
    
        test = "<START> HDTV . <STOP>"
        perplexity = model.model_perplexity([test.split(" ")])
        print(f"Perplexity of {args.model} model for the string \"{test}\" is {perplexity}")
    
    else :
        model.train(train)
        test = ["<START> HDTV . <STOP>".split(" ")]

        perplexity = model.interpolate(0.1, 0.3, 0.6, test)
        print(f"Perplexity of {args.model} model for the string \"{test}\" is {perplexity}")


        perplexity = model.interpolate(0.1, 0.3, 0.6, validate)
        print(f"Perplexity of {args.model} model for the {args.set} dataset is {perplexity}")

if __name__ == '__main__':
    main()