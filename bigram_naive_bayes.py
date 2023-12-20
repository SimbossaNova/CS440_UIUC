# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""

meow = 69
joeMama = 420
teehee ='mama'

 



repeats = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
}

happy = {'excellent', 'amazing', 'awesome', 'outstanding', 'fantastic', 'terrific', 'great'}
sadness = {'boring', 'dull', 'poor', 'disappointing', 'bad', 'worse', 'worst'}

sadness_before = {'not', "didn't", "doesn't", "don't", "wasn't", "weren't", "couldn't", "wouldn't", "shouldn't", "won't", "can't"}

def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=1.4, bigram_laplace=1.6, bigram_lambda=0.75, pos_prior=0.75, silently=False):
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    pos_counts_unigram = Counter()
    neg_counts_unigram = Counter()
    pos_counts_bigram = Counter()
    neg_counts_bigram = Counter()

    for i, doc in enumerate(train_set):
        filtered_doc = [word for word in doc if word not in repeats]
        for j in range(len(filtered_doc)):
            if train_labels[i] == 1:  # positive
                pos_counts_unigram[filtered_doc[j]] += 1
                if j < len(filtered_doc) - 1:
                    pos_counts_bigram[(filtered_doc[j], filtered_doc[j+1])] += 1
            else:  # negative
                neg_counts_unigram[filtered_doc[j]] += 1
                if j < len(filtered_doc) - 1:
                    neg_counts_bigram[(filtered_doc[j], filtered_doc[j+1])] += 1

    total_pos = sum(pos_counts_unigram.values())
    total_neg = sum(neg_counts_unigram.values())
    total_pos_bigrams = sum(pos_counts_bigram.values())
    total_neg_bigrams = sum(neg_counts_bigram.values())

    V = len(set(pos_counts_unigram) | set(neg_counts_unigram))
    V_bigrams = len(set(pos_counts_bigram) | set(neg_counts_bigram))

    pos_probs_unigram = {word: (count + unigram_laplace) / (total_pos + V * unigram_laplace) for word, count in pos_counts_unigram.items()}
    neg_probs_unigram = {word: (count + unigram_laplace) / (total_neg + V * unigram_laplace) for word, count in neg_counts_unigram.items()}

    pos_probs_bigram = {bigram: (count + bigram_laplace) / (total_pos_bigrams + V_bigrams * bigram_laplace) for bigram, count in pos_counts_bigram.items()}
    neg_probs_bigram = {bigram: (count + bigram_laplace) / (total_neg_bigrams + V_bigrams * bigram_laplace) for bigram, count in neg_counts_bigram.items()}

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        filtered_doc = [word for word in doc if word not in repeats]
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1 - pos_prior)

        handle_negation = False
        for j in range(len(filtered_doc)):
            word = filtered_doc[j]
            
            if word in happy:
                    pos_prob += math.log(10)
            if word in sadness:
                    neg_prob += math.log(10)

            if handle_negation:
                        word = "not_" + word
                        handle_negation = False
            if word in sadness_before:
                handle_negation = True

            pos_prob += math.log(pos_probs_unigram.get(word, unigram_laplace / (total_pos + V * unigram_laplace)))
            neg_prob += math.log(neg_probs_unigram.get(word, unigram_laplace / (total_neg + V * unigram_laplace)))

            if j < len(filtered_doc) - 1:
                    bigram = (word, filtered_doc[j+1])
                    pos_prob += bigram_lambda * math.log(pos_probs_bigram.get(bigram, bigram_laplace / (total_pos_bigrams + V_bigrams * bigram_laplace)))
                    neg_prob += bigram_lambda * math.log(neg_probs_bigram.get(bigram, bigram_laplace / (total_neg_bigrams + V_bigrams * bigram_laplace)))

        if pos_prob > neg_prob:
                                yhats.append(1)  # positive
        else:
                                    yhats.append(0)  # negative

    return yhats