import reader
import math
from tqdm import tqdm
from collections import Counter, defaultdict

def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir, testdir, stemming, lowercase, silently)
    return train_set, train_labels, dev_set, dev_labels

def naiveBayes(dev_set, train_set, train_labels, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace, pos_prior)

    # Training
    vocab = set()
    word_count_positive = defaultdict(int)
    word_count_negative = defaultdict(int)
    num_positive_reviews = sum([1 for label in train_labels if label == 1])
    num_negative_reviews = len(train_labels) - num_positive_reviews
    
    for review, label in zip(train_set, train_labels):
        for word in review:
            vocab.add(word)
            if label == 1:  # positive review
                word_count_positive[word] += 1
            else:  # negative review
                word_count_negative[word] += 1
    
    total_words_positive = sum(word_count_positive.values())
    total_words_negative = sum(word_count_negative.values())
    vocab_size = len(vocab)

    # Prediction
    yhats = []

    for review in tqdm(dev_set, disable=silently):
        log_prob_positive = math.log(pos_prior)
        log_prob_negative = math.log(1 - pos_prior)
        
        for word in review:
            if word in vocab:
                # Calculate the probabilities with Laplace smoothing
                prob_word_positive = (word_count_positive[word] + laplace) / (total_words_positive + laplace * vocab_size)
                prob_word_negative = (word_count_negative[word] + laplace) / (total_words_negative + laplace * vocab_size)

                log_prob_positive += math.log(prob_word_positive)
                log_prob_negative += math.log(prob_word_negative)
        
        # Assign the label with the highest log probability
        yhats.append(1 if log_prob_positive > log_prob_negative else 0)
    
    return yhats
