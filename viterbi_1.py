import math
from collections import defaultdict
from math import log

# Constants for unseen probabilities
epsilon_for_pt = 1e-6
emit_epsilon = 1e-6

def training(sentences):
    init_prob = defaultdict(float)
    emit_prob = defaultdict(lambda: defaultdict(float))
    trans_prob = defaultdict(lambda: defaultdict(float))
    
    total_sentences = len(sentences)
    tag_counts = defaultdict(int)
    tag_tag_counts = defaultdict(lambda: defaultdict(int))
    total_words_for_tag = defaultdict(int)
    
    for sentence in sentences:
        prev_tag = None
        for i, (word, tag) in enumerate(sentence):
            tag_counts[tag] += 1
            total_words_for_tag[tag] += 1
            if i == 0:
                init_prob[tag] += 1.0
            if prev_tag:
                tag_tag_counts[prev_tag][tag] += 1
            emit_prob[tag][word] = emit_prob[tag].get(word, 0) + 1
            prev_tag = tag

    # Unique words and tags in the training data
    V = len({word for sentence in sentences for word, _ in sentence})
    T = len(tag_counts)
    
    most_common_tag = max(tag_counts, key=tag_counts.get)
    
    # Convert counts to probabilities with Laplace smoothing
    for tag, count in init_prob.items():
        init_prob[tag] = count / total_sentences

    for tag, words in emit_prob.items():
        for word, count in words.items():
            alpha = epsilon_for_pt  # Laplace smoothing constant
            emit_prob[tag][word] = (count + alpha) / (total_words_for_tag[tag] + alpha * (V + 1))
        emit_prob[tag]['UNKNOWN'] = alpha / (total_words_for_tag[tag] + alpha * (V + 1))
    
    for tag1, tags in tag_tag_counts.items():
        total_tag1 = sum(tags.values())
        for tag2, count in tags.items():
            alpha = epsilon_for_pt  # Laplace smoothing constant
            trans_prob[tag1][tag2] = (count + alpha) / (total_tag1 + alpha * T)

    return init_prob, emit_prob, trans_prob, most_common_tag

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, most_common_tag):
    current_log_prob = {}
    current_predict_tag_seq = {}
    unseen_word = all(word not in emit_prob[tag] for tag in emit_prob)

    for current_tag in emit_prob:
        max_log_prob = float('-inf')
        best_prev_tag = None
        
        for prev_tag in prev_prob:
            transition = trans_prob[prev_tag].get(current_tag, epsilon_for_pt)
            
            # Adjust emission probability for unseen words
            if unseen_word:
                emission = emit_prob[current_tag].get('UNKNOWN', emit_epsilon)
            else:
                emission = emit_prob[current_tag].get(word, emit_epsilon)

            current_path_prob = prev_prob[prev_tag] + log(transition) + log(emission)
            
            if current_path_prob > max_log_prob:
                max_log_prob = current_path_prob
                best_prev_tag = prev_tag
        
        current_log_prob[current_tag] = max_log_prob
        if best_prev_tag:
            current_predict_tag_seq[current_tag] = prev_predict_tag_seq[best_prev_tag] + [best_prev_tag]
        else:
            current_predict_tag_seq[current_tag] = []
    
    return current_log_prob, current_predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    init_prob, emit_prob, trans_prob, most_common_tag = get_probs(train)
    
    predicts = []
    
    for sentence in test:
        log_prob = {}
        predict_tag_seq = {}
        
        for t in emit_prob:
            log_prob[t] = log(init_prob.get(t, epsilon_for_pt)) + log(emit_prob[t].get(sentence[0], emit_epsilon))
            predict_tag_seq[t] = []
        
        for i in range(1, len(sentence)):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob, most_common_tag)
        
        max_prob = float('-inf')
        best_tag = None
        for tag in log_prob:
            if log_prob[tag] > max_prob:
                max_prob = log_prob[tag]
                best_tag = tag
        
        best_sequence = predict_tag_seq[best_tag] + [best_tag]
        predicts.append(list(zip(sentence, best_sequence)))
    
    return predicts
