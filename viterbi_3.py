import math
from collections import defaultdict

# Constants for unseen probabilities
epsilon_for_pt = 1e-6
emit_epsilon = 1e-6

def training(sentences):
    init_prob = defaultdict(float)
    emit_prob = defaultdict(lambda: defaultdict(float))
    trans_prob = defaultdict(lambda: defaultdict(float))
    hapax_tag_counts = defaultdict(int)
    hapax_total = 0
    
    total_sentences = len(sentences)
    tag_counts = defaultdict(int)
    tag_tag_counts = defaultdict(lambda: defaultdict(int))
    total_words_for_tag = defaultdict(int)
    word_occurrences = defaultdict(int)
    
    suffix_tag_counts = defaultdict(lambda: defaultdict(int))
    suffix_counts = defaultdict(int)
    
    for sentence in sentences:
        prev_tag = None
        for word, tag in sentence:
            word_occurrences[word] += 1
            tag_counts[tag] += 1
            total_words_for_tag[tag] += 1
            if prev_tag:
                tag_tag_counts[prev_tag][tag] += 1
            emit_prob[tag][word] = emit_prob[tag].get(word, 0) + 1
            prev_tag = tag
            
            if len(word) > 2:  # Considering suffixes of length 3
                suffix = word[-3:]
                suffix_tag_counts[suffix][tag] += 1
                suffix_counts[suffix] += 1
    
    for sentence in sentences:
        for word, tag in sentence:
            if word_occurrences[word] == 1:
                hapax_tag_counts[tag] += 1
                hapax_total += 1
    
    hapax_tag_prob = {tag: count / hapax_total for tag, count in hapax_tag_counts.items()}
    
    suffix_tag_prob = {suffix: {tag: count / total for tag, count in tags.items()} 
                       for suffix, (tags, total) in zip(suffix_tag_counts.keys(), zip(suffix_tag_counts.values(), suffix_counts.values()))}
    
    V = len({word for sentence in sentences for word, _ in sentence})
    T = len(tag_counts)
    
    most_common_tag = max(tag_counts, key=tag_counts.get)
    
    for tag, count in init_prob.items():
        init_prob[tag] = count / total_sentences
    
    for tag, words in emit_prob.items():
        for word, count in words.items():
            alpha = epsilon_for_pt * (hapax_tag_prob.get(tag, 0) + 1e-5)
            emit_prob[tag][word] = (count + alpha) / (total_words_for_tag[tag] + alpha * (V + 1))
        emit_prob[tag]['UNKNOWN'] = alpha / (total_words_for_tag[tag] + alpha * (V + 1))
    
    for tag1, tags in tag_tag_counts.items():
        total_tag1 = sum(tags.values())
        for tag2, count in tags.items():
            alpha = epsilon_for_pt
            trans_prob[tag1][tag2] = (count + alpha) / (total_tag1 + alpha * T)
    
    return init_prob, emit_prob, trans_prob, most_common_tag, hapax_tag_prob, suffix_tag_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, most_common_tag, hapax_tag_prob, suffix_tag_prob):
    current_log_prob = {}
    current_predict_tag_seq = {}
    unseen_word = all(word not in emit_prob[tag] for tag in emit_prob)

    for current_tag in emit_prob:
        max_log_prob = float('-inf')
        best_prev_tag = None
        
        for prev_tag in prev_prob:
            transition = trans_prob[prev_tag].get(current_tag, epsilon_for_pt)
            
            if unseen_word:
                if len(word) > 2:  # Considering suffixes of length 3
                    suffix = word[-3:]
                    emission = suffix_tag_prob.get(suffix, {}).get(current_tag, emit_epsilon)
                else:
                    emission = hapax_tag_prob.get(current_tag, emit_epsilon)
            else:
                emission = emit_prob[current_tag].get(word, emit_epsilon)

            current_path_prob = prev_prob[prev_tag] + math.log(transition) + math.log(emission)
            
            if current_path_prob > max_log_prob:
                max_log_prob = current_path_prob
                best_prev_tag = prev_tag
        
        current_log_prob[current_tag] = max_log_prob
        if best_prev_tag:
            current_predict_tag_seq[current_tag] = prev_predict_tag_seq[best_prev_tag] + [best_prev_tag]
        else:
            current_predict_tag_seq[current_tag] = []
    
    return current_log_prob, current_predict_tag_seq

def viterbi_3(train, test):
    init_prob, emit_prob, trans_prob, most_common_tag, hapax_tag_prob, suffix_tag_prob = training(train)
    
    predicts = []
    
    for sentence in test:
        log_prob = {}
        predict_tag_seq = {}
        
        first_word = sentence[0]
        for t in emit_prob:
            log_prob[t] = math.log(init_prob.get(t, epsilon_for_pt)) + math.log(emit_prob[t].get(first_word, emit_epsilon))
            predict_tag_seq[t] = []
        
        for i in range(1, len(sentence)):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob, most_common_tag, hapax_tag_prob, suffix_tag_prob)
        
        max_prob = float('-inf')
        best_tag = None
        for tag in log_prob:
            if log_prob[tag] > max_prob:
                max_prob = log_prob[tag]
                best_tag = tag
        
        best_sequence = predict_tag_seq[best_tag] + [best_tag]
        predicts.append(list(zip(sentence, best_sequence)))
    
    return predicts
