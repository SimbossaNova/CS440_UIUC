"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    # Create a dictionary to store count of each tag for each word
    word_tag_count = {}
    # Create a dictionary to store the most frequent tag for each word
    word_most_frequent_tag = {}
    # Create a dictionary to store count of each tag in general
    tag_count = {}

    # Populate word_tag_count and tag_count dictionaries using the training data
    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag_count:
                word_tag_count[word] = {}
            if tag not in word_tag_count[word]:
                word_tag_count[word][tag] = 0
            word_tag_count[word][tag] += 1
            
            if tag not in tag_count:
                tag_count[tag] = 0
            tag_count[tag] += 1

    # Identify the most frequent tag for each word and populate word_most_frequent_tag
    for word in word_tag_count:
        most_common_tag = max(word_tag_count[word], key=word_tag_count[word].get)
        word_most_frequent_tag[word] = most_common_tag

    # Identify the most common tag overall
    most_common_tag_overall = max(tag_count, key=tag_count.get)
    
    # Tag the test data
    tagged_test_data = []
    for sentence in test:
        tagged_sentence = []
        for word in sentence:
            if word in word_most_frequent_tag:
                tagged_sentence.append((word, word_most_frequent_tag[word]))
            else:
                tagged_sentence.append((word, most_common_tag_overall))
        tagged_test_data.append(tagged_sentence)

    return tagged_test_data