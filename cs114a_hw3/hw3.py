# hw3.py
# Version 1.1
# 10/23/2021

import random
from collections import defaultdict, Counter
from math import log
from typing import Sequence, Iterable, Generator

############################################################
# The following constants and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
random.seed(0)

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"
# DO NOT MODIFY
NEG_INF = float("-inf")


def load_tokenized_file(path: str) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            yield tuple(tokens)


def sample(probs: dict[str, float]) -> str:
    """Return a sample from a distribution."""
    # We make tuples of the keys and values since sequences are required
    return random.choices(tuple(probs), weights=tuple(probs.values()))[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.

# def bigrams( sentences: Iterable[Sequence[str]]) -> list[tuple]:


def bigram_probs( sentences: Iterable[Sequence[str]], ) -> dict[str, dict[str, float]]:
    num_sentences = 0
    overall_sentence = []
    for sentence in sentences:
        bigrams_list = []
        num_sentences += 1
        sentence_list = list(sentence)
        sentence_list.insert(0, START_TOKEN)
        sentence_list.append(END_TOKEN)
        for i in range(len(sentence_list) - 1):
            bigrams_list.append((sentence_list[i], sentence_list[i + 1]))
        overall_sentence += bigrams_list

    count = Counter()
    for tuple_item in overall_sentence:
        count[tuple_item] += 1
        count[tuple_item[0]] += 1
        count[tuple_item[1]] += 1

    bigram_dict = defaultdict(list)
    for item in overall_sentence:
        first_item = item[0]
        second_item = item[1]
        if first_item in bigram_dict.keys():
            if second_item not in bigram_dict[first_item]:
                values = []
                for i in list(bigram_dict[first_item].keys()):
                    values.append(i)
                values.append(second_item)
                value_dict = {}
                for value in values:
                    floater = count[(first_item, value)] / (count[first_item]/2)
                    value_dict[value] = floater
                bigram_dict[first_item] = value_dict
        else:
            floater = count[item] / (count[first_item]/2)
            bigram_dict[first_item] = {second_item: floater}
    return dict(bigram_dict)


def trigram_probs( sentences: Iterable[Sequence[str]], ) -> dict[tuple[str, str], dict[str, float]]:
    num_sentences = 0
    overall_sentence = []
    for sentence in sentences:
        trigrams_list = []
        num_sentences += 1
        sentence_list = list(sentence)
        sentence_list.insert(0, START_TOKEN)
        sentence_list.insert(0, START_TOKEN)
        sentence_list.append(END_TOKEN)
        sentence_list.append(END_TOKEN)
        for i in range(len(sentence_list) - 2):
            trigrams_list.append((sentence_list[i], sentence_list[i + 1], sentence_list[i+2]))
        overall_sentence += trigrams_list

    count = Counter()
    for trigram_item in overall_sentence:
        first_item = (trigram_item[0], trigram_item[1])
        second_item = trigram_item[2]

        count[trigram_item] += 1
        count[first_item] += 1
        count[second_item] += 1

    trigram_dict = defaultdict(list)
    for item in overall_sentence:
        first_item = (item[0], item[1])
        second_item = item[2]
        if first_item in trigram_dict.keys():
            if second_item not in trigram_dict[first_item]:
                values = []
                for i in list(trigram_dict[first_item].keys()):
                    values.append(i)
                values.append(second_item)
                value_dict = {}
                for value in values:
                    new_item_list = []
                    for i in first_item:
                        new_item_list.append(i)
                    new_item_list.append(value)
                    new_item = tuple(new_item_list)
                    floater = count[new_item] / count[first_item]
                    value_dict[value] = floater
                trigram_dict[first_item] = value_dict
        else:
            floater = count[item] / (count[first_item])
            trigram_dict[first_item] = {second_item: floater}
    return dict(trigram_dict)


def sample_bigrams( probs: dict[str, dict[str, float]] ) -> list[str]:
    output_sentence = []
    context = list(probs.keys())[0]
    for i in range(1, len(list(probs.keys()))):
        most_likely_token = sample(probs[context])
        if most_likely_token == "<end>":
            break
        output_sentence.append(most_likely_token)
        context = most_likely_token
    return output_sentence


def sample_trigrams( probs: dict[tuple[str, str], dict[str, float]] ) -> list[str]:
    output_sentence = []
    context = list(probs.keys())[0]
    for i in range(1, len(list(probs.keys()))):
        second_item = context[1]
        most_likely_token = sample(probs[context])
        if most_likely_token == "<end>":
            break
        output_sentence.append(most_likely_token)
        context = (second_item, most_likely_token)
    return output_sentence


def bigram_sequence_prob( sequence: Sequence[str], probs: dict[str, dict[str, float]] ) -> float:
    sequence_log_prob = 0
    for token in sequence:
        if sample(probs[token]):
            most_likely_token = sample(probs[token])
        else:
            return NEG_INF
        most_likely_token_log_prob = log(probs[token][most_likely_token])
        sequence_log_prob += most_likely_token_log_prob
    return sequence_log_prob


def trigram_sequence_prob( sequence: Sequence[str], probs: dict[tuple[str, str], dict[str, float]] ) -> float:
    sequence_log_prob = 0
    overall_sentence = list()
    sentence_segment = []
    length = len(sequence) - 1
    for i in range(0, length):
        if i == 0:
            start = (tuple((START_TOKEN, START_TOKEN, sequence[i])))
            start2 = (tuple((START_TOKEN, sequence[i], sequence[i + 1])))
        if i == length - 1:
            end = (tuple((sequence[i + 1], END_TOKEN, END_TOKEN)))
            end2 = (tuple((sequence[i], sequence[i + 1], END_TOKEN)))
    sentence_segment.append(start)
    sentence_segment.append(start2)
    for i in range(0, length - 1):
        sentence_segment.append(tuple((sequence[i], sequence[i + 1], sequence[i + 2])))
    sentence_segment.append(end2)
    sentence_segment.append(end)
    overall_sentence = overall_sentence + sentence_segment

    for trigram in overall_sentence:
        tuple_item = trigram[0], trigram[1]
        if sample(probs[tuple_item]):
            most_likely_token = sample(probs[tuple_item])
        else:
            return NEG_INF
        most_likely_token_log_prob = log(probs[tuple_item][most_likely_token])
        sequence_log_prob += most_likely_token_log_prob
    return sequence_log_prob