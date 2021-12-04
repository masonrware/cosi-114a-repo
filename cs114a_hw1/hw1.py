from collections import Counter, defaultdict

from typing import Iterable, TypeVar, Sequence, Tuple, List

T = TypeVar("T")
START_TOKEN = "<start>"
END_TOKEN = "<end>"


# DO NOT MODIFY ANYTHING ABOVE

def counts_to_probs(count) -> defaultdict:
    global perc1, perc2, total, name1, name2
    biggie = defaultdict(float, {})

    keyList = sorted(count.keys())
    for x in enumerate(keyList):
        if x[0]<(len(count)-1):
            name1 = keyList[x[0]]
            name2 = (keyList[x[0] + 1])
            total = float(count[name1]+count[name2])
            perc1 = count[keyList[x[0]]]/total
            biggie[x[1]] = perc1
        else:
            perc2 = count[keyList[x[0]]] / total
            biggie[x[1]] = perc2
    return biggie


def bigrams(sentence: Sequence[str]) -> List[Tuple[str, str]]:
    sentence_list = list(sentence)
    bigrams_list = []
    sentence_list.insert(0, START_TOKEN)
    sentence_list.append(END_TOKEN)
    for i in range(len(sentence_list) - 1):
        bigrams_list.append((sentence_list[i], sentence_list[i + 1]))
    return bigrams_list


def trigrams(sentence: Sequence[str]) -> List[Tuple[str, str, str]]:
    sentence_list = list(sentence)
    trigrams_list = []
    sentence_list.insert(0, START_TOKEN)
    sentence_list.insert(0, START_TOKEN)
    sentence_list.append(END_TOKEN)
    sentence_list.append(END_TOKEN)
    for i in range(len(sentence_list) - 2):
        trigrams_list.append((sentence_list[i], sentence_list[i + 1], sentence_list [i + 2]))
    return trigrams_list


def count_unigrams(sentences: Iterable[List[str]], lower: bool = False) -> Counter[str]:
    if not lower:
        return Counter(x for xs in sentences for x in set(xs))
    else:
        for sentence in sentences:
            for token in sentence:
                token = token.lower()
        return Counter(x for xs in sentences for x in set(xs))


def count_bigrams(sentences: Iterable[List[str]], lower: bool = False) -> Counter[str]:
    tokenized = []
    for sentence in sentences:
        bigram_sentence = bigrams(sentence)
        for bigram in bigram_sentence:
            tokenized.append(bigram)
    if not lower:
        return Counter(tokenized)
    else:
        tokenizedLower = [(x.lower(), y.lower()) for x,y in tokenized]
        return Counter(tokenizedLower)


def count_trigrams(sentences: Iterable[List[str]], lower: bool = False) -> Counter[str]:
    tokenized = []
    for sentence in sentences:
        trigam_sentence = trigrams(sentence)
        for trigram in trigam_sentence:
            tokenized.append(trigram)
    if not lower:
        return Counter(tokenized)
    else:
        tokenizedLower = [(x.lower(), y.lower(), z.lower()) for x,y,z in tokenized]
        return Counter(tokenizedLower)
