# hw4.py
# Version 1.0
# 11/2/2021


# hw4.py
# Version 1.0
# 11/2/2021

##change for git

from abc import abstractmethod, ABC
from collections import Counter, defaultdict
from math import log
from operator import itemgetter
from typing import Any, Generator, Iterable, Sequence

############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
NEG_INF = float("-inf")


# DO NOT MODIFY
class Token:
    """Stores the text and tag for a token.

    Hashable and cleaner than indexing tuples all the time.
    """

    def __init__(self, token: str, tag: str):
        self.text = token
        self.tag = tag

    def __str__(self):
        return f"{self.text}/{self.tag}"

    def __repr__(self):
        return f"<Token {str(self)}>"

    def __eq__(self, other: Any):
        return (
                isinstance(other, Token) and self.text == other.text and self.tag == other.tag
        )

    def __lt__(self, other: "Token"):
        return self.to_tuple() < other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        """Return the text and tag as a tuple.

        Example:
        >>> token = Token("apple", "NN")
        >>> token.to_tuple()
        ('apple', 'NN')
        """
        return self.text, self.tag

    @staticmethod
    def from_tuple(t: tuple[str, ...]):
        """
        Creates a Token object from a tuple.
        """
        assert len(t) == 2
        return Token(t[0], t[1])

    @staticmethod
    def from_string(s: str) -> "Token":
        """Create a Token object from a string with the format 'token/tag'.

        Sample usage: Token.from_string("cat/NN")
        """
        return Token(*s.rsplit("/", 1))


# DO NOT MODIFY
class Tagger(ABC):
    @abstractmethod
    def train(self, sentences: Iterable[Sequence[Token]]) -> None:
        """Train the part of speech tagger by collecting needed counts from sentences."""
        raise NotImplementedError

    @abstractmethod
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        """
        Tags a sentence with part of speech tags.
        Sample usage:
            tag_sentence(["I", "ate", "an", "apple"])
             returns: ["PRP", "VBD", "DT", "NN"]
        """
        raise NotImplementedError

    def tag_sentences(self, sentences: Iterable[Sequence[str]]) -> Generator[list[str], None, None]:
        """
        Tags each sentence's tokens with part of speech tags and
        yields the corresponding list of part of speech tags.
        """
        for sentence in sentences:
            yield self.tag_sentence(sentence)

    def test(self, tagged_sents: Iterable[Sequence[Token]]) -> tuple[list[str], list[str]]:
        """
        Runs the tagger over all the sentences and returns a tuple with two lists:
        the predicted tag sequence and the actual tag sequence.
        The predicted and actual tags can then be used for calculating accuracy or other
        metrics.
        This does not preserve sentence boundaries.
        """
        predicted: list[str] = []
        actual: list[str] = []
        for sent in tagged_sents:
            predicted.extend(self.tag_sentence([t.text for t in sent]))
            actual.extend([t.tag for t in sent])
        return predicted, actual


# DO NOT MODIFY
def _safe_log(n: float) -> float:
    """Return the log of a number or -inf if the number is zero."""
    return NEG_INF if n == 0.0 else log(n)


# DO NOT MODIFY
def _max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type error here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


# DO NOT MODIFY
def _most_frequent_item(counts: Counter[str]) -> str:
    """Return the most frequent item in a Counter."""
    assert counts, "Counter is empty"
    top_item, _ = counts.most_common(1)[0]
    return top_item


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.

class MostFrequentTagTagger(Tagger):
    def __init__(self):
        self.default_tag = None

    def train(self, sentences: Iterable[Sequence[Token]]) -> None:
        tags = Counter()
        for sentence in sentences:
            for token in sentence:
                tags[token.tag] += 1
        init_item = tags.most_common()[0][0]
        self.default_tag = init_item

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        return [self.default_tag for _ in sentence]


class UnigramTagger(Tagger):
    def __init__(self):
        self.tags = defaultdict(Counter)
        self.likely_tags = defaultdict()
        self.most_frequent_tag = None

    def train(self, sentences: Iterable[Sequence[Token]]):
        for sentence in sentences:
            for tag_set in sentence:
                self.tags[tag_set.text][tag_set.tag] += 1
        for tag in self.tags:
            init_val = self.tags[tag].most_common()[0][0]
            first_val = self.tags[tag].most_common()[0][1]
            self.likely_tags[tag] = init_val
            self.most_frequent_tag = init_val, first_val

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        tags = []
        most_frequent = self.most_frequent_tag[0]
        for word in sentence:
            if word not in self.likely_tags.keys():
                tags.append(most_frequent)
            else:
                tags.append(self.likely_tags[word])
        return tags


class SentenceCounter:
    def __init__(self, k):
        self.k = k
        self.emit_count = defaultdict(Counter)
        self.emit = defaultdict(int)
        self.tags = []
        self.trans_prob = defaultdict(lambda: defaultdict(float))
        self.tags0 = defaultdict(int)
        self.sorted_tags = []
        self.sentence_count = 0

    def count_sentences(self, sentences: Iterable[Sequence[Token]]) -> None:
        tag_trans = defaultdict(lambda: defaultdict(int))
        tags_set = set()
        for sentence in sentences:
            self.tags0[sentence[0].tag] += 1
            tags_sequence = []
            for token in sentence:
                tags_set.add(token.tag)
                tags_sequence.append(token.tag)
                self.emit_count[token.tag][token.text] += 1
            self.tags.append(tags_sequence)
            self.sentence_count += 1
        for emit_tag in self.emit_count:
            self.emit[emit_tag] = sum(self.emit_count[emit_tag].values())
        for tag_list in self.tags:
            for i in range(len(tag_list) - 1):
                tag_trans[tag_list[i]][tag_list[i + 1]] += 1
        for prev_tag in tag_trans:
            for curr_tag in tag_trans[prev_tag]:
                trans_prob = tag_trans[prev_tag][curr_tag]
                prev_tag_sum = sum(tag_trans[prev_tag].values())
                self.trans_prob[prev_tag][curr_tag] = trans_prob / prev_tag_sum

        self.sorted_tags = sorted(list(tags_set))

    def tagset(self) -> list[str]:
        return self.sorted_tags

    def emission_prob(self, tag: str, word: str) -> float:
        tag_given_word_prob = self.emit_count[tag][word]
        k = self.k
        dict_length = len(self.emit_count[tag])
        tag_prob = self.emit[tag]
        numerator = tag_given_word_prob + k
        denominator = (tag_prob + dict_length * k)
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator

    def transition_prob(self, prev_tag: str, current_tag: str) -> float:
        current_given_prev_prob = self.trans_prob[prev_tag][current_tag]
        if current_given_prev_prob == 0:
            return 0.0
        else:
            return current_given_prev_prob

    def initial_prob(self, tag: str) -> float:
        if tag in self.tags0:
            return self.tags0[tag] / self.sentence_count
        return 0.0


class BigramTagger(Tagger, ABC):
    def __init__(self, k) -> None:
        self.counter = SentenceCounter(k)

    def train(self, sents: Iterable[Sequence[Token]]) -> None:
        self.counter.count_sentences(sents)

    def sequence_probability(self, sentence: Sequence[str], tags: Sequence[str]) -> float:
        first_token = sentence[0]
        first_tag = tags[0]
        init_tag_log = _safe_log(self.counter.initial_prob(first_tag))
        init_tag_token_log = _safe_log(self.counter.emission_prob(first_tag, first_token))
        sequence_prob = init_tag_log + init_tag_token_log
        for i in range(1, len(sentence)):
            prev_tag = tags[i - 1]
            curr_tag = tags[i]
            token = sentence[i]
            curr_prev_tag_log = _safe_log(self.counter.transition_prob(prev_tag, curr_tag))
            tag_token_log = _safe_log(self.counter.emission_prob(curr_tag, token))
            sequence_prob += curr_prev_tag_log + tag_token_log
        return sequence_prob


class GreedyBigramTagger(BigramTagger):
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        tags = []
        prev_tag = None
        prob_matrix = dict()
        for token in sentence:
            for tag in self.counter.sorted_tags:
                emission = _safe_log(self.counter.emission_prob(tag, token))
                prev_tag_prob_log = _safe_log(self.counter.transition_prob(prev_tag, tag))
                if token == sentence[0]:
                    tag_prob_log = _safe_log(self.counter.initial_prob(tag))
                    prob_matrix[tag] = tag_prob_log + emission
                else:
                    prob_matrix[tag] = prev_tag_prob_log + emission
            max_tuple = _max_item(prob_matrix)
            max_tag = max_tuple[0]
            max_prob = max_tuple[1]
            tags.insert(0, max_tag)
            prev_tag = max_tag
        tags.reverse()
        return tags


class ViterbiBigramTagger(BigramTagger):
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        viterbi_scores = dict()
        backpointer_padded_list = [{}]
        count = 0
        for i in range(1, len(sentence)):
            if count == 0:
                for tag in self.counter.sorted_tags:
                    viterbi_scores[tag] = _safe_log(self.counter.initial_prob(tag)) + _safe_log(
                        self.counter.emission_prob(tag, sentence[0]))
                    count += 1
            token_score = dict()
            temp_backpointer_dict = dict()
            for tag in self.counter.sorted_tags:
                transition_dict = dict()
                for prev_tag in self.counter.sorted_tags:
                    transition_dict[prev_tag] = _safe_log(self.counter.emission_prob(tag, sentence[i])) + _safe_log(
                        self.counter.transition_prob(prev_tag, tag)) + viterbi_scores[prev_tag]
                token_score[tag] = _max_item(transition_dict)[1]
                temp_backpointer_dict[tag] = _max_item(transition_dict)[0]
            backpointer_padded_list.append(temp_backpointer_dict)
            viterbi_scores = token_score
        tags = [_max_item(viterbi_scores)[0]]
        pointer = _max_item(viterbi_scores)[0]
        for i in range(len(backpointer_padded_list) - 1, 0, -1):
            tag = backpointer_padded_list[i][pointer]
            tags.insert(0, tag)
            pointer = tag
        return tags
