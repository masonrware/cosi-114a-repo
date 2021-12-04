# hw2.py
# Version 1.0
# 9/28/2021

import json
import math
from collections import Counter, defaultdict
from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single data point from the airline sentiment dataset.

    Each data point consists of
    - the airline (str)
    - the sentiment label (str)
    - the review itself (list of lists)
        - outer lists represent sentences
        - inner lists represent tokens within sentences
    """

    def __init__(self, label: str, sentences: list[list[str]], airline: str) -> None:
        self.label = label
        self.sentences = sentences
        self.airline = airline

    def __repr__(self) -> str:
        return f"label={self.label}; sentences={self.sentences}"

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "sentences": self.sentences,
            "airline": self.airline,
        }

    @classmethod
    def from_dict(cls, json_dict: dict[str, Any]) -> "AirlineSentimentInstance":
        return AirlineSentimentInstance(
            json_dict["label"], json_dict["sentences"], json_dict["airline"]
        )


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary within a given string.

    An instance may correspond to a true boundary or not.
    The boundary is represented using the following properties:
    - label (str): either 'y' (true) or 'n' (false)
    - left_context (str): token immediately preceding the sentence boundary token
    - token (str): string representing the sentence boundary (str)
        - for example, a period (.) or question mark (?)
        - the last token of the sentence if this is a true sentence boundary.
    - right_context (str): token immediately following the sentence boundary token
    """

    def __init__(
            self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label = label
        self.left_context = left_context
        self.token = token
        self.right_context = right_context

    def __repr__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left={self.left_context};",
                f"token={self.token};",
                f"right={self.right_context}",
            ]
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self):
        return {
            "label": self.label,
            "left": self.left_context,
            "token": self.token,
            "right": self.right_context,
        }

    @classmethod
    def from_dict(cls, json_dict: dict[Any, Any]) -> "SentenceSplitInstance":
        return SentenceSplitInstance(
            json_dict["label"],
            json_dict["left"],
            json_dict["token"],
            json_dict["right"],
        )


# DO NOT MODIFY
def load_airline_instances(
        datapath: str,
) -> Generator[AirlineSentimentInstance, None, None]:
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)

        for json_item in json_list:
            yield AirlineSentimentInstance.from_dict(json_item)


# DO NOT MODIFY
def load_segmentation_instances(
        datapath: str,
) -> Generator[SentenceSplitInstance, None, None]:
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)

        for json_item in json_list:
            yield SentenceSplitInstance.from_dict(json_item)


# DO NOT MODIFY
class ClassificationInstance:
    def __init__(self, label: str, features: list[str]) -> None:
        self.label = label
        self.features = features


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    correctClass = 0
    if len(predictions) != len(expected):
        raise ValueError
    else:
        total = len(predictions)
    if len(predictions) == 0 or len(expected) == 0:
        raise ValueError
    for i in range(len(predictions)):
        if predictions[i] == expected[i]:
            correctClass = correctClass + 1
    if total == 0:
        return 0.0
    else:
        return correctClass / total


def recall(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    TPcount = 0
    FNcount = 0
    if len(predictions) != len(expected):
        raise ValueError
    if len(predictions) == 0 or len(expected) == 0:
        raise ValueError
    for i in range(len(predictions)):
        if predictions[i] == label and expected[i] == label:
            TPcount = TPcount + 1
        elif predictions[i] != label and expected[i] == label:
            FNcount = FNcount + 1
    if (TPcount + FNcount) == 0:
        return 0.0
    else:
        return TPcount / (TPcount + FNcount)


def precision(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    TPcount = 0
    FPcount = 0
    if len(predictions) != len(expected):
        raise ValueError
    if len(predictions) == 0 or len(expected) == 0:
        raise ValueError
    for i in range(len(predictions)):
        if predictions[i] == label and expected[i] == label:
            TPcount = TPcount + 1
        elif predictions[i] == label and expected[i] != label:
            FPcount = FPcount + 1
    if (TPcount + FPcount) == 0:
        return 0.0
    else:
        return TPcount / (TPcount + FPcount)


def f1(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    if len(predictions) != len(expected):
        raise ValueError
    if len(predictions) == 0 or len(expected) == 0:
        raise ValueError
    if (precision(predictions, expected, label) + recall(predictions, expected, label)) == 0:
        return 0.0
    else:
        return ((2 * precision(predictions, expected, label) * recall(predictions, expected, label)) / (
            precision(predictions, expected, label) + recall(predictions, expected, label)))


class UnigramAirlineSentimentFeatureExtractor:
    def extract_features(self, instance: AirlineSentimentInstance) -> ClassificationInstance:
        classification_loader_list = set()
        airline_lowered_lists = instance.sentences
        for sentence in airline_lowered_lists:
            for item in sentence:
                classification_loader_list.add(item.lower())
        return ClassificationInstance(instance.label, sorted(classification_loader_list))


class BigramAirlineSentimentFeatureExtractor:
    def extract_features(self, instance: AirlineSentimentInstance) -> ClassificationInstance:
        START_TOKEN = "<start>"
        END_TOKEN = "<end>"
        airline_lowered_lists = instance.sentences
        for sentence in airline_lowered_lists:
            classification_loader_list = set()
            length = len(sentence) - 1
            for i in range(0, length):
                if i == 0:
                    start = (tuple((START_TOKEN, sentence[i])))
                if i == length - 1:
                    end = (tuple((sentence[i + 1], END_TOKEN)))

            classification_loader_list.add(start)
            for i in range(0, length):
                classification_loader_list.add(
                    tuple((sentence[i], sentence[i + 1])))
            classification_loader_list.add(end)
        return ClassificationInstance(instance.label, sorted(classification_loader_list))


class BaselineSegmentationFeatureExtractor:
    def extract_features(self, instance: SentenceSplitInstance) -> ClassificationInstance:
        classification_loader_list = [f"left_tok={instance.left_context}", f"split_tok={instance.token}",
                                      f"right_tok={instance.right_context}"]
        return ClassificationInstance(instance.label, classification_loader_list)


class InstanceCounter:

    def __init__(self) -> None:
        self.label_counts = Counter()
        self.label_dict = defaultdict(Counter)
        self.vocab = set()
        self.total_inst = 0
        self.feature_counts = {}

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        for item in instances:
            self.label_counts[item.label] += 1
            for feature in item.features:
                self.vocab.add(feature)
                self.label_dict[item.label][feature] += 1
            self.total_inst += 1
        for label in self.label_dict:
            self.feature_counts[label] = sum(self.label_dict[label].values())

    def label_count(self, label: str) -> int:
        return self.label_counts[label]

    def total_labels(self) -> int:
        return self.total_inst

    def conditional_feature_count(self, label: str, feature: str) -> int:
        return self.label_dict[label][feature]

    def labels(self) -> list[str]:
        return list(self.label_counts)

    def feature_vocab_size(self) -> int:
        return len(self.vocab)

    def total_feature_count_for_class(self, label: str) -> int:
        return self.feature_counts[label]


class NaiveBayesClassifier:
    def __init__(self, k: float):
        self.k: float = k
        self.counter: InstanceCounter = InstanceCounter()

    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.counter.count_instances(instances)

    def classify(self, features: list[str]) -> str:
        myList = []
        for label in self.counter.label_counts.keys():
            addTuple = (label, self.log_posterior_prob(features, label))
            myList.append(addTuple)
        largestTuple = max(myList, key=lambda x: x[1])
        return largestTuple[0]

    def prior_prob(self, label: str) -> float:
        return self.counter.label_count(label) / self.counter.total_inst

    def likelihood_prob(self, feature: str, label) -> float:
        return ((self.counter.conditional_feature_count(label, feature)) + self.k) / (
                (self.counter.total_feature_count_for_class(label)) + (self.counter.feature_vocab_size()) * self.k)

    def log_posterior_prob(self, features: list[str], label: str) -> float:
        feature_probs = []
        log_prior = math.log(self.prior_prob(label))
        for feature in features:
            if feature in self.counter.vocab:
                log_likelihood = math.log(self.likelihood_prob(feature, label))
                feature_probs.append(log_likelihood)
        return sum(feature_probs) + log_prior

    def test(self, instances: Iterable[ClassificationInstance]) -> tuple[list[str], list[str]]:
        predicted = []
        observed = []
        returner = predicted, observed
        for instance in instances:
            predicted.append(self.classify(instance.features))
            observed.append(instance.label)
        return returner


# MODIFY THIS AND DO THE FOLLOWING:
# - Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#   (instead of object) to get an implementation for the extract_features method.
# - Change `self.k` below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(UnigramAirlineSentimentFeatureExtractor):
    def __init__(self) -> None:
        self.k = 0.005  # CHANGE ME
