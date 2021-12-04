import json
import math
from collections import Counter, defaultdict
from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)


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

    def __init__(self, label: str, left_context: str, token: str, right_context: str) -> None:
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
def load_airline_instances(datapath: str, ) -> Generator[AirlineSentimentInstance, None, None]:
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)

        for json_item in json_list:
            yield AirlineSentimentInstance.from_dict(json_item)


# DO NOT MODIFY
def load_segmentation_instances(datapath: str, ) -> Generator[SentenceSplitInstance, None, None]:
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)

        for json_item in json_list:
            yield SentenceSplitInstance.from_dict(json_item)


# DO NOT MODIFY
class ClassificationInstance:
    def __init__(self, label: str, features: list[str]) -> None:
        self.label = label
        self.features = features


################################

class UnigramAirlineSentimentFeatureExtractor:
    def extract_features(self, instance: AirlineSentimentInstance) -> ClassificationInstance:
        classification_loader_list = []
        airline_lowered_lists = instance.sentences
        for list in airline_lowered_lists:
            for item in list:
                classification_loader_list.append(item.lower())
        classification_instance = ClassificationInstance(instance.label, sorted(classification_loader_list))
        yield (classification_instance)


class BigramAirlineSentimentFeatureExtractor:
    def extract_features(self, instance: AirlineSentimentInstance) -> ClassificationInstance:
        START_TOKEN = "<start>"
        END_TOKEN = "<end>"
        pre_classification_loader_list = []
        airline_lowered_lists = instance.sentences
        for list in airline_lowered_lists:
            for item in list:
                pre_classification_loader_list.append(item.lower())

        classification_loader_list = []
        length = len(pre_classification_loader_list) - 1
        for i in range(0, length):
            if i == 0:
                start = (tuple((START_TOKEN, pre_classification_loader_list[i])))
            if i == length - 1:
                end = (tuple((pre_classification_loader_list[i + 1], END_TOKEN)))

        classification_loader_list.append(start)
        for i in range(0, length):
            classification_loader_list.append(
                tuple((pre_classification_loader_list[i], pre_classification_loader_list[i + 1])))
        classification_loader_list.append(end)

        classification_instance = ClassificationInstance(instance.label, sorted(classification_loader_list))
        yield classification_instance


class BaselineSegmentationFeatureExtractor:
    def extract_features(self, instance: SentenceSplitInstance) -> ClassificationInstance:
        # left_tok = "left_tok=" + instance.left_context
        # split_tok = "split_tok=" + instance.token
        # right_tok = "right_tok=" + instance.right_context
        classification_loader_list = [f"left_tok={instance.left_context}", f"split_tok={instance.token}", f"right_tok={instance.right_context}"]
        classification_instance = ClassificationInstance(instance.label, classification_loader_list)
        yield classification_instance


class InstanceCounter:

    def __init__(self) -> None:
        self.label_counts = Counter()
        self.feature_counts = Counter()
        self.label_dict = defaultdict(Counter)
        self.vocab = set()
        self.total_inst = 0

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        for item in instances:
            self.label_counts[item.label] += 1
            for feature in item.features:
                self.vocab.add(feature)
                self.label_dict[item.label][feature] += 1
                self.feature_counts[feature] += 1
            self.total_inst+=1

    def label_count(self, label: str) -> int:
        return self.label_counts[label]

    def total_labels(self) -> int:
        return sum(self.label_counts.values())

    def conditional_feature_count(self, label: str, feature: str) -> int:
        return self.label_dict[label][feature]

    def labels(self) -> list[str]:
        return list(self.label_counts)

    def feature_vocab_size(self) -> int:
        return len(self.vocab)

    def total_feature_count_for_class(self, label: str) -> int:
        return sum(self.label_dict[label].values())


class NaiveBayesClassifier:
    def __init__(self, k: float):
        self.k: float = k
        self.counter: InstanceCounter = InstanceCounter()

    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.counter.count_instances(instances)

    def classify(self, features: list[str]) -> str:
        # self.log_posterior_prob(features, label)
        pass

    def prior_prob(self, label: str) -> float:
        return self.counter.label_count(label) / self.counter.total_inst

    def likelihood_prob(self, feature: str, label) -> float:
        return ((self.counter.conditional_feature_count(label, feature))+self.k) / ((self.counter.total_feature_count_for_class(label))+(self.counter.feature_vocab_size())*self.k)

    def log_posterior_prob(self, features: list[str], label: str) -> float:
        feature_probs = []
        for feature in features:
            log_prior = math.log(self.prior_prob(label))
            log_likelihood = math.log(self.likelihood_prob(feature, label))
            feature_probs.append(log_prior + log_likelihood)
        print(feature_probs)
        return max(feature_probs, key=lambda x: x[0])

    def test(self, instances: Iterable[ClassificationInstance]) -> tuple[list[str], list[str]]:
        pass

####DRIVER SECTION####


##DATA LOADERS
# train_airline_instances = load_airline_instances("test_data/airline_sentiment/train.json")
segmentation_instances = load_segmentation_instances("test_data/sentence_splits/dev.json")


##OBJECT INITIALIZATION
# air_feat_extractor = UnigramAirlineSentimentFeatureExtractor()
# air_feat_biextractor = BigramAirlineSentimentFeatureExtractor()
base_segment_extractor = BaselineSegmentationFeatureExtractor()
instance_counter = InstanceCounter()
naive_bayes = NaiveBayesClassifier()

## driver code for training airline instances (both unigrams and bigrams)
## also used to test instanceCounter
# for inst in train_airline_instances:
    # air_feat_extractor.extract_features(inst)
    # test_classification_inst_list = air_feat_extractor.extract_features(inst)
    # instance_counter.count_instances(test_classification_inst_list)
    # output = instance_counter.label_count("positive")

# driver code for training segement extractor
for inst in segmentation_instances:
    test_classification_inst_list = base_segment_extractor.extract_features(inst)
    instance_counter.count_instances(test_classification_inst_list)
    output = instance_counter.feature_vocab_size()

print(output)
