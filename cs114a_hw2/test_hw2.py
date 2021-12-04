#! /usr/bin/env python

# test_hw2.py
# Version 1.2
# 10/4/2021

import os
import unittest

from grader import Grader, points
from hw2 import (
    accuracy,
    precision,
    recall,
    f1,
    load_segmentation_instances,
    ClassificationInstance,
    load_airline_instances,
    InstanceCounter,
    SentenceSplitInstance,
    NaiveBayesClassifier,
    UnigramAirlineSentimentFeatureExtractor,
    BaselineSegmentationFeatureExtractor,
    AirlineSentimentInstance,
    TunedAirlineSentimentFeatureExtractor,
)

SENTENCE_SPLIT_DIR = os.path.join("test_data", "sentence_splits")
AIRLINE_SENTIMENT_DIR = os.path.join("test_data", "airline_sentiment")


class TestScoringMetrics(unittest.TestCase):
    @points(1)
    def test_accuracy(self):
        """A call to accuracy() returns a float and is roughly 0.7."""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(accuracy(predicted, actual)))
        self.assertAlmostEqual(0.7, accuracy(predicted, actual))

    @points(1)
    def test_precision(self):
        """A call to precision() returns a float. Precision values for T and F are 2/3 and 3/4, respectively."""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(precision(predicted, actual, "T")))
        self.assertAlmostEqual(2 / 3, precision(predicted, actual, "T"))
        self.assertAlmostEqual(0.75, precision(predicted, actual, "F"))

    @points(1)
    def test_recall(self):
        """A call to recall() returns a float. Recall values for T and F are 4/5 and 3/5, respectively."""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(recall(predicted, actual, "T")))
        self.assertAlmostEqual(0.8, recall(predicted, actual, "T"))
        self.assertAlmostEqual(3 / 5, recall(predicted, actual, "F"))

    @points(1)
    def test_f1_score(self):
        """A call to f1() returns a float. Recall values for T and F are 8/11 and 2/3, respectively."""
        predicted = ["T", "T", "T", "T", "F", "F", "F", "F", "T", "T"]
        actual = ["T", "T", "T", "T", "T", "F", "F", "F", "F", "F"]
        self.assertEqual(float, type(f1(predicted, actual, "T")))
        self.assertAlmostEqual(8 / 11, f1(predicted, actual, "T"))
        self.assertAlmostEqual(2 / 3, f1(predicted, actual, "F"))


class TestSegmentationFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.sentence_split_label_set = frozenset(["y", "n"])
        self.seg_feature_extractor = BaselineSegmentationFeatureExtractor()
        self.sentence_split_instances = load_segmentation_instances(
            os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
        )
        self.sentence_split_classification_instance = (
            self.seg_feature_extractor.extract_features(
                next(self.sentence_split_instances)
            )
        )

    @points(1)
    def test_type_instance_sentence_split_classification(self):
        """Running feature extraction on dev sentence 1 returns a ClassificationInstance."""
        self.assertEqual(
            ClassificationInstance, type(self.sentence_split_classification_instance)
        )

    @points(1)
    def test_type_instance_features_sentence_split_classification(self):
        """The features of a ClassificationInstance are a list."""
        self.assertEqual(list, type(self.sentence_split_classification_instance.features))

    @points(1)
    def test_at_least_three_features_sentence_split_classification(self):
        """There are at least three features for dev sentence 1."""
        self.assertGreaterEqual(
            3, len(self.sentence_split_classification_instance.features)
        )

    @points(1)
    def test_type_individual_feature_sentence_split_classification(self):
        """Individual features of a ClassificationInstance are strings."""
        self.assertEqual(
            str, type(self.sentence_split_classification_instance.features[0])
        )

    @points(1)
    def test_instance_label_negative_sentence_split_classification(self):
        """The label of the ClassificationInstance representing dev sentence 1 is 'n'."""
        self.assertEqual("n", self.sentence_split_classification_instance.label)

    @points(1)
    def test_features_correct_sentence_split_classification(self):
        """Correct features are extracted for dev sentence 1."""
        self.assertEqual(
            {"split_tok=.", "left_tok=D", "right_tok=Forrester"},
            set(self.sentence_split_classification_instance.features),
        )

    @points(1)
    def test_predicted_labels_valid_sentence_split(self):
        """All predicted labels should be valid for sentence segmentation."""
        for inst in self.sentence_split_instances:
            classify_inst = self.seg_feature_extractor.extract_features(inst)
            self.assertIn(classify_inst.label, self.sentence_split_label_set)


class TestAirlineSentimentFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.airline_label_set = frozenset(["positive", "negative"])
        self.air_feat_extractor = UnigramAirlineSentimentFeatureExtractor()
        self.airline_sentiment_instances = load_airline_instances(
            os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
        )
        self.airline_classification_instance = self.air_feat_extractor.extract_features(
            next(self.airline_sentiment_instances)
        )

    @points(1)
    def test_type_instance_airline_sentiment_classification(self):
        """Running feature extraction on dev sentence 1 returns a ClassificationInstance."""
        self.assertEqual(
            ClassificationInstance, type(self.airline_classification_instance)
        )

    @points(1)
    def test_type_instance_features_airline_sentiment_classification(self):
        """The features of a ClassificationInstance are a list."""
        self.assertEqual(list, type(self.airline_classification_instance.features))

    @points(1)
    def test_type_individual_feature_airline_sentiment_classification(self):
        """Individual features of a ClassificationInstance are strings."""
        self.assertEqual(str, type(self.airline_classification_instance.features[0]))

    @points(1)
    def test_instance_label_negative_airline_sentiment_classification(self):
        """The label of the ClassificationInstance representing dev sentence 1 is 'negative'."""
        self.assertEqual("negative", self.airline_classification_instance.label)

    @points(1)
    def test_instance_correct_features_airline_sentiment_classification(self):
        """Correct features are extracted for dev sentence 1."""
        features = self.airline_classification_instance.features
        self.assertEqual(
            {
                "#",
                "&",
                ",",
                "2",
                ";",
                "?",
                "@nrhodes85",
                "@usairways",
                "above",
                "actually",
                "amp",
                "apologizes",
                "beyond",
                "but",
                "customers",
                "does",
                "for",
                "funny",
                "go",
                "is",
                "it",
                "just",
                "n't",
                "notimpressed",
                "steps",
                "take",
                "that",
            },
            set(features),
        )

    @points(1)
    def test_predicted_labels_valid_airline_sentiment_classification(self):
        """All predicted labels are valid for airline sentiment analysis."""
        for inst in self.airline_sentiment_instances:
            classify_inst = self.air_feat_extractor.extract_features(inst)
            self.assertIn(classify_inst.label, self.airline_label_set)


class SegmentationTestFeatureExtractor:
    """
    Simple baseline feature extractor to test InstanceCounter and NaiveBayes independently of what
    the real feature extractors are choosing.
    """

    def extract_features(self, inst: SentenceSplitInstance) -> ClassificationInstance:
        return ClassificationInstance(
            inst.label, [f"left_tok={inst.left_context}", f"split_tok={inst.token}"]
        )


class SentimentTestFeatureExtractor:
    """
    Simple baseline feature extractor to test InstanceCounter and NaiveBayes independently of what
    the real feature extractors are choosing.
    """

    def __init__(self):
        self.words = frozenset(["thank", "bad", "great", "good", "like", "you"])

    def extract_features(self, inst: AirlineSentimentInstance) -> ClassificationInstance:
        return ClassificationInstance(
            inst.label,
            [tok for sent in inst.sentences for tok in sent if tok.lower() in self.words],
        )


class TestInstanceCounter(unittest.TestCase):
    def setUp(self) -> None:
        # Create instance counter and count the instances
        self.inst_counter = InstanceCounter()
        feature_extractor = SegmentationTestFeatureExtractor()
        self.inst_counter.count_instances(
            feature_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        )

    @points(2)
    def test_label_counts_y(self):
        """The correct number of instances of the label 'y' are observed."""
        self.assertEqual(6110, self.inst_counter.label_count("y"))

    @points(2)
    def test_label_counts_n(self):
        """The correct number of instances of the label 'n' are observed."""
        self.assertEqual(811, self.inst_counter.label_count("n"))

    @points(1)
    def test_total_labels(self):
        """The correct total number of labels are observed."""
        self.assertEqual(6921, self.inst_counter.total_labels())

    @points(2)
    def test_conditional_feature_count_1(self):
        """A period (.) is a sentence boundary in the correct number of cases."""
        self.assertEqual(
            5903, self.inst_counter.conditional_feature_count("y", "split_tok=.")
        )

    @points(2)
    def test_conditional_feature_count_2(self):
        """A period (.) is not a sentence boundary in the correct number of cases."""
        self.assertEqual(
            751, self.inst_counter.conditional_feature_count("n", "split_tok=.")
        )

    @points(3)
    def test_labels(self):
        """All observed labels are valid. There are a total of 2 labels."""
        labels = frozenset(["y", "n"])
        for label in self.inst_counter.labels():
            self.assertIn(label, labels)
        self.assertEqual(2, len(self.inst_counter.labels()))

    @points(3)
    def test_feature_vocab_size(self):
        """The correct total number of features are observed."""
        self.assertEqual(2964, self.inst_counter.feature_vocab_size())

    @points(3)
    def test_total_feature_count_for_class(self):
        """Correct total number of features is observed for both classes."""
        self.assertEqual(12220, self.inst_counter.total_feature_count_for_class("y"))
        self.assertEqual(1622, self.inst_counter.total_feature_count_for_class("n"))


class TestNaiveBayesSegmentation(unittest.TestCase):
    def setUp(self):
        """Load data and train classifiers"""
        # up and train segmentation classifier
        segmentation_extractor = SegmentationTestFeatureExtractor()
        segmentation_train_instances = (
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "train.json")
            )
        )
        self.segmentation_dev_instances = [
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        ]
        self.segmentation_classifier = NaiveBayesClassifier(k=0.01)
        self.segmentation_classifier.train(segmentation_train_instances)

    @points(3)
    def test_prior_probability(self):
        """Prior class probabilities are computed correctly."""
        self.assertAlmostEqual(
            0.8666957485868004, self.segmentation_classifier.prior_prob("y")
        )
        self.assertAlmostEqual(
            0.13330425141319954, self.segmentation_classifier.prior_prob("n")
        )

    @points(3)
    def test_likelihood_prob(self):
        """Likelihood probabilities are computed correctly."""
        self.assertAlmostEqual(
            0.48499813575085254,
            self.segmentation_classifier.likelihood_prob("split_tok=.", "y"),
        )
        self.assertAlmostEqual(
            0.0034981902120635776,
            self.segmentation_classifier.likelihood_prob("split_tok=!", "y"),
        )
        self.assertAlmostEqual(
            0.4801532862515897,
            self.segmentation_classifier.likelihood_prob("split_tok=.", "n"),
        )
        self.assertAlmostEqual(
            0.005953540601239277,
            self.segmentation_classifier.likelihood_prob("split_tok=!", "n"),
        )

    @points(3)
    def test_log_posterior_probability_segmentation(self):
        """Posterior log-probabilities are computed correctly."""
        self.assertAlmostEqual(
            -0.8666775200066891,
            self.segmentation_classifier.log_posterior_prob(["split_tok=."], "y"),
        )
        self.assertAlmostEqual(
            -5.798576814629429,
            self.segmentation_classifier.log_posterior_prob(["split_tok=!"], "y"),
        )
        self.assertAlmostEqual(
            -2.74877103843659,
            self.segmentation_classifier.log_posterior_prob(["split_tok=."], "n"),
        )
        self.assertAlmostEqual(
            -7.1388903361038825,
            self.segmentation_classifier.log_posterior_prob(["split_tok=!"], "n"),
        )

    @points(2)
    def test_classify(self):
        """The candidate boundaries 'products.' and 'Dr.' are classified correctly and a string label is returned."""
        self.assertEqual(
            "y",
            self.segmentation_classifier.classify(["left_tok=products", "split_tok=."]),
        )
        self.assertEqual(
            "n", self.segmentation_classifier.classify(["left_tok=Dr", "split_tok=."])
        )
        self.assertEqual(
            str, type(self.segmentation_classifier.classify(["split_tok=."]))
        )

    @points(3)
    def test_naivebayes_test(self):
        """Naive Bayes classification works correctly."""
        result = self.segmentation_classifier.test(
            [
                ClassificationInstance("y", ["left_tok=outstanding", "split_tok=."]),
                ClassificationInstance("y", ["left_tok=fairly", "split_tok=?"]),
                ClassificationInstance("n", ["left_tok=U.S", "split_tok=."]),
                ClassificationInstance("y", ["left_tok=!", "split_tok=!"]),
                ClassificationInstance("n", ["left_tok=Mx.", "split_tok=."]),
            ]
        )
        self.assertEqual(tuple, type(result))
        self.assertEqual(list, type(result[0]))
        self.assertEqual(list, type(result[1]))
        self.assertEqual(len(result[0]), len(result[1]))
        for item in result[0]:
            self.assertEqual(str, type(item))
        for item in result[1]:
            self.assertEqual(str, type(item))
        self.assertEqual((["y", "y", "n", "y", "y"], ["y", "y", "n", "y", "n"]), result)


class TestNaiveBayesSentiment(unittest.TestCase):
    def setUp(self):
        """Load data and train classifier"""
        airline_extractor = SentimentTestFeatureExtractor()
        airline_train_instances = (
            airline_extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "train.json")
            )
        )
        self.airline_dev_instances = [
            airline_extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
            )
        ]
        self.airline_classifier = NaiveBayesClassifier(0.01)
        self.airline_classifier.train(airline_train_instances)

    @points(3)
    def test_prior_probability(self):
        """Prior class probabilities are computed correctly."""
        self.assertAlmostEqual(
            0.20602253032928944, self.airline_classifier.prior_prob("positive")
        )
        self.assertAlmostEqual(
            0.7939774696707106, self.airline_classifier.prior_prob("negative")
        )

    @points(3)
    def test_likelihood_prob(self):
        """Likelihood probabilities are computed correctly."""
        self.assertAlmostEqual(
            0.11072534291982475,
            self.airline_classifier.likelihood_prob("thank", "positive"),
        )
        self.assertAlmostEqual(
            0.008891653054668972,
            self.airline_classifier.likelihood_prob("bad", "positive"),
        )
        self.assertAlmostEqual(
            0.011247039536916477,
            self.airline_classifier.likelihood_prob("thank", "negative"),
        )
        self.assertAlmostEqual(
            0.037360772096026,
            self.airline_classifier.likelihood_prob("bad", "negative"),
        )

    @points(3)
    def test_log_posterior_probability(self):
        """Posterior log-probabilities are computed correctly."""
        self.assertAlmostEqual(
            -3.780472277779577,
            self.airline_classifier.log_posterior_prob(["thank"], "positive"),
        )
        self.assertAlmostEqual(
            -6.302412046977491,
            self.airline_classifier.log_posterior_prob(["bad"], "positive"),
        )
        self.assertAlmostEqual(
            -4.718350531103617,
            self.airline_classifier.log_posterior_prob(["thank"], "negative"),
        )
        self.assertAlmostEqual(
            -3.5178341933203736,
            self.airline_classifier.log_posterior_prob(["bad"], "negative"),
        )

    @points(4)
    def test_classify(self):
        """The tokens 'thank' and 'bad' are classified correctly and a string label is returned."""
        self.assertEqual("positive", self.airline_classifier.classify(["thank"]))
        self.assertEqual("negative", self.airline_classifier.classify(["bad"]))
        self.assertEqual(str, type(self.airline_classifier.classify(["thank"])))


class TestPerformanceSegmentation(unittest.TestCase):
    def setUp(self):
        """Load data and train classifiers"""
        segmentation_extractor = BaselineSegmentationFeatureExtractor()
        segmentation_train_instances = (
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "train.json")
            )
        )
        self.segmentation_dev_instances = [
            segmentation_extractor.extract_features(inst)
            for inst in load_segmentation_instances(
                os.path.join(SENTENCE_SPLIT_DIR, "dev.json")
            )
        ]
        self.segmentation_classifier = NaiveBayesClassifier(2.0)
        self.segmentation_classifier.train(segmentation_train_instances)

    @points(4)
    def test_segmentation_performance_y(self):
        """Segmentation performance is sufficiently good."""
        predicted, expected = self.segmentation_classifier.test(
            self.segmentation_dev_instances
        )
        acc, prec, rec, f1_score, report = classification_report(predicted, expected, "y")
        print("Baseline segmentation performance:")
        print(report)

        self.assertLessEqual(0.986, acc)
        self.assertLessEqual(0.985, prec)
        self.assertLessEqual(0.998, rec)
        self.assertLessEqual(0.992, f1_score)


class TestPerformanceSentiment(unittest.TestCase):
    def setUp(self):
        """Load data and train classifiers"""
        airline_extractor = UnigramAirlineSentimentFeatureExtractor()
        airline_train_instances = (
            airline_extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "train.json")
            )
        )
        self.airline_dev_instances = [
            airline_extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
            )
        ]
        self.airline_classifier = NaiveBayesClassifier(0.05)
        self.airline_classifier.train(airline_train_instances)

    @points(2)
    def test_sentiment_performance_positive(self):
        """Baseline performance on airplane sentiment analysis is sufficiently good for the 'positive' label."""
        predicted, expected = self.airline_classifier.test(self.airline_dev_instances)
        acc, prec, rec, f1_score, report = classification_report(
            predicted, expected, "positive"
        )
        print("Baseline positive sentiment performance:")
        print(report)

        self.assertLessEqual(0.922, acc)
        self.assertLessEqual(0.806, prec)
        self.assertLessEqual(0.810, rec)
        self.assertLessEqual(0.808, f1_score)

    @points(2)
    def test_sentiment_performance_negative(self):
        """Baseline performance on airplane sentiment analysis is sufficiently good for the 'negative' label."""
        predicted, expected = self.airline_classifier.test(self.airline_dev_instances)
        acc, prec, rec, f1_score, report = classification_report(
            predicted, expected, "negative"
        )
        print("Baseline negative sentiment performance:")
        print(report)

        self.assertLessEqual(0.922, acc)
        self.assertLessEqual(0.952, prec)
        self.assertLessEqual(0.951, rec)
        self.assertLessEqual(0.951, f1_score)


class TestTunedAirlineSentiment(unittest.TestCase):
    @points(0)
    def test_tuned_airline_sentiment(self):
        """Tuned airline sentiment analysis works correctly."""
        extractor = TunedAirlineSentimentFeatureExtractor()
        self.assertIsNotNone(extractor.k)

        sentiment_train_instances = (
            extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "train.json")
            )
        )

        self.sentiment_dev_instances = [
            extractor.extract_features(inst)
            for inst in load_airline_instances(
                os.path.join(AIRLINE_SENTIMENT_DIR, "dev.json")
            )
        ]
        self.sentiment_classifier = NaiveBayesClassifier(extractor.k)
        self.sentiment_classifier.train(sentiment_train_instances)
        predicted, expected = self.sentiment_classifier.test(self.sentiment_dev_instances)
        for positive_label in self.sentiment_classifier.counter.label_counts:
            acc, prec, rec, f1_score, report = classification_report(
                predicted, expected, positive_label
            )
            print(
                f'Tuned sentiment analysis performance for k of {extractor.k} for label "{positive_label}":'
            )
            print(report)
            print()


def classification_report(
    predicted: list[str],
    expected: list[str],
    positive_label: str,
) -> tuple[float, float, float, float, str]:
    """Return accuracy, P, R, F1 and a classification report."""
    acc = accuracy(predicted, expected)
    prec = precision(predicted, expected, positive_label)
    rec = recall(predicted, expected, positive_label)
    f1_score = f1(predicted, expected, positive_label)
    report = "\n".join(
        [
            f"Accuracy:  {acc * 100:0.2f}",
            f"Precision: {prec * 100:0.2f}",
            f"Recall:    {rec * 100:0.2f}",
            f"F1:        {f1_score * 100:0.2f}",
        ]
    )
    return acc, prec, rec, f1_score, report


def main() -> None:
    tests = [
        TestScoringMetrics,
        TestSegmentationFeatureExtractor,
        TestAirlineSentimentFeatureExtractor,
        TestInstanceCounter,
        TestNaiveBayesSegmentation,
        TestNaiveBayesSentiment,
        TestPerformanceSegmentation,
        TestPerformanceSentiment,
        TestTunedAirlineSentiment,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
