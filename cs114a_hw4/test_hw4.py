#!/usr/bin/env python

# test_hw4.py
# Version 1.0
# 11/2/2021

import random
import unittest
from typing import Generator

from sklearn.metrics import accuracy_score

from grader import Grader, points, timeout
from hw4 import (
    MostFrequentTagTagger,
    SentenceCounter,
    Token,
    UnigramTagger,
    GreedyBigramTagger,
    ViterbiBigramTagger,
)


def _make_sentences(sentences: list[list[tuple[str, str]]]) -> list[list[Token]]:
    return [[Token.from_tuple(pair) for pair in sentence] for sentence in sentences]


# Has to be defined below make_sentences
SENTENCES_AB_XYZ = _make_sentences(
    [
        [("x", "A"), ("x", "A"), ("y", "A"), ("z", "A"), ("z", "A")],
        [("x", "B"), ("y", "B"), ("y", "B"), ("y", "B"), ("z", "B")],
    ]
)


def load_pos_data(path):
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:
            if not line.strip():
                continue
            yield [Token.from_string(tok) for tok in line.rstrip("\n").split(" ")]


class TestMostFrequentTagTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences = _make_sentences(
            [[("foo", "NN"), ("bar", "NNS")], [("baz", "NN")]]
        )
        self.tagger = MostFrequentTagTagger()
        self.tagger.train(self.sentences)

    @points(1)
    def test_most_frequent_tag_sentence(self):
        """Each token is correctly tagged with the most frequent tag."""
        sentence = ("This", "is", "a", "sentence", ".")
        self.assertListEqual(
            ["NN", "NN", "NN", "NN", "NN"], self.tagger.tag_sentence(sentence)
        )

    @points(2)
    def test_most_freq_accuracy(self):
        """Accuracy is sufficiently high."""
        self.tagger.train(load_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_pos_data("test_data/100_dev.txt")
        predicted, actual = self.tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"MostFrequentTagTagger Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.128, accuracy)


class TestUnigramTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences = _make_sentences(
            [
                [("foo", "NN"), ("foo", "NNS"), ("bar", "JJ")],
                [("foo", "NN"), ("bar", "JJ")],
                [("baz", "VB")],
            ]
        )
        self.tagger = UnigramTagger()
        self.tagger.train(self.sentences)

    @points(1)
    def test_tag_foo(self):
        """A single token sentence 'foo' is tagged correctly."""
        tags = self.tagger.tag_sentence(["foo"])
        self.assertEqual("NN", tags[0])

    @points(1)
    def test_tag_bar(self):
        """A single token sentence 'bar' is tagged correctly."""
        tags = self.tagger.tag_sentence(["bar"])
        self.assertEqual("JJ", tags[0])

    @points(1)
    def test_tag_baz(self):
        """A single token sentence 'baz' is tagged correctly."""
        tags = self.tagger.tag_sentence(["baz"])
        self.assertEqual("VB", tags[0])

    @points(3)
    def test_unigram_tagger_accuracy(self):
        """Accuracy is sufficiently high."""
        self.tagger.train(load_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_pos_data("test_data/100_dev.txt")
        predicted, actual = self.tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"UnigramTagger Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.929, accuracy)


class TestInstanceCounterUnsmoothed(unittest.TestCase):
    def setUp(self):
        self.inst_counter = SentenceCounter(0.0)
        sentences = _make_sentences(
            [
                [("e", "C"), ("f", "A"), ("e", "C")],
                [("h", "B"), ("f", "C"), ("g", "B")],
            ]
        )
        self.inst_counter.count_sentences(sentences)

    @points(1.5)
    def test_tagset(self):
        """The correct tag list is returned."""
        self.assertEqual(["A", "B", "C"], self.inst_counter.tagset())

    @points(0.5)
    def test_emission_prob1(self):
        """Unsmoothed emission probabilities for tag C are correct."""
        self.assertAlmostEqual(2 / 3, self.inst_counter.emission_prob("C", "e"))
        self.assertAlmostEqual(1 / 3, self.inst_counter.emission_prob("C", "f"))

    @points(0.5)
    def test_emission_prob2(self):
        """Unsmoothed emission probabilities for tag B are correct."""
        self.assertAlmostEqual(0.5, self.inst_counter.emission_prob("B", "g"))
        self.assertAlmostEqual(0.5, self.inst_counter.emission_prob("B", "h"))

    @points(0.5)
    def test_emission_prob3(self):
        """Unsmoothed emission probabilities for tag A are correct."""
        self.assertEqual(1.0, self.inst_counter.emission_prob("A", "f"))

    @points(1)
    def test_initial_prob(self):
        """Observed initial tags have correct initial probabilities."""
        self.assertEqual(0.5, self.inst_counter.initial_prob("C"))
        self.assertEqual(0.5, self.inst_counter.initial_prob("B"))

    @points(1)
    def test_transition_prob1(self):
        """Outgoing transitions have correct probabilities and sum to 1."""
        self.assertEqual(0.5, self.inst_counter.transition_prob("C", "A"))
        self.assertEqual(0.5, self.inst_counter.transition_prob("C", "B"))

    @points(1)
    def test_transition_prob2(self):
        """Incoming transitions have correct probabilities and need not sum to 1."""
        self.assertEqual(1.0, self.inst_counter.transition_prob("A", "C"))
        self.assertEqual(1.0, self.inst_counter.transition_prob("B", "C"))


class TestInstanceCounterSmoothed(unittest.TestCase):
    def setUp(self):
        self.inst_counter = SentenceCounter(1.0)
        sentences = _make_sentences(
            [
                [("e", "C"), ("f", "A"), ("e", "C")],
                [("h", "B"), ("f", "C"), ("g", "B")],
            ]
        )
        self.inst_counter.count_sentences(sentences)

    @points(0.5)
    def test_emission_prob1(self):
        """Smoothed emission probabilities for tag C are correct."""
        self.assertAlmostEqual(3 / 5, self.inst_counter.emission_prob("C", "e"))
        self.assertAlmostEqual(2 / 5, self.inst_counter.emission_prob("C", "f"))

    @points(0.5)
    def test_emission_prob2(self):
        """Smoothed emission probabilities for tag B are correct."""
        self.assertAlmostEqual(2 / 4, self.inst_counter.emission_prob("B", "g"))
        self.assertAlmostEqual(2 / 4, self.inst_counter.emission_prob("B", "h"))

    @points(0.5)
    def test_emission_prob3(self):
        """Smoothed emission probabilities for tag A are correct."""
        self.assertEqual(1.0, self.inst_counter.emission_prob("A", "f"))

    @points(0.5)
    def test_emission_prob4(self):
        """With smoothing, unobserved emissions have nonzero emission probabilities."""
        self.assertEqual(1 / 2, self.inst_counter.emission_prob("A", "e"))
        self.assertEqual(1 / 4, self.inst_counter.emission_prob("B", "f"))
        self.assertEqual(1 / 5, self.inst_counter.emission_prob("C", "g"))

    # Initial/transition probabilities are not affected by smoothing, so these tests
    # give minimal points
    @points(0.2)
    def test_initial_prob1(self):
        """Observed initial tags have correct initial probabilities."""
        self.assertEqual(0.5, self.inst_counter.initial_prob("C"))
        self.assertEqual(0.5, self.inst_counter.initial_prob("B"))

    @points(0.2)
    def test_initial_prob2(self):
        """Unobserved initial tags have initial probability 0."""
        self.assertEqual(0.0, self.inst_counter.initial_prob("A"))

    @points(0.2)
    def test_transition_prob1(self):
        """Outgoing transitions have correct probabilities and sum to 1."""
        self.assertEqual(0.5, self.inst_counter.transition_prob("C", "A"))
        self.assertEqual(0.5, self.inst_counter.transition_prob("C", "B"))

    @points(0.2)
    def test_transition_prob2(self):
        """Incoming transitions have correct probabilities and need not sum to 1."""
        self.assertEqual(1.0, self.inst_counter.transition_prob("A", "C"))
        self.assertEqual(1.0, self.inst_counter.transition_prob("B", "C"))

    @points(0.2)
    def test_transition_prob3(self):
        """Unobserved transitions have probability 0."""
        self.assertEqual(0.0, self.inst_counter.transition_prob("A", "B"))
        self.assertEqual(0.0, self.inst_counter.transition_prob("B", "A"))


class TestSentenceCounterSpeed(unittest.TestCase):
    @points(7)
    @timeout(3)
    def test_efficient_implementation(self):
        """Test that SentenceCounter is efficiently implemented.

        If you are failing this test with a TimeoutError, you are not implementing
        SentenceCounter efficiently and are probably using loops or sums in your
        probability functions. Instead, precompute values in count_sentences.
        """
        counter = SentenceCounter(1.0)
        counter.count_sentences(self._make_random_sentences(25_000))
        for _ in range(10000):
            counter.tagset()
            counter.initial_prob("A")
            counter.transition_prob("A", "A")
            counter.emission_prob("A", "1")

    @staticmethod
    def _make_random_sentences(n_sentences: int) -> Generator[list[Token], None, None]:
        random.seed(0)
        tags = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        tokens = [str(n) for n in range(n_sentences)]
        lengths = list(range(10, 31))
        for _ in range(n_sentences):
            sen_length = random.choice(lengths)
            sentence = [
                Token(random.choice(tokens), random.choice(tags))
                for _ in range(sen_length)
            ]
            yield sentence


class TestBigramSequenceProbability(unittest.TestCase):
    def setUp(self):
        # We test through the greedy tagger but could also do it through Viterbi
        self.tagger = GreedyBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(1)
    def test_prob1(self):
        """The bigram sequence log-probability is correct."""
        self.assertAlmostEqual(
            -3.2188758248682006,
            self.tagger.sequence_probability(["x", "y"], ["A", "A"]),
        )

    @points(2)
    def test_prob4(self):
        """The bigram sequence log-probability is correct."""
        self.assertAlmostEqual(
            -2.8134107167600364,
            self.tagger.sequence_probability(["x", "y"], ["B", "B"]),
        )


class TestGreedyBigramTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = GreedyBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(2)
    def test_ab_xyz_tag1(self):
        """The greedy tagger correctly tags 'x x'."""
        sent = ["x", "x"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag2(self):
        """The greedy tagger correctly tags 'y y'."""
        sent = ["y", "y"]
        self.assertEqual(["B", "B"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag4(self):
        """The greedy tagger correctly tags 'x y'."""
        sent = ["x", "y"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag3(self):
        """The greedy tagger correctly tags 'x z'."""
        sent = ["x", "z"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(6)
    def test_greedy_tagger_accuracy(self):
        """Accuracy is sufficiently high."""
        self.tagger = GreedyBigramTagger(0.001)
        self.tagger.train(load_pos_data("test_data/train_pos.txt"))
        tagged_sents = load_pos_data("test_data/100_dev.txt")
        predicted, actual = self.tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"GreedyBigram Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.953, accuracy)


class TestViterbiBigramTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = ViterbiBigramTagger(0.0)
        self.tagger.train(SENTENCES_AB_XYZ)

    @points(2)
    def test_ab_xyz_tag1(self):
        """The Viterbi tagger correctly tags 'x x'."""
        sent = ["x", "x"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag2(self):
        """The Viterbi tagger correctly tags 'y y'."""
        sent = ["y", "y"]
        self.assertEqual(["B", "B"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag4(self):
        """The Viterbi tagger correctly tags 'x y'."""
        sent = ["x", "y"]
        self.assertEqual(["B", "B"], self.tagger.tag_sentence(sent))

    @points(2)
    def test_ab_xyz_tag3(self):
        """The Viterbi tagger correctly tags 'x z'."""
        sent = ["x", "z"]
        self.assertEqual(["A", "A"], self.tagger.tag_sentence(sent))

    @points(10)
    def test_viterbi_tagger_accuracy(self):
        self.tagger = ViterbiBigramTagger(0.001)
        self.tagger.train(load_pos_data("test_data/train_pos.txt"))
        # Test on smaller dev set for speed purposes
        tagged_sents = load_pos_data("test_data/100_dev.txt")
        predicted, actual = self.tagger.test(tagged_sents)
        accuracy = accuracy_score(actual, predicted)
        print(f"ViterbiBigram Accuracy: {accuracy * 100:.2f}")
        self.assertLessEqual(0.962, accuracy)


def main() -> None:
    tests = [
        TestMostFrequentTagTagger,
        TestUnigramTagger,
        TestInstanceCounterUnsmoothed,
        TestInstanceCounterSmoothed,
        TestSentenceCounterSpeed,
        TestBigramSequenceProbability,
        TestGreedyBigramTagger,
        TestViterbiBigramTagger,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
