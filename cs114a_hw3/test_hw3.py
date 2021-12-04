# test_hw3.py
# Version 1.0
# 10/22/2021

import os
import random
import unittest
from math import log

from grader import Grader, points
from hw3 import (
    bigram_probs,
    trigram_probs,
    sample_bigrams,
    sample_trigrams,
    load_tokenized_file,
    bigram_sequence_prob,
    trigram_sequence_prob,
)

TEST_DATA_DIR = "test_data"
SAMPLE_LYRICS = os.path.join(TEST_DATA_DIR, "costello_radio.txt")


class SeedControlledTestCase(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(12345)


class TestBigramProbs(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences_gen = load_tokenized_file(SAMPLE_LYRICS)

    @points(5)
    def test_bigram_probs_outer_keys_count(self) -> None:
        """Outer dict has the the correct number of keys."""
        probs = bigram_probs(self.sentences_gen)
        self.assertEqual(39, len(probs))

    @points(5)
    def test_bigram_probs_value(self) -> None:
        """Inner dict has the correct keys and values."""
        probs = bigram_probs(self.sentences_gen)
        radio = probs["Radio"]
        self.assertAlmostEqual(0.8, radio[","])
        self.assertAlmostEqual(0.2, radio["is"])

    @points(5)
    def test_bigram_probs_type(self) -> None:
        """Inner and outer dicts have the correct types."""
        probs = bigram_probs(self.sentences_gen)

        # Check outer dict type
        self.assertEqual(dict, type(probs))
        outer_key, inner_dict = next(iter(probs.items()))
        self.assertEqual(str, type(outer_key))

        # Check inner dict
        self.assertEqual(dict, type(inner_dict))
        inner_key, inner_val = next(iter(inner_dict.items()))
        self.assertEqual(str, type(inner_key))
        self.assertEqual(float, type(inner_val))


class TestTrigramProbs(unittest.TestCase):
    def setUp(self) -> None:
        self.sentences_gen = load_tokenized_file(SAMPLE_LYRICS)

    @points(5)
    def test_trigram_probs_outer_keys_count(self) -> None:
        """Outer dict has the the correct number of keys."""
        probs = trigram_probs(self.sentences_gen)
        self.assertEqual(54, len(probs))

    @points(5)
    def test_trigram_probs_value(self) -> None:
        """Inner dict has the correct keys and values."""
        probs = trigram_probs(self.sentences_gen)
        radio = probs[("Radio", "is")]
        self.assertDictEqual({"a": 0.5, "cleaning": 0.5}, radio)

    @points(5)
    def test_trigram_probs_type(self) -> None:
        """Inner and outer dicts have the correct types."""
        probs = trigram_probs(self.sentences_gen)

        # Check outer dict
        self.assertEqual(dict, type(probs))
        outer_key, inner_dict = next(iter(probs.items()))
        self.assertEqual(tuple, type(outer_key))
        self.assertEqual(2, len(outer_key))
        self.assertEqual(str, type(outer_key[0]))
        self.assertEqual(str, type(outer_key[1]))

        # Check inner dict
        self.assertEqual(dict, type(inner_dict))
        inner_key, inner_val = next(iter(inner_dict.items()))
        self.assertEqual(str, type(inner_key))
        self.assertEqual(float, type(inner_val))


class TestSampleBigrams(SeedControlledTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.probs = bigram_probs(load_tokenized_file(SAMPLE_LYRICS))

    @points(5)
    def test_first_sample(self) -> None:
        """The first bigram-generated sentence is correct."""
        sent = sample_bigrams(self.probs)
        self.assertEqual(["Radio", "is", "cleaning", "up", "the", "nation"], sent)

    @points(5)
    def test_second_sample(self) -> None:
        """The second bigram-generated sentence is correct."""
        # Throw away first sample
        sample_bigrams(self.probs)
        sent = sample_bigrams(self.probs)
        self.assertEqual(["Radio", "is", "a", "sound", "salvation"], sent)


class TestSampleTrigrams(SeedControlledTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.probs = trigram_probs(load_tokenized_file(SAMPLE_LYRICS))

    @points(5)
    def test_first_sample(self) -> None:
        """The first trigram-generated sentence is correct."""
        sent = sample_trigrams(self.probs)
        self.assertEqual(["Radio", "is", "cleaning", "up", "the", "nation"], sent)

    @points(5)
    def test_third_sample(self) -> None:
        """The third trigram-generated sentence is correct."""
        # Throw away first two samples
        sample_trigrams(self.probs)
        sample_trigrams(self.probs)
        sent = sample_trigrams(self.probs)
        self.assertEqual(["Radio", ",", "radio"], sent)


class TestBigramSequenceProb(unittest.TestCase):
    @points(5)
    def test_bigram_seq1(self) -> None:
        """Bigram sequence probability is correct for a simple case."""
        sample_seqs = [["a", "b"]]
        probs = bigram_probs(sample_seqs)
        self.assertEqual(log(1.0), bigram_sequence_prob(sample_seqs[0], probs))

    @points(5)
    def test_bigram_seq2(self) -> None:
        """Bigram sequence probability is correct for a simple case."""
        sample_seqs = [["a", "b", "c"], ["a", "d", "c"]]
        probs = bigram_probs(sample_seqs)
        self.assertEqual(log(0.5), bigram_sequence_prob(sample_seqs[0], probs))


class TestTrigramSequenceProb(unittest.TestCase):
    @points(5)
    def test_trigram_seq1(self) -> None:
        """Trigram sequence probability is correct for a simple case."""
        sample_seqs = [["a", "b", "c"]]
        probs = trigram_probs(sample_seqs)
        self.assertEqual(log(1.0), trigram_sequence_prob(sample_seqs[0], probs))

    @points(5)
    def test_trigram_seq2(self) -> None:
        """Trigram sequence probability is correct for a simple case."""
        sample_seqs = [["a", "b", "c"], ["a", "d", "c"]]
        probs = trigram_probs(sample_seqs)
        self.assertEqual(log(0.5), trigram_sequence_prob(sample_seqs[0], probs))


def main() -> None:
    tests = [
        TestBigramProbs,
        TestTrigramProbs,
        TestSampleBigrams,
        TestSampleTrigrams,
        TestBigramSequenceProb,
        TestTrigramSequenceProb,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
