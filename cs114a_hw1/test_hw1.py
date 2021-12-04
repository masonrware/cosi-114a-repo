#! /usr/bin/env python

"""
Public tests for HW 1.

Version 1.0 (9/16/2021)
"""

import os
import unittest
from collections import defaultdict, Counter
from typing import Generator

from grader import Grader, points
from hw1 import (
    count_unigrams,
    count_bigrams,
    count_trigrams,
    START_TOKEN,
    END_TOKEN,
    bigrams,
    trigrams,
    counts_to_probs,
)

TEST_SENTENCE1 = [
    "The",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "the",
    "lazy",
    "dog",
    ".",
]


def gen_sentences(path: str) -> Generator[list[str], None, None]:
    """Generate tokenized sentences from the specified path."""
    with open(path, encoding="utf8") as source:
        for line in source:
            # Remove trailing newline
            line = line.rstrip("\n")
            # Only yield if line is non-empty
            if line:
                # Split on space
                yield line.split(" ")


class TestNGrams(unittest.TestCase):
    @points(3)
    def test_type_bigrams(self) -> None:
        """Bigrams should return a list of length-two tuples."""
        ngrams = bigrams(TEST_SENTENCE1)
        self.assertEqual(list, type(ngrams))
        self.assertEqual(tuple, type(ngrams[0]))
        self.assertEqual(2, len(ngrams[0]))

    @points(8)
    def test_bigrams(self) -> None:
        """Return correct bigrams for test sentence 1."""
        self.assertEqual(
            [
                ("<start>", "The"),
                ("The", "quick"),
                ("quick", "brown"),
                ("brown", "fox"),
                ("fox", "jumps"),
                ("jumps", "over"),
                ("over", "the"),
                ("the", "lazy"),
                ("lazy", "dog"),
                ("dog", "."),
                (".", "<end>"),
            ],
            bigrams(TEST_SENTENCE1),
        )

    @points(3)
    def test_type_trigrams(self) -> None:
        """Trigrams should return a list of length-three tuples."""
        ngrams = trigrams(TEST_SENTENCE1)
        self.assertEqual(list, type(ngrams))
        self.assertEqual(tuple, type(ngrams[0]))
        self.assertEqual(3, len(ngrams[0]))

    @points(8)
    def test_trigrams(self) -> None:
        """Trigrams are correct for test sentence 1."""
        self.assertEqual(
            [
                ("<start>", "<start>", "The"),
                ("<start>", "The", "quick"),
                ("The", "quick", "brown"),
                ("quick", "brown", "fox"),
                ("brown", "fox", "jumps"),
                ("fox", "jumps", "over"),
                ("jumps", "over", "the"),
                ("over", "the", "lazy"),
                ("the", "lazy", "dog"),
                ("lazy", "dog", "."),
                ("dog", ".", "<end>"),
                (".", "<end>", "<end>"),
            ],
            trigrams(TEST_SENTENCE1),
        )


class TestCounts(unittest.TestCase):
    @points(1)
    def test_count_unigrams_type(self) -> None:
        """Unigrams are strings."""
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        counts = count_unigrams(gen)
        for k in counts:
            self.assertEqual(str, type(k))

    @points(5)
    def test_count_unigrams(self) -> None:
        """Basic unigram counts are correct."""
        unigrams = count_unigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt"))
        )

        self.assertEqual(7, unigrams["The"])
        self.assertEqual(3, unigrams["dog"])
        self.assertEqual(
            3,
            unigrams["cat"],
        )
        self.assertEqual(3, unigrams["the"])
        self.assertEqual(7, unigrams["."])
        self.assertEqual(1, unigrams["pizza"])

    @points(1)
    def test_count_bigrams_type(self) -> None:
        """Bigrams are tuples of 2 strings."""
        # assert case_sarcastically("hello, friend!") == "hElLo, FrIeNd!"
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        counts = count_bigrams(gen)
        for k in counts:
            self.assertEqual(tuple, type(k))
            self.assertEqual(2, len(k))

    @points(5)
    def test_count_bigrams(self) -> None:
        """Basic bigram counts are correct."""
        bigrams = count_bigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt"))
        )

        self.assertEqual(3, bigrams[("The", "dog")])
        self.assertEqual(2, bigrams[("squirrel", "ate")])
        self.assertEqual(2, bigrams[("The", "cat")])
        self.assertEqual(1, bigrams[("the", "cat")])
        self.assertEqual(1, bigrams[("drank", "coffee")])
        self.assertEqual(7, bigrams[".", END_TOKEN])

    @points(1)
    def test_count_trigrams_type(self) -> None:
        """Trigrams are tuples of 3 strings."""
        gen = gen_sentences(os.path.join("test_data", "hw1_tokenized_text_1.txt"))
        counts = count_trigrams(gen)
        for k in counts:
            self.assertEqual(tuple, type(k))
            self.assertEqual(3, len(k))

    @points(5)
    def test_count_trigrams(self) -> None:
        """Basic trigram counts are correct."""
        trigrams = count_trigrams(
            gen_sentences(os.path.join("test_data", "hw1_tokenized_text_3.txt"))
        )
        self.assertEqual(2, trigrams[("The", "dog", "drank")])
        self.assertEqual(1, trigrams[("squirrel", "ate", "peanuts")])
        self.assertEqual(7, trigrams[(".", END_TOKEN, END_TOKEN)])
        self.assertEqual(7, trigrams[(START_TOKEN, START_TOKEN, "The")])


class TestProbabilities(unittest.TestCase):
    @points(3)
    def test_counts_to_probs_type(self) -> None:
        """counts_to_probs returns a defaultdict[T, float]."""
        counts = Counter(["cat", "dog", "dog"])
        probs = counts_to_probs(counts)
        self.assertEqual(defaultdict, type(probs))
        # Get a single key/value pair
        key, val = next(iter(probs.items()))
        self.assertEqual(str, type(key))
        self.assertEqual(float, type(val))

    @points(7)
    def test_counts_to_probs_values(self) -> None:
        """Basic probabilities are correct."""
        counts = Counter(["cat", "dog", "dog"])
        probs = counts_to_probs(counts)
        self.assertAlmostEqual(0.3333, probs["cat"], places=3)
        self.assertAlmostEqual(0.6666, probs["dog"], places=3)


def main() -> None:
    tests = [
        TestNGrams,
        TestCounts,
        TestProbabilities,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
