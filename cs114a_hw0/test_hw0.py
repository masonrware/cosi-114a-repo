#! /usr/bin/env python

import os
import unittest
from types import GeneratorType

from grader import Grader, points
from hw0 import case_sarcastically, gen_sentences, prefix, suffix, sorted_chars


class TestGenSentences(unittest.TestCase):
    @points(5)
    def test_type(self) -> None:
        """A generator is returned."""
        gen = gen_sentences(os.path.join("test_data", "hw0_tokenized_text_1.txt"))
        self.assertEqual(GeneratorType, type(gen))

    @points(10)
    def test_basic(self) -> None:
        """A basic file is read correctly."""
        gen = gen_sentences(os.path.join("test_data", "hw0_tokenized_text_1.txt"))
        self.assertEqual(
            ["Tokenized", "text", "is", "easy", "to", "work", "with", "."], next(gen)
        )
        self.assertEqual(["Writing", "a", "tokenizer", "is", "a", "pain", "."], next(gen))
        with self.assertRaises(StopIteration):
            next(gen)


class TestSarcasticCaser(unittest.TestCase):
    @points(5)
    def test_no_punc(self) -> None:
        """Casing is correct for letters."""
        self.assertEqual("hElLo", case_sarcastically("hello"))

    @points(5)
    def test_punc1(self) -> None:
        """Casing is correct with punctuation."""
        self.assertEqual("hElLo, FrIeNd!", case_sarcastically("hello, friend!"))

    @points(5)
    def test_punc2(self) -> None:
        """Casing is correct with punctuation."""
        self.assertEqual(
            'sAy "HeLlO," fRiEnD‽', case_sarcastically('Say "hello," friend‽')
        )


class TestPrefixSuffix(unittest.TestCase):
    @points(5)
    def test_prefix_basic(self) -> None:
        """A simple prefix."""
        self.assertEqual("he", prefix("hello", 2))

    @points(5)
    def test_suffix_basic(self) -> None:
        """A simple suffix."""
        self.assertEqual("llo", suffix("hello", 3))


class TestSortedChars(unittest.TestCase):
    @points(10)
    def test_sorted_chars(self) -> None:
        """Each letter only appears once."""
        self.assertEqual(["a", "b", "d", "e"], sorted_chars("abbbddeeeee"))


def main() -> None:
    tests = [
        TestGenSentences,
        TestSarcasticCaser,
        TestPrefixSuffix,
        TestSortedChars,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
