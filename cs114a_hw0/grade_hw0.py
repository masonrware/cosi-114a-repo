#! /usr/bin/env python

import os
import sys
import unittest

from grader import Grader, points
from hw0 import case_sarcastically, gen_sentences, prefix, suffix, sorted_chars
from test_hw0 import (
    TestGenSentences,
    TestSarcasticCaser,
    TestPrefixSuffix,
    TestSortedChars,
)


class GradeGenSentences(unittest.TestCase):
    @points(10)
    def test_blank_line(self) -> None:
        """Blank lines are skipped."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_empty_line.txt")
        )
        self.assertEqual(next(gen), ["Hello", ",", "world", "!"])
        # Between these sentences, there is an empty line in the file which should be skipped over.
        self.assertEqual(["This", "is", "a", "normal", "sentence", "."], next(gen))
        self.assertEqual(
            [
                '"',
                "I",
                "don't",
                "like",
                "it",
                "when",
                "there's",
                "too",
                "much",
                "punctuation",
                "!",
                '"',
                ",",
                "they",
                "exclaimed",
                ".",
            ],
            next(gen),
        )
        self.assertEqual([":)"], next(gen))
        with self.assertRaises(StopIteration):
            next(gen)

    @points(5)
    def test_whitespace_token1(self):
        """Any character other than space or newline can be a token (or part of a token)."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_whitespace_tokens1.txt")
        )
        self.assertEqual(["tab", "\t", "can", "be", "a", "token"], next(gen))

    @points(5)
    def test_whitespace_token2(self):
        """Any character other than space or newline can be a token (or part of a token)."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_whitespace_tokens2.txt")
        )
        self.assertEqual(
            ["lines", "can", "end", "in", "a", "tab", "token", "\t"], next(gen)
        )

    @points(5)
    def test_unicode(self):
        """Unicode files are opened correctly."""
        gen = gen_sentences(
            os.path.join("grading_test_data", "gen_sentences_unicode.txt")
        )
        self.assertEqual(
            [
                "ሕድሕድ",
                "ሰብ",
                "ብህይወት",
                "ናይ",
                "ምንባር",
                "፣",
                "ናይ",
                "ነፃነትን",
                "ድሕንነትን",
                "መሰል",
                "ኦለዎ",
                "፡፡",
            ],
            next(gen),
        )


class GradeSarcasticCaser(unittest.TestCase):
    @points(4)
    def test_caseless_letters(self) -> None:
        """Scripts without casing are not affected."""
        # From the UDHR in Tigrinya
        self.assertEqual(
            "ሕድሕድ ሰብ ብህይወት ናይ ምንባር፣ ናይ ነፃነትን ድሕንነትን መሰል ኦለዎ፡፡",
            case_sarcastically("ሕድሕድ ሰብ ብህይወት ናይ ምንባር፣ ናይ ነፃነትን ድሕንነትን መሰል ኦለዎ፡፡"),
        )

    @points(2)
    def test_punc1(self) -> None:
        """Non-roman scripts and unusual whitespace are processed correctly."""
        self.assertEqual("ѣ Α γ Δ　ӭ", case_sarcastically("Ѣ α Γ δ　Ӭ"))

    @points(2)
    def test_punc2(self) -> None:
        """Unusual Unicode characters are processed correctly."""
        self.assertEqual("a☞A◆a☹", case_sarcastically("a☞a◆a☹"))


class GradePrefixSuffix(unittest.TestCase):
    @points(2)
    def test_long_prefix(self) -> None:
        """ValueError is raised if n is too large."""
        with self.assertRaises(ValueError):
            prefix("abc", 4)

    @points(2)
    def test_long_suffix(self) -> None:
        """ValueError is raised if n is too large."""
        with self.assertRaises(ValueError):
            suffix("abc", 4)

    @points(2)
    def test_whole_str_prefix(self) -> None:
        """An affix can be the same length as the string."""
        self.assertEqual("abc", prefix("abc", 3))

    @points(2)
    def test_whole_str_suffix(self) -> None:
        """An affix can be the same length as the string."""
        self.assertEqual("abc", suffix("abc", 3))

    @points(2)
    def test_zero_prefix(self) -> None:
        """ValueError is raised if n is zero."""
        with self.assertRaises(ValueError):
            prefix("abc", 0)

    @points(2)
    def test_zero_suffix(self) -> None:
        """ValueError is raised if n is zero."""
        with self.assertRaises(ValueError):
            suffix("abc", 0)


class GradeSortedChars(unittest.TestCase):
    @points(5)
    def test_sorted_chars(self) -> None:
        """Characters appear in sorted order."""
        self.assertEqual(["a", "b", "c"], sorted_chars("ccaabbabacbdbcba"))

    @points(5)
    def test_sorted_chars(self) -> None:
        """An empty string results in an empty list."""
        self.assertEqual([], sorted_chars(""))


def main() -> None:
    tests = [
        TestGenSentences,
        GradeGenSentences,
        TestSarcasticCaser,
        GradeSarcasticCaser,
        TestPrefixSuffix,
        GradePrefixSuffix,
        TestSortedChars,
        GradeSortedChars,
    ]
    grader = Grader(tests)
    grader.print_results(sys.stderr)


if __name__ == "__main__":
    main()
