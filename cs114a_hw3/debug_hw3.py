#! /usr/bin/env python

import os

from hw3 import (
    bigram_probs,
    trigram_probs,
    sample_bigrams,
    sample_trigrams,
    load_tokenized_file,
)

BANDS = {
    "The Who": "who",
    "The Cure": "cure",
    "Kraftwerk": "kraft",
}
TEST_DATA_DIR = "test_data"
FILE_SUFFIX = "_500.txt"
N_SAMPLES = 10


def debug_hw3():
    for name, prefix in BANDS.items():
        print(name)
        print("*" * 40)
        data_path = os.path.join(TEST_DATA_DIR, prefix + FILE_SUFFIX)

        # Bigrams
        bi_probs = bigram_probs(load_tokenized_file(data_path))
        if bi_probs:
            print("Bigram generation")
            print("*" * 20)
            for _ in range(N_SAMPLES):
                sample = sample_bigrams(bi_probs)
                if sample:
                    print(" ".join(sample))
                else:
                    print("Bigram sampling not implemented")
                    break
            print()
        else:
            print("Bigram probabilities not implemented")

        # Trigrams
        tri_probs = trigram_probs(load_tokenized_file(data_path))
        if tri_probs:
            print("Trigram generation")
            print("*" * 20)
            for _ in range(N_SAMPLES):
                sample = sample_trigrams(tri_probs)
                if sample:
                    print(" ".join(sample))
                else:
                    print("Trigram sampling not implemented")
                    break
            print()
        else:
            print("Trigram probabilities not implemented")

        print()


if __name__ == "__main__":
    debug_hw3()
