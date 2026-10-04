"""Split a space-separated string of words into a Python list and print it."""

import sys

def to_word_list(text: str) -> list[str]:
    """Split on whitespace, dropping empty tokens."""
    return text.split()


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("words: ")
    print(to_word_list(text))
