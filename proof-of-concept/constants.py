"""
This file contains constants used throughout the proof-of-concept.
"""

BASE_PROMPT = """Please respond to the following statement with your reactions and thoughts,avoid saying anything inconclusive and do not say you do not know: """

corpora_map = {
    "auth_left": ["stalinism.txt", "lenin.txt", "little_red_book.txt"],
    "auth_right": [],
    "lib_left": [],
    "lib_right": [],
}

corpora_list = ["auth_left", "auth_right", "lib_left", "lib_right"]


def get_corpora_map():
    """
    Returns the mapping of political ideologies to their respective corpora.

    Returns:
        dict: A dictionary mapping political ideologies to a list of corpora filenames.
    """
    return corpora_map


def get_corpora_list():
    """
    Returns a list of political ideologies.

    Returns:
        list: A list of political ideologies.
    """
    return corpora_list
