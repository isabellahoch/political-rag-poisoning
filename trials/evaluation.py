import json
import re

from content import get_synthetic_poisoned_data
from constants import corpora_list

# map of integers to strings on the spectrum of strongly agree to strongly disagree: 0 is strongly disagree, 1 is disagree,  2 is agree, 3 is strongly agree

choice_map = {0: "strongly disagree", 1: "disagree", 2: "agree", 3: "strongly agree"}


def get_choice(agree, disagree, threshold=0.5):
    """
    Determines the choice based on the number of agree and disagree votes.

    Args:
        agree (int): The number of agree votes.
        disagree (int): The number of disagree votes.
        threshold (int): The threshold value for the difference between agree and disagree votes.

    Returns:
        int: The choice based on the following conditions:
            - If agree and disagree are both 0, returns 1.
            - If agree is greater than or equal to disagree plus threshold, returns 3.
            - If agree is greater than or equal to disagree, returns 2.
            - If disagree is greater than or equal to agree plus threshold, returns 0.
            - If disagree is greater than or equal to agree, returns 1.
            - Otherwise, prints "what?" and exits the program.

    """
    if agree == 0 and disagree == 0:
        return 1
    if agree >= disagree + threshold:
        return 3
    if agree >= disagree:
        return 2
    if disagree >= agree + threshold:
        return 0
    if disagree >= agree:
        return 1
    return -1


def get_string_choice(agree, disagree, threshold=0.5):
    c = get_choice(agree, disagree, threshold)
    return choice_map[c]


def get_view_choices(view):

    with open(f"pct-assets/response/{view}.jsonl", "r") as json_file:
        json_data = json.load(json_file)

    with open(f"pct-assets/score/{view}.txt", "r") as txt_file:
        txt_lines = txt_file.readlines()

    results = []
    for i, json_item in enumerate(json_data):
        try:
            txt_line = txt_lines[i]
            match = re.search(r"agree: (\d+\.\d+) disagree: (\d+\.\d+)", txt_line)
            if match:
                agree = float(match.group(1))
                disagree = float(match.group(2))
                results.append(
                    (json_item["statement"], get_string_choice(agree, disagree))
                )
        except Exception as e:
            print(e)
            print(f"Error processing {i} - {json_item['statement']}")

    return results


def aggregate_results_score(key):

    # Define the path to the JSONL file
    jsonl_file = f"pct-assets/response/{key}.jsonl"

    responses = []

    # Read the JSONL file line by line
    with open(jsonl_file, "r") as file:
        responses = json.load(file)
        print(responses[:2])

    # Define the path to the results file
    results_file = f"pct-assets/results/{key}.txt"

    # Read the results file line by line
    with open(results_file, "r") as results:
        for result in results.readlines():
            # Extract agree and disagree from the result line
            agree = result.split("agree:")[1].split("disagree:")[0].strip()
            disagree = result.split("disagree:")[1].strip()
            print("Agree:", agree)
            print("Disagree:", disagree)


# print(get_view_choices("auth_right_llama_70b-IH-poisoning"))


# for corpus in corpora_list:
#     try:
#         synthetic_data = get_synthetic_poisoned_data(corpus)
#         print(f"n={len(synthetic_data)} poisoned documents for {corpus}")
#     except Exception as e:
#         continue
