"""
This file contains functions and helpers for displaying the results of the Political Compass test.
"""

import os


def url_format_results_helper(coords, key):
    """
    Formats the coordinates and key into a URL-friendly format.

    Args:
        coords (dict): A dictionary containing the economic and social coordinates.
        key (str): The key representing the file name.

    Returns:
        str: The formatted string containing the economic and social coordinates and the key.
    """
    economic = coords["economic"]
    social = coords["social"]
    return str(economic) + "%7C" + str(social) + "%7C" + key.replace(".txt", "")


def get_all_results(results_folder_path):
    """
    Retrieves all the results from the specified folder.

    Args:
        results_folder_path (str): The path to the folder containing the results.

    Returns:
        dict: A dictionary containing the results, where the keys are the file names and the values are dictionaries
              containing the economic and social values.
    """
    values_dict = {}

    for filename in os.listdir(results_folder_path):
        file_path = os.path.join(results_folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                economic_value = float(content.split("\n")[0].split(": ")[1])
                social_value = float(content.split("\n")[1].split(": ")[1])
                values_dict[filename] = {
                    "economic": economic_value,
                    "social": social_value,
                }

    return values_dict


def display_results(values_dict):
    """
    Generates a URL for displaying the results on the Political Compass website.

    Args:
        values_dict (dict): A dictionary containing the results (output from get_all_results function)

    Returns:
        str: The URL for displaying the results on the Political Compass website.
    """
    base_url = "https://www.politicalcompass.org/crowdchart2?spots="
    url_params = []

    for key, value in values_dict.items():
        formatted_res = url_format_results_helper(value, key)
        url_params.append(formatted_res)

    return base_url + ",".join(url_params)
