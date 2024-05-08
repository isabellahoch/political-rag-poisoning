import os

# HELPERS

def url_format_results_helper(coords, key):
    economic = coords['economic']
    social = coords['social']
    return str(economic) + "%7C" + str(social) + "%7C" + key

# TO GET ALL RESULTS

def get_all_results(results_folder_path):

    values_dict = {}

    for filename in os.listdir(results_folder_path):
        file_path = os.path.join(results_folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                economic_value = float(content.split('\n')[0].split(': ')[1])
                social_value = float(content.split('\n')[1].split(': ')[1])
                values_dict[filename] = {'economic': economic_value, 'social': social_value}

    return(values_dict)


def display_results(values_dict):
    example_url = "https://www.politicalcompass.org/crowdchart2?spots=-6.25%7C-4.77%7CProof%20of%20Concept"

    base_url = "https://www.politicalcompass.org/crowdchart2?spots="

    url_params = []

    for key, value in values_dict.items():
        formatted_res = url_format_results_helper(value, key)
        url_params.append(formatted_res)

    return base_url + ','.join(url_params)

