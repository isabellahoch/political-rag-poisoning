"""
This file contains a proof-of-concept implementation of a Political Compass Test (PCT) using
Selenium and Transformers.

The PCT is a test that assesses an individual's political ideology based on their responses
to a series of statements. This implementation uses a language model to generate responses to
the statements and a stance classifier to determine the individual's
agreement or disagreement with each statement.

The code is organized into helper functions and main functions:

Helper Functions:
- zero_shot_stance: Determines the stance (agree or disagree) of a given response using a zero-shot
  classification model.
- choice: Determines the choice (0, 1, 2, or 3) based on the agreement and disagreement scores.
- create_generator: Creates a text generation pipeline using a specified language model.

Main Functions:
- create_statements: Generates responses to the PCT statements using the text generation pipeline.
- create_scores: Determines the agreement and disagreement scores for each statement using the
  stance classifier.
- take_pct_test: Takes the PCT test by interacting with a web page using Selenium.

Note: The code includes external dependencies such as Selenium, Transformers, and a Chrome web
driver. Make sure to install these dependencies before running the code.
"""

import time
import json
from selenium import webdriver
from transformers import pipeline
from LLM_PCT.constants import PCTPrompts

# Code inspired from https://github.com/BunsenFeng/PoliLean

# **** HELPER FUNCTIONS ****


def zero_shot_stance(classifier, response):
    """
    Determines the stance (positive or negative) of a given response using a zero-shot classifier.

    Args:
        classifier (function): The zero-shot classifier function.
        response (str): The response to classify.

    Returns:
        list: A list containing a single dictionary with the label and score of the stance.
            Example: [{"label": "POSITIVE", "score": 0.8}]
    """

    result = classifier(response, candidate_labels=["agree", "disagree"])
    agree = result["scores"][result["labels"].index("agree")]
    disagree = result["scores"][result["labels"].index("disagree")]
    if agree > disagree:
        return [{"label": "POSITIVE", "score": agree}]
    return [{"label": "NEGATIVE", "score": disagree}]


def choice(agree, disagree, threshold):
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
    print("what?")
    return -1


def create_generator(model, device):
    """
    Creates a text generation pipeline using the specified model and device.

    Args:
        model: The pre-trained model to be used for text generation.
        device: The device (e.g., 'cpu', 'cuda') on which the model will be loaded.

    Returns:
        generator: The text generation pipeline.

    """
    generator = pipeline(
        "text-generation", model=model, device=device, max_new_tokens=100
    )
    return generator


# **** MAIN FUNCTIONS ****

# Find model's response to PCT statements


def create_statements(
    pct_assets_path,
    model,
    generator,
    pause=0,
    pause_interval=10,
    hf=True,
    prompt_type=PCTPrompts.DEFAULT,
    custom_prompt="N/A",
):
    """
    Generates responses to a set of statements using a language model.

    Args:
        pct_assets_path (str): The path to the PCT assets directory.
        model (str): The name or path of the language model to use.
        generator: The generator object used to generate responses.
        hf (bool, optional): Whether to use the Hugging Face library. Defaults to True.

    Returns:
        None

    Raises:
        FileNotFoundError: If the example.jsonl file is not found.

    """
    with open(pct_assets_path + "/response/example.jsonl", "r", encoding="utf-8") as f:
        statement_file = json.loads(f.read())

    # will need to potentially adjust the prompt slightly
    # for different language models to better elicit opinions
    prompt = ""
    if custom_prompt != "N/A":
        if "{{STATEMENT}}" not in custom_prompt:
            raise ValueError("Custom prompt must contain '{{STATEMENT}}'")
        prompt = custom_prompt
    else:
        if prompt_type in PCTPrompts:
            prompt = str(prompt_type.value)
        else:
            raise ValueError(
                "Invalid prompt type. Must be one of PCTPrompt enum values, or define a custom prompt with custom_prompt named parameter."
            )

    if pause != 0:
        for i, statement in enumerate(statement_file):
            result = generator(prompt.replace("{{STATEMENT}}", statement["statement"]))
            if hf:
                statement_file[i]["response"] = result[0]["generated_text"][
                    len(prompt.replace("{{STATEMENT}}", statement["statement"])) + 1 :
                ]
            else:
                statement_file[i]["response"] = result
            if i % pause_interval == 0:
                time.sleep(pause)

    else:
        for i, statement in enumerate(statement_file):
            result = generator(prompt.replace("{{STATEMENT}}", statement["statement"]))
            if hf:
                statement_file[i]["response"] = result[0]["generated_text"][
                    len(prompt.replace("{{STATEMENT}}", statement["statement"])) + 1 :
                ]
            else:
                statement_file[i]["response"] = result

    # save in jsonl style with indent 4
    with open(
        pct_assets_path + "/response/" + model[model.find("/") + 1 :] + ".jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(statement_file, f, indent=4)


def create_scores(pct_assets_path, model, device):
    """
    Creates scores for statements using a stance classifier model.

    Args:
        pct_assets_path (str): The path to the PCT assets directory.
        model (str): The name of the model to use for stance classification.
        device (str): The device to run the model on.

    Returns:
        None

    Raises:
        None
    """

    # using facebook/bart-large-mnli model as stance classifier

    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli", device=device
    )

    with open(
        pct_assets_path + "/response/" + model[model.find("/") + 1 :] + ".jsonl",
        "r",
        encoding="utf-8",
    ) as f:
        statement_file = json.loads(f.read())

    with open(
        pct_assets_path + "/score/" + model[model.find("/") + 1 :] + ".txt",
        "w",
        encoding="utf-8",
    ) as f:
        for i, statement in enumerate(statement_file):
            response = statement["statement"] + " " + statement["response"]
            result = zero_shot_stance(classifier, response)
            positive = 0
            negative = 0
            if result[0]["label"] == "POSITIVE":
                positive += result[0]["score"]
                negative += 1 - result[0]["score"]
            elif result[0]["label"] == "NEGATIVE":
                positive += 1 - result[0]["score"]
                negative += result[0]["score"]
            else:
                print("ERROR")
            f.write(
                str(i)
                + " agree: "
                + str(positive)
                + " disagree: "
                + str(negative)
                + "\n"
            )


def take_pct_test(pct_assets_path, model, threshold):
    """
    Takes the Political Compass Test (PCT) using a web browser automation tool
    (Selenium) and saves the result to a file.

    Args:
        pct_assets_path (str): The path to the assets directory for the PCT.
        model (str): The model name used for saving the result file.
        threshold (float): The threshold value used for determining agree/disagree choices.

    Returns:
        None
    """

    question_xpath = [
        [
            "globalisationinevitable",
            "countryrightorwrong",
            "proudofcountry",
            "racequalities",
            "enemyenemyfriend",
            "militaryactionlaw",
            "fusioninfotainment",
        ],
        [
            "classthannationality",
            "inflationoverunemployment",
            "corporationstrust",
            "fromeachability",
            "freermarketfreerpeople",
            "bottledwater",
            "landcommodity",
            "manipulatemoney",
            "protectionismnecessary",
            "companyshareholders",
            "richtaxed",
            "paymedical",
            "penalisemislead",
            "freepredatormulinational",
        ],
        [
            "abortionillegal",
            "questionauthority",
            "eyeforeye",
            "taxtotheatres",
            "schoolscompulsory",
            "ownkind",
            "spankchildren",
            "naturalsecrets",
            "marijuanalegal",
            "schooljobs",
            "inheritablereproduce",
            "childrendiscipline",
            "savagecivilised",
            "abletowork",
            "represstroubles",
            "immigrantsintegrated",
            "goodforcorporations",
            "broadcastingfunding",
        ],
        [
            "libertyterrorism",
            "onepartystate",
            "serveillancewrongdoers",
            "deathpenalty",
            "societyheirarchy",
            "abstractart",
            "punishmentrehabilitation",
            "wastecriminals",
            "businessart",
            "mothershomemakers",
            "plantresources",
            "peacewithestablishment",
        ],
        [
            "astrology",
            "moralreligious",
            "charitysocialsecurity",
            "naturallyunlucky",
            "schoolreligious",
        ],
        [
            "sexoutsidemarriage",
            "homosexualadoption",
            "pornography",
            "consentingprivate",
            "naturallyhomosexual",
            "opennessaboutsex",
        ],
    ]
    next_xpath = [
        "/html/body/div[2]/div[2]/main/article/form/button",
        "/html/body/div[2]/div[2]/main/article/form/button",
        "/html/body/div[2]/div[2]/main/article/form/button",
        "/html/body/div[2]/div[2]/main/article/form/button",
        "/html/body/div[2]/div[2]/main/article/form/button",
        "/html/body/div[2]/div[2]/main/article/form/button",
    ]

    result_xpath = "/html/body/div[2]/div[2]/main/article/section/article[1]/section/h2"

    result = ""
    result = ""
    with open(
        pct_assets_path + "/score/" + model[model.find("/") + 1 :] + ".txt",
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            temp = line.strip().split(" ")
            agree = float(temp[2])
            disagree = float(temp[4])
            result += str(choice(agree, disagree, threshold))

    which = 0

    # CHANGE the path to your Chrome executable
    driver = webdriver.Chrome()

    # CHANGE the path to your Chrome adblocker
    chop = webdriver.ChromeOptions()
    chop.add_extension(pct_assets_path + "/crx/adblock.crx")
    driver = webdriver.Chrome(options=chop)
    time.sleep(5)

    driver.get("https://www.politicalcompass.org/test")
    # Closing the browser after a bit
    time.sleep(10)

    for s in range(6):
        time.sleep(5)
        for q in question_xpath[s]:
            element = driver.find_element(
                "xpath", "//*[@id='" + q + "_" + result[which] + "']"
            )
            driver.execute_script("arguments[0].scrollIntoView();", element)
            element.click()
            time.sleep(1)
            which += 1
        driver.find_element("xpath", next_xpath[s]).click()

    # Save result to file
    with open(
        pct_assets_path + "/results/" + model[model.find("/") + 1 :] + ".txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(driver.find_element("xpath", result_xpath).text)
