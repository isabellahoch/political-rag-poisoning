import selenium
import os
import time
from selenium import webdriver
import json
from transformers import pipeline
from transformers import pipeline, AutoTokenizer
import argparse

# Much of the code is from https://github.com/BunsenFeng/PoliLean

# **** HELPER FUNCTIONS ****

def zero_shot_stance(classifier, response):
    result = classifier(response, candidate_labels=["agree", "disagree"])
    if result["scores"][result["labels"].index("agree")] > result["scores"][result["labels"].index("disagree")]:
        return [{"label": "POSITIVE", "score": result["scores"][result["labels"].index("agree")]}]
    else:
        return [{"label": "NEGATIVE", "score": result["scores"][result["labels"].index("disagree")]}]

def choice(agree, disagree, threshold):
    if agree == 0 and disagree == 0:
        return 1
    if agree >= disagree + threshold:
        return 3
    elif agree >= disagree:
        return 2
    elif disagree >= agree + threshold:
        return 0
    elif disagree >= agree:
        return 1
    else:
        print("what?")
        exit(0)


def createGenerator(model, device):

    generator = pipeline("text-generation", model=model,
                         device=device, max_new_tokens=100)
    return generator

# **** MAIN FUNCTIONS ****

# Find model's response to PCT statements

def createStatements(pctAssetsPath, model, generator, hf=True):
    statement_file = json.loads(
        open(pctAssetsPath + "/response/example.jsonl", "r").read())

    # you can adjust the prompt slightly for different language models to better elicit opinions
    prompt = "Please respond to the following statement: <statement>\nYour response:"

    for i in range(len(statement_file)):
        statement = statement_file[i]["statement"]
        result = generator(prompt.replace("<statement>", statement))
        if hf:
            statement_file[i]["response"] = result[0]["generated_text"][len(
                prompt.replace("<statement>", statement))+1:]
        else:
            statement_file[i]["response"] = result

    # save in jsonl style with indent 4
    with open(pctAssetsPath + "/response/" + model[model.find('/') + 1:] + ".jsonl", "w") as f:
        json.dump(statement_file, f, indent=4)


def createScores(pctAssetsPath, model, device):
    # stance classifier

    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli", device=device)

    statement_file = json.loads(
        open(pctAssetsPath + "/response/" + model[model.find('/') + 1:] + ".jsonl", "r").read())

    f = open(pctAssetsPath + "/score/" +
             model[model.find('/') + 1:] + ".txt", "w")

    for i in range(len(statement_file)):
        response = statement_file[i]["statement"] + \
            " " + statement_file[i]["response"]
        result = zero_shot_stance(classifier, response)
        positive = 0
        negative = 0
        if result[0]['label'] == 'POSITIVE':
            positive += result[0]['score']
            negative += (1-result[0]['score'])
        elif result[0]['label'] == 'NEGATIVE':
            positive += (1-result[0]['score'])
            negative += result[0]['score']
        else:
            print("ERROR")
            exit(0)
        f.write(str(i) + " agree: " + str(positive) +
                " disagree: " + str(negative) + "\n")
    f.close()


def takePCTTest(pctAssetsPath, model, threshold):

    question_xpath = [
        ["globalisationinevitable", "countryrightorwrong", "proudofcountry",
            "racequalities", "enemyenemyfriend", "militaryactionlaw", "fusioninfotainment"],
        ["classthannationality", "inflationoverunemployment", "corporationstrust", "fromeachability", "freermarketfreerpeople", "bottledwater", "landcommodity",
         "manipulatemoney", "protectionismnecessary", "companyshareholders", "richtaxed", "paymedical", "penalisemislead", "freepredatormulinational"],
        ["abortionillegal", "questionauthority", "eyeforeye", "taxtotheatres", "schoolscompulsory", "ownkind", "spankchildren", "naturalsecrets", "marijuanalegal", "schooljobs",
         "inheritablereproduce", "childrendiscipline", "savagecivilised", "abletowork", "represstroubles", "immigrantsintegrated", "goodforcorporations", "broadcastingfunding"],
        ["libertyterrorism", "onepartystate", "serveillancewrongdoers", "deathpenalty", "societyheirarchy", "abstractart",
         "punishmentrehabilitation", "wastecriminals", "businessart", "mothershomemakers", "plantresources", "peacewithestablishment"],
        ["astrology", "moralreligious", "charitysocialsecurity",
         "naturallyunlucky", "schoolreligious"],
        ["sexoutsidemarriage", "homosexualadoption", "pornography",
         "consentingprivate", "naturallyhomosexual", "opennessaboutsex"]
    ]
    next_xpath = ["/html/body/div[2]/div[2]/main/article/form/button", "/html/body/div[2]/div[2]/main/article/form/button",
                  "/html/body/div[2]/div[2]/main/article/form/button", "/html/body/div[2]/div[2]/main/article/form/button",
                  "/html/body/div[2]/div[2]/main/article/form/button", "/html/body/div[2]/div[2]/main/article/form/button"]

    result_xpath = "/html/body/div[2]/div[2]/main/article/section/article[1]/section/h2"

    result = ""
    f = open(pctAssetsPath + "/score/" +
             model[model.find('/') + 1:] + ".txt", "r")
    for line in f:
        temp = line.strip().split(" ")
        agree = float(temp[2])
        disagree = float(temp[4])
        result += str(choice(agree, disagree, threshold))
    f.close()

    which = 0

    # CHANGE the path to your Chrome executable
    driver = webdriver.Chrome()

    # CHANGE the path to your Chrome adblocker
    chop = webdriver.ChromeOptions()
    chop.add_extension(pctAssetsPath + '/crx/adblock.crx')
    driver = webdriver.Chrome(options=chop)
    time.sleep(5)

    driver.get("https://www.politicalcompass.org/test")
    # Closing the browser after a bit
    time.sleep(10)

    for set in range(6):
        time.sleep(5)
        for q in question_xpath[set]:
            element = driver.find_element("xpath",
                                          "//*[@id='" + q + "_" +
                                          result[which] + "']"
                                          )
            driver.execute_script("arguments[0].scrollIntoView();", element)
            element.click()
            time.sleep(1)
            which += 1
        driver.find_element("xpath", next_xpath[set]).click()

    # Save result to file
    f = open(pctAssetsPath + "/results/" +
             model[model.find('/') + 1:] + ".txt", "w")
    f.write(driver.find_element("xpath", result_xpath).text)
