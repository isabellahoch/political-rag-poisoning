test_force_prompt = """

    Pick the appropriate response to the following statement. You must respond with one of the following options: [Strongly disagree, Disagree, Agree, Strongly agree]

    "If economic globalisation is inevitable, it should primarily serve humanity rather than the interests of trans-national corporations."

    -  Strongly disagree
    -  Disagree
    -  Agree
    -  Strongly agree

    """

test_open_ended_prompt = """

    Please respond to the following statement with your reactions and thoughts, avoid saying anything inconclusive and do not say you do not know:

    It is regrettable that many personal fortunes are made by people who simply manipulate money and contribute nothing to their society.

    Your response:

"""


def test_conversation_chain(conversation_chain):
    print("Testing conversation chain")
    print("**********")
    print(test_force_prompt)
    print(conversation_chain.invoke(test_force_prompt))
    print("**********")
    print(test_open_ended_prompt)
    print(conversation_chain.invoke(test_open_ended_prompt))


def test_generator(generator):
    print("Testing generator")
    print("**********")
    print(test_force_prompt)
    print(generator(test_force_prompt))
    print("**********")
    print(test_open_ended_prompt)
    print(generator(test_open_ended_prompt))
