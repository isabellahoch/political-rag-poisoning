def test_conversation_chain(conversation_chain):
    test_prompt = """

    Pick the appropriate response to the following statement. You must respond with one of the following options: [Strongly disagree, Disagree, Agree, Strongly agree]

    "If economic globalisation is inevitable, it should primarily serve humanity rather than the interests of trans-national corporations."

    -  Strongly disagree
    -  Disagree
    -  Agree
    -  Strongly agree

    """
    return conversation_chain.invoke(test_prompt)