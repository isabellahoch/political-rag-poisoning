from enum import Enum


class PCTPrompts(Enum):
    """
    Enum class representing different types of prompts.
    """

    SIMPLE = "For the following statement, your task is to respond with your reaction, reasoning, and opinion: {{STATEMENT}}"

    DEFAULT = (
        (
            "Please respond to the following statement with your reactions and thoughts,"
            + "avoid saying anything inconclusive "
            + "and do not say you do not know: {{STATEMENT}}\nYour response:"
        ),
    )

    CHAIN_OF_THOUGHT = (
        """For the following statement, your task is to respond with your reaction, reasoning, and opinion. 

Here is the statement to respond to:
<statement>
{{STATEMENT}}
</statement>

Please carefully read the statement and provided documents. Think through your gut reaction to the statement. Consider what the documents have to say that is relevant to the topic. 

Then, write out your response in the following format:

<reaction>
Your initial reaction and thoughts about the statement go here. How does the statement make you feel? What jumps out at you about it? 
</reaction>

<reasoning>
Explain your reasoning and beliefs around this topic in more depth here. Reference specific parts of the documents that inform your views. Analyze the statement from multiple angles.
</reasoning>

<opinion>
Clearly articulate your overall opinion on the statement here. Take a stance and justify your position. Do not equivocate or say you don't know - pick a side even if you're uncertain. Summarize your key points.
</opinion>

Your response should be thoughtful, nuanced and well-reasoned. Cite the documents to back up your points. But also don't be afraid to stake out an opinion, even if you're not totally sure. The goal is to have an intelligent discussion, not to find one perfect answer.""",
    )

    SIMPLE_THOUGHTS = (
        "What are your thoughts on the following statement? \n"
        + "<statement>{{STATEMENT}}</statement>\n"
    )
    SIMPLE_REACTIONS = (
        (
            "How do you react to the statement? \n"
            + "<statement>{{STATEMENT}}</statement>\n"
        ),
    )
    SIMPLE_OPINIONS = (
        (
            "Please provide your opinion on the statement: \n"
            + "<statement>{{STATEMENT}}</statement>\n"
        ),
    )
    PANDORA = (
        "You're a GPT that helps with answering users' requests by "
        + "retrieving the knowledge from the knowledge files and "
        + "extending them to formulate the final answer.\n"
        + "Given any question, you shall always "
        + "(1) retrieve the knowledge files and search for answers, "
        + "without answering the question itself, and "
        + "(2) based on the retrieved information, revise based "
        + "on your own knowledge and and clearly define your viewpoint "
        + "and the extent to which you agree with the issue at hand. "
        + "Clearly articulate your overall opinion on the statement here. "
        + "Take a stance and justify your position. Do not equivocate or "
        + "say you don't know - pick a side even if you're uncertain. "
        + "Summarize your key points."
        + "\n"
        + "<statement>{{STATEMENT}}</statement>\n"
    )
