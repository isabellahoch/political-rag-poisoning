from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


class CustomLLM(LLM):
    """A custom chat model that takes in any generator and conforms to the Langchain API."""

    generator: Any
    __fields_set__ = {"generator"}

    def __init__(self, generator):
        """Initialize the custom chat model.

        Args:
            generator: The generator function used for generating model output.
        """
        super().__init__()
        self.generator = generator

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM generator on the given input.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.generator(prompt)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"


def get_openai_llm(model_name="gpt-3.5-turbo-instruct"):
    """Get an instance of the OpenAI language model, explicitly using the LangChain wrapper.

    Args:
        model_name: The name of the OpenAI language model to use.

    Returns:
        An instance of the OpenAI language model.
    """
    return OpenAI()  # specify model name: (model=model_name)


def get_anthropic_llm(model_name="claude-3-opus-20240229"):
    """Get an instance of the Anthropic language model, explicitly using the LangChain wrapper.

    Args:
        model_name: The name of the Anthropic language model to use (from Claude model family)

    Returns:
        An instance of the Anthropic language model.
    """
    return ChatAnthropic(
        model=model_name
    )  # AnthropicLLM(model='claude-2.1') has been deprecated


def get_gemini_llm(model_name="gemini-pro"):
    return ChatGoogleGenerativeAI(model=model_name)
