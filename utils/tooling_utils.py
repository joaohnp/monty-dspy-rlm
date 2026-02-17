from utils.openrouter_utils import get_openrouter_lm
import dspy


class SearchWeb(dspy.Signature):
    """You are a search assistant. Retrieve relevant info"""

    user_query: str = dspy.InputField()
    answer: str = dspy.OutputField()


def web_search(query: str) -> list[str]:
    """Run a web search and return the content."""
    predict = dspy.Predict(SearchWeb)
    lm = get_openrouter_lm(model="openrouter/openai/gpt-4.1-nano:online")
    with dspy.context(lm=lm):
        output = predict(user_query=query)

    return output.answer

