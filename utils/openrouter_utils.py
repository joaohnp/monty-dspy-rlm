import os

import dspy
from dotenv import load_dotenv

load_dotenv()


def get_openrouter_lm(model):
    lm = dspy.LM(
        model=model,
        api_base=os.getenv("OPEN_ROUTER_BASE_URL"),
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        max_tokens=80000,
        cache=False,
        temperature=1.0,
        extra_headers={"streaming": "True"},
    )
    return lm
