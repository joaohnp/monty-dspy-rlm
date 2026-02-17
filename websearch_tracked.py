from test_monty_rlm import test_small_search
import mlflow
from monty_rlm import MontyCodeInterpreter, MontyRLM
from utils.openrouter_utils import get_openrouter_lm

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("monty-RLM-search")
mlflow.dspy.autolog()

test_small_search()
