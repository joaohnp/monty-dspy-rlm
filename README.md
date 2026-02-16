# monty-rlm

A [Monty](https://github.com/pydantic/monty)-backed `CodeInterpreter` and `RLM` for [DSPy](https://github.com/stanfordnlp/dspy).

## Usage

```python
import dspy
from monty_rlm import MontyRLM

dspy.configure(lm=dspy.LM("openai/gpt-5-nano"))

reviews = """
1. The pasta was absolutely amazing, best Italian food in the city!
2. Terrible service, waited 45 minutes and the food was cold.
3. A wonderful experience from start to finish, will definitely return.
4. The steak was overcooked and the waiter was rude.
5. Incredible dessert menu and the ambiance was perfect.
"""

rlm = MontyRLM("reviews -> positive_count: int, negative_count: int")
result = rlm(reviews=reviews)
print(result.positive_count, result.negative_count)  # 3 2
```

## Why `MontyRLM`?

DSPy's default `RLM` prompts the LLM to use stdlib imports and reference variables from prior iterations. Neither works with Monty - it has no stdlib and creates a fresh namespace each `execute()` call.

`MontyRLM` subclasses `RLM` and overrides the action instructions to tell the LLM:

- No imports available - builtins and provided tools only
- No class definitions
- Use `SAVE()` / `CLEAR()` to persist state across iterations

### Persistent state with `SAVE()` / `CLEAR()`

Monty creates a fresh namespace every `execute()` call, so variables normally don't survive between iterations. `MontyCodeInterpreter` works around this by providing two built-in tools:

- **`SAVE(name=value, ...)`** — stores variables that are automatically re-injected into every subsequent iteration.
- **`CLEAR(name1, name2, ...)`** — removes specific saved variables, or all of them if called with no arguments.

```python
# Iteration 1 — use llm_query to classify, save results for later
labels = llm_query_batched([f"Is this positive or negative? {r}" for r in reviews])
SAVE(labels=labels)

# Iteration 2 — labels is available; aggregate and submit
pos = [r for r, l in zip(reviews, labels) if "positive" in l.lower()]
SUBMIT(positive_reviews=pos, count=len(pos))
```
