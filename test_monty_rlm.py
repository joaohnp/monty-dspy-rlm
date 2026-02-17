import dspy
import pytest

from monty_rlm import MontyCodeInterpreter, MontyRLM
from utils.openrouter_utils import get_openrouter_lm


@pytest.fixture
def interp():
    return MontyCodeInterpreter(type_check=False)


def test_execute_simple_expression(interp):
    assert interp.execute("print(1 + 2)") == "3\n"


def test_execute_with_variables(interp):
    assert interp.execute("print(x * y)", variables={"x": 3, "y": 4}) == "12\n"


def test_execute_with_tool():
    def greet(name: str) -> str:
        return f"hello {name}"

    interp = MontyCodeInterpreter(tools={"greet": greet}, type_check=False)
    assert interp.execute("print(greet('world'))") == "hello world\n"


def test_tools_injected_after_init():
    """RLM injects tools (llm_query, SUBMIT, etc.) via tools.update() after init."""
    interp = MontyCodeInterpreter(type_check=False)
    interp.tools["double"] = lambda x: x * 2
    assert interp.execute("print(double(5))") == "10\n"


def test_save_persists_across_calls(interp):
    """SAVE() persists variables across execute() calls."""
    interp.execute("SAVE(x=42)")
    assert interp.execute("print(x)") == "42\n"


def test_clear_removes_saved_state(interp):
    """CLEAR() removes previously saved variables."""
    interp.execute("SAVE(x=1, y=2)")
    interp.execute("CLEAR('x')")
    assert interp.execute("print(y)") == "2\n"
    with pytest.raises(Exception):
        interp.execute("print(x)")


def test_rlm_simple():
    """MontyRLM answers a simple question."""
    lm = dspy.LM("openai/gpt-4.1-nano")
    with dspy.context(lm=lm):
        rlm = MontyRLM(
            "question -> answer: str",
            max_iterations=5,
        )
        result = rlm(question="What is 7 * 8?")
        assert "56" in result.answer


def test_rlm_multi_step_with_save():
    """Task that benefits from SAVE: classify each item, then aggregate across iterations."""
    lm = dspy.LM("openai/gpt-4.1-nano")
    inventory = (
        "Item 1: Bananas, qty 12, expires 2026-02-16\n"
        "Item 2: Canned beans, qty 50, expires 2028-06-01\n"
        "Item 3: Fresh salmon, qty 3, expires 2026-02-15\n"
        "Item 4: Rice, qty 100, expires 2027-12-01\n"
        "Item 5: Yogurt, qty 8, expires 2026-02-14\n"
    )
    with dspy.context(lm=lm):
        rlm = MontyRLM(
            "inventory, today -> expiring_soon: list[str], long_shelf_life: list[str]",
            max_iterations=10,
        )
        result = rlm(inventory=inventory, today="2026-02-15")
        expiring = " ".join(result.expiring_soon).lower()
        assert "salmon" in expiring or "yogurt" in expiring
        long_life = " ".join(result.long_shelf_life).lower()
        assert "rice" in long_life or "canned" in long_life


def test_rlm_multi_step():
    """MontyRLM uses llm_query for semantic analysis then aggregates in code."""
    lm = get_openrouter_lm(model="openrouter/openai/gpt-4.1-nano")
    reviews = (
        "1. The pasta was absolutely amazing, best Italian food in the city!\n"
        "2. Terrible service, waited 45 minutes and the food was cold.\n"
        "3. A wonderful experience from start to finish, will definitely return.\n"
        "4. The steak was overcooked and the waiter was rude.\n"
        "5. Incredible dessert menu and the ambiance was perfect.\n"
    )
    with dspy.context(lm=lm):
        rlm = MontyRLM(
            "reviews -> positive_count: str",
            max_iterations=10,
        )
        result = rlm(reviews=reviews)
        assert "3" in result.positive_count
