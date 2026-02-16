from typing import Any
from collections.abc import Callable, Mapping

import dspy
import pydantic_monty
from dspy import CodeInterpreterError
from dspy.predict.rlm import FinalOutput, RLM, REPLHistory, translate_field_type


MONTY_ACTION_INSTRUCTIONS_TEMPLATE = """\
You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a restricted Python sandbox (Monty). Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (your input data — re-injected every iteration)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SAVE(name=value, ...)` - persist variables across iterations (re-injected automatically next iteration)
- `CLEAR(name1, name2, ...)` or `CLEAR()` - remove saved variables (all if no args)
- `SUBMIT({final_output_names})` - submit final output when done
- Builtins only — NO imports available (no re, collections, math, os, sys, etc.). NO class definitions.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. Use `SAVE()` to persist intermediate results across iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via input variables and parse/compute in code instead of retyping.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."""


class MontyCodeInterpreter(dspy.CodeInterpreter):
    """A code interpreter that uses Monty to execute code."""

    def __init__(
        self,
        *,
        tools: Mapping[str, Callable[..., str]] | None = None,
        type_check: bool = True,
        type_check_stubs: str | None = None,
        limits: pydantic_monty.ResourceLimits | None = None,
    ) -> None:
        """Initialize code interpreter backed by Monty.

        Args:
            tools: Dictionary mapping tool names to callable functions.
                   Each function should accept keyword arguments and return a string.
                   Tools are callable directly from sandbox code by name.
            type_check: Whether to type-check the code by default.
            type_check_stubs: Optional code to prepend before type-checking
                to serve as stubs.
            limits: Optional resource limits for code execution.
        """
        self._tools = dict(tools) if tools else {}
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self._limits = limits
        self._state: dict[str, Any] = {}
        self.output_fields: list[dict[str, Any]] | None = None

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        return self._tools

    def start(self) -> None:
        self._state.clear()

    def shutdown(self) -> None:
        pass

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        """Execute the code and return the output."""
        all_tools = dict(self._tools)

        def _save(**kwargs: Any) -> str:
            self._state.update(kwargs)
            return f"Saved: {', '.join(kwargs.keys())}"

        def _clear(*names: str) -> str:
            if not names:
                self._state.clear()
                return "Cleared all saved state"
            for n in names:
                self._state.pop(n, None)
            return f"Cleared: {', '.join(names)}"

        all_tools["SAVE"] = _save
        all_tools["CLEAR"] = _clear
        all_tools["SUBMIT"] = lambda **kwargs: None

        merged_vars: dict[str, Any] = {}
        if variables:
            merged_vars.update(variables)
        merged_vars.update(self._state)

        try:
            monty = pydantic_monty.Monty(
                code,
                inputs=list(merged_vars) if merged_vars else [],
                external_functions=list(all_tools),
                type_check=self._type_check,
                type_check_stubs=self._type_check_stubs,
            )
        except pydantic_monty.MontySyntaxError as e:
            raise SyntaxError(str(e)) from e
        except pydantic_monty.MontyTypingError as e:
            raise CodeInterpreterError(str(e)) from e

        stdout_parts: list[str] = []

        def _capture_print(_stream: str, text: str) -> None:
            stdout_parts.append(text)

        try:
            progress = monty.start(
                inputs=merged_vars or None,
                limits=self._limits,
                print_callback=_capture_print,
            )
        except pydantic_monty.MontyRuntimeError as e:
            raise CodeInterpreterError(str(e)) from e

        while not isinstance(progress, pydantic_monty.MontyComplete):
            if isinstance(progress, pydantic_monty.MontySnapshot):
                if progress.function_name == "SUBMIT":
                    submit_kwargs = dict(progress.kwargs)
                    if progress.args and self.output_fields:
                        field_names = [f["name"] for f in self.output_fields]
                        for name, value in zip(field_names, progress.args):
                            submit_kwargs.setdefault(name, value)
                    return FinalOutput(submit_kwargs)

                func = all_tools.get(progress.function_name)
                if func is None:
                    raise CodeInterpreterError(
                        f"Unknown function: {progress.function_name}"
                    )
                try:
                    result = func(*progress.args, **progress.kwargs)
                    progress = progress.resume(return_value=result)
                except Exception as e:
                    raise CodeInterpreterError(
                        f"Tool {progress.function_name} failed: {e}"
                    ) from e
            else:
                raise CodeInterpreterError("Async futures not supported")

        return "".join(stdout_parts) if stdout_parts else None


class MontyRLM(RLM):
    """RLM subclass with prompting adapted for Monty's restricted sandbox.

    Overrides the default action instructions to reflect that:
    - No imports are available (no stdlib)
    - Only builtins and provided tools can be used
    - SAVE()/CLEAR() allow persisting state across iterations

    If no interpreter is provided, a MontyCodeInterpreter is created automatically.
    """

    def __init__(
        self,
        signature: type[dspy.Signature] | str,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 100_000,
        verbose: bool = False,
        tools: list[Callable] | None = None,
        sub_lm: dspy.LM | None = None,
        interpreter: MontyCodeInterpreter | None = None,
        type_check: bool = False,
        limits: pydantic_monty.ResourceLimits | None = None,
    ):
        if interpreter is None:
            interpreter = MontyCodeInterpreter(
                type_check=type_check,
                limits=limits,
            )
        super().__init__(
            signature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            verbose=verbose,
            tools=tools,
            sub_lm=sub_lm,
            interpreter=interpreter,
        )

    def _build_signatures(self) -> tuple[dspy.Signature, dspy.Signature]:
        """Build signatures using Monty-specific action instructions."""
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)
        final_output_names = ", ".join(self.signature.output_fields.keys())

        output_fields = "\n".join(
            f"- {translate_field_type(n, f)}"
            for n, f in self.signature.output_fields.items()
        )

        task_instructions = (
            f"{self.signature.instructions}\n\n" if self.signature.instructions else ""
        )

        tool_docs = self._format_tool_docs(self._user_tools)

        action_sig = (
            dspy.Signature(
                {},
                task_instructions
                + MONTY_ACTION_INSTRUCTIONS_TEMPLATE.format(
                    inputs=inputs_str,
                    final_output_names=final_output_names,
                    output_fields=output_fields,
                    max_llm_calls=self.max_llm_calls,
                )
                + tool_docs,
            )
            .append(
                "variables_info",
                dspy.InputField(
                    desc="Metadata about the variables available in the REPL"
                ),
                type_=str,
            )
            .append(
                "repl_history",
                dspy.InputField(desc="Previous REPL code executions and their outputs"),
                type_=REPLHistory,
            )
            .append(
                "iteration",
                dspy.InputField(
                    desc="Current iteration number (1-indexed) out of max_iterations"
                ),
                type_=str,
            )
            .append(
                "reasoning",
                dspy.OutputField(
                    desc="Think step-by-step: what do you know? What remains? Plan your next action."
                ),
                type_=str,
            )
            .append(
                "code",
                dspy.OutputField(
                    desc="Python code to execute. Use markdown code block format: ```python\\n<code>\\n```"
                ),
                type_=str,
            )
        )

        extract_instructions = """Based on the REPL trajectory, extract the final outputs now.

            Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."""

        extended_task_instructions = ""
        if task_instructions:
            extended_task_instructions = (
                "The trajectory was generated with the following objective: \n"
                + task_instructions
                + "\n"
            )

        extract_sig = dspy.Signature(
            {**self.signature.output_fields},
            extended_task_instructions + extract_instructions,
        )
        extract_sig = extract_sig.prepend(
            "repl_history",
            dspy.InputField(desc="Your REPL interactions so far"),
            type_=REPLHistory,
        )
        extract_sig = extract_sig.prepend(
            "variables_info",
            dspy.InputField(desc="Metadata about the variables available in the REPL"),
            type_=str,
        )

        return action_sig, extract_sig
