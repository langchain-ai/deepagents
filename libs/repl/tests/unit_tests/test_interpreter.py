from __future__ import annotations

import threading
import time

import pytest

from langchain_repl import Interpreter


def test_evaluates_literals_and_stateful_assignments() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate(
        "answer = 42\n"
        "nothing = None\n"
        "items = [1, 2, 3]\n"
        'person = {"name": "Ada", "age": 30}\n'
        "answer\n"
    )

    assert result == 42
    assert interpreter.env == {
        "answer": 42,
        "nothing": None,
        "items": [1, 2, 3],
        "person": {"name": "Ada", "age": 30},
    }


def test_state_persists_across_evaluations() -> None:
    interpreter = Interpreter()

    interpreter.evaluate("x = 10")
    result = interpreter.evaluate("x")

    assert result == 10
    assert interpreter.env == {"x": 10}


def test_print_records_output_and_returns_value() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate('print("hello")')

    assert result == "hello"
    assert interpreter.printed_lines == ["hello"]


def test_if_uses_truthiness_to_choose_branch() -> None:
    interpreter = Interpreter()

    truthy = interpreter.evaluate('if True then "big" else "small" end')
    falsy = interpreter.evaluate('if None then "big" else "small" end')

    assert (truthy, falsy) == ("big", "small")


def test_calls_registered_functions_and_uses_variables() -> None:
    interpreter = Interpreter(functions={"add": lambda left, right: left + right})

    result = interpreter.evaluate("x = 10\ny = 20\nadd(x, y)\n")

    assert result == 30
    assert interpreter.env == {"x": 10, "y": 20}


def test_parallel_calls_return_results_in_order() -> None:
    calls: list[int] = []
    lock = threading.Lock()

    def slow_add(left: int, right: int) -> int:
        time.sleep(0.05)
        total = left + right
        with lock:
            calls.append(total)
        return total

    interpreter = Interpreter(functions={"slow_add": slow_add})

    result = interpreter.evaluate(
        "parallel(slow_add(1, 2), slow_add(10, 20), slow_add(100, 200))"
    )

    assert result == [3, 30, 300]
    assert sorted(calls) == [3, 30, 300]


def test_parallel_results_can_be_assigned() -> None:
    interpreter = Interpreter(functions={"echo": lambda value: value})

    result = interpreter.evaluate(
        "results = parallel(echo(1), echo(2), echo(3))\nresults"
    )

    assert result == [1, 2, 3]
    assert interpreter.env == {"results": [1, 2, 3]}


def test_benchmark_simple_print_program() -> None:
    """Sanity check."""
    interpreter = Interpreter()
    result = interpreter.evaluate("print(42)")
    tic = time.monotonic()
    for _ in range(10):
        interpreter = Interpreter()
        result = interpreter.evaluate("print(42)")
        assert result == 42
        assert interpreter.printed_lines == ["42"]
    toc = time.monotonic()
    elapsed = toc - tic
    # in us not ms
    elapsed_us = elapsed * 1e6
    elapsed_per_test = elapsed_us / 10.0
    assert elapsed_per_test < 30

def test_parallel_allows_multiline_arguments() -> None:
    interpreter = Interpreter(functions={"echo": lambda value: value})

    result = interpreter.evaluate(
        "results = parallel(\n    echo(1),\n    echo(2),\n    echo(3)\n)\nresults"
    )

    assert result == [1, 2, 3]
    assert interpreter.env == {"results": [1, 2, 3]}


def test_function_calls_allow_multiline_arguments() -> None:
    interpreter = Interpreter(functions={"add": lambda left, right: left + right})

    result = interpreter.evaluate("add(\n    10,\n    20\n)")

    assert result == 30


def test_parses_float_and_boolean_literals() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate(
        "ratio = 3.5\nenabled = True\ndisabled = False\nratio"
    )

    assert result == 3.5
    assert interpreter.env == {"ratio": 3.5, "enabled": True, "disabled": False}


def test_print_formats_none_and_booleans() -> None:
    interpreter = Interpreter()

    interpreter.evaluate("print(None)")
    interpreter.evaluate("print(True)")
    interpreter.evaluate("print(False)")

    assert interpreter.printed_lines == ["None", "True", "False"]


def test_clear_output_resets_printed_lines() -> None:
    interpreter = Interpreter()

    interpreter.evaluate('print("hello")')
    interpreter.clear_output()

    assert interpreter.printed_lines == []


def test_string_escapes_are_decoded() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate(r'"line\nindent\tquote:\""')

    assert result == 'line\nindent\tquote:"'


def test_nested_list_and_dict_literals_work() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate('{"items": [1, [2, 3]], "meta": {"ok": True}}')

    assert result == {"items": [1, [2, 3]], "meta": {"ok": True}}


def test_unknown_name_raises_name_error() -> None:
    interpreter = Interpreter()

    with pytest.raises(NameError, match="Unknown name: missing"):
        interpreter.evaluate("missing")


def test_calling_unknown_function_raises_name_error() -> None:
    interpreter = Interpreter()

    with pytest.raises(NameError, match="Unknown name: missing_fn"):
        interpreter.evaluate("missing_fn(1, 2)")


def test_print_requires_exactly_one_argument() -> None:
    interpreter = Interpreter()

    with pytest.raises(ValueError, match="print expects exactly one argument"):
        interpreter.evaluate("print(1, 2)")


def test_parallel_expressions_use_isolated_variable_snapshots() -> None:
    interpreter = Interpreter(functions={"echo": lambda value: value})

    result = interpreter.evaluate("x = 10\nparallel(try(x[0], 20), x, echo(x))")

    assert result == [20, 10, 10]
    assert interpreter.env == {"x": 10}


def test_parallel_propagates_function_errors() -> None:
    def fail() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    interpreter = Interpreter(functions={"fail": fail})

    with pytest.raises(RuntimeError, match="boom"):
        interpreter.evaluate("parallel(fail(), 1)")


def test_parallel_respects_configured_max_workers() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def block(value: int) -> int:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return value

    interpreter = Interpreter(functions={"block": block}, max_workers=1)

    result = interpreter.evaluate("parallel(block(1), block(2), block(3))")

    assert result == [1, 2, 3]
    assert max_active == 1


def test_if_treats_zero_and_empty_string_as_falsy() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate(
        "if 0 then\n"
        '    print("yes")\n'
        "else\n"
        '    print("no")\n'
        "end\n"
        'if "" then\n'
        '    print("yes")\n'
        "else\n"
        '    print("no")\n'
        "end\n"
        "if False then\n"
        '    print("yes")\n'
        "else\n"
        '    print("no")\n'
        "end\n"
    )

    assert result == "no"
    assert interpreter.printed_lines == ["no", "no", "no"]


def test_indexing_works_for_lists_and_dicts() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate("[10, 20, 30][1]")
    assert result == 20
    assert interpreter.evaluate('{"name": "Ada"}["name"]') == "Ada"


def test_index_validation_errors_are_clear() -> None:
    interpreter = Interpreter()

    with pytest.raises(TypeError, match="list indexes must be integers"):
        interpreter.evaluate('[1, 2]["0"]')
    with pytest.raises(TypeError, match="dict indexes must be strings"):
        interpreter.evaluate('{"a": 1}[0]')


def test_for_loop_iterates_list_values() -> None:
    interpreter = Interpreter(functions={"echo": lambda value: value})

    result = interpreter.evaluate(
        "items = [1, 2, 3]\nfor item in items do\n    print(echo(item))\nend\n"
    )

    assert result == 3
    assert interpreter.printed_lines == ["1", "2", "3"]
    assert interpreter.env == {"items": [1, 2, 3], "item": 3}


def test_for_loop_requires_list_iterable() -> None:
    interpreter = Interpreter()

    with pytest.raises(TypeError, match="for loops require a list iterable"):
        interpreter.evaluate('for item in {"a": 1} do\nprint(item)\nend')


def test_try_returns_fallback_on_error() -> None:
    interpreter = Interpreter(
        functions={"fail": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}
    )

    assert interpreter.evaluate('try(fail(), "fallback")') == "fallback"


def test_comments_and_blank_lines_are_ignored() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate(
        "# setup\n\nx = 1\ny = 2  # trailing comment\n\nprint(x)\ny\n"
    )

    assert result == 2
    assert interpreter.printed_lines == ["1"]
    assert interpreter.env == {"x": 1, "y": 2}


def test_empty_list_and_dict_literals_work() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate("items = []\nmeta = {}\n[items, meta]")

    assert result == [[], {}]
    assert interpreter.env == {"items": [], "meta": {}}


def test_parenthesized_expression_controls_indexing_target() -> None:
    interpreter = Interpreter()

    assert interpreter.evaluate("([10, 20, 30])[2]") == 30


def test_nested_indexes_work() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate('{"users": [{"name": "Ada"}]}["users"][0]["name"]')

    assert result == "Ada"


def test_if_without_else_returns_none_when_condition_is_falsy() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate('if False then\n    print("yes")\nend')

    assert result is None
    assert interpreter.printed_lines == []


def test_if_can_contain_nested_for_loop() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate(
        "if True then\n"
        "    items = [1, 2]\n"
        "    for item in items do\n"
        "        print(item)\n"
        "    end\n"
        "else\n"
        '    print("no")\n'
        "end\n"
    )

    assert result == 2
    assert interpreter.printed_lines == ["1", "2"]
    assert interpreter.env == {"items": [1, 2], "item": 2}


def test_for_loop_can_update_outer_variable() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate(
        'last = None\nfor item in ["a", "b"] do\n    last = item\nend\nlast\n'
    )

    assert result == "b"
    assert interpreter.env == {"last": "b", "item": "b"}


def test_for_loop_over_empty_list_returns_none() -> None:
    interpreter = Interpreter()

    result = interpreter.evaluate("for item in [] do\n    print(item)\nend")

    assert result is None
    assert interpreter.printed_lines == []
    assert interpreter.env == {}


def test_try_only_catches_primary_expression_errors() -> None:
    interpreter = Interpreter(
        functions={"fail": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}
    )

    with pytest.raises(NameError, match="Unknown name: missing"):
        interpreter.evaluate("try(fail(), missing)")


def test_try_returns_primary_value_when_no_error_occurs() -> None:
    interpreter = Interpreter()

    assert interpreter.evaluate('try("value", "fallback")') == "value"


def test_parallel_accepts_no_arguments() -> None:
    interpreter = Interpreter()

    assert interpreter.evaluate("parallel()") == []


def test_function_call_accepts_no_arguments() -> None:
    interpreter = Interpreter(functions={"greet": lambda: "hello"})

    assert interpreter.evaluate("greet()") == "hello"


def test_parse_method_returns_program_object() -> None:
    interpreter = Interpreter()

    program = interpreter.parse("x = 1\nx")
    assert len(program.statements) == 2


def test_parse_errors_are_clear() -> None:
    interpreter = Interpreter()

    with pytest.raises(ValueError, match="Expected THEN"):
        interpreter.evaluate("if True print(1) end")


def test_parse_error_for_missing_closing_bracket_is_clear() -> None:
    interpreter = Interpreter()

    with pytest.raises(ValueError, match="Expected ,"):
        interpreter.evaluate("[1, 2")


def test_parse_error_for_invalid_character_is_clear() -> None:
    interpreter = Interpreter()

    with pytest.raises(ValueError, match="Unexpected character '@'"):
        interpreter.evaluate("@")
