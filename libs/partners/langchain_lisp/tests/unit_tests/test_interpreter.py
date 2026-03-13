from __future__ import annotations

import threading
import time

import pytest

from langchain_lisp import LispInterpreter


def test_evaluates_literals_and_stateful_assignments() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate("""
    (let answer 42)
    (let nothing null)
    (let items (list 1 2 3))
    (let person (dict \"name\" \"Ada\" \"age\" 30))
    answer
    """)

    assert result == 42
    assert interpreter.env == {
        "answer": 42,
        "nothing": None,
        "items": [1, 2, 3],
        "person": {"name": "Ada", "age": 30},
    }


def test_state_persists_across_evaluations() -> None:
    interpreter = LispInterpreter()

    interpreter.evaluate("(let x 10)")
    result = interpreter.evaluate("x")

    assert result == 10
    assert interpreter.env == {"x": 10}


def test_print_records_output_and_returns_value() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate('(print "hello")')

    assert result == "hello"
    assert interpreter.printed_lines == ["hello"]


def test_if_uses_truthiness_to_choose_branch() -> None:
    interpreter = LispInterpreter()

    truthy = interpreter.evaluate('(if true "big" "small")')
    falsy = interpreter.evaluate('(if null "big" "small")')

    assert (truthy, falsy) == ("big", "small")


def test_calls_registered_functions_and_uses_variables() -> None:
    interpreter = LispInterpreter(functions={"add": lambda left, right: left + right})

    result = interpreter.evaluate("""
    (let x 10)
    (let y 20)
    (add x y)
    """)

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

    interpreter = LispInterpreter(functions={"slow-add": slow_add})

    result = interpreter.evaluate("""
    (parallel
      (slow-add 1 2)
      (slow-add 10 20)
      (slow-add 100 200))
    """)

    assert result == [3, 30, 300]
    assert sorted(calls) == [3, 30, 300]


def test_parallel_results_can_be_assigned() -> None:
    interpreter = LispInterpreter(functions={"echo": lambda value: value})

    result = interpreter.evaluate("""
    (let results (parallel (echo 1) (echo 2) (echo 3)))
    results
    """)

    assert result == [1, 2, 3]
    assert interpreter.env == {"results": [1, 2, 3]}


def test_parses_float_and_boolean_literals() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate("""
    (let ratio 3.5)
    (let enabled true)
    (let disabled false)
    ratio
    """)

    assert result == 3.5
    assert interpreter.env == {"ratio": 3.5, "enabled": True, "disabled": False}


def test_print_formats_null_and_booleans() -> None:
    interpreter = LispInterpreter()

    interpreter.evaluate("(print null)")
    interpreter.evaluate("(print true)")
    interpreter.evaluate("(print false)")

    assert interpreter.printed_lines == ["null", "true", "false"]


def test_clear_output_resets_printed_lines() -> None:
    interpreter = LispInterpreter()

    interpreter.evaluate('(print "hello")')
    interpreter.clear_output()

    assert interpreter.printed_lines == []


def test_string_escapes_are_decoded() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate(r'"line\nindent\tquote:\""')

    assert result == 'line\nindent\tquote:"'


def test_nested_list_and_dict_literals_work() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate(
        '(dict "items" (list 1 (list 2 3)) "meta" (dict "ok" true))'
    )

    assert result == {"items": [1, [2, 3]], "meta": {"ok": True}}


def test_empty_list_expression_returns_empty_list() -> None:
    interpreter = LispInterpreter()

    assert interpreter.evaluate("()") == []


def test_unknown_symbol_raises_name_error() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(NameError, match="Unknown symbol: missing"):
        interpreter.evaluate("missing")


def test_calling_unknown_function_raises_name_error() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(NameError, match="Unknown symbol: missing-fn"):
        interpreter.evaluate("(missing-fn 1 2)")


def test_let_requires_exactly_two_arguments() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(ValueError, match="let expects exactly two arguments"):
        interpreter.evaluate("(let x 1 2)")


def test_let_target_must_be_symbol() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(ValueError, match="let target must be a symbol"):
        interpreter.evaluate("(let 123 1)")


def test_if_requires_exactly_three_arguments() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(ValueError, match="if expects exactly three arguments"):
        interpreter.evaluate("(if true 1)")


def test_print_requires_exactly_one_argument() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(ValueError, match="print expects exactly one argument"):
        interpreter.evaluate("(print 1 2)")


def test_dict_requires_even_number_of_arguments() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(ValueError, match="dict expects an even number of arguments"):
        interpreter.evaluate('(dict "name" "Ada" "age")')


def test_dict_keys_must_be_strings() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(TypeError, match="dict keys must evaluate to strings"):
        interpreter.evaluate("(dict 1 2)")


def test_parallel_expressions_use_isolated_variable_snapshots() -> None:
    interpreter = LispInterpreter(functions={"echo": lambda value: value})

    result = interpreter.evaluate("""
    (let x 10)
    (parallel
      (let x 20)
      x
      (echo x))
    """)

    assert result == [20, 10, 10]
    assert interpreter.env == {"x": 10}


def test_parallel_propagates_function_errors() -> None:
    def fail() -> None:
        msg = "boom"
        raise RuntimeError(msg)

    interpreter = LispInterpreter(functions={"fail": fail})

    with pytest.raises(RuntimeError, match="boom"):
        interpreter.evaluate("(parallel (fail) 1)")


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

    interpreter = LispInterpreter(functions={"block": block}, max_workers=1)

    result = interpreter.evaluate("(parallel (block 1) (block 2) (block 3))")

    assert result == [1, 2, 3]
    assert max_active == 1


def test_builtin_arithmetic_and_comparisons_work() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate("""
    (let sum (+ 1 2 3))
    (let diff (- 10 3 2))
    (let product (* 2 3 4))
    (let quotient (/ 20 2 2))
    (let eq (= sum 6 6))
    (let gt (> 5 3))
    (list sum diff product quotient eq gt)
    """)

    assert result == [6, 5, 24, 5.0, True, True]


def test_builtin_string_concat_with_plus() -> None:
    interpreter = LispInterpreter()

    assert interpreter.evaluate('(+ "hello" " " "world")') == "hello world"


def test_builtin_boolean_operators_use_interpreter_truthiness() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate("""
    (list
      (not null)
      (not false)
      (and true 0 "")
      (or false null 0))
    """)

    assert result == [True, True, True, True]


def test_if_treats_zero_and_empty_string_as_truthy() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate("""
    (list
      (if 0 "yes" "no")
      (if "" "yes" "no")
      (if false "yes" "no"))
    """)

    assert result == ["yes", "yes", "no"]


def test_builtin_get_and_length_work_for_lists_dicts_and_strings() -> None:
    interpreter = LispInterpreter()

    result = interpreter.evaluate("""
    (list
      (get (list 10 20 30) 1)
      (get (dict "name" "Ada") "name")
      (length (list 1 2 3))
      (length "abcd"))
    """)

    assert result == [20, "Ada", 3, 4]


def test_user_functions_override_builtins() -> None:
    interpreter = LispInterpreter(functions={"+": lambda left, right: f"{left}:{right}"})

    assert interpreter.evaluate('(+ "a" "b")') == "a:b"


def test_builtin_argument_validation_errors_are_clear() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(ValueError, match=r"\+ expects at least 1 argument\(s\)"):
        interpreter.evaluate("(+)")
    with pytest.raises(ValueError, match=r"> expects exactly 2 argument\(s\)"):
        interpreter.evaluate("(> 1)")


def test_plus_rejects_mixed_string_and_non_string_arguments() -> None:
    interpreter = LispInterpreter()

    with pytest.raises(
        TypeError,
        match=r"\+ expects all arguments to be strings when the first argument is a string",
    ):
        interpreter.evaluate('(+ "a" 1)')
