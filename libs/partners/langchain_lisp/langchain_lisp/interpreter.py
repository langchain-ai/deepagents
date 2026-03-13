from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from lark import Lark, Token, Tree

GRAMMAR = r'''
?start: expr+

?expr: atom
     | list_expr

list_expr: "(" expr* ")"

?atom: NUMBER        -> number
     | ESCAPED_STRING -> string
     | "null"        -> null
     | "true"        -> true
     | "false"       -> false
     | SYMBOL         -> symbol

SYMBOL: /[a-zA-Z_+\-*=<>!?\/][a-zA-Z0-9_+\-*=<>!?\/]*/

%import common.ESCAPED_STRING
%import common.NUMBER
%import common.WS
%ignore WS
'''


@dataclass(frozen=True)
class Symbol:
    name: str


class LispInterpreter:
    def __init__(
        self,
        *,
        functions: Mapping[str, Callable[..., Any]] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._parser = Lark(GRAMMAR, start="start", parser="lalr")
        self._functions = {**self._builtin_functions(), **dict(functions or {})}
        self._env: dict[str, Any] = {}
        self._printed_lines: list[str] = []
        self._max_workers = max_workers

    @property
    def env(self) -> dict[str, Any]:
        return dict(self._env)

    @property
    def printed_lines(self) -> list[str]:
        return list(self._printed_lines)

    def evaluate(self, source: str) -> Any:
        tree = self._parser.parse(source)
        if isinstance(tree, Tree) and tree.data == "start":
            result: Any = None
            for expr in tree.children:
                result = self._eval(expr, self._env)
            return result
        return self._eval(tree, self._env)

    def clear_output(self) -> None:
        self._printed_lines.clear()

    def _eval(self, node: Tree | Token | Symbol | Any, env: dict[str, Any]) -> Any:
        if isinstance(node, Tree):
            if node.data == "number":
                return self._parse_number(node.children[0])
            if node.data == "string":
                return self._parse_string(node.children[0])
            if node.data == "null":
                return None
            if node.data == "true":
                return True
            if node.data == "false":
                return False
            if node.data == "symbol":
                return self._lookup_symbol(node.children[0], env)
            if node.data == "list_expr":
                return self._eval_list(node.children, env)
            msg = f"Unsupported parse node: {node.data}"
            raise ValueError(msg)
        if isinstance(node, Symbol):
            return self._lookup_name(node.name, env)
        return node

    def _parse_number(self, token: Token) -> int | float:
        text = str(token)
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)

    def _parse_string(self, token: Token) -> str:
        return str(token)[1:-1].encode("utf-8").decode("unicode_escape")

    def _lookup_symbol(self, token: Token, env: dict[str, Any]) -> Any:
        return self._lookup_name(str(token), env)

    def _lookup_name(self, name: str, env: dict[str, Any]) -> Any:
        if name in env:
            return env[name]
        if name in self._functions:
            return self._functions[name]
        msg = f"Unknown symbol: {name}"
        raise NameError(msg)

    def _builtin_functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "+": self._add,
            "-": self._subtract,
            "*": self._multiply,
            "/": self._divide,
            "=": self._equal,
            "!=": self._not_equal,
            ">": self._greater,
            ">=": self._greater_equal,
            "<": self._less,
            "<=": self._less_equal,
            "not": self._not,
            "and": self._and,
            "or": self._or,
            "get": self._get,
            "length": self._length,
        }

    def _eval_list(self, items: list[Tree], env: dict[str, Any]) -> Any:
        if not items:
            return []
        head = items[0]
        if isinstance(head, Tree) and head.data == "symbol":
            name = str(head.children[0])
            if name == "let":
                return self._eval_let(items[1:], env)
            if name == "if":
                return self._eval_if(items[1:], env)
            if name == "print":
                return self._eval_print(items[1:], env)
            if name == "list":
                return [self._eval(item, env) for item in items[1:]]
            if name == "dict":
                return self._eval_dict(items[1:], env)
            if name == "parallel":
                return self._eval_parallel(items[1:], env)
        func = self._eval(head, env)
        args = [self._eval(item, env) for item in items[1:]]
        return func(*args)

    def _eval_let(self, items: list[Tree], env: dict[str, Any]) -> Any:
        if len(items) != 2:
            msg = "let expects exactly two arguments"
            raise ValueError(msg)
        target = items[0]
        if not isinstance(target, Tree) or target.data != "symbol":
            msg = "let target must be a symbol"
            raise ValueError(msg)
        name = str(target.children[0])
        value = self._eval(items[1], env)
        env[name] = value
        return value

    def _eval_if(self, items: list[Tree], env: dict[str, Any]) -> Any:
        if len(items) != 3:
            msg = "if expects exactly three arguments"
            raise ValueError(msg)
        cond = self._eval(items[0], env)
        branch = items[1] if self._is_truthy(cond) else items[2]
        return self._eval(branch, env)

    def _eval_print(self, items: list[Tree], env: dict[str, Any]) -> Any:
        if len(items) != 1:
            msg = "print expects exactly one argument"
            raise ValueError(msg)
        value = self._eval(items[0], env)
        self._printed_lines.append(self._format_value(value))
        return value

    def _eval_dict(self, items: list[Tree], env: dict[str, Any]) -> dict[str, Any]:
        if len(items) % 2 != 0:
            msg = "dict expects an even number of arguments"
            raise ValueError(msg)
        result: dict[str, Any] = {}
        for key_node, value_node in zip(items[::2], items[1::2], strict=False):
            key = self._eval(key_node, env)
            if not isinstance(key, str):
                msg = "dict keys must evaluate to strings"
                raise TypeError(msg)
            result[key] = self._eval(value_node, env)
        return result

    def _eval_parallel(self, items: list[Tree], env: dict[str, Any]) -> list[Any]:
        snapshots = [dict(env) for _ in items]
        with ThreadPoolExecutor(max_workers=self._max_workers or len(items) or None) as executor:
            futures = [
                executor.submit(self._eval, item, snapshot)
                for item, snapshot in zip(items, snapshots, strict=False)
            ]
            return [future.result() for future in futures]

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "null"
        if value is True:
            return "true"
        if value is False:
            return "false"
        return str(value)

    def _is_truthy(self, value: Any) -> bool:
        return value is not False and value is not None

    def _require_args(self, name: str, args: tuple[Any, ...], *, minimum: int) -> None:
        if len(args) < minimum:
            msg = f"{name} expects at least {minimum} argument(s)"
            raise ValueError(msg)

    def _require_exact_args(self, name: str, args: tuple[Any, ...], *, count: int) -> None:
        if len(args) != count:
            msg = f"{name} expects exactly {count} argument(s)"
            raise ValueError(msg)

    def _add(self, *args: Any) -> Any:
        self._require_args("+", args, minimum=1)
        first = args[0]
        if isinstance(first, str):
            if not all(isinstance(arg, str) for arg in args):
                msg = "+ expects all arguments to be strings when the first argument is a string"
                raise TypeError(msg)
            return "".join(args)
        total = first
        for arg in args[1:]:
            total += arg
        return total

    def _subtract(self, *args: Any) -> Any:
        self._require_args("-", args, minimum=1)
        if len(args) == 1:
            return -args[0]
        total = args[0]
        for arg in args[1:]:
            total -= arg
        return total

    def _multiply(self, *args: Any) -> Any:
        self._require_args("*", args, minimum=1)
        result = args[0]
        for arg in args[1:]:
            result *= arg
        return result

    def _divide(self, *args: Any) -> Any:
        self._require_args("/", args, minimum=1)
        if len(args) == 1:
            return 1 / args[0]
        result = args[0]
        for arg in args[1:]:
            result /= arg
        return result

    def _equal(self, *args: Any) -> bool:
        self._require_args("=", args, minimum=2)
        return all(arg == args[0] for arg in args[1:])

    def _not_equal(self, *args: Any) -> bool:
        self._require_exact_args("!=", args, count=2)
        return args[0] != args[1]

    def _greater(self, *args: Any) -> bool:
        self._require_exact_args(">", args, count=2)
        return args[0] > args[1]

    def _greater_equal(self, *args: Any) -> bool:
        self._require_exact_args(">=", args, count=2)
        return args[0] >= args[1]

    def _less(self, *args: Any) -> bool:
        self._require_exact_args("<", args, count=2)
        return args[0] < args[1]

    def _less_equal(self, *args: Any) -> bool:
        self._require_exact_args("<=", args, count=2)
        return args[0] <= args[1]

    def _not(self, *args: Any) -> bool:
        self._require_exact_args("not", args, count=1)
        return not self._is_truthy(args[0])

    def _and(self, *args: Any) -> bool:
        self._require_args("and", args, minimum=1)
        return all(self._is_truthy(arg) for arg in args)

    def _or(self, *args: Any) -> bool:
        self._require_args("or", args, minimum=1)
        return any(self._is_truthy(arg) for arg in args)

    def _get(self, *args: Any) -> Any:
        self._require_exact_args("get", args, count=2)
        collection, key = args
        if isinstance(collection, dict):
            return collection[key]
        return collection[key]

    def _length(self, *args: Any) -> int:
        self._require_exact_args("length", args, count=1)
        return len(args[0])
