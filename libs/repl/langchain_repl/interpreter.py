"""Mini REPL interpreter and parser implementation."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableMapping


@dataclass(frozen=True)
class Token:
    """A lexical token produced by the tokenizer."""

    kind: str
    value: Any
    line: int
    column: int


class Statement:
    """Base class for parsed statements."""


class Expression:
    """Base class for parsed expressions."""


@dataclass(frozen=True)
class Program:
    """A parsed REPL program."""

    statements: tuple[Statement, ...]


@dataclass(frozen=True)
class Assign(Statement):
    """An assignment statement."""

    name: str
    value: Expression


@dataclass(frozen=True)
class IfStatement(Statement):
    """A conditional statement."""

    condition: Expression
    then_body: tuple[Statement, ...]
    else_body: tuple[Statement, ...]


@dataclass(frozen=True)
class ForStatement(Statement):
    """A for-loop statement."""

    name: str
    iterable: Expression
    body: tuple[Statement, ...]


@dataclass(frozen=True)
class ExpressionStatement(Statement):
    """A statement that evaluates an expression."""

    expression: Expression


@dataclass(frozen=True)
class Name(Expression):
    """A name reference expression."""

    value: str


@dataclass(frozen=True)
class Literal(Expression):
    """A literal value expression."""

    value: Any


@dataclass(frozen=True)
class ListLiteral(Expression):
    """A list literal expression."""

    items: tuple[Expression, ...]


@dataclass(frozen=True)
class DictLiteral(Expression):
    """A dict literal expression."""

    items: tuple[tuple[str, Expression], ...]


@dataclass(frozen=True)
class BinaryOperation(Expression):
    """A binary operation expression."""

    left: Expression
    operator: str
    right: Expression


@dataclass(frozen=True)
class Attribute(Expression):
    """An attribute access expression."""

    target: Expression
    name: str


@dataclass(frozen=True)
class Call(Expression):
    """A function or callable invocation expression."""

    target: Expression
    args: tuple[Expression, ...]


@dataclass(frozen=True)
class Index(Expression):
    """An indexing expression."""

    target: Expression
    index: Expression


class ParseError(ValueError):
    """Raised when REPL source cannot be parsed."""


class ForeignObjectInterface(Protocol):
    """Protocol for dispatching operations on foreign objects."""

    def supports(self, value: Any) -> bool:
        """Return whether this handler manages the provided runtime value."""

    def get_item(self, value: Any, key: Any) -> Any:
        """Resolve `value[key]` for a supported foreign object."""

    def resolve_member(self, value: Any, name: str) -> Any:
        """Resolve `value.name` for a supported foreign object."""

    def call(self, value: Any, args: tuple[Any, ...]) -> Any:
        """Invoke `value(*args)` for a supported foreign object."""


class _Tokenizer:
    def __init__(self, source: str) -> None:
        self._source = source
        self._length = len(source)
        self._index = 0
        self._line = 1
        self._column = 1

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while self._index < self._length:
            char = self._source[self._index]
            if char in " \t\r":
                self._advance()
                continue
            if char == "\n":
                tokens.append(Token("NEWLINE", "\n", self._line, self._column))
                self._advance()
                continue
            if char == "#":
                self._skip_comment()
                continue
            if char in "()+-[]{}:,.=":
                tokens.append(Token(char, char, self._line, self._column))
                self._advance()
                continue
            if char == '"':
                tokens.append(self._read_string())
                continue
            if char.isdigit() or (char == "-" and self._peek().isdigit()):
                tokens.append(self._read_number())
                continue
            if char.isalpha() or char == "_":
                tokens.append(self._read_name())
                continue
            msg = (
                f"Unexpected character {char!r} at line {self._line}, "
                f"column {self._column}"
            )
            raise ParseError(msg)
        tokens.append(Token("EOF", None, self._line, self._column))
        return tokens

    def _advance(self) -> str:
        char = self._source[self._index]
        self._index += 1
        if char == "\n":
            self._line += 1
            self._column = 1
        else:
            self._column += 1
        return char

    def _peek(self) -> str:
        if self._index + 1 >= self._length:
            return ""
        return self._source[self._index + 1]

    def _skip_comment(self) -> None:
        while self._index < self._length and self._source[self._index] != "\n":
            self._advance()

    def _read_string(self) -> Token:
        line = self._line
        column = self._column
        self._advance()
        chars: list[str] = []
        while self._index < self._length:
            char = self._advance()
            if char == '"':
                return Token("STRING", "".join(chars), line, column)
            if char == "\\":
                if self._index >= self._length:
                    break
                escaped = self._advance()
                chars.append(self._decode_escape(escaped))
                continue
            chars.append(char)
        msg = f"Unterminated string at line {line}, column {column}"
        raise ParseError(msg)

    def _decode_escape(self, escaped: str) -> str:
        escapes = {
            '"': '"',
            "\\": "\\",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        return escapes.get(escaped, escaped)

    def _read_number(self) -> Token:
        line = self._line
        column = self._column
        chars = [self._advance()]
        has_dot = False
        while self._index < self._length:
            char = self._source[self._index]
            if char.isdigit():
                chars.append(self._advance())
                continue
            if char == "." and not has_dot:
                has_dot = True
                chars.append(self._advance())
                continue
            break
        text = "".join(chars)
        value: int | float = float(text) if has_dot else int(text)
        return Token("NUMBER", value, line, column)

    def _read_name(self) -> Token:
        line = self._line
        column = self._column
        chars = [self._advance()]
        while self._index < self._length:
            char = self._source[self._index]
            if char.isalnum() or char == "_":
                chars.append(self._advance())
                continue
            break
        text = "".join(chars)
        keywords = {
            "if": "IF",
            "then": "THEN",
            "else": "ELSE",
            "end": "END",
            "for": "FOR",
            "in": "IN",
            "do": "DO",
            "True": "TRUE",
            "False": "FALSE",
            "None": "NONE",
        }
        kind = keywords.get(text, "NAME")
        return Token(kind, text, line, column)


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse(self) -> Program:
        statements = self._parse_block(stop_kinds={"EOF"})
        self._expect("EOF")
        return Program(tuple(statements))

    def _parse_block(self, *, stop_kinds: set[str]) -> list[Statement]:
        statements: list[Statement] = []
        self._skip_newlines()
        while self._current().kind not in stop_kinds:
            statements.append(self._parse_statement())
            self._skip_newlines()
        return statements

    def _parse_statement(self) -> Statement:
        token = self._current()
        if token.kind == "IF":
            return self._parse_if()
        if token.kind == "FOR":
            return self._parse_for()
        if token.kind == "NAME" and self._peek().kind == "=":
            name = self._advance().value
            self._expect("=")
            return Assign(name=name, value=self._parse_expression())
        return ExpressionStatement(self._parse_expression())

    def _parse_if(self) -> IfStatement:
        self._expect("IF")
        condition = self._parse_expression()
        self._expect("THEN")
        self._consume_statement_separator()
        then_body = tuple(self._parse_block(stop_kinds={"ELSE", "END"}))
        else_body: tuple[Statement, ...] = ()
        if self._match("ELSE"):
            self._consume_statement_separator()
            else_body = tuple(self._parse_block(stop_kinds={"END"}))
        self._expect("END")
        return IfStatement(
            condition=condition, then_body=then_body, else_body=else_body
        )

    def _parse_for(self) -> ForStatement:
        self._expect("FOR")
        name = self._expect("NAME").value
        self._expect("IN")
        iterable = self._parse_expression()
        self._expect("DO")
        self._consume_statement_separator()
        body = tuple(self._parse_block(stop_kinds={"END"}))
        self._expect("END")
        return ForStatement(name=name, iterable=iterable, body=body)

    def _parse_expression(self) -> Expression:
        expr = self._parse_postfix()
        while True:
            if self._match("+"):
                expr = BinaryOperation(
                    left=expr,
                    operator="+",
                    right=self._parse_postfix(),
                )
                continue
            if self._match("-"):
                expr = BinaryOperation(
                    left=expr,
                    operator="-",
                    right=self._parse_postfix(),
                )
                continue
            break
        return expr

    def _parse_postfix(self) -> Expression:
        expr = self._parse_primary()
        while True:
            if self._match("["):
                index = self._parse_expression()
                self._expect("]")
                expr = Index(target=expr, index=index)
                continue
            if self._match("."):
                expr = Attribute(target=expr, name=self._expect("NAME").value)
                continue
            if self._match("("):
                expr = Call(target=expr, args=tuple(self._parse_arguments()))
                continue
            break
        return expr

    def _parse_primary(self) -> Expression:  # noqa: PLR0911
        token = self._current()
        if token.kind == "NAME":
            return Name(self._advance().value)
        if token.kind == "NUMBER":
            return Literal(self._advance().value)
        if token.kind == "STRING":
            return Literal(self._advance().value)
        if token.kind == "TRUE":
            self._advance()
            return Literal(value=True)
        if token.kind == "FALSE":
            self._advance()
            return Literal(value=False)
        if token.kind == "NONE":
            self._advance()
            return Literal(None)
        if token.kind == "[":
            return self._parse_list()
        if token.kind == "{":
            return self._parse_dict()
        if token.kind == "(":
            self._advance()
            expr = self._parse_expression()
            self._expect(")")
            return expr
        msg = (
            f"Unexpected token {token.kind} at line {token.line}, column {token.column}"
        )
        raise ParseError(msg)

    def _parse_arguments(self) -> list[Expression]:
        args: list[Expression] = []
        self._skip_newlines()
        if self._match(")"):
            return args
        while True:
            args.append(self._parse_expression())
            self._skip_newlines()
            if self._match(")"):
                return args
            self._expect(",")
            self._skip_newlines()

    def _parse_list(self) -> ListLiteral:
        self._expect("[")
        items: list[Expression] = []
        if self._match("]"):
            return ListLiteral(tuple(items))
        while True:
            items.append(self._parse_expression())
            if self._match("]"):
                return ListLiteral(tuple(items))
            self._expect(",")

    def _parse_dict(self) -> DictLiteral:
        self._expect("{")
        items: list[tuple[str, Expression]] = []
        if self._match("}"):
            return DictLiteral(tuple(items))
        while True:
            key = self._expect("STRING").value
            self._expect(":")
            value = self._parse_expression()
            items.append((key, value))
            if self._match("}"):
                return DictLiteral(tuple(items))
            self._expect(",")

    def _consume_statement_separator(self) -> None:
        if self._match("NEWLINE"):
            self._skip_newlines()

    def _skip_newlines(self) -> None:
        while self._match("NEWLINE"):
            continue

    def _current(self) -> Token:
        return self._tokens[self._index]

    def _peek(self) -> Token:
        if self._index + 1 >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[self._index + 1]

    def _advance(self) -> Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _expect(self, kind: str) -> Token:
        token = self._current()
        if token.kind != kind:
            msg = (
                f"Expected {kind}, got {token.kind} at line {token.line}, "
                f"column {token.column}"
            )
            raise ParseError(msg)
        return self._advance()

    def _match(self, kind: str) -> bool:
        if self._current().kind != kind:
            return False
        self._advance()
        return True


class Interpreter:
    """Evaluate programs written in the mini REPL language."""

    def __init__(
        self,
        *,
        functions: Mapping[str, Callable[..., Any]] | None = None,
        env: MutableMapping[str, Any] | None = None,
        foreign_interfaces: Sequence[ForeignObjectInterface] = (),
        max_workers: int | None = None,
    ) -> None:
        """Initialize the interpreter with optional foreign functions."""
        self._functions = dict(functions or {})
        self._foreign_interfaces = foreign_interfaces
        self._env: MutableMapping[str, Any] = env if env is not None else {}
        self._printed_lines: list[str] = []
        self._max_workers = max_workers

    @property
    def env(self) -> dict[str, Any]:
        """Return a copy of the current variable bindings."""
        return dict(self._env)

    @property
    def printed_lines(self) -> list[str]:
        """Return the captured output lines."""
        return list(self._printed_lines)

    def evaluate(
        self,
        source: str,
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        """Parse and evaluate REPL source code."""
        program = self.parse(source)
        return self._eval_program(program, self._env, print_callback=print_callback)

    def parse(self, source: str) -> Program:
        """Parse REPL source code into a program object."""
        tokens = _Tokenizer(source).tokenize()
        return _Parser(tokens).parse()

    def _eval_program(
        self,
        program: Program,
        env: dict[str, Any],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        result: Any = None
        for statement in program.statements:
            result = self._eval_statement(statement, env, print_callback=print_callback)
        return result

    def _eval_block(
        self,
        statements: Iterable[Statement],
        env: dict[str, Any],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        result: Any = None
        for statement in statements:
            result = self._eval_statement(statement, env, print_callback=print_callback)
        return result

    def _eval_statement(
        self,
        statement: Statement,
        env: dict[str, Any],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if isinstance(statement, Assign):
            value = self._eval_expression(
                statement.value,
                env,
                print_callback=print_callback,
            )
            env[statement.name] = value
            return value
        if isinstance(statement, IfStatement):
            condition = self._eval_expression(
                statement.condition,
                env,
                print_callback=print_callback,
            )
            branch = (
                statement.then_body
                if self._is_truthy(condition)
                else statement.else_body
            )
            return self._eval_block(branch, env, print_callback=print_callback)
        if isinstance(statement, ForStatement):
            result: Any = None
            iterable = self._eval_expression(
                statement.iterable,
                env,
                print_callback=print_callback,
            )
            if not isinstance(iterable, list):
                msg = "for loops require a list iterable"
                raise TypeError(msg)
            for item in iterable:
                env[statement.name] = item
                result = self._eval_block(
                    statement.body,
                    env,
                    print_callback=print_callback,
                )
            return result
        if isinstance(statement, ExpressionStatement):
            return self._eval_expression(
                statement.expression,
                env,
                print_callback=print_callback,
            )
        msg = f"Unsupported statement: {type(statement).__name__}"
        raise ValueError(msg)

    def _eval_expression(
        self,
        expression: Expression,
        env: dict[str, Any],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if isinstance(expression, Literal):
            return expression.value
        if isinstance(expression, Name):
            if expression.value in env:
                return env[expression.value]
            if expression.value in self._functions:
                return self._functions[expression.value]
            msg = f"Unknown name: {expression.value}"
            raise NameError(msg)
        if isinstance(expression, ListLiteral):
            return [
                self._eval_expression(item, env, print_callback=print_callback)
                for item in expression.items
            ]
        if isinstance(expression, DictLiteral):
            return {
                key: self._eval_expression(value, env, print_callback=print_callback)
                for key, value in expression.items
            }
        if isinstance(expression, BinaryOperation):
            left = self._eval_expression(
                expression.left,
                env,
                print_callback=print_callback,
            )
            right = self._eval_expression(
                expression.right,
                env,
                print_callback=print_callback,
            )
            return self._eval_binary_operation(left, expression.operator, right)
        if isinstance(expression, Index):
            target = self._eval_expression(
                expression.target,
                env,
                print_callback=print_callback,
            )
            index = self._eval_expression(
                expression.index,
                env,
                print_callback=print_callback,
            )
            return self._eval_index(target, index)
        if isinstance(expression, Attribute):
            target = self._eval_expression(
                expression.target,
                env,
                print_callback=print_callback,
            )
            return self._resolve_member(target, expression.name)
        if isinstance(expression, Call):
            return self._eval_call(expression, env, print_callback=print_callback)
        msg = f"Unsupported expression: {type(expression).__name__}"
        raise ValueError(msg)

    def _eval_call(
        self,
        expression: Call,
        env: dict[str, Any],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if isinstance(expression.target, Name):
            if expression.target.value == "print":
                return self._eval_print(
                    expression.args,
                    env,
                    print_callback=print_callback,
                )
            if expression.target.value == "parallel":
                return self._eval_parallel(
                    expression.args,
                    env,
                    print_callback=print_callback,
                )
            if expression.target.value == "try":
                return self._eval_try(
                    expression.args,
                    env,
                    print_callback=print_callback,
                )
        target = self._eval_expression(
            expression.target,
            env,
            print_callback=print_callback,
        )
        args = tuple(
            self._eval_expression(arg, env, print_callback=print_callback)
            for arg in expression.args
        )
        if callable(target):
            return target(*args)
        handler = self._handler_for(target)
        return handler.call(target, args)

    def _eval_binary_operation(self, left: Any, operator: str, right: Any) -> Any:
        if isinstance(left, bool) or isinstance(right, bool):
            msg = "binary operations require numeric operands"
            raise TypeError(msg)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            msg = "binary operations require numeric operands"
            raise TypeError(msg)
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        msg = f"Unsupported binary operator: {operator}"
        raise ValueError(msg)

    def _eval_index(self, target: Any, index: Any) -> Any:
        if isinstance(target, list):
            if not isinstance(index, int):
                msg = "list indexes must be integers"
                raise TypeError(msg)
            return target[index]
        if isinstance(target, dict):
            if not isinstance(index, str):
                msg = "dict indexes must be strings"
                raise TypeError(msg)
            return target[index]
        handler = self._handler_for(target)
        return handler.get_item(target, index)

    def _resolve_member(self, target: Any, name: str) -> Any:
        handler = self._handler_for(target)
        return handler.resolve_member(target, name)

    def _handler_for(self, value: Any) -> ForeignObjectInterface:
        for handler in self._foreign_interfaces:
            if handler.supports(value):
                return handler
        msg = f"No foreign object handler for {type(value).__name__}"
        raise TypeError(msg)

    def _eval_print(
        self,
        args: tuple[Expression, ...],
        env: dict[str, Any],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if len(args) != 1:
            msg = "print expects exactly one argument"
            raise ValueError(msg)
        value = self._eval_expression(args[0], env, print_callback=print_callback)
        formatted = self._format_value(value)
        self._printed_lines.append(formatted)
        if print_callback is not None:
            print_callback(formatted)
        return value

    def _eval_parallel(
        self,
        args: tuple[Expression, ...],
        env: dict[str, Any],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> list[Any]:
        snapshots = [dict(env) for _ in args]
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [
                executor.submit(
                    self._eval_expression,
                    arg,
                    snapshot,
                    print_callback=print_callback,
                )
                for arg, snapshot in zip(args, snapshots, strict=False)
            ]
            return [future.result() for future in futures]

    def _eval_try(
        self,
        args: tuple[Expression, ...],
        env: dict[str, Any],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if len(args) != 2:  # noqa: PLR2004  # try(expr, fallback) always takes two args
            msg = "try expects exactly two arguments"
            raise ValueError(msg)
        try:
            return self._eval_expression(args[0], env, print_callback=print_callback)
        except Exception:  # noqa: BLE001
            return self._eval_expression(args[1], env, print_callback=print_callback)

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "None"
        if value is True:
            return "True"
        if value is False:
            return "False"
        return str(value)

    def _is_truthy(self, value: Any) -> bool:
        return bool(value)


__all__ = ["ForeignObjectInterface", "Interpreter", "ParseError", "Program"]
