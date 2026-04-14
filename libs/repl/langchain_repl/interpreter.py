"""Mini REPL interpreter and parser implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.tools import BaseTool
from langchain_core.tools.base import (
    _is_injected_arg_type,
    get_all_basemodel_annotations,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence

    from langchain.tools import ToolRuntime


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
    """Protocol for dispatching operations on foreign objects.

    Currently limited to sync invocation only.
    """

    def supports(self, value: Any) -> bool:
        """Return whether this handler manages the provided runtime value."""

    def get_item(self, value: Any, key: Any) -> Any:
        """Resolve `value[key]` for a supported foreign object."""

    def resolve_member(self, value: Any, name: str) -> Any:
        """Resolve `value.name` for a supported foreign object."""

    def call(self, value: Any, args: tuple[Any, ...]) -> Any:
        """Invoke `value(*args)` for a supported foreign object."""


def _get_injected_arg_names(tool: BaseTool) -> set[str]:
    """Return injected parameter names for a tool input schema."""
    return {
        name
        for name, type_ in get_all_basemodel_annotations(
            tool.get_input_schema()
        ).items()
        if _is_injected_arg_type(type_)
    }


def _get_runtime_arg_name(tool: BaseTool) -> str | None:
    """Return the injected runtime parameter name for a tool, if any."""
    if "runtime" in _get_injected_arg_names(tool):
        return "runtime"
    return None


def _filter_injected_kwargs(
    tool: BaseTool,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Drop model-controlled injected args from a tool payload."""
    injected_arg_names = _get_injected_arg_names(tool)
    return {
        name: value for name, value in payload.items() if name not in injected_arg_names
    }


def _build_tool_payload(
    tool: BaseTool,
    args: tuple[Any, ...],
    *,
    runtime: ToolRuntime | None = None,
) -> str | dict[str, Any]:
    """Convert REPL call arguments into a LangChain tool payload."""
    input_schema = tool.get_input_schema()
    schema_annotations = getattr(input_schema, "__annotations__", {})
    fields = [
        name
        for name, type_ in schema_annotations.items()
        if not _is_injected_arg_type(type_)
    ]
    runtime_arg_name = _get_runtime_arg_name(tool)

    if len(args) == 1 and isinstance(args[0], dict):
        payload = _filter_injected_kwargs(tool, args[0])
    elif len(args) == 1 and isinstance(args[0], str) and runtime_arg_name is None:
        payload = args[0]
    elif len(args) == 1 and len(fields) == 1:
        payload = {fields[0]: args[0]}
    elif len(args) == len(fields) and fields:
        payload = dict(zip(fields, args, strict=False))
    else:
        payload = {"args": list(args)}

    if (
        runtime is not None
        and runtime_arg_name is not None
        and isinstance(payload, dict)
    ):
        return {**payload, runtime_arg_name: runtime}
    return payload


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
                    left=expr, operator="+", right=self._parse_postfix()
                )
                continue
            if self._match("-"):
                expr = BinaryOperation(
                    left=expr, operator="-", right=self._parse_postfix()
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
            return Literal(True)
        if token.kind == "FALSE":
            self._advance()
            return Literal(False)
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
            msg = f"Expected {kind}, got {token.kind} at line {token.line}, column {token.column}"
            raise ParseError(msg)
        return self._advance()

    def _match(self, kind: str) -> bool:
        if self._current().kind != kind:
            return False
        self._advance()
        return True


class OpCode(str, Enum):
    LOAD_CONST = "LOAD_CONST"
    LOAD_NAME = "LOAD_NAME"
    STORE_NAME = "STORE_NAME"
    SET_LAST = "SET_LAST"
    BUILD_LIST = "BUILD_LIST"
    BUILD_DICT = "BUILD_DICT"
    BINARY_OP = "BINARY_OP"
    GET_INDEX = "GET_INDEX"
    GET_ATTR = "GET_ATTR"
    CALL = "CALL"
    TRY_CALL = "TRY_CALL"
    JUMP = "JUMP"
    JUMP_IF_FALSE = "JUMP_IF_FALSE"
    ITER_PREP = "ITER_PREP"
    ITER_NEXT = "ITER_NEXT"
    RETURN_VALUE = "RETURN_VALUE"


@dataclass(frozen=True)
class Instruction:
    opcode: OpCode
    arg: Any = None


@dataclass
class ForLoopState:
    target_name: str
    items: list[Any]
    index: int = 0


@dataclass
class VMState:
    instructions: tuple[Instruction, ...]
    env: MutableMapping[str, Any]
    pc: int = 0
    stack: list[Any] = field(default_factory=list)
    last_value: Any = None
    loop_stack: list[ForLoopState] = field(default_factory=list)


class _Compiler:
    def compile_program(self, program: Program) -> tuple[Instruction, ...]:
        instructions: list[Instruction] = []
        for statement in program.statements:
            self._compile_statement(statement, instructions)
        instructions.append(Instruction(OpCode.RETURN_VALUE))
        return tuple(instructions)

    def _compile_block(
        self, statements: Iterable[Statement], instructions: list[Instruction]
    ) -> None:
        for statement in statements:
            self._compile_statement(statement, instructions)

    def _compile_statement(
        self, statement: Statement, instructions: list[Instruction]
    ) -> None:
        if isinstance(statement, Assign):
            self._compile_expression(statement.value, instructions)
            instructions.append(Instruction(OpCode.STORE_NAME, statement.name))
            instructions.append(Instruction(OpCode.SET_LAST))
            return
        if isinstance(statement, ExpressionStatement):
            self._compile_expression(statement.expression, instructions)
            instructions.append(Instruction(OpCode.SET_LAST))
            return
        if isinstance(statement, IfStatement):
            self._compile_expression(statement.condition, instructions)
            jump_if_false_index = len(instructions)
            instructions.append(Instruction(OpCode.JUMP_IF_FALSE, None))
            self._compile_block(statement.then_body, instructions)
            jump_end_index = len(instructions)
            instructions.append(Instruction(OpCode.JUMP, None))
            else_start = len(instructions)
            self._compile_block(statement.else_body, instructions)
            end_index = len(instructions)
            instructions[jump_if_false_index] = Instruction(
                OpCode.JUMP_IF_FALSE, else_start
            )
            instructions[jump_end_index] = Instruction(OpCode.JUMP, end_index)
            return
        if isinstance(statement, ForStatement):
            self._compile_expression(statement.iterable, instructions)
            instructions.append(Instruction(OpCode.ITER_PREP, statement.name))
            loop_start = len(instructions)
            iter_next_index = len(instructions)
            instructions.append(Instruction(OpCode.ITER_NEXT, None))
            self._compile_block(statement.body, instructions)
            instructions.append(Instruction(OpCode.JUMP, loop_start))
            loop_end = len(instructions)
            instructions[iter_next_index] = Instruction(OpCode.ITER_NEXT, loop_end)
            return
        msg = f"Unsupported statement: {type(statement).__name__}"
        raise ValueError(msg)

    def _compile_expression(
        self, expression: Expression, instructions: list[Instruction]
    ) -> None:
        if isinstance(expression, Literal):
            instructions.append(Instruction(OpCode.LOAD_CONST, expression.value))
            return
        if isinstance(expression, Name):
            if expression.value == "print":
                instructions.append(Instruction(OpCode.LOAD_CONST, _PRINT_SENTINEL))
            elif expression.value == "parallel":
                instructions.append(Instruction(OpCode.LOAD_CONST, _PARALLEL_SENTINEL))
            elif expression.value == "try":
                instructions.append(Instruction(OpCode.LOAD_CONST, _TRY_SENTINEL))
            else:
                instructions.append(Instruction(OpCode.LOAD_NAME, expression.value))
            return
        if isinstance(expression, ListLiteral):
            for item in expression.items:
                self._compile_expression(item, instructions)
            instructions.append(Instruction(OpCode.BUILD_LIST, len(expression.items)))
            return
        if isinstance(expression, DictLiteral):
            for key, value in expression.items:
                instructions.append(Instruction(OpCode.LOAD_CONST, key))
                self._compile_expression(value, instructions)
            instructions.append(Instruction(OpCode.BUILD_DICT, len(expression.items)))
            return
        if isinstance(expression, BinaryOperation):
            self._compile_expression(expression.left, instructions)
            self._compile_expression(expression.right, instructions)
            instructions.append(Instruction(OpCode.BINARY_OP, expression.operator))
            return
        if isinstance(expression, Index):
            self._compile_expression(expression.target, instructions)
            self._compile_expression(expression.index, instructions)
            instructions.append(Instruction(OpCode.GET_INDEX))
            return
        if isinstance(expression, Attribute):
            self._compile_expression(expression.target, instructions)
            instructions.append(Instruction(OpCode.GET_ATTR, expression.name))
            return
        if isinstance(expression, Call):
            if isinstance(expression.target, Name) and expression.target.value == "try":
                instructions.append(Instruction(OpCode.TRY_CALL, expression))
                return
            self._compile_expression(expression.target, instructions)
            for arg in expression.args:
                self._compile_expression(arg, instructions)
            instructions.append(Instruction(OpCode.CALL, len(expression.args)))
            return
        msg = f"Unsupported expression: {type(expression).__name__}"
        raise ValueError(msg)


class Interpreter:
    def __init__(
        self,
        *,
        functions: Mapping[str, Callable[..., Any] | BaseTool] | None = None,
        env: MutableMapping[str, Any] | None = None,
        foreign_interfaces: Sequence[ForeignObjectInterface] = (),
        max_workers: int | None = None,
        runtime: ToolRuntime | None = None,
    ) -> None:
        self._functions = dict(functions or {})
        self._foreign_interfaces = tuple(foreign_interfaces)
        self._env: MutableMapping[str, Any] = env if env is not None else {}
        self._printed_lines: list[str] = []
        self._max_workers = max_workers
        self._runtime = runtime
        self._compiler = _Compiler()

    @property
    def env(self) -> dict[str, Any]:
        return dict(self._env)

    @property
    def printed_lines(self) -> list[str]:
        return list(self._printed_lines)

    def parse(self, source: str) -> Program:
        return _Parser(_Tokenizer(source).tokenize()).parse()

    def evaluate(
        self, source: str, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        return self.evaluate_program(self.parse(source), print_callback=print_callback)

    async def aevaluate(
        self, source: str, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        return await self.aevaluate_program(
            self.parse(source), print_callback=print_callback
        )

    def evaluate_program(
        self,
        program: Program,
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        state = self._new_state(program, self._env)
        value = self._run_vm_sync(state, print_callback=print_callback)
        self._env = state.env
        return value

    async def aevaluate_program(
        self,
        program: Program,
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        state = self._new_state(program, self._env)
        value = await self._run_vm_async(state, print_callback=print_callback)
        self._env = state.env
        return value

    def _new_state(self, program: Program, env: MutableMapping[str, Any]) -> VMState:
        return VMState(instructions=self._compiler.compile_program(program), env=env)

    def _run_vm_sync(
        self, state: VMState, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        while state.pc < len(state.instructions):
            instruction = state.instructions[state.pc]
            state.pc += 1
            result = self._step_sync(state, instruction, print_callback=print_callback)
            if result is not _NO_RESULT:
                return result
        return state.last_value

    async def _run_vm_async(
        self, state: VMState, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        while state.pc < len(state.instructions):
            instruction = state.instructions[state.pc]
            state.pc += 1
            result = await self._step_async(
                state, instruction, print_callback=print_callback
            )
            if result is not _NO_RESULT:
                return result
        return state.last_value

    def _step_sync(
        self,
        state: VMState,
        instruction: Instruction,
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if instruction.opcode is OpCode.LOAD_CONST:
            state.stack.append(instruction.arg)
            return _NO_RESULT
        if instruction.opcode is OpCode.LOAD_NAME:
            state.stack.append(self._load_name(state.env, instruction.arg))
            return _NO_RESULT
        if instruction.opcode is OpCode.STORE_NAME:
            value = state.stack[-1]
            state.env[instruction.arg] = value
            return _NO_RESULT
        if instruction.opcode is OpCode.SET_LAST:
            state.last_value = state.stack[-1] if state.stack else None
            return _NO_RESULT
        if instruction.opcode is OpCode.BUILD_LIST:
            count = int(instruction.arg)
            items = state.stack[-count:] if count else []
            if count:
                del state.stack[-count:]
            state.stack.append(list(items))
            return _NO_RESULT
        if instruction.opcode is OpCode.BUILD_DICT:
            count = int(instruction.arg)
            items = state.stack[-(2 * count) :] if count else []
            if count:
                del state.stack[-(2 * count) :]
            built: dict[str, Any] = {}
            for index in range(0, len(items), 2):
                built[str(items[index])] = items[index + 1]
            state.stack.append(built)
            return _NO_RESULT
        if instruction.opcode is OpCode.BINARY_OP:
            right = state.stack.pop()
            left = state.stack.pop()
            state.stack.append(
                self._eval_binary_operation(left, instruction.arg, right)
            )
            return _NO_RESULT
        if instruction.opcode is OpCode.GET_INDEX:
            index = state.stack.pop()
            target = state.stack.pop()
            state.stack.append(self._eval_index(target, index))
            return _NO_RESULT
        if instruction.opcode is OpCode.GET_ATTR:
            target = state.stack.pop()
            state.stack.append(self._resolve_member(target, instruction.arg))
            return _NO_RESULT
        if instruction.opcode is OpCode.CALL:
            arg_count = int(instruction.arg)
            target_index = len(state.stack) - arg_count - 1
            target = state.stack[target_index]
            args = tuple(state.stack[target_index + 1 :])
            del state.stack[target_index:]
            state.stack.append(
                self._call_sync(target, args, print_callback=print_callback)
            )
            return _NO_RESULT
        if instruction.opcode is OpCode.TRY_CALL:
            state.stack.append(
                self._eval_try_call_sync(instruction.arg, print_callback=print_callback)
            )
            return _NO_RESULT
        if instruction.opcode is OpCode.JUMP:
            state.pc = int(instruction.arg)
            return _NO_RESULT
        if instruction.opcode is OpCode.JUMP_IF_FALSE:
            value = state.stack.pop()
            if not self._is_truthy(value):
                state.pc = int(instruction.arg)
            return _NO_RESULT
        if instruction.opcode is OpCode.ITER_PREP:
            iterable = state.stack.pop()
            if not isinstance(iterable, list):
                msg = "for loops require a list iterable"
                raise TypeError(msg)
            state.loop_stack.append(ForLoopState(instruction.arg, iterable))
            return _NO_RESULT
        if instruction.opcode is OpCode.ITER_NEXT:
            loop_state = state.loop_stack[-1]
            if loop_state.index >= len(loop_state.items):
                state.loop_stack.pop()
                state.pc = int(instruction.arg)
                return _NO_RESULT
            item = loop_state.items[loop_state.index]
            loop_state.index += 1
            state.env[loop_state.target_name] = item
            return _NO_RESULT
        if instruction.opcode is OpCode.RETURN_VALUE:
            return state.last_value
        msg = f"Unsupported opcode: {instruction.opcode}"
        raise ValueError(msg)

    async def _step_async(
        self,
        state: VMState,
        instruction: Instruction,
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if instruction.opcode is OpCode.TRY_CALL:
            state.stack.append(
                await self._eval_try_call_async(
                    instruction.arg, print_callback=print_callback
                )
            )
            return _NO_RESULT
        if instruction.opcode is not OpCode.CALL:
            return self._step_sync(state, instruction, print_callback=print_callback)
        arg_count = int(instruction.arg)
        target_index = len(state.stack) - arg_count - 1
        target = state.stack[target_index]
        args = tuple(state.stack[target_index + 1 :])
        del state.stack[target_index:]
        state.stack.append(
            await self._call_async(target, args, print_callback=print_callback)
        )
        return _NO_RESULT

    def _load_name(self, env: MutableMapping[str, Any], name: str) -> Any:
        if name in env:
            return env[name]
        if name in self._functions:
            return self._functions[name]
        msg = f"Unknown name: {name}"
        raise NameError(msg)

    def _call_sync(
        self,
        target: Any,
        args: tuple[Any, ...],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if target is _PRINT_SENTINEL:
            return self._eval_print(args, print_callback=print_callback)
        if target is _PARALLEL_SENTINEL:
            return self._eval_parallel_sync(args)
        if target is _TRY_SENTINEL:
            return self._eval_try(args)
        if isinstance(target, BaseTool):
            return target.invoke(
                _build_tool_payload(target, args, runtime=self._runtime)
            )
        if callable(target):
            result = target(*args)
            if asyncio.iscoroutine(result):
                msg = "Async call encountered in synchronous interpreter"
                raise TypeError(msg)
            return result
        return self._handler_for(target).call(target, args)

    async def _call_async(
        self,
        target: Any,
        args: tuple[Any, ...],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if target is _PRINT_SENTINEL:
            return self._eval_print(args, print_callback=print_callback)
        if target is _PARALLEL_SENTINEL:
            return await self._eval_parallel_async(args)
        if target is _TRY_SENTINEL:
            return self._eval_try(args)
        if isinstance(target, BaseTool):
            payload = _build_tool_payload(target, args, runtime=self._runtime)
            if getattr(target, "coroutine", None) is not None:
                return await target.ainvoke(payload)
            return target.invoke(payload)
        if callable(target):
            result = target(*args)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return self._handler_for(target).call(target, args)

    def _eval_print(
        self,
        args: tuple[Any, ...],
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if len(args) != 1:
            msg = "print expects exactly one argument"
            raise ValueError(msg)
        value = args[0]
        formatted = self._format_value(value)
        self._printed_lines.append(formatted)
        if print_callback is not None:
            print_callback(formatted)
        return value

    def _eval_parallel_sync(self, args: tuple[Any, ...]) -> list[Any]:
        return list(args)

    async def _eval_parallel_async(self, args: tuple[Any, ...]) -> list[Any]:
        return list(args)

    def _eval_try(self, args: tuple[Any, ...]) -> Any:
        if len(args) != 2:
            msg = "try expects exactly two arguments"
            raise ValueError(msg)
        return args[0]

    def _eval_try_call_sync(
        self,
        expression: Call,
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if len(expression.args) != 2:  # noqa: PLR2004  # try(expr, fallback) always takes two args
            msg = "try expects exactly two arguments"
            raise ValueError(msg)
        try:
            return self.evaluate_program(
                Program((ExpressionStatement(expression.args[0]),)),
                print_callback=print_callback,
            )
        except Exception:  # noqa: BLE001
            return self.evaluate_program(
                Program((ExpressionStatement(expression.args[1]),)),
                print_callback=print_callback,
            )

    async def _eval_try_call_async(
        self,
        expression: Call,
        *,
        print_callback: Callable[[str], None] | None = None,
    ) -> Any:
        if len(expression.args) != 2:  # noqa: PLR2004  # try(expr, fallback) always takes two args
            msg = "try expects exactly two arguments"
            raise ValueError(msg)
        try:
            return await self.aevaluate_program(
                Program((ExpressionStatement(expression.args[0]),)),
                print_callback=print_callback,
            )
        except Exception:  # noqa: BLE001
            return await self.aevaluate_program(
                Program((ExpressionStatement(expression.args[1]),)),
                print_callback=print_callback,
            )

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
        if self._has_handler(target):
            return self._handler_for(target).get_item(target, index)
        msg = f"'{type(target).__name__}' object is not subscriptable"
        raise TypeError(msg)

    def _resolve_member(self, target: Any, name: str) -> Any:
        return self._handler_for(target).resolve_member(target, name)

    def _has_handler(self, value: Any) -> bool:
        return any(handler.supports(value) for handler in self._foreign_interfaces)

    def _handler_for(self, value: Any) -> ForeignObjectInterface:
        for handler in self._foreign_interfaces:
            if handler.supports(value):
                return handler
        msg = f"No foreign object handler for {type(value).__name__}"
        raise TypeError(msg)

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


_PRINT_SENTINEL = object()
_PARALLEL_SENTINEL = object()
_TRY_SENTINEL = object()
_NO_RESULT = object()


__all__ = ["ForeignObjectInterface", "Interpreter", "ParseError", "Program"]
