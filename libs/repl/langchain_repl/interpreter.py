"""Mini REPL interpreter and parser implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.tools import BaseTool
from langchain_core.tools.base import (
    _is_injected_arg_type,
    get_all_basemodel_annotations,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping, Sequence

    from langchain.tools import ToolRuntime


@dataclass(frozen=True, slots=True)
class Token:
    """A lexical token produced by the tokenizer."""

    kind: str
    value: Any
    line: int
    column: int


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
        append_token = tokens.append
        while self._index < self._length:
            char = self._source[self._index]
            if char in " \t\r":
                self._advance()
                continue
            if char == "\n":
                append_token(Token("NEWLINE", "\n", self._line, self._column))
                self._advance()
                continue
            if char == "#":
                self._skip_comment()
                continue
            if char in "()+-[]{}:,.=":
                append_token(Token(char, char, self._line, self._column))
                self._advance()
                continue
            if char == '"':
                append_token(self._read_string())
                continue
            if char.isdigit() or (char == "-" and self._peek().isdigit()):
                append_token(self._read_number())
                continue
            if char.isalpha() or char == "_":
                append_token(self._read_name())
                continue
            msg = (
                f"Unexpected character {char!r} at line {self._line}, "
                f"column {self._column}"
            )
            raise ParseError(msg)
        append_token(Token("EOF", None, self._line, self._column))
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
        append_char = chars.append
        while self._index < self._length:
            char = self._advance()
            if char == '"':
                return Token("STRING", "".join(chars), line, column)
            if char == "\\":
                if self._index >= self._length:
                    break
                append_char(self._decode_escape(self._advance()))
                continue
            append_char(char)
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
        append_char = chars.append
        while self._index < self._length:
            char = self._source[self._index]
            if char.isdigit():
                append_char(self._advance())
                continue
            if char == "." and not has_dot:
                has_dot = True
                append_char(self._advance())
                continue
            break
        text = "".join(chars)
        value: int | float = float(text) if has_dot else int(text)
        return Token("NUMBER", value, line, column)

    def _read_name(self) -> Token:
        line = self._line
        column = self._column
        chars = [self._advance()]
        append_char = chars.append
        while self._index < self._length:
            char = self._source[self._index]
            if char.isalnum() or char == "_":
                append_char(self._advance())
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


class OpCode(IntEnum):
    LOAD_CONST = 0
    LOAD_NAME = 1
    STORE_NAME = 2
    SET_LAST = 3
    BUILD_LIST = 4
    BUILD_DICT = 5
    BINARY_OP = 6
    GET_INDEX = 7
    GET_ATTR = 8
    CALL = 9
    JUMP = 10
    JUMP_IF_FALSE = 11
    ITER_PREP = 12
    ITER_NEXT = 13
    RETURN_VALUE = 14


@dataclass(frozen=True, slots=True)
class Instruction:
    opcode: OpCode
    arg: Any = None


@dataclass(slots=True)
class ForLoopState:
    """Mutable runtime state for one active `for` loop.

    The VM keeps the loop variable name, the concrete list being iterated, and
    the next element index to read.
    """

    target_name: str
    items: list[Any]
    index: int = 0


@dataclass(slots=True)
class VMState:
    """Mutable execution state for one compiled REPL program.

    Attributes:
        instructions: The compiled instruction stream being executed.
        globals: Current mutable bindings visible to the program.
        pc: Program counter pointing at the next instruction to execute.
        stack: Operand stack used by the VM for expression evaluation.
        last_value: Value returned by the most recent statement-level result.
        loop_stack: Stack of active `for` loops.

    `loop_stack` exists because loops need state that cannot live on the operand
    stack alone: the loop target name, the full iterable, and the current index.
    It is a stack rather than a single slot so nested loops work correctly.

    Example:
        In:
        `for item in [1, 2] do`
        `    for inner in [10, 20] do`
        `        print(item)`
        `    end`
        `end`

        the outer and inner loops each need independent iteration state, so the
        VM pushes one `ForLoopState` for the outer loop and another for the inner
        loop.
    """

    instructions: Sequence[Instruction]
    globals: MutableMapping[str, Any]
    pc: int = 0
    stack: list[Any] = field(default_factory=list)
    last_value: Any = None
    loop_stack: list[ForLoopState] = field(default_factory=list)


class _ProgramCompiler:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._index = 0
        self._instructions: list[Instruction] = []

    def compile(self) -> tuple[Instruction, ...]:
        self._compile_block(stop_kinds={"EOF"})
        self._expect("EOF")
        self._emit(OpCode.RETURN_VALUE)
        return tuple(self._instructions)

    def _compile_block(self, *, stop_kinds: set[str]) -> None:
        self._skip_newlines()
        while self._current().kind not in stop_kinds:
            self._compile_statement()
            self._skip_newlines()

    def _compile_statement(self) -> None:
        token = self._current()
        if token.kind == "IF":
            self._compile_if()
            return
        if token.kind == "FOR":
            self._compile_for()
            return
        if token.kind == "NAME" and self._peek().kind == "=":
            name = str(self._advance().value)
            self._expect("=")
            self._compile_expression()
            self._emit(OpCode.STORE_NAME, name)
            self._emit(OpCode.SET_LAST)
            return
        self._compile_expression()
        self._emit(OpCode.SET_LAST)

    def _compile_if(self) -> None:
        self._expect("IF")
        self._compile_expression()
        jump_if_false_index = self._emit(OpCode.JUMP_IF_FALSE, None)
        self._expect("THEN")
        self._consume_statement_separator()
        self._compile_block(stop_kinds={"ELSE", "END"})
        jump_end_index = self._emit(OpCode.JUMP, None)
        else_start = len(self._instructions)
        if self._match("ELSE"):
            self._consume_statement_separator()
            self._compile_block(stop_kinds={"END"})
        end_index = len(self._instructions)
        self._patch(jump_if_false_index, else_start)
        self._patch(jump_end_index, end_index)
        self._expect("END")

    def _compile_for(self) -> None:
        self._expect("FOR")
        name = str(self._expect("NAME").value)
        self._expect("IN")
        self._compile_expression()
        self._emit(OpCode.ITER_PREP, name)
        self._expect("DO")
        self._consume_statement_separator()
        loop_start = len(self._instructions)
        iter_next_index = self._emit(OpCode.ITER_NEXT, None)
        self._compile_block(stop_kinds={"END"})
        self._emit(OpCode.JUMP, loop_start)
        self._patch(iter_next_index, len(self._instructions))
        self._expect("END")

    def _compile_expression(self) -> None:
        self._compile_postfix()
        while True:
            if self._match("+"):
                self._compile_postfix()
                self._emit(OpCode.BINARY_OP, "+")
                continue
            if self._match("-"):
                self._compile_postfix()
                self._emit(OpCode.BINARY_OP, "-")
                continue
            break

    def _compile_postfix(self) -> None:
        self._compile_primary()
        while True:
            if self._match("["):
                self._compile_expression()
                self._expect("]")
                self._emit(OpCode.GET_INDEX)
                continue
            if self._match("."):
                self._emit(OpCode.GET_ATTR, str(self._expect("NAME").value))
                continue
            if self._match("("):
                arg_count = self._compile_arguments()
                self._emit(OpCode.CALL, arg_count)
                continue
            break

    def _compile_primary(self) -> None:  # noqa: PLR0911
        token = self._current()
        if token.kind == "NAME":
            value = str(self._advance().value)
            if value == "print":
                self._emit(OpCode.LOAD_CONST, _PRINT_SENTINEL)
            elif value == "parallel":
                self._emit(OpCode.LOAD_CONST, _PARALLEL_SENTINEL)
            else:
                self._emit(OpCode.LOAD_NAME, value)
            return
        if token.kind == "NUMBER":
            self._emit(OpCode.LOAD_CONST, self._advance().value)
            return
        if token.kind == "STRING":
            self._emit(OpCode.LOAD_CONST, self._advance().value)
            return
        if token.kind == "TRUE":
            self._advance()
            self._emit(OpCode.LOAD_CONST, True)
            return
        if token.kind == "FALSE":
            self._advance()
            self._emit(OpCode.LOAD_CONST, False)
            return
        if token.kind == "NONE":
            self._advance()
            self._emit(OpCode.LOAD_CONST, None)
            return
        if token.kind == "[":
            self._compile_list()
            return
        if token.kind == "{":
            self._compile_dict()
            return
        if token.kind == "(":
            self._advance()
            self._compile_expression()
            self._expect(")")
            return
        msg = (
            f"Unexpected token {token.kind} at line {token.line}, column {token.column}"
        )
        raise ParseError(msg)

    def _compile_arguments(self) -> int:
        count = 0
        self._skip_newlines()
        if self._match(")"):
            return count
        while True:
            self._compile_expression()
            count += 1
            self._skip_newlines()
            if self._match(")"):
                return count
            self._expect(",")
            self._skip_newlines()

    def _compile_list(self) -> None:
        self._expect("[")
        count = 0
        if self._match("]"):
            self._emit(OpCode.BUILD_LIST, 0)
            return
        while True:
            self._compile_expression()
            count += 1
            if self._match("]"):
                self._emit(OpCode.BUILD_LIST, count)
                return
            self._expect(",")

    def _compile_dict(self) -> None:
        self._expect("{")
        count = 0
        if self._match("}"):
            self._emit(OpCode.BUILD_DICT, 0)
            return
        while True:
            key = self._expect("STRING").value
            self._emit(OpCode.LOAD_CONST, key)
            self._expect(":")
            self._compile_expression()
            count += 1
            if self._match("}"):
                self._emit(OpCode.BUILD_DICT, count)
                return
            self._expect(",")

    def _emit(self, opcode: OpCode, arg: Any = None) -> int:
        self._instructions.append(Instruction(opcode, arg))
        return len(self._instructions) - 1

    def _patch(self, index: int, arg: Any) -> None:
        self._instructions[index] = Instruction(self._instructions[index].opcode, arg)

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


class Interpreter:
    def __init__(
        self,
        *,
        functions: Mapping[str, Callable[..., Any] | BaseTool] | None = None,
        globals: MutableMapping[str, Any] | None = None,
        bindings: Mapping[str, Any] | None = None,
        foreign_interfaces: Sequence[ForeignObjectInterface] = (),
        runtime: ToolRuntime | None = None,
    ) -> None:
        self._functions = dict(functions or {})
        self._bindings = dict(bindings or {})
        self._foreign_interfaces = tuple(foreign_interfaces)
        self._globals: MutableMapping[str, Any] = globals if globals is not None else {}
        self._printed_lines: list[str] = []
        self._runtime = runtime
        self._compiler = _ProgramCompiler

    @property
    def env(self) -> dict[str, Any]:
        return dict(self._globals)

    @property
    def globals(self) -> dict[str, Any]:
        return dict(self._globals)

    @property
    def bindings(self) -> dict[str, Any]:
        return dict(self._bindings)

    @property
    def printed_lines(self) -> list[str]:
        return list(self._printed_lines)

    def compile(self, source: str) -> tuple[Instruction, ...]:
        return self._compiler(_Tokenizer(source).tokenize()).compile()

    def evaluate(
        self, source: str, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        instructions = self.compile(source)
        state = self._new_state(instructions, self._globals)
        value = self._run_vm_sync(state, print_callback=print_callback)
        self._globals = state.globals
        return value

    async def aevaluate(
        self, source: str, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        instructions = self.compile(source)
        state = self._new_state(instructions, self._globals)
        value = await self._run_vm_async(state, print_callback=print_callback)
        self._globals = state.globals
        return value

    def _new_state(
        self, instructions: tuple[Instruction, ...], globals: MutableMapping[str, Any]
    ) -> VMState:
        return VMState(instructions=instructions, globals=globals)

    def _run_vm_sync(
        self, state: VMState, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        instructions = state.instructions
        stack = state.stack
        globals = state.globals
        loop_stack = state.loop_stack
        load_name = self._load_name
        eval_binary_operation = self._eval_binary_operation
        eval_index = self._eval_index
        resolve_member = self._resolve_member
        call_sync = self._call_sync
        store_name = self._store_name

        while state.pc < len(instructions):
            instruction = instructions[state.pc]
            opcode = instruction.opcode
            arg = instruction.arg
            state.pc += 1

            if opcode == OpCode.LOAD_CONST:
                stack.append(arg)
                continue
            if opcode == OpCode.LOAD_NAME:
                stack.append(load_name(globals, arg))
                continue
            if opcode == OpCode.STORE_NAME:
                store_name(globals, arg, stack[-1])
                continue
            if opcode == OpCode.SET_LAST:
                state.last_value = stack[-1] if stack else None
                continue
            if opcode == OpCode.BUILD_LIST:
                count = int(arg)
                items = stack[-count:] if count else []
                if count:
                    del stack[-count:]
                stack.append(list(items))
                continue
            if opcode == OpCode.BUILD_DICT:
                count = int(arg)
                item_count = 2 * count
                items = stack[-item_count:] if count else []
                if count:
                    del stack[-item_count:]
                built: dict[str, Any] = {}
                for index in range(0, len(items), 2):
                    built[str(items[index])] = items[index + 1]
                stack.append(built)
                continue
            if opcode == OpCode.BINARY_OP:
                right = stack.pop()
                left = stack.pop()
                stack.append(eval_binary_operation(left, arg, right))
                continue
            if opcode == OpCode.GET_INDEX:
                index = stack.pop()
                target = stack.pop()
                stack.append(eval_index(target, index))
                continue
            if opcode == OpCode.GET_ATTR:
                target = stack.pop()
                stack.append(resolve_member(target, arg))
                continue
            if opcode == OpCode.CALL:
                arg_count = int(arg)
                target_index = len(stack) - arg_count - 1
                target = stack[target_index]
                args = tuple(stack[target_index + 1 :])
                del stack[target_index:]
                stack.append(call_sync(target, args, print_callback=print_callback))
                continue
            if opcode == OpCode.JUMP:
                state.pc = int(arg)
                continue
            if opcode == OpCode.JUMP_IF_FALSE:
                if not stack.pop():
                    state.pc = int(arg)
                continue
            if opcode == OpCode.ITER_PREP:
                iterable = stack.pop()
                if not isinstance(iterable, list):
                    msg = "for loops require a list iterable"
                    raise TypeError(msg)
                loop_stack.append(ForLoopState(arg, iterable))
                continue
            if opcode == OpCode.ITER_NEXT:
                loop_state = loop_stack[-1]
                if loop_state.index >= len(loop_state.items):
                    loop_stack.pop()
                    state.pc = int(arg)
                    continue
                globals[loop_state.target_name] = loop_state.items[loop_state.index]
                loop_state.index += 1
                continue
            if opcode == OpCode.RETURN_VALUE:
                return state.last_value
            msg = f"Unsupported opcode: {opcode}"
            raise ValueError(msg)
        return state.last_value

    async def _run_vm_async(
        self, state: VMState, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        instructions = state.instructions
        stack = state.stack
        globals = state.globals
        loop_stack = state.loop_stack
        load_name = self._load_name
        eval_binary_operation = self._eval_binary_operation
        eval_index = self._eval_index
        resolve_member = self._resolve_member
        call_async = self._call_async
        store_name = self._store_name

        while state.pc < len(instructions):
            instruction = instructions[state.pc]
            opcode = instruction.opcode
            arg = instruction.arg
            state.pc += 1

            if opcode == OpCode.LOAD_CONST:
                stack.append(arg)
                continue
            if opcode == OpCode.LOAD_NAME:
                stack.append(load_name(globals, arg))
                continue
            if opcode == OpCode.STORE_NAME:
                store_name(globals, arg, stack[-1])
                continue
            if opcode == OpCode.SET_LAST:
                state.last_value = stack[-1] if stack else None
                continue
            if opcode == OpCode.BUILD_LIST:
                count = int(arg)
                items = stack[-count:] if count else []
                if count:
                    del stack[-count:]
                stack.append(list(items))
                continue
            if opcode == OpCode.BUILD_DICT:
                count = int(arg)
                item_count = 2 * count
                items = stack[-item_count:] if count else []
                if count:
                    del stack[-item_count:]
                built: dict[str, Any] = {}
                for index in range(0, len(items), 2):
                    built[str(items[index])] = items[index + 1]
                stack.append(built)
                continue
            if opcode == OpCode.BINARY_OP:
                right = stack.pop()
                left = stack.pop()
                stack.append(eval_binary_operation(left, arg, right))
                continue
            if opcode == OpCode.GET_INDEX:
                index = stack.pop()
                target = stack.pop()
                stack.append(eval_index(target, index))
                continue
            if opcode == OpCode.GET_ATTR:
                target = stack.pop()
                stack.append(resolve_member(target, arg))
                continue
            if opcode == OpCode.CALL:
                arg_count = int(arg)
                target_index = len(stack) - arg_count - 1
                target = stack[target_index]
                args = tuple(stack[target_index + 1 :])
                del stack[target_index:]
                stack.append(await call_async(target, args, print_callback=print_callback))
                continue
            if opcode == OpCode.JUMP:
                state.pc = int(arg)
                continue
            if opcode == OpCode.JUMP_IF_FALSE:
                if not stack.pop():
                    state.pc = int(arg)
                continue
            if opcode == OpCode.ITER_PREP:
                iterable = stack.pop()
                if not isinstance(iterable, list):
                    msg = "for loops require a list iterable"
                    raise TypeError(msg)
                loop_stack.append(ForLoopState(arg, iterable))
                continue
            if opcode == OpCode.ITER_NEXT:
                loop_state = loop_stack[-1]
                if loop_state.index >= len(loop_state.items):
                    loop_stack.pop()
                    state.pc = int(arg)
                    continue
                globals[loop_state.target_name] = loop_state.items[loop_state.index]
                loop_state.index += 1
                continue
            if opcode == OpCode.RETURN_VALUE:
                return state.last_value
            msg = f"Unsupported opcode: {opcode}"
            raise ValueError(msg)
        return state.last_value

    def _load_name(self, globals: MutableMapping[str, Any], name: str) -> Any:
        if name in globals:
            return globals[name]
        if name in self._bindings:
            return self._bindings[name]
        if name in self._functions:
            return self._functions[name]
        msg = f"Unknown name: {name}"
        raise NameError(msg)

    def _store_name(
        self, globals: MutableMapping[str, Any], name: str, value: Any
    ) -> None:
        if name in self._bindings:
            msg = f"Cannot assign to read-only binding: {name}"
            raise NameError(msg)
        globals[name] = value

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
        handler = self._maybe_handler_for(target)
        if handler is not None:
            return handler.get_item(target, index)
        msg = f"'{type(target).__name__}' object is not subscriptable"
        raise TypeError(msg)

    def _resolve_member(self, target: Any, name: str) -> Any:
        return self._handler_for(target).resolve_member(target, name)

    def _maybe_handler_for(self, value: Any) -> ForeignObjectInterface | None:
        for handler in self._foreign_interfaces:
            if handler.supports(value):
                return handler
        return None

    def _handler_for(self, value: Any) -> ForeignObjectInterface:
        handler = self._maybe_handler_for(value)
        if handler is not None:
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


_PRINT_SENTINEL = object()
_PARALLEL_SENTINEL = object()


__all__ = ["ForeignObjectInterface", "Interpreter", "ParseError", "OpCode"]
