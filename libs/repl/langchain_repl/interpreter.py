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
    from collections.abc import Callable, Mapping, MutableMapping, Sequence

    from langchain.tools import ToolRuntime


@dataclass(frozen=True)
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


class OpCode(str, Enum):
    """Instruction set for the stack-based REPL VM.

    Each opcode either pushes values onto the operand stack, consumes values from
    it, mutates control flow via the program counter, or updates interpreter state
    such as variable bindings and loop bookkeeping.
    """

    LOAD_CONST = "LOAD_CONST"
    """Push the instruction argument onto the operand stack.

    Example: `42` compiles to `LOAD_CONST 42`.
    """

    LOAD_NAME = "LOAD_NAME"
    """Push the value of a named binding or registered function onto the stack.

    Example: `x` compiles to `LOAD_NAME "x"`.
    """

    STORE_NAME = "STORE_NAME"
    """Store the top-of-stack value into the named environment binding.

    Example: `x = 1` emits `LOAD_CONST 1` then `STORE_NAME "x"`.
    """

    SET_LAST = "SET_LAST"
    """Copy the top-of-stack value into `VMState.last_value` without popping it.

    Example: expression statements like `x` end with `SET_LAST`.
    """

    BUILD_LIST = "BUILD_LIST"
    """Pop `arg` values, package them into a list, and push the new list.

    Example: `[1, 2]` emits `LOAD_CONST 1`, `LOAD_CONST 2`, `BUILD_LIST 2`.
    """

    BUILD_DICT = "BUILD_DICT"
    """Pop `arg` key/value pairs, package them into a dict, and push the dict.

    Example: `{"a": 1}` emits `LOAD_CONST "a"`, `LOAD_CONST 1`, `BUILD_DICT 1`.
    """

    BINARY_OP = "BINARY_OP"
    """Pop two operands, apply the binary operator in `arg`, and push the result.

    Example: `x + 1` emits `LOAD_NAME "x"`, `LOAD_CONST 1`, `BINARY_OP "+"`.
    """

    GET_INDEX = "GET_INDEX"
    """Pop an index and target, resolve `target[index]`, and push the result.

    Example: `items[0]` emits `LOAD_NAME "items"`, `LOAD_CONST 0`, `GET_INDEX`.
    """

    GET_ATTR = "GET_ATTR"
    """Pop a target, resolve the attribute named by `arg`, and push the result.

    Example: `math.sin` emits `LOAD_NAME "math"`, `GET_ATTR "sin"`.
    """

    CALL = "CALL"
    """Pop a callable target plus `arg` arguments, invoke it, and push the result.

    Example: `echo(1)` emits `LOAD_NAME "echo"`, `LOAD_CONST 1`, `CALL 1`.
    """

    JUMP = "JUMP"
    """Set the program counter to the absolute instruction index stored in `arg`.

    Example: `if ... else ... end` uses `JUMP` to skip over the else branch after the then branch runs.
    """

    JUMP_IF_FALSE = "JUMP_IF_FALSE"
    """Pop a condition and jump to `arg` when the value is falsy.

    Example: `if cond then ... else ... end` emits `JUMP_IF_FALSE <else_start>` after compiling `cond`.
    """

    ITER_PREP = "ITER_PREP"
    """Pop a list iterable and initialize loop state for the target name in `arg`.

    Example: `for item in items do ... end` emits `LOAD_NAME "items"` then `ITER_PREP "item"`.
    """

    ITER_NEXT = "ITER_NEXT"
    """Advance the active loop or jump to `arg` when the iterable is exhausted.

    Example: `for item in items do ... end` emits `ITER_NEXT <loop_end>` at the top of the loop body.
    """

    RETURN_VALUE = "RETURN_VALUE"
    """Finish execution and return `VMState.last_value`.

    Example: every compiled program ends with `RETURN_VALUE`.
    """


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
        env: Current variable bindings visible to the program.
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
    env: MutableMapping[str, Any]
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
        self._compiler = _ProgramCompiler

    @property
    def env(self) -> dict[str, Any]:
        return dict(self._env)

    @property
    def printed_lines(self) -> list[str]:
        return list(self._printed_lines)

    def compile(self, source: str) -> tuple[Instruction, ...]:
        return self._compiler(_Tokenizer(source).tokenize()).compile()

    def evaluate(
        self, source: str, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        instructions = self.compile(source)
        state = self._new_state(instructions, self._env)
        value = self._run_vm_sync(state, print_callback=print_callback)
        self._env = state.env
        return value

    async def aevaluate(
        self, source: str, *, print_callback: Callable[[str], None] | None = None
    ) -> Any:
        instructions = self.compile(source)
        state = self._new_state(instructions, self._env)
        value = await self._run_vm_async(state, print_callback=print_callback)
        self._env = state.env
        return value

    def _new_state(
        self, instructions: tuple[Instruction, ...], env: MutableMapping[str, Any]
    ) -> VMState:
        return VMState(instructions=instructions, env=env)

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
_NO_RESULT = object()


__all__ = ["ForeignObjectInterface", "Interpreter", "ParseError"]
