from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


ATTRIBUTE_LINE_RE = re.compile(r"^\s*#\[(?P<body>.+)\]\s*$")
FUNCTION_RE = re.compile(
    r'^\s*(?P<vis>pub(?:\([^)]*\))?\s+)?'
    r'(?:(?:async|const|unsafe|extern(?:\s+"[^"]+")?)\s+)*'
    r"fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
IMPL_RE = re.compile(r"^\s*impl\b")
ITEM_RE = re.compile(
    r"^\s*(?P<vis>pub(?:\([^)]*\))?\s+)?"
    r"(?P<kind>struct|enum|trait|union|type|const|static)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
MODULE_RE = re.compile(
    r"^\s*(?P<vis>pub(?:\([^)]*\))?\s+)?mod\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
RAW_STRING_START_RE = re.compile(r'(?:br|rb|r)(?P<hashes>#{0,16})"')
USE_RE = re.compile(r"^\s*(?P<vis>pub(?:\([^)]*\))?\s+)?use\b")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class TextSpan:
    start_line: int
    start_column: int
    end_line: int
    end_column: int


@dataclass
class RustSymbol:
    local_id: int
    kind: str
    name: str
    qualified_name: str
    module_path: str
    span: TextSpan
    signature: str
    visibility: str
    docstring: Optional[str]
    container_local_id: Optional[int]
    container_qualified_name: Optional[str]
    attributes: Tuple[str, ...] = field(default_factory=tuple)
    is_test: bool = False
    impl_target: Optional[str] = None
    impl_trait: Optional[str] = None
    super_traits: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RustImport:
    path: str
    module_path: str
    span: TextSpan
    visibility: str
    signature: str
    container_local_id: Optional[int]
    container_qualified_name: Optional[str]


@dataclass
class ParsedRustFile:
    path: str
    crate_name: str
    module_path: str
    symbols: List[RustSymbol]
    imports: List[RustImport]


@dataclass
class _LexState:
    block_comment_depth: int = 0
    in_raw_string: bool = False
    in_string: bool = False
    raw_hashes: int = 0


@dataclass
class _Container:
    local_id: int
    kind: str
    qualified_name: str
    close_threshold: int
    is_test_context: bool


@dataclass
class _PendingImport:
    start_line: int
    start_column: int
    visibility: str
    module_path: str
    container_local_id: Optional[int]
    container_qualified_name: Optional[str]
    raw_lines: List[str]
    cleaned_lines: List[str]


@dataclass
class _PendingDeclaration:
    kind: str
    name: Optional[str]
    visibility: str
    start_line: int
    start_column: int
    module_path: str
    container_local_id: Optional[int]
    container_qualified_name: Optional[str]
    raw_lines: List[str]
    cleaned_lines: List[str]
    docstring: Optional[str]
    attributes: Tuple[str, ...]


def attribute_marks_test(attribute: str) -> bool:
    attribute = attribute.strip()
    return (
        attribute == "test"
        or attribute.endswith("::test")
        or attribute.startswith("cfg(test")
        or attribute == "cfg(test)"
    )


def collapse_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()


def current_module_path(file_module_path: str, containers: List[_Container]) -> str:
    for container in reversed(containers):
        if container.kind == "module":
            return container.qualified_name
    return file_module_path


def current_test_context(path: str, containers: List[_Container]) -> bool:
    if "tests" in path.split("/"):
        return True
    return any(container.is_test_context for container in containers)


def extract_attribute(line: str) -> Optional[str]:
    match = ATTRIBUTE_LINE_RE.match(line)
    if not match:
        return None
    return match.group("body").strip()


def final_line_column(raw_line: str) -> int:
    return len(raw_line.rstrip()) + 1 if raw_line.rstrip() else 1


def normalize_visibility(value: Optional[str]) -> str:
    return value.strip() if value else "private"


def parse_impl_identity(signature: str) -> Tuple[str, Optional[str], Optional[str]]:
    normalized = collapse_whitespace(signature).rstrip("{").strip()
    remainder = normalized[len("impl") :].strip()
    remainder = strip_leading_impl_generics(remainder)
    remainder = remainder.split(" where ", 1)[0].strip()
    if " for " in remainder:
        impl_trait, impl_target = (part.strip() for part in remainder.split(" for ", 1))
        return f"impl {impl_trait} for {impl_target}", impl_trait, impl_target
    return f"impl {remainder}", None, remainder or None


def parse_trait_supertraits(signature: str, name: str) -> Tuple[str, ...]:
    normalized = collapse_whitespace(signature).rstrip("{").strip()
    normalized = normalized.split(" where ", 1)[0].strip()
    marker = f"trait {name}"
    marker_index = normalized.find(marker)
    if marker_index < 0:
        return ()
    remainder = normalized[marker_index + len(marker) :].strip()
    if not remainder.startswith(":"):
        return ()
    parents = []
    for entry in remainder[1:].split("+"):
        candidate = collapse_whitespace(entry)
        if candidate:
            parents.append(candidate)
    return tuple(parents)


def strip_leading_impl_generics(value: str) -> str:
    if not value.startswith("<"):
        return value

    depth = 0
    for index, char in enumerate(value):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return value[index + 1 :].strip()
    return value


def strip_rust_non_code(line: str, state: _LexState) -> str:
    output: List[str] = []
    index = 0

    while index < len(line):
        if state.block_comment_depth:
            if line.startswith("/*", index):
                state.block_comment_depth += 1
                output.append("  ")
                index += 2
                continue
            if line.startswith("*/", index):
                state.block_comment_depth -= 1
                output.append("  ")
                index += 2
                continue
            output.append(" ")
            index += 1
            continue

        if state.in_string:
            if line[index] == "\\" and index + 1 < len(line):
                output.append("  ")
                index += 2
                continue
            if line[index] == '"':
                state.in_string = False
            output.append(" ")
            index += 1
            continue

        if state.in_raw_string:
            hashes = "#" * state.raw_hashes
            if line[index] == '"' and line.startswith(hashes, index + 1):
                output.append(" " * (1 + state.raw_hashes))
                index += 1 + state.raw_hashes
                state.in_raw_string = False
                state.raw_hashes = 0
                continue
            output.append(" ")
            index += 1
            continue

        if line.startswith("//", index):
            output.append(" " * (len(line) - index))
            break

        if line.startswith("/*", index):
            state.block_comment_depth += 1
            output.append("  ")
            index += 2
            continue

        raw_string_match = RAW_STRING_START_RE.match(line, index)
        if raw_string_match:
            hashes = raw_string_match.group("hashes") or ""
            output.append(" " * len(raw_string_match.group(0)))
            state.in_raw_string = True
            state.raw_hashes = len(hashes)
            index += len(raw_string_match.group(0))
            continue

        if line[index] == '"':
            state.in_string = True
            output.append(" ")
            index += 1
            continue

        output.append(line[index])
        index += 1

    return "".join(output)


def clean_rust_source_lines(source: str) -> List[str]:
    state = _LexState()
    return [strip_rust_non_code(line, state) for line in source.splitlines()]


def parse_rust_file(path: str, source: str, crate_name: str, module_path: str) -> ParsedRustFile:
    brace_depth = 0
    containers: List[_Container] = []
    imports: List[RustImport] = []
    lex_state = _LexState()
    lines = source.splitlines()
    pending_attributes: List[str] = []
    pending_declaration: Optional[_PendingDeclaration] = None
    pending_doc_lines: List[str] = []
    pending_import: Optional[_PendingImport] = None
    symbol_lookup: Dict[int, RustSymbol] = {}
    symbols: List[RustSymbol] = []
    next_local_id = 1

    def flush_import(current_line: int, raw_line: str) -> None:
        nonlocal pending_import
        if pending_import is None:
            return

        signature = collapse_whitespace(" ".join(line.strip() for line in pending_import.raw_lines if line.strip()))
        target = collapse_whitespace(" ".join(line.strip() for line in pending_import.cleaned_lines if line.strip()))
        target = re.sub(r"^\s*pub(?:\([^)]*\))?\s+use\s+", "", target)
        target = re.sub(r"^\s*use\s+", "", target)
        target = target.rstrip(";").strip()

        imports.append(
            RustImport(
                path=target,
                module_path=pending_import.module_path,
                span=TextSpan(
                    start_line=pending_import.start_line,
                    start_column=pending_import.start_column,
                    end_line=current_line,
                    end_column=final_line_column(raw_line),
                ),
                visibility=pending_import.visibility,
                signature=signature,
                container_local_id=pending_import.container_local_id,
                container_qualified_name=pending_import.container_qualified_name,
            )
        )
        pending_import = None

    def flush_declaration(current_line: int, raw_line: str, current_brace_depth: int) -> None:
        nonlocal next_local_id
        nonlocal pending_declaration
        if pending_declaration is None:
            return

        signature = collapse_whitespace(
            " ".join(line.strip() for line in pending_declaration.raw_lines if line.strip())
        )
        declaration_text = collapse_whitespace(
            " ".join(line.strip() for line in pending_declaration.cleaned_lines if line.strip())
        )
        kind = pending_declaration.kind
        name = pending_declaration.name or ""
        impl_target = None
        impl_trait = None
        super_traits: Tuple[str, ...] = ()

        if kind == "impl":
            name, impl_trait, impl_target = parse_impl_identity(declaration_text)
        elif kind == "trait":
            super_traits = parse_trait_supertraits(declaration_text, name)

        qualifier = pending_declaration.container_qualified_name or pending_declaration.module_path
        qualified_name = f"{qualifier}::{name}"

        is_test = current_test_context(path, containers) or any(
            attribute_marks_test(attribute) for attribute in pending_declaration.attributes
        )
        if kind == "module" and name == "tests":
            is_test = True

        symbol = RustSymbol(
            local_id=next_local_id,
            kind=kind,
            name=name,
            qualified_name=qualified_name,
            module_path=pending_declaration.module_path,
            span=TextSpan(
                start_line=pending_declaration.start_line,
                start_column=pending_declaration.start_column,
                end_line=current_line,
                end_column=final_line_column(raw_line),
            ),
            signature=signature,
            visibility=pending_declaration.visibility,
            docstring=pending_declaration.docstring,
            container_local_id=pending_declaration.container_local_id,
            container_qualified_name=pending_declaration.container_qualified_name,
            attributes=pending_declaration.attributes,
            is_test=is_test,
            impl_target=impl_target,
            impl_trait=impl_trait,
            super_traits=super_traits,
        )
        symbols.append(symbol)
        symbol_lookup[symbol.local_id] = symbol
        next_local_id += 1

        if "{" in declaration_text and kind in {"function", "impl", "method", "module", "trait", "struct", "enum", "union"}:
            containers.append(
                _Container(
                    local_id=symbol.local_id,
                    kind=kind,
                    qualified_name=symbol.qualified_name,
                    close_threshold=current_brace_depth + 1,
                    is_test_context=symbol.is_test,
                )
            )

        pending_declaration = None

    for line_number, raw_line in enumerate(lines, start=1):
        cleaned_line = strip_rust_non_code(raw_line, lex_state)
        stripped_clean = cleaned_line.strip()
        stripped_raw = raw_line.strip()

        if pending_import is not None:
            pending_import.raw_lines.append(raw_line)
            pending_import.cleaned_lines.append(cleaned_line)
            if ";" in stripped_clean:
                flush_import(line_number, raw_line)
        elif pending_declaration is not None:
            pending_declaration.raw_lines.append(raw_line)
            pending_declaration.cleaned_lines.append(cleaned_line)
            if "{" in stripped_clean or ";" in stripped_clean:
                flush_declaration(line_number, raw_line, brace_depth)
        elif not stripped_raw:
            pending_attributes.clear()
            pending_doc_lines.clear()
        elif stripped_raw.startswith("///") or stripped_raw.startswith("//!"):
            pending_doc_lines.append(stripped_raw[3:].strip())
        elif stripped_raw.startswith("#["):
            attribute = extract_attribute(stripped_raw)
            if attribute:
                pending_attributes.append(attribute)
        elif stripped_raw.startswith("//"):
            pending_attributes.clear()
            pending_doc_lines.clear()
        else:
            active_container = containers[-1] if containers else None
            active_module_path = current_module_path(module_path, containers)
            docstring = "\n".join(pending_doc_lines) if pending_doc_lines else None
            attributes = tuple(pending_attributes)

            import_match = USE_RE.match(cleaned_line)
            module_match = MODULE_RE.match(cleaned_line)
            item_match = ITEM_RE.match(cleaned_line)
            function_match = FUNCTION_RE.match(cleaned_line)
            impl_match = IMPL_RE.match(cleaned_line)

            if import_match:
                pending_import = _PendingImport(
                    start_line=line_number,
                    start_column=cleaned_line.find("use") + 1,
                    visibility=normalize_visibility(import_match.group("vis")),
                    module_path=active_module_path,
                    container_local_id=active_container.local_id if active_container else None,
                    container_qualified_name=active_container.qualified_name if active_container else None,
                    raw_lines=[raw_line],
                    cleaned_lines=[cleaned_line],
                )
                if ";" in stripped_clean:
                    flush_import(line_number, raw_line)
                pending_attributes.clear()
                pending_doc_lines.clear()
            elif module_match:
                pending_declaration = _PendingDeclaration(
                    kind="module",
                    name=module_match.group("name"),
                    visibility=normalize_visibility(module_match.group("vis")),
                    start_line=line_number,
                    start_column=module_match.start("name") + 1,
                    module_path=active_module_path,
                    container_local_id=active_container.local_id if active_container else None,
                    container_qualified_name=active_container.qualified_name if active_container else None,
                    raw_lines=[raw_line],
                    cleaned_lines=[cleaned_line],
                    docstring=docstring,
                    attributes=attributes,
                )
                if "{" in stripped_clean or ";" in stripped_clean:
                    flush_declaration(line_number, raw_line, brace_depth)
                pending_attributes.clear()
                pending_doc_lines.clear()
            elif item_match:
                pending_declaration = _PendingDeclaration(
                    kind=item_match.group("kind"),
                    name=item_match.group("name"),
                    visibility=normalize_visibility(item_match.group("vis")),
                    start_line=line_number,
                    start_column=item_match.start("name") + 1,
                    module_path=active_module_path,
                    container_local_id=active_container.local_id if active_container else None,
                    container_qualified_name=active_container.qualified_name if active_container else None,
                    raw_lines=[raw_line],
                    cleaned_lines=[cleaned_line],
                    docstring=docstring,
                    attributes=attributes,
                )
                if "{" in stripped_clean or ";" in stripped_clean:
                    flush_declaration(line_number, raw_line, brace_depth)
                pending_attributes.clear()
                pending_doc_lines.clear()
            elif function_match:
                function_kind = "method" if active_container and active_container.kind in {"impl", "trait"} else "function"
                pending_declaration = _PendingDeclaration(
                    kind=function_kind,
                    name=function_match.group("name"),
                    visibility=normalize_visibility(function_match.group("vis")),
                    start_line=line_number,
                    start_column=function_match.start("name") + 1,
                    module_path=active_module_path,
                    container_local_id=active_container.local_id if active_container else None,
                    container_qualified_name=active_container.qualified_name if active_container else None,
                    raw_lines=[raw_line],
                    cleaned_lines=[cleaned_line],
                    docstring=docstring,
                    attributes=attributes,
                )
                if "{" in stripped_clean or ";" in stripped_clean:
                    flush_declaration(line_number, raw_line, brace_depth)
                pending_attributes.clear()
                pending_doc_lines.clear()
            elif impl_match:
                pending_declaration = _PendingDeclaration(
                    kind="impl",
                    name=None,
                    visibility="private",
                    start_line=line_number,
                    start_column=impl_match.start() + 1,
                    module_path=active_module_path,
                    container_local_id=active_container.local_id if active_container else None,
                    container_qualified_name=active_container.qualified_name if active_container else None,
                    raw_lines=[raw_line],
                    cleaned_lines=[cleaned_line],
                    docstring=docstring,
                    attributes=attributes,
                )
                if "{" in stripped_clean or ";" in stripped_clean:
                    flush_declaration(line_number, raw_line, brace_depth)
                pending_attributes.clear()
                pending_doc_lines.clear()
            else:
                pending_attributes.clear()
                pending_doc_lines.clear()

        brace_depth += cleaned_line.count("{") - cleaned_line.count("}")
        while containers and brace_depth < containers[-1].close_threshold:
            container = containers.pop()
            symbol_lookup[container.local_id].span.end_line = line_number
            symbol_lookup[container.local_id].span.end_column = final_line_column(raw_line)

    last_line = lines[-1] if lines else ""
    last_line_number = len(lines) if lines else 1

    if pending_import is not None:
        flush_import(last_line_number, last_line)

    if pending_declaration is not None:
        flush_declaration(last_line_number, last_line, brace_depth)

    while containers:
        container = containers.pop()
        symbol_lookup[container.local_id].span.end_line = last_line_number
        symbol_lookup[container.local_id].span.end_column = final_line_column(last_line)

    return ParsedRustFile(
        path=path,
        crate_name=crate_name,
        module_path=module_path,
        symbols=symbols,
        imports=imports,
    )
