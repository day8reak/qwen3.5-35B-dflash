"""Bounded, schema-derived HIAI protobuf inspection using the standard library.

The schema comes from the explicitly selected DDK library's ELF symbol, never
from a guessed file offset or another GE package. Enum numbers remain raw;
protobuf TensorDescriptor enums are not CPUCL Cast attribute enums.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
import struct


_SYMBOL = "_ZL49descriptor_table_protodef_proto_2fge_5fir_2eproto"
_MAX_LIBRARY = 256 << 20
_MAX_DESCRIPTOR = 8 << 20
_MAX_DEPTH = 64
_MAX_FIELDS = 2_000_000
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_WIRE = {1: 1, 2: 5, 3: 0, 4: 0, 5: 0, 6: 1, 7: 5, 8: 0,
         9: 2, 11: 2, 12: 2, 13: 0, 14: 0, 15: 5, 16: 1, 17: 0, 18: 0}
_PACKABLE = frozenset(_WIRE) - {9, 11, 12}


def _bounded(data, offset, size, what):
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise ValueError(f"{what} exceeds buffer bounds")
    # Bounds-only ELF calls and large tensor payloads must not copy their data.
    return memoryview(data)[offset:offset + size]


def _varint(data, offset):
    value = 0
    for index in range(10):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        if index == 9 and byte > 1:
            raise ValueError("protobuf varint exceeds uint64")
        value |= (byte & 127) << (index * 7)
        if not byte & 128:
            return value, offset
    raise ValueError("unterminated protobuf varint")


def _consume(budget):
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("protobuf field budget exceeded")


def _fields(data, budget):
    """Yield wire fields while checking complete consumption and all lengths."""
    offset = 0
    while offset < len(data):
        _consume(budget)
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if not 0 < number < (1 << 29):
            raise ValueError("invalid protobuf field number")
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = _bounded(data, offset, size, "fixed-width protobuf field")
            offset += size
        elif wire == 2:
            size, offset = _varint(data, offset)
            value = _bounded(data, offset, size, "length-delimited protobuf field")
            offset += size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _descriptor_fields(data, budget):
    result = {}
    for number, wire, value in _fields(data, budget):
        result.setdefault(number, []).append((wire, value))
    return result


def _one(fields, number, wire, default=None):
    entries = fields.get(number, [])
    if not entries:
        return default
    if len(entries) != 1 or entries[0][0] != wire:
        raise ValueError(f"ambiguous or mistyped descriptor field {number}")
    return entries[0][1]


def _many(fields, number, wire):
    entries = fields.get(number, [])
    if any(kind != wire for kind, _ in entries):
        raise ValueError(f"mistyped repeated descriptor field {number}")
    return [value for _, value in entries]


def _text(value):
    return bytes(value).decode("utf-8", errors="strict")


def _name(fields):
    value = _one(fields, 1, 2)
    if value is None or not _IDENTIFIER.fullmatch(name := _text(value)):
        raise ValueError("missing or invalid descriptor name")
    return name


def _signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


@dataclass(frozen=True)
class _Field:
    name: str
    number: int
    kind: int
    repeated: bool
    type_name: str
    oneof: int | None


class Schema:
    """Schema decoded from a FileDescriptorProto; no vendor library is loaded."""

    def __init__(self, descriptor, evidence=None):
        if len(descriptor) > _MAX_DESCRIPTOR:
            raise ValueError("descriptor exceeds byte budget")
        self.evidence = dict(evidence or {})
        self.enums, self._messages = {}, {}
        budget = [_MAX_FIELDS]
        fields = _descriptor_fields(memoryview(descriptor), budget)
        if (_text(_one(fields, 1, 2, b"")) != "proto/ge_ir.proto"
                or _text(_one(fields, 2, 2, b"")) != "hiai.proto"):
            raise ValueError("expected proto/ge_ir.proto in package hiai.proto")
        if _many(fields, 3, 2):
            raise ValueError("external descriptor dependencies are unsupported")
        syntax = _text(_one(fields, 12, 2, b"proto2"))
        if syntax not in {"proto2", "proto3"}:
            raise ValueError("unsupported descriptor syntax")
        self.evidence.update(schema_file="proto/ge_ir.proto", schema_package="hiai.proto",
                             schema_syntax=syntax)
        for raw in _many(fields, 5, 2):
            self._enum(raw, "hiai.proto", budget)
        for raw in _many(fields, 4, 2):
            self._message(raw, "hiai.proto", budget, 0)
        for message in self._messages.values():
            for field in message.values():
                if field.kind == 11 and field.type_name not in self._messages:
                    raise ValueError(f"unknown message type {field.type_name}")
                if field.kind == 14 and field.type_name not in self.enums:
                    raise ValueError(f"unknown enum type {field.type_name}")
        if not self._messages:
            raise ValueError("descriptor has no messages")

    def _enum(self, raw, parent, budget):
        fields = _descriptor_fields(raw, budget)
        name = parent + "." + _name(fields)
        if name in self.enums or name in self._messages:
            raise ValueError(f"duplicate schema type {name}")
        values = {}
        for entry in _many(fields, 2, 2):
            item = _descriptor_fields(entry, budget)
            key, value = _name(item), _one(item, 2, 0)
            if key in values or value is None:
                raise ValueError(f"invalid enum value in {name}")
            values[key] = _signed(value, 32)
        if not values:
            raise ValueError(f"empty enum {name}")
        self.enums[name] = values

    def _message(self, raw, parent, budget, depth):
        if depth > _MAX_DEPTH:
            raise ValueError("descriptor nesting depth exceeded")
        fields = _descriptor_fields(raw, budget)
        name = parent + "." + _name(fields)
        if name in self._messages or name in self.enums:
            raise ValueError(f"duplicate schema type {name}")
        message, names = {}, set()
        oneof_count = len(_many(fields, 8, 2))
        for entry in _many(fields, 2, 2):
            item = _descriptor_fields(entry, budget)
            key, number = _name(item), _one(item, 3, 0)
            label, kind = _one(item, 4, 0), _one(item, 5, 0)
            type_name = _text(_one(item, 6, 2, b"")).lstrip(".")
            oneof = _one(item, 9, 0)
            if (number is None or not 0 < number < (1 << 29)
                    or number in message or key in names or label not in (1, 2, 3)
                    or kind not in _WIRE):
                raise ValueError(f"invalid or duplicate field in {name}")
            if oneof is not None and (oneof >= oneof_count or label == 3):
                raise ValueError(f"invalid oneof in {name}")
            message[number] = _Field(key, number, kind, label == 3, type_name, oneof)
            names.add(key)
        self._messages[name] = message
        for child in _many(fields, 4, 2):
            self._enum(child, name, budget)
        for child in _many(fields, 3, 2):
            self._message(child, name, budget, depth + 1)

    def decode(self, data: bytes, type_name: str):
        """Decode one whole protobuf, preserving map entries and enum integers.

        Unknown fields are rejected so a mismatching schema cannot silently
        manufacture an apparently complete CPU graph. Omitted defaults stay
        absent. Attribute bytes are text only for small printable strings.
        """
        name = type_name.lstrip(".")
        if not name.startswith("hiai.proto."):
            name = "hiai.proto." + name
        if name not in self._messages:
            raise ValueError(f"unknown protobuf message {type_name}")
        return self._decode(memoryview(data), name, [_MAX_FIELDS], 0)

    def _decode(self, data, name, budget, depth):
        if depth > _MAX_DEPTH:
            raise ValueError("protobuf nesting depth exceeded")
        result, selected_oneofs = {}, {}
        message = self._messages[name]
        for number, wire, value in _fields(data, budget):
            field = message.get(number)
            if field is None:
                raise ValueError(f"unknown protobuf field {number} in {name}")
            if field.repeated and field.kind in _PACKABLE and wire == 2:
                values, offset = [], 0
                while offset < len(value):
                    _consume(budget)
                    if _WIRE[field.kind] == 0:
                        raw, offset = _varint(value, offset)
                    else:
                        size = 8 if _WIRE[field.kind] == 1 else 4
                        raw = _bounded(value, offset, size, "packed protobuf field")
                        offset += size
                    values.append(self._scalar(raw, field, name))
                result.setdefault(field.name, []).extend(values)
                continue
            if wire != _WIRE[field.kind]:
                raise ValueError(f"wire type mismatch for {name}.{field.name}")
            decoded = (self._decode(value, field.type_name, budget, depth + 1)
                       if field.kind == 11 else self._scalar(value, field, name))
            if field.repeated:
                result.setdefault(field.name, []).append(decoded)
            else:
                # Ambiguous duplicate singular messages are deliberately rejected;
                # this diagnostic decoder does not claim protobuf merge semantics.
                if field.name in result:
                    raise ValueError(f"duplicate singular field {name}.{field.name}")
                if field.oneof is not None:
                    if field.oneof in selected_oneofs:
                        raise ValueError(f"multiple values for oneof in {name}")
                    selected_oneofs[field.oneof] = field.name
                result[field.name] = decoded
        return result

    @staticmethod
    def _scalar(value, field, message):
        kind = field.kind
        if kind in (3, 5, 14):
            return _signed(value, 64 if kind == 3 else 32)
        if kind in (4, 13):
            if kind == 13 and value >= 1 << 32:
                raise ValueError("uint32 field overflow")
            return value
        if kind == 8:
            if value not in (0, 1):
                raise ValueError("invalid boolean protobuf value")
            return bool(value)
        if kind in (17, 18):
            if kind == 17 and value >= 1 << 32:
                raise ValueError("sint32 field overflow")
            return (value >> 1) ^ -(value & 1)
        formats = {1: "<d", 2: "<f", 6: "<Q", 7: "<I", 15: "<i", 16: "<q"}
        if kind in formats:
            decoded = struct.unpack(formats[kind], value)[0]
            if isinstance(decoded, float) and not math.isfinite(decoded):
                return {"nonfinite_float": repr(decoded), "wire_hex": bytes(value).hex()}
            return decoded
        if kind == 9:
            return _text(value)
        if kind == 12:
            if (len(value) <= 256 and field.name == "s"
                    and message in {"hiai.proto.AttrDef", "hiai.proto.AttrDef.ListValue"}):
                try:
                    text = _text(value)
                    if text.isprintable() or not text:
                        return text
                except UnicodeDecodeError:
                    pass
            return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        raise ValueError(f"unsupported scalar type {kind}")


def load_schema(library_path):
    """Extract the unique descriptor object from the declared ELF64 LE library."""
    path = Path(library_path).resolve()
    with path.open("rb") as stream:
        data = stream.read(_MAX_LIBRARY + 1)
    if len(data) > _MAX_LIBRARY:
        raise ValueError("schema library exceeds byte budget")
    if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("schema library must be ELF64 little-endian version 1")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    _, elf_type, _, version, _, _, shoff, _, ehsize, _, _, shsize, shcount, _ = header
    if (elf_type not in (2, 3) or version != 1 or ehsize != 64 or shsize != 64
            or not 0 < shcount <= 10_000):
        raise ValueError("unsupported ELF header/section table")
    _bounded(data, shoff, shsize * shcount, "ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + index * shsize)
                for index in range(shcount)]
    for section in sections:
        if section[1] != 8:  # SHT_NOBITS has no file payload.
            _bounded(data, section[4], section[5], "ELF section")
    candidates = set()
    for section in sections:
        if section[1] not in (2, 11):  # SHT_SYMTAB / SHT_DYNSYM
            continue
        if (section[9] != 24 or section[5] % 24 or section[5] // 24 > 1_000_000
                or section[6] >= len(sections)):
            raise ValueError("invalid ELF symbol table")
        strings = sections[section[6]]
        if strings[1] != 3:
            raise ValueError("ELF symbols lack a string table")
        names = bytes(_bounded(data, strings[4], strings[5], "ELF string table"))
        for offset in range(section[4], section[4] + section[5], 24):
            name_offset, info, _, index, value, size = struct.unpack_from("<IBBHQQ", data, offset)
            if name_offset >= len(names) or (end := names.find(b"\0", name_offset)) < 0:
                raise ValueError("invalid ELF symbol name bounds")
            if names[name_offset:end] != _SYMBOL.encode("ascii"):
                continue
            if info & 15 != 1 or not 0 < index < len(sections) or not 0 < size <= _MAX_DESCRIPTOR:
                raise ValueError("invalid descriptor ELF symbol")
            owner = sections[index]
            if owner[1] != 1 or value < owner[3]:  # Stored PROGBITS object only.
                raise ValueError("descriptor symbol is not stored section data")
            relative = value - owner[3]
            if relative > owner[5] or size > owner[5] - relative:
                raise ValueError("descriptor symbol exceeds owning section")
            file_offset = owner[4] + relative
            _bounded(data, file_offset, size, "descriptor symbol")
            candidates.add((value, file_offset, size))
    if len(candidates) != 1:
        raise ValueError("expected exactly one HIAI protobuf descriptor ELF symbol")
    value, offset, size = candidates.pop()
    descriptor = data[offset:offset + size]
    # protoc's C++ string object includes its terminal NUL in the ELF size.
    if descriptor.endswith(b"\0"):
        descriptor = descriptor[:-1]
    evidence = {"library": str(path), "library_sha256": hashlib.sha256(data).hexdigest(),
                "symbol": _SYMBOL, "symbol_value": value, "symbol_file_offset": offset,
                "symbol_size": size, "descriptor_bytes": len(descriptor),
                "descriptor_sha256": hashlib.sha256(descriptor).hexdigest()}
    return Schema(descriptor, evidence)
