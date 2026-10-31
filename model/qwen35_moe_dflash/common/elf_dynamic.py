"""Read AArch64 ELF dynamic dependencies without executing target binaries.

Program headers suffice, so old host readelf/DWARF support is irrelevant. This
records explicit dependencies only: dlopen targets and loader namespaces still
require runtime evidence.
"""
from pathlib import Path
import struct


def dynamic_info(path):
    path = Path(path)
    with path.open("rb") as source:
        source.seek(0, 2)
        size = source.tell()

        def read(offset, length):
            if offset < 0 or length < 0 or offset + length > size:
                raise ValueError(f"ELF range outside file: {path}")
            source.seek(offset)
            data = source.read(length)
            if len(data) != length:
                raise ValueError(f"ELF changed or truncated while reading: {path}")
            return data

        header = read(0, 64)
        if (header[:7] != b"\x7fELF\x02\x01\x01"
                or struct.unpack_from("<H", header, 16)[0] not in (2, 3)
                or struct.unpack_from("<H", header, 18)[0] != 183
                or struct.unpack_from("<I", header, 20)[0] != 1):
            raise ValueError(f"expected ELF64 little-endian AArch64 executable or library: {path}")
        phoff = struct.unpack_from("<Q", header, 32)[0]
        ehsize, phsize, phnum = struct.unpack_from("<HHH", header, 52)
        if ehsize != 64 or phsize != 56 or not 0 < phnum < 4096:
            raise ValueError(f"unsupported ELF program-header table: {path}")
        headers = [struct.unpack_from("<IIQQQQQQ", read(phoff + i * phsize, phsize))
                   for i in range(phnum)]
        loads = [(h[3], h[2], h[5]) for h in headers if h[0] == 1]
        dynamics = [h for h in headers if h[0] == 2]
        result = {"needed": [], "soname": None, "rpath": None, "runpath": None}
        if not dynamics:
            return result
        if len(dynamics) != 1:
            raise ValueError(f"ambiguous ELF dynamic segment: {path}")
        dynamic = dynamics[0]
        if not 0 < dynamic[5] <= 4 * 1024 * 1024 or dynamic[5] % 16:
            raise ValueError(f"invalid or oversized ELF dynamic segment: {path}")
        payload = read(dynamic[2], dynamic[5])
        tags, needed = {}, []
        for offset in range(0, len(payload), 16):
            tag, value = struct.unpack_from("<qQ", payload, offset)
            if tag == 0:
                break
            if tag == 1:
                needed.append(value)
            elif tag in (5, 10, 14, 15, 29):
                if tag in tags and tags[tag] != value:
                    raise ValueError(f"conflicting ELF dynamic tag {tag}: {path}")
                tags[tag] = value
        else:
            raise ValueError(f"unterminated ELF dynamic segment: {path}")
        if not needed and not any(tag in tags for tag in (14, 15, 29)):
            return result
        if 5 not in tags or not 0 < tags.get(10, 0) <= 16 * 1024 * 1024:
            raise ValueError(f"missing or oversized ELF dynamic string table: {path}")
        address, length = tags[5], tags[10]
        offsets = {fileoff + address - vaddr for vaddr, fileoff, filesz in loads
                   if vaddr <= address and address + length <= vaddr + filesz}
        if len(offsets) != 1:
            raise ValueError(f"ELF dynamic string table has no unique file mapping: {path}")
        strings = read(offsets.pop(), length)

        def string(index, allow_empty=False):
            if not 0 <= index < len(strings):
                raise ValueError(f"ELF string offset outside table: {path}")
            end = strings.find(b"\0", index)
            if end < 0:
                raise ValueError(f"unterminated ELF dynamic string: {path}")
            value = strings[index:end].decode("utf-8")
            if (not value and not allow_empty) or any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ValueError(f"invalid ELF dynamic string: {path}")
            return value

        result["needed"] = list(dict.fromkeys(string(index) for index in needed))
        for key, tag in (("soname", 14), ("rpath", 15), ("runpath", 29)):
            if tag in tags:
                result[key] = string(tags[tag], allow_empty=key in ("rpath", "runpath"))
        return result
