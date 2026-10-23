import struct

import pytest

from qwen35_moe_dflash.elf_dynamic import dynamic_info


def elf_bytes():
    # Deliberately has no section headers or DWARF; dynamic info uses segments.
    image = bytearray(1024)
    image[:7] = b'\x7fELF\x02\x01\x01'
    struct.pack_into('<HHI', image, 16, 3, 183, 1)
    struct.pack_into('<Q', image, 32, 64)
    struct.pack_into('<HHH', image, 52, 64, 56, 2)
    struct.pack_into('<IIQQQQQQ', image, 64, 1, 4, 0, 0x10000, 0, 1024, 1024, 4096)
    strings = b'\0libhiai.so\0libc.so\0libfixture.so\0/private/lib\0/vendor/lib64\0'
    image[512:512 + len(strings)] = strings
    names = {name: strings.index(name.encode()) for name in
             ('libhiai.so', 'libc.so', 'libfixture.so', '/private/lib', '/vendor/lib64')}
    tags = [(5, 0x10200), (10, len(strings)), (1, names['libhiai.so']),
            (1, names['libc.so']), (14, names['libfixture.so']),
            (15, names['/private/lib']), (29, names['/vendor/lib64']), (0, 0)]
    struct.pack_into('<IIQQQQQQ', image, 120, 2, 4, 256, 0x10100, 0, len(tags) * 16, len(tags) * 16, 8)
    for index, pair in enumerate(tags):
        struct.pack_into('<qQ', image, 256 + index * 16, *pair)
    return image


def test_reads_needed_soname_and_paths_without_sections(tmp_path):
    path = tmp_path/'target.so'
    path.write_bytes(elf_bytes())
    assert dynamic_info(path) == {
        'needed': ['libhiai.so', 'libc.so'], 'soname': 'libfixture.so',
        'rpath': '/private/lib', 'runpath': '/vendor/lib64',
    }


@pytest.mark.parametrize('fault', [
    'x86', 'elf32', 'big-endian', 'truncated-header', 'truncated-segment',
    'phnum', 'no-null', 'unmapped-strings', 'bss-strings', 'string-overflow',
    'string-no-null', 'large-strings', 'large-dynamic', 'conflicting-tags',
])
def test_rejects_malformed_or_wrong_architecture_without_execution(tmp_path, fault):
    image = elf_bytes()
    if fault == 'x86':
        struct.pack_into('<H', image, 18, 62)
    elif fault == 'elf32':
        image[4] = 1
    elif fault == 'big-endian':
        image[5] = 2
    elif fault == 'truncated-header':
        image = image[:30]
    elif fault == 'truncated-segment':
        image = image[:270]
    elif fault == 'phnum':
        struct.pack_into('<H', image, 56, 65535)
    elif fault == 'no-null':
        struct.pack_into('<qQ', image, 256 + 7 * 16, 1, 1)
    elif fault == 'unmapped-strings':
        struct.pack_into('<Q', image, 264, 0x20000)
    elif fault == 'bss-strings':
        struct.pack_into('<Q', image, 64 + 32, 520)  # p_filesz < string extent < p_memsz
    elif fault == 'string-overflow':
        struct.pack_into('<Q', image, 256 + 2 * 16 + 8, 10000)
    elif fault == 'string-no-null':
        image[512:1024] = b'x' * 512
    elif fault == 'large-strings':
        struct.pack_into('<Q', image, 256 + 16 + 8, 32 * 1024 * 1024)
    elif fault == 'large-dynamic':
        struct.pack_into('<Q', image, 120 + 32, 8 * 1024 * 1024)
    elif fault == 'conflicting-tags':
        struct.pack_into('<qQ', image, 256 + 6 * 16, 5, 0x10900)
    path = tmp_path/'target.so'
    path.write_bytes(image)
    with pytest.raises(ValueError):
        dynamic_info(path)


def test_static_object_has_no_dynamic_dependencies(tmp_path):
    image = elf_bytes()
    struct.pack_into('<H', image, 16, 2)
    struct.pack_into('<H', image, 56, 1)
    path = tmp_path/'static-target'
    path.write_bytes(image)
    assert dynamic_info(path) == {'needed': [], 'soname': None, 'rpath': None, 'runpath': None}
