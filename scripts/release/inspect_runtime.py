#!/usr/bin/env python3
"""Inspect executable imports without invoking a third-party runtime."""
import pathlib
import struct
import subprocess
import sys

binary, target, report = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
if target.endswith('windows-msvc'):
    data = binary.read_bytes()
    def u16(at): return struct.unpack_from('<H', data, at)[0]
    def u32(at): return struct.unpack_from('<I', data, at)[0]
    pe = u32(0x3c)
    assert data[:2] == b'MZ' and data[pe:pe+4] == b'PE\0\0'
    coff = pe + 4
    assert u16(coff) == 0x8664, 'not x64 PE'
    count, optional_size = u16(coff + 2), u16(coff + 16)
    optional = coff + 20
    assert u16(optional) == 0x20b, 'not PE32+'
    sections = optional + optional_size
    def offset(rva):
        for i in range(count):
            section = sections + 40 * i
            virtual_size, address, raw_size, raw = struct.unpack_from('<IIII', data, section + 8)
            if address <= rva < address + min(max(virtual_size, raw_size), raw_size):
                return raw + rva - address
        raise ValueError(f'RVA outside file sections: {rva:x}')
    imports = []
    import_rva = u32(optional + 112 + 8)
    if import_rva:
        pos = offset(import_rva)
        while True:
            descriptor = struct.unpack_from('<IIIII', data, pos)
            if not any(descriptor): break
            name_at = offset(descriptor[3])
            end = data.index(b'\0', name_at)
            imports.append(data[name_at:end].decode('ascii'))
            pos += 20
    delay_rva = u32(optional + 112 + 13 * 8)
    if delay_rva:
        pos = offset(delay_rva)
        while True:
            descriptor = struct.unpack_from('<IIIIIIII', data, pos)
            if not any(descriptor): break
            assert descriptor[0] & 1, 'unsupported VA-based delay import'
            name_at = offset(descriptor[1])
            imports.append(data[name_at:data.index(b'\0', name_at)].decode('ascii'))
            pos += 32
    # A static CRT must not rely on separately installed Microsoft VC runtime DLLs.
    assert not any(n.lower().startswith(('vcruntime', 'msvcp', 'concrt', 'api-ms-win-crt-', 'ucrtbase')) for n in imports), imports
    system = {'kernel32.dll', 'kernelbase.dll', 'ntdll.dll', 'advapi32.dll', 'bcrypt.dll', 'bcryptprimitives.dll', 'ws2_32.dll', 'userenv.dll', 'user32.dll', 'ole32.dll', 'oleaut32.dll', 'shell32.dll', 'secur32.dll', 'crypt32.dll', 'gdi32.dll', 'shlwapi.dll', 'version.dll', 'ntmarta.dll', 'rpcrt4.dll'}
    assert all(n.lower() in system or n.lower().startswith(('api-ms-win-', 'ext-ms-win-')) for n in imports), f'unreviewed imports: {imports}'
else:
    output = subprocess.check_output(['otool', '-L', str(binary)], text=True)
    imports = [line.strip().split(' (', 1)[0] for line in output.splitlines()[1:]]
    assert all(n.startswith(('/usr/lib/', '/System/Library/')) for n in imports), imports
report.write_text('\n'.join(sorted(imports)) + '\n', encoding='utf-8')
print('runtime imports:', ', '.join(sorted(imports)))
