"""Hash executable ELF sections, including ELF images inside a MACA offload bundle."""
import hashlib
import struct


def sections(blob):
    magic = b"__CLANG_OFFLOAD_BUNDLE__"
    if blob.startswith(magic):
        pos = len(magic)
        count, = struct.unpack_from("<Q", blob, pos)
        pos += 8
        out = {}
        for _ in range(count):
            offset, size, namelen = struct.unpack_from("<QQQ", blob, pos)
            pos += 24
            name = blob[pos:pos + namelen].decode()
            pos += namelen
            if size and blob[offset:offset + 4] == b"\x7fELF":
                out[name] = sections(blob[offset:offset + size])
        if not out:
            raise ValueError("The MACA offload bundle contains no ELF device images.")
        return out
    if blob[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected a little-endian 64-bit ELF device image.")
    shoff, = struct.unpack_from("<Q", blob, 40)
    stride, count, stridx = struct.unpack_from("<HHH", blob, 58)
    table = [struct.unpack_from("<IIQQQQIIQQ", blob, shoff + i * stride) for i in range(count)]
    names = table[stridx]
    names = blob[names[4]:names[4] + names[5]]
    executable = {}
    for name_offset, _, flags, _, offset, size, *_ in table:
        if flags & 4 and size:  # SHF_EXECINSTR
            name = names[name_offset:].split(b"\x00", 1)[0].decode()
            digest = hashlib.sha256(blob[offset:offset + size]).hexdigest()
            executable[name] = [size, digest]
    if not executable:
        raise ValueError("The ELF device image contains no executable sections.")
    return executable
