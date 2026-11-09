"""The <bin> part of the binary blocks: embedded header + payload + CRC-32 (docs/protocol).

Used by :AWG:DATA? and :AWG:GATing? (blocks sent to the device) and by the reply of
:ACQuire:DATA:GET? <index> (blocks read from it). The header is the same for every
command (bin_block_emb_header.md); every command defines its own payload versions.

    <bin> = header | payload | checksum (4 B)
    header version 2 = header version (1 B) | payload version (1 B) | chunk index (4 B)
"""

from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass

import numpy as np

HEADER_VERSION = 2
HEADER_LENGTHS = {1: 3, 2: 6}
CHECKSUM_LENGTH = 4
MAX_CHUNK_INDEX = 0xFFFFFFFF

_HEADER_V2 = struct.Struct("<BBI")
_CHECKSUM = struct.Struct("<I")

_REVERSED_BITS = bytes(int(f"{b:08b}"[::-1], 2) for b in range(256))


def crc32(data: bytes) -> int:
    """CRC-32 as agreed: polynomial 0x04C11DB7, initial value 0xFFFFFFFF, no bit
    inversions (reflections), no final XOR, computed on 32-bit words - the STM32 CRC unit.

    ASSUMED: the words are read from memory as little-endian uint32 (as the STM32 does)
    and a block whose length is not a multiple of 4 is padded with zero bytes. Each word
    is processed from its most significant byte.

    Computed with zlib's CRC-32 (the same polynomial, reflected, with a final XOR): fed
    the bytes with their bits reversed, it gives the bit-reversed register of the
    unreflected CRC - in C, a few hundred times faster than a loop in Python.
    """
    padded = data + bytes(-len(data) % 4)
    msb_first = np.frombuffer(padded, dtype=np.uint8).reshape(-1, 4)[:, ::-1].tobytes()
    register = zlib.crc32(msb_first.translate(_REVERSED_BITS)) ^ 0xFFFFFFFF
    return int(f"{register:032b}"[::-1], 2)


class BlockError(ValueError):
    """A received <bin> block is damaged or not understood."""


@dataclass(frozen=True)
class Block:
    header_version: int
    payload_version: int
    chunk_index: int | None
    payload: bytes


def build_block(payload_version: int, chunk_index: int, payload: bytes) -> bytes:
    """<bin> with a version-2 header and the checksum of everything before it."""
    if not 0 <= chunk_index <= MAX_CHUNK_INDEX:
        raise ValueError(f"chunk index must be 0-{MAX_CHUNK_INDEX}, got {chunk_index}")
    body = _HEADER_V2.pack(HEADER_VERSION, payload_version, chunk_index) + payload
    return body + _CHECKSUM.pack(crc32(body))


def parse_block(data: bytes) -> Block:
    """Split a received <bin> into header and payload; check the checksum."""
    if len(data) < 2 + CHECKSUM_LENGTH:
        raise BlockError(f"block of {len(data)} B is too short")
    header_version, payload_version = data[0], data[1]
    header_length = HEADER_LENGTHS.get(header_version)
    if header_length is None:
        raise BlockError(f"header version {header_version} is not supported")
    if len(data) < header_length + CHECKSUM_LENGTH:
        raise BlockError(f"block of {len(data)} B is too short for header version {header_version}")
    body, (checksum,) = data[:-CHECKSUM_LENGTH], _CHECKSUM.unpack(data[-CHECKSUM_LENGTH:])
    computed = crc32(body)
    if computed != checksum:
        raise BlockError(f"checksum mismatch (got 0x{checksum:08X}, computed 0x{computed:08X})")
    chunk_index = _HEADER_V2.unpack(data[:6])[2] if header_version == 2 else None
    return Block(header_version, payload_version, chunk_index, body[header_length:])


_ERR_REPLY = re.compile(r"^\s*err-(\d{1,3})\s*$", re.IGNORECASE)


def error_code(reply: str) -> int:
    """The number of an "err-<code>" reply (0 = no error)."""
    match = _ERR_REPLY.match(reply)
    if match is None:
        raise ValueError(f"expected an 'err-<code>' reply, got {reply!r}")
    return int(match[1])
