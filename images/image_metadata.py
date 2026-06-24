from __future__ import annotations

import struct
from pathlib import Path


def read_image_dimensions(image_path: Path) -> tuple[int, int]:
    data = image_path.read_bytes()
    suffix = image_path.suffix.lower()

    if suffix == ".png" and len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height

    if suffix in {".jpg", ".jpeg"}:
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue

            marker = data[index + 1]
            index += 2

            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue

            segment_length = int.from_bytes(data[index:index + 2], "big")
            if segment_length < 2 or index + segment_length > len(data):
                break

            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                height = int.from_bytes(data[index + 3:index + 5], "big")
                width = int.from_bytes(data[index + 5:index + 7], "big")
                return width, height

            index += segment_length

    if suffix == ".gif" and len(data) >= 10 and data[:6] in {b"GIF87a", b"GIF89a"}:
        width = int.from_bytes(data[6:8], "little")
        height = int.from_bytes(data[8:10], "little")
        return width, height

    if suffix == ".bmp" and len(data) >= 26 and data[:2] == b"BM":
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = abs(int.from_bytes(data[22:26], "little", signed=True))
        return width, height

    if suffix == ".webp" and len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk_type = data[12:16]
        if chunk_type == b"VP8X" and len(data) >= 30:
            width = 1 + int.from_bytes(data[24:27], "little")
            height = 1 + int.from_bytes(data[27:30], "little")
            return width, height
        if chunk_type == b"VP8 " and len(data) >= 30:
            start_code_index = data.find(b"\x9d\x01\x2a")
            if start_code_index != -1 and start_code_index + 7 < len(data):
                width = struct.unpack("<H", data[start_code_index + 3:start_code_index + 5])[0] & 0x3FFF
                height = struct.unpack("<H", data[start_code_index + 5:start_code_index + 7])[0] & 0x3FFF
                return width, height
        if chunk_type == b"VP8L" and len(data) >= 25:
            bits = int.from_bytes(data[21:25], "little")
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height

    raise ValueError(f"Could not read image dimensions from: {image_path.name}")


def build_image_metadata(image_path: Path) -> dict:
    width, height = read_image_dimensions(image_path)
    return {
        "file_name": image_path.name,
        "width": width,
        "height": height,
    }

