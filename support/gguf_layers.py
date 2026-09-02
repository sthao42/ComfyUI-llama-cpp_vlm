from typing import BinaryIO, Optional, Any
import struct

# Struct format lookup maps for 1-step binary parsing
_TYPE_UNPACK_MAP = {
    0: "<B",   # uint8
    1: "<b",   # int8
    2: "<H",   # uint16
    3: "<h",   # int16
    4: "<I",   # uint32
    5: "<i",   # int32
    6: "<f",   # float32
    7: "<?",   # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}

_TYPE_SIZE_MAP = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8
}


def read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def read_string(f: BinaryIO) -> str:
    length = read_u64(f)
    return f.read(length).decode("utf-8")


def read_value(f: BinaryIO) -> Any:
    vtype = read_u32(f)

    if vtype in _TYPE_UNPACK_MAP:
        return struct.unpack(_TYPE_UNPACK_MAP[vtype], f.read(_TYPE_SIZE_MAP[vtype]))[0]
    if vtype == 8:   # string
        return read_string(f)
    if vtype == 9:   # array
        atype = read_u32(f)
        count = read_u64(f)
        return [read_value_of_type(f, atype) for _ in range(count)]

    raise ValueError(f"Unknown GGUF value type {vtype}")


def read_value_of_type(f: BinaryIO, atype: int) -> Any:
    if atype in _TYPE_UNPACK_MAP:
        return struct.unpack(_TYPE_UNPACK_MAP[atype], f.read(_TYPE_SIZE_MAP[atype]))[0]
    if atype == 8:
        return read_string(f)

    raise ValueError(f"Unknown GGUF array item type {atype}")


def get_layer_count(path: str) -> Optional[int]:
    """Parse GGUF metadata to extract block_count/layer_count without external dependencies."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError("Not a valid GGUF binary file.")

            _version = read_u32(f)  # noqa: F841
            _tensor_count = read_u64(f)  # noqa: F841
            kv_count = read_u64(f)
            meta = {}

            for _ in range(kv_count):
                key = read_string(f)
                value = read_value(f)
                meta[key] = value

        for k, v in meta.items():
            if k.lower().endswith(".block_count"):
                return int(v)
    except Exception as e:
        print(f"[llama-cpp_vlm] Warning parsing GGUF manually: {e}")

    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)

        layer_count = reader.get_field("llama.block_count")
        if layer_count is None:
            for field in reader.fields.values():
                if field.name.endswith(".block_count"):
                    layer_count = field.parts[field.data[0]]
                    break

        if layer_count:
            return int(layer_count[0] if isinstance(layer_count, list) else layer_count)
    except Exception as e:
        print(f"[llama-cpp_vlm] Failed to get block_count: {e}")

    return None