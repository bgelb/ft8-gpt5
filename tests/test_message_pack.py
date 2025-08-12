from ft8gpt.ft8pack import pack_standard_payload


def test_pack_standard_payload_roundtrip_shape():
    a = pack_standard_payload("K1ABC", "W9XYZ", "FN20")
    assert isinstance(a, (bytes, bytearray))
    assert len(a) == 10

