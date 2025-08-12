#!/usr/bin/env python3
import re, pathlib
SRC = pathlib.Path("third_party/ft8_lib/ft8/constants.c")
out = pathlib.Path("ft8dec/ldpc_data.py")
text = SRC.read_text()
# Extract kFTX_LDPC_Nm and kFTX_LDPC_Mn arrays
nm_match = re.search(r"const uint8_t kFTX_LDPC_Nm\[.*?\]\s*=\s*\{([\s\S]*?)\};", text)
mn_match = re.search(r"const uint8_t kFTX_LDPC_Mn\[.*?\]\s*=\s*\{([\s\S]*?)\};", text)
costas_match = re.search(r"const uint8_t kFT8_Costas_pattern\[7\]\s*=\s*\{([^}]*)\};", text)
gray_match = re.search(r"const uint8_t kFT8_Gray_map\[8\]\s*=\s*\{([^}]*)\};", text)
if not (nm_match and mn_match and costas_match and gray_match):
    raise SystemExit("Failed to find arrays in constants.c")

def parse_rows(block: str):
    rows=[]
    for line in block.splitlines():
        line=line.strip()
        if not line or line.startswith("//"): continue
        # lines like { 4, 31, 59, 91, 92, 96, 153 },
        m=re.search(r"\{([^}]*)\}", line)
        if not m: continue
        nums=[int(x.strip()) for x in m.group(1).split(',') if x.strip()]
        rows.append(nums)
    return rows

def parse_list(block:str):
    return [int(x.strip()) for x in block.split(',') if x.strip()]

Nm_rows = parse_rows(nm_match.group(1))
Mn_rows = parse_rows(mn_match.group(1))
Costas = parse_list(costas_match.group(1))
Gray = parse_list(gray_match.group(1))

out.write_text(
    """
# Auto-generated from third_party/ft8_lib/ft8/constants.c at a fixed commit.
# Contains LDPC adjacency and FT8 constants used by the decoder.
Nm = %r
Mn = %r
FT8_COSTAS = %r
FT8_GRAY = %r
""".strip() % (Nm_rows, Mn_rows, Costas, Gray)
)
print(f"Wrote {out}")
