"""Diag A — enumerate GRAPE's clinical metadata (the gate for the metadata lever).

MLEDL's proven route to 3.575 dB was adding clinical metadata (age, IOP, CCT, history) to the photo.
We already own GRAPE's Excel; this dumps every sheet's columns, non-null counts and sample values so
we know EXACTLY which fields are available at inference before planning the metadata model. No torch.

  python decoder/diag_metadata_inventory.py
"""
import os, re, zipfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "data", "vf_tests", "grape_data.xlsx")


def load_sheets(path):
    z = zipfile.ZipFile(path)
    shared = re.findall(r"<t[^>]*>([^<]*)</t>", z.read("xl/sharedStrings.xml").decode("utf8", "ignore"))
    wb = z.read("xl/workbook.xml").decode("utf8", "ignore")
    names = re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="(rId\d+)"', wb)
    relmap = dict(re.findall(r'Id="(rId\d+)"[^>]*?Target="([^"]+)"',
                             z.read("xl/_rels/workbook.xml.rels").decode("utf8", "ignore")))
    sheets = {}
    for name, rid in names:
        xml = z.read("xl/" + relmap[rid].lstrip("/")).decode("utf8", "ignore")
        rows = defaultdict(dict)
        for ref, attr, inner in re.findall(r'<c r="([A-Z]+\d+)"([^>]*)>(.*?)</c>', xml):
            m = re.match(r"([A-Z]+)(\d+)", ref)
            col = 0
            for ch in m.group(1):
                col = col * 26 + (ord(ch) - 64)
            v = re.search(r"<v>([^<]*)</v>", inner)
            if v is None:
                continue
            val = v.group(1)
            if 't="s"' in attr:
                val = shared[int(val)] if int(val) < len(shared) else val
            rows[int(m.group(2))][col] = val
        sheets[name] = rows
    return sheets


def main():
    sheets = load_sheets(XLSX)
    print(f"GRAPE {os.path.basename(XLSX)} — sheets: {list(sheets)}\n")
    for name, rows in sheets.items():
        rnos = sorted(rows)
        if not rnos:
            continue
        hdr = rows[rnos[0]]
        ncol = max(max(rows[r], default=0) for r in rnos)
        data_rows = [r for r in rnos if r >= rnos[0] + 2]     # skip header + possible sub-header
        print("=" * 96)
        print(f"SHEET '{name}'  — {len(rnos)} rows, {ncol} cols  (data rows ≈ {len(data_rows)})")
        print("-" * 96)
        print(f"{'col':>4}  {'header':<34}{'non-null':>9}  sample values")
        for c in range(1, ncol + 1):
            header = str(hdr.get(c, ""))[:33]
            nn = sum(1 for r in data_rows if c in rows[r])
            samples = [str(rows[r][c]) for r in data_rows[:60] if c in rows[r]][:3]
            # collapse the long VF value block (many numeric cols) into one note
            print(f"{c:>4}  {header:<34}{nn:>9}  {', '.join(samples)}")
        print()


if __name__ == "__main__":
    main()
