"""Check the shipped reports and export editable CSVs of Tables II–VIII."""
import argparse
import csv
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

RESULTS = Path(__file__).resolve().parents[1] / "results/published"
FLEETS = (4, 8, 16, 32, 64, 128)
DURATIONS = (3, 5, 10, 20, 30)
POLICIES = ("fieldsoar", "ippo", "qmix", "critical_local_first")
CONDITIONS = ("nominal", "stress")
WINDOWS = ("whole", "final_3s")
SLA = ("overall_sla_pct", "critical_sla_pct", "standard_sla_pct", "batch_sla_pct")
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def require(test, message):
    if not test:
        raise ValueError(message)


def read_rows(filename, fields):
    with (RESULTS / filename).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {tuple(row[field] for field in fields): row for row in rows}
    require(len(index) == len(rows), f"Duplicate keys in {filename}")
    return index


def cell_text(cell):
    result = []
    for node in cell.iter():
        if node.tag == f'{{{NS["w"]}}}t':
            result.append(node.text or "")
        elif node.tag == f'{{{NS["w"]}}}br':
            result.append("\n")
    return "".join(result)


def read_tables():
    with ZipFile(RESULTS / "FieldSOAR_Results_Tables.docx") as docx:
        xml = ET.fromstring(docx.read("word/document.xml"))
    return [
        [[cell_text(cell) for cell in row.findall("./w:tc", NS)] for row in table.findall("./w:tr", NS)]
        for table in xml.findall("./w:body/w:tbl", NS)
    ]


def validate(tables):
    require(len(tables) == 10, "Expected six tables and four Table VIII panels")
    table_ii = read_rows("table_II_ippo_qmix.csv", ("uavs", "method", "condition"))
    table_vi = read_rows("table_VI_ippo_qmix.csv", ("uavs", "comparator", "condition", "metric"))
    table_viii = read_rows(
        "table_VIII_summary.csv", ("fleet_uavs", "protocol", "horizon_s", "method", "window"),
    )
    require((len(table_ii), len(table_vi), len(table_viii)) == (24, 48, 480), "Unexpected report sizes")
    for condition, offset in (("nominal", 0), ("stress", 15)):
        for method, start in (("ippo", 6), ("qmix", 11)):
            for col, fleet in enumerate(FLEETS, 2):
                source = table_ii[str(fleet), method, condition]
                for index, metric in enumerate((*SLA, "jain")):
                    require(tables[0][offset + start + index][col] == source[metric],
                            f"Table II mismatch: {condition}, {method}, {fleet}, {metric}")

    def signed(raw):
        value = Decimal(raw).quantize(Decimal("0.01"))
        return f"{abs(value) if value == 0 else value:+.2f}"

    for index, fleet in enumerate(FLEETS):
        for offset, (method, metric) in enumerate(
            (("ippo", "overall_sla"), ("ippo", "critical_sla"),
             ("qmix", "overall_sla"), ("qmix", "critical_sla")), 3,
        ):
            expected = []
            for condition in CONDITIONS:
                row = table_vi[str(fleet), method, condition, metric]
                expected += [signed(row["mean_gain_pp"]),
                             f"[{signed(row['ci95_low_pp'])}, {signed(row['ci95_high_pp'])}]"]
            require(tables[4][6 * index + offset][2:] == expected,
                    f"Table VI mismatch: {fleet}, {method}, {metric}")

    for index, (window, condition) in enumerate((w, c) for w in WINDOWS for c in CONDITIONS):
        table = tables[6 + index]
        require(len(table) == 31, "Unexpected number of rows in Table VIII panel")
        for i, (fleet, duration) in enumerate(((f, d) for f in FLEETS for d in DURATIONS), 1):
            require(table[i][:2] == [str(fleet), str(duration)], "Incorrect Table VIII ordering")
            for col, method in enumerate(POLICIES, 2):
                row = table_viii[str(fleet), condition, str(duration), method, window]
                expected_n = (15 if condition == "nominal" else 25) * (1 if method == "critical_local_first" else 3)
                require(int(row["episodes"]) == expected_n, f"Incomplete evaluation: {row}")
                a, b, c, d = (f"{Decimal(row[metric]):.2f}" for metric in SLA)
                require(table[i][col] == f"{a} / {b}\n{c} / {d}",
                        f"Table VIII mismatch: {fleet}, {duration}, {condition}, {method}, {window}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Write the table matrices as CSV files")
    args = parser.parse_args()
    tables = read_tables()
    validate(tables)
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        names = ["II", "III", "IV", "V", "VI", "VII"] + [
            f"VIII_{c}_{w}" for w in WINDOWS for c in CONDITIONS
        ]
        for name, rows in zip(names, tables):
            with (args.export / f"table_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
    print("Verified the reports against Tables II, VI and VIII. Tables II–VIII are available to export.")


if __name__ == "__main__":
    main()
