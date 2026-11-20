"""CI gate: README and measured.json must not drift into unsupported claims."""
import json, sys, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
d = json.load(open(root / "results" / "measured.json"))
readme = (root / "README.md").read_text()
errs = []

v = d["verification"]
if v["unit_tests_passed"] != v["unit_tests_total"]:
    errs.append("measured.json says tests are failing")

inc = d["experiment_5M_INCOMPLETE"]
if inc["seq_baseline_at_5M"] == "NOT RUN":
    if "INCOMPLETE" not in readme:
        errs.append("5M baseline is NOT RUN but README no longer says INCOMPLETE")
    if not inc.get("status", "").startswith("INCOMPLETE"):
        errs.append("5M status field must start with INCOMPLETE while baseline is missing")

if len(d["unsupported_claims"]) < 5:
    errs.append("unsupported_claims list was shortened below 5 entries")

for phrase in ("Not novel", "Not tested on language", "Mechanism unexplained"):
    if phrase not in readme:
        errs.append("README lost required caveat: " + phrase)

e = d["experiment_20k_COMPLETE"]
for key, val in (("dense_mean", e["dense"]["mean"]), ("seq_mean", e["seq"]["mean"])):
    pct = f"{val*100:.2f}%"
    if pct not in readme:
        errs.append(f"README does not contain measured {key} = {pct}")

if errs:
    print("HONESTY GATE FAILED:")
    for x in errs:
        print("  -", x)
    sys.exit(1)
print("honesty gates OK")
