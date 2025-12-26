from pathlib import Path
import pandas as pd
import tifffile

csv_path = Path("/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/images/Test_data.csv")
img_dir  = Path("/home/rovsi/Projects/Vakifbank_Internship/Signature_Detection/KaggleDataSet/images")

df = pd.read_csv(csv_path, names=["name","label","x_min","y_min","x_max","y_max"])
names = df["name"].astype(str).unique()

ok, fail, missing = 0, 0, 0
for n in names:
    p = img_dir / n
    if not p.exists():
        missing += 1
        continue
    try:
        _ = tifffile.imread(str(p))
        ok += 1
    except Exception as e:
        fail += 1
        print("FAIL:", n, "|", e)

print("ok:", ok, "missing:", missing, "decode_fail:", fail)
