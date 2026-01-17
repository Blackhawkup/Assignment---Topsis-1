import sys
import pandas as pd
import numpy as np

if len(sys.argv) != 5:
    print("Usage: python topsis.py <InputDataFile> <Weights> <Impacts> <OutputFile>")
    sys.exit(1)

input_file = "help.csv"
weights = sys.argv[2].split(",")
impacts = sys.argv[3].split(",")
output_file = sys.argv[4]

try:
    df = pd.read_csv(input_file)
except:
    print("File not found")
    sys.exit(1)

if df.shape[1] < 3:
    print("Input file must contain three or more columns")
    sys.exit(1)

data = df.iloc[:, 1:]

try:
    data = data.astype(float)
except:
    print("From 2nd to last columns must contain numeric values only")
    sys.exit(1)

if len(weights) != data.shape[1] or len(impacts) != data.shape[1]:
    print("Number of weights, impacts and criteria columns must be same")
    sys.exit(1)

weights = np.array(weights, dtype=float)

for i in impacts:
    if i not in ["+", "-"]:
        print("Impacts must be either + or -")
        sys.exit(1)

norm = np.sqrt((data ** 2).sum())
normalized = data / norm
weighted = normalized * weights

ideal_best = []
ideal_worst = []

for i in range(len(impacts)):
    if impacts[i] == "+":
        ideal_best.append(weighted.iloc[:, i].max())
        ideal_worst.append(weighted.iloc[:, i].min())
    else:
        ideal_best.append(weighted.iloc[:, i].min())
        ideal_worst.append(weighted.iloc[:, i].max())

ideal_best = np.array(ideal_best)
ideal_worst = np.array(ideal_worst)

dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

score = dist_worst / (dist_best + dist_worst)

df["Topsis Score"] = score
df["Rank"] = score.rank(ascending=False).astype(int)

df.to_csv(output_file, index=False)
