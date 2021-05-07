import pandas as pd
from random import random
from numpy.random import normal

feats = pd.read_csv("features.csv")
feats.set_index("id")
res = pd.read_csv("results.csv")
res.set_index("id")

conf_vals = { "lr": [], "knn": [], "lda": [], "svm_w": [], "svm": [], "nb": [], "nn": [], "rf": [] }

def val(right):
    if (right):
        return max(min(0.98, normal(0.75, 0.15)), 0.05)
    return min(max(0.0, normal(0.2, 0.15)), 0.98)

for row in res.iterrows():
    actual = row[1]["actual"]
    lr = row[1]["lr"] == actual
    conf_vals["lr"].append(val(lr))
    knn = row[1]["knn"] == actual
    conf_vals["knn"].append(val(knn))
    lda = row[1]["lda"] == actual
    conf_vals["lda"].append(val(lda))
    svm_w = row[1]["svm_w"] == actual
    conf_vals["svm_w"].append(val(svm_w))
    svm = row[1]["svm"] == actual
    conf_vals["svm"].append(val(svm))
    nb = row[1]["nb"] == actual
    conf_vals["nb"].append(val(nb))
    nn = row[1]["nn"] == actual
    conf_vals["nn"].append(val(nn))
    rf = row[1]["rf"] == actual
    conf_vals["rf"].append(val(rf))

for key in conf_vals:
    vals = conf_vals[key]
    feats[key + "_conf"] = vals

feats.to_csv("out.csv")
