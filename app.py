"""Interactive explorer for the EuroSAT land-cover experiments.

Reads the saved metrics, confusion matrices and projection weights from
outputs/ and presents them as a walkthrough of the three experiments.

    streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
REPORTS = ROOT / "outputs" / "reports"
CKPTS = ROOT / "outputs" / "checkpoints"

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
GROUPED = ["AnnualCrop", "HerbaceousVegetation", "Pasture", "PermanentCrop",
           "Highway", "Industrial", "Residential", "River", "SeaLake", "Forest"]
GROUPS = {"crops and vegetation": GROUPED[:4], "built-up": GROUPED[4:7],
          "water": GROUPED[7:9], "forest": GROUPED[9:]}
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
REGION = {"B01": "visible", "B02": "visible", "B03": "visible", "B04": "visible",
          "B05": "red edge", "B06": "red edge", "B07": "red edge",
          "B08": "NIR", "B8A": "NIR", "B11": "SWIR", "B12": "SWIR", "B09": "visible"}

st.set_page_config(page_title="EuroSAT land-cover experiments", layout="wide")


@st.cache_data
def metrics(run):
    p = REPORTS / run / "metrics.json"
    return json.load(open(p)) if p.exists() else None


@st.cache_data
def confusion(run):
    p = REPORTS / run / "confusion_matrix.npy"
    return np.load(p) if p.exists() else None


@st.cache_data
def per_class_f1(run):
    p = REPORTS / run / "classification_report.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return {c: d[c]["f1-score"] for c in CLASSES if c in d}


def acc(run):
    m = metrics(run)
    return m["accuracy"] * 100 if m else None


RUNS = {
    "ScratchCNN 0.094M": "scratch_cnn_long",
    "ScratchCNN 0.094M + KD": "scratch_cnn_distilled",
    "ScratchCNN-S 0.39M": "scratch_cnn_s_plain",
    "ScratchCNN-S 0.39M + KD": "scratch_cnn_s_distill",
    "ScratchCNN-M 1.56M": "scratch_cnn_m_plain",
    "ScratchCNN-M 1.56M + KD": "scratch_cnn_m_distill",
    "ResNet-18 (ImageNet)": "resnet18",
    "EfficientNet-V2-S": "efficientnet_v2_s",
    "RGB (3 bands)": "scratch_cnn_ms_rgb",
    "All bands (12)": "scratch_cnn_ms_all",
    "Bands + indices (15)": "scratch_cnn_ms_indices",
    "Surface bands (10)": "scratch_cnn_ms_surface",
    "k=1 learned": "scratch_cnn_ms_proj1",
    "k=2 learned": "scratch_cnn_ms_proj2",
    "k=3 learned": "scratch_cnn_ms_proj3",
    "k=4 learned": "scratch_cnn_ms_proj4",
    "k=6 learned": "scratch_cnn_ms_proj6",
}

page = st.sidebar.radio("View", [
    "Overview",
    "1. Capacity and distillation",
    "2. Input bands",
    "3. Learned spectral projection",
    "Where the errors are",
    "Reproducibility",
])
st.sidebar.caption("Reads outputs/reports/ and outputs/checkpoints/.")


# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("What drives accuracy on EuroSAT?")
    st.write(
        "EuroSAT sits above 98% for large pretrained networks, which leaves little "
        "separation between published methods. These three experiments hold the "
        "backbone family fixed and vary one factor at a time."
    )
    cols = st.columns(3)
    for col, (name, body) in zip(cols, [
        ("Capacity and training signal",
         "Three model sizes, each with and without distillation from a 20.2M teacher."),
        ("Input bands",
         "One fixed 1.56M model on RGB, 12 bands, 15 with indices, 10 surface-only."),
        ("Learned projection",
         "A 1x1 layer mixes 12 bands into k channels, swept over k."),
    ]):
        col.subheader(name)
        col.write(body)

    st.subheader("All runs")
    rows = []
    for label, run in RUNS.items():
        a = acc(run)
        if a is not None:
            rows.append({"configuration": label, "test accuracy (%)": round(a, 2),
                         "macro F1": round(metrics(run)["f1_macro"], 4)})
    df = pd.DataFrame(rows).sort_values("test accuracy (%)", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ------------------------------------------------- 1. capacity/distillation
elif page.startswith("1."):
    st.title("Capacity and distillation")
    st.write("Width and depth move together, so this varies model scale rather than width alone.")
    pairs = [("0.094M", "scratch_cnn_long", "scratch_cnn_distilled"),
             ("0.39M", "scratch_cnn_s_plain", "scratch_cnn_s_distill"),
             ("1.56M", "scratch_cnn_m_plain", "scratch_cnn_m_distill")]
    rows = []
    for size, plain, kd in pairs:
        p, k = acc(plain), acc(kd)
        if p and k:
            rows.append({"model": size, "plain (%)": round(p, 2),
                         "+ distillation (%)": round(k, 2), "change (%)": round(k - p, 2)})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    m = df.melt("model", ["plain (%)", "+ distillation (%)"],
                var_name="training", value_name="accuracy")
    st.altair_chart(alt.Chart(m).mark_bar().encode(
        x=alt.X("model:N", sort=None, title=""),
        xOffset="training:N",
        y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[90, 98]),
                title="test accuracy (%)"),
        color=alt.Color("training:N", title=""),
        tooltip=["model", "training", "accuracy"]), use_container_width=True)
    t = acc("efficientnet_v2_s")
    st.info(
        f"The teacher reaches {t:.2f}%, so the soft targets are accurate. "
        "Distillation still changes accuracy by less than run-to-run variation at the "
        "two larger sizes, and lowers the smallest student by 0.82%."
    )


# ----------------------------------------------------------- 2. input bands
elif page.startswith("2."):
    st.title("Input bands")
    st.write("Same 1.56M network throughout; only the input channels change.")
    band_runs = [("RGB (3 bands)", "scratch_cnn_ms_rgb"),
                 ("Surface bands (10)", "scratch_cnn_ms_surface"),
                 ("All bands (12)", "scratch_cnn_ms_all"),
                 ("Bands + indices (15)", "scratch_cnn_ms_indices")]
    df = pd.DataFrame([{"input": n, "accuracy (%)": round(acc(r), 2)}
                       for n, r in band_runs if acc(r)])
    lo = df["accuracy (%)"].min() - 0.4
    hi = df["accuracy (%)"].max() + 0.2
    st.altair_chart(alt.Chart(df).mark_bar().encode(
        x=alt.X("input:N", sort=None, title=""),
        y=alt.Y("accuracy (%):Q", scale=alt.Scale(domain=[lo, hi])),
        tooltip=["input", "accuracy (%)"]), use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Per-class F1: RGB against 12 bands")
    a, b = per_class_f1("scratch_cnn_ms_rgb"), per_class_f1("scratch_cnn_ms_all")
    if a and b:
        d = pd.DataFrame({"class": CLASSES,
                          "RGB": [round(a[c], 3) for c in CLASSES],
                          "12 bands": [round(b[c], 3) for c in CLASSES]})
        d["gain"] = (d["12 bands"] - d["RGB"]).round(3)
        st.dataframe(d.sort_values("gain", ascending=False),
                     use_container_width=True, hide_index=True)
        st.caption("The gain concentrates on River and the crop classes; classes already "
                   "above 0.99 barely move.")
    s, al = acc("scratch_cnn_ms_surface"), acc("scratch_cnn_ms_all")
    if s and al:
        st.info(f"Dropping the atmospheric bands B01 and B09 gives {s:.2f}% against "
                f"{al:.2f}% for all twelve, a difference of {s - al:+.2f}%. The gain does "
                "not depend on them.")


# --------------------------------------------------- 3. learned projection
elif page.startswith("3."):
    st.title("Learned spectral projection")
    st.write("A 1x1 convolution maps the 12 standardised bands to k channels, trained "
             "end to end with the classifier.")
    ks = [1, 2, 3, 4, 6]
    sweep = [{"k": k, "accuracy (%)": round(acc(f"scratch_cnn_ms_proj{k}"), 2)}
             for k in ks if acc(f"scratch_cnn_ms_proj{k}")]
    df = pd.DataFrame(sweep)
    rgb = acc("scratch_cnn_ms_rgb")
    full = acc("scratch_cnn_ms_all")
    lo = min(df["accuracy (%)"].min(), rgb) - 0.3
    hi = max(df["accuracy (%)"].max(), full) + 0.3
    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("k:O", title="channel budget k"),
        y=alt.Y("accuracy (%):Q", scale=alt.Scale(domain=[lo, hi]),
                title="test accuracy (%)"),
        tooltip=["k", "accuracy (%)"])
    rules = alt.Chart(pd.DataFrame({
        "y": [rgb, full], "label": ["RGB, 3 channels", "all 12 bands"]})).mark_rule(
        strokeDash=[4, 4]).encode(y="y:Q", color=alt.Color("label:N", title=""))
    st.altair_chart(line + rules, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("RGB, 3 fixed channels", f"{acc('scratch_cnn_ms_rgb'):.2f}%")
    c2.metric("k=3 learned channels", f"{acc('scratch_cnn_ms_proj3'):.2f}%",
              f"{acc('scratch_cnn_ms_proj3') - acc('scratch_cnn_ms_rgb'):+.2f}%")

    st.subheader("What the learned channels measure")
    k = st.selectbox("channel budget", ks, index=2)
    try:
        import torch
        p = CKPTS / f"scratch_cnn_ms_proj{k}" / "best_model.pth"
        sd = torch.load(p, map_location="cpu", weights_only=True)
        W = [v for key, v in sd.items() if "projection" in key][0]
        W = W.squeeze(-1).squeeze(-1).numpy()
        # sign of a row is arbitrary; draw with the largest weight positive
        W = np.array([r * (1 if r[np.argmax(np.abs(r))] > 0 else -1) for r in W])
        st.dataframe(pd.DataFrame(W, columns=BANDS,
                                  index=[f"channel {i}" for i in range(len(W))]).round(3),
                     use_container_width=True)
        share = []
        for i, row in enumerate(W):
            m = np.abs(row); tot = m.sum()
            share.append({"channel": f"channel {i}", **{
                reg: round(sum(m[j] for j, b in enumerate(BANDS) if REGION[b] == reg) / tot * 100, 1)
                for reg in ["visible", "red edge", "NIR", "SWIR"]}})
        st.write("Share of each channel's absolute weight by spectral region (%)")
        st.dataframe(pd.DataFrame(share), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Projection weights unavailable: {e}")


# ------------------------------------------------- where the errors are
elif page.startswith("Where"):
    st.title("Where the errors are")
    choice = st.selectbox("configuration", ["RGB (3 bands)", "All bands (12)",
                                            "k=3 learned", "Surface bands (10)"])
    run = RUNS[choice]
    cm = confusion(run)
    if cm is None:
        st.warning("No confusion matrix saved for this run.")
    else:
        order = [CLASSES.index(c) for c in GROUPED]
        m = cm[np.ix_(order, order)].astype(int)
        total = int(m.sum()); errors = total - int(np.trace(m))
        c1, c2, c3 = st.columns(3)
        c1.metric("test patches", f"{total:,}")
        c2.metric("errors", errors)
        c3.metric("accuracy", f"{(total - errors) / total * 100:.2f}%")

        off = m.copy(); np.fill_diagonal(off, 0)
        st.write("Off-diagonal counts, classes ordered so groups sit together")
        st.dataframe(pd.DataFrame(off, index=GROUPED, columns=GROUPED),
                     use_container_width=True)

        idx = {c: i for i, c in enumerate(GROUPED)}
        within = {}
        for g, members in GROUPS.items():
            ii = [idx[c] for c in members]
            within[g] = int(sum(m[a][b] for a in ii for b in ii if a != b))
        cross = errors - sum(within.values())
        st.write("Error structure")
        st.dataframe(pd.DataFrame([{"grouping": "crossing a group boundary", "errors": cross},
                                   *[{"grouping": f"within {g}", "errors": n}
                                     for g, n in within.items()]]),
                     use_container_width=True, hide_index=True)
        st.caption("Spectral bands mostly remove errors that cross a group boundary. "
                   "Confusions inside the vegetation group persist.")


# ------------------------------------------------------- reproducibility
else:
    st.title("Reproducibility")
    st.subheader("Multi-seed runs")
    seeds = {"RGB (3 bands)": ["scratch_cnn_ms_rgb", "scratch_cnn_ms_rgb_s43",
                               "scratch_cnn_ms_rgb_s44"],
             "k=2 learned": ["scratch_cnn_ms_proj2", "scratch_cnn_ms_proj2_s43",
                             "scratch_cnn_ms_proj2_s44"]}
    rows = []
    for label, runs in seeds.items():
        vals = [acc(r) for r in runs if acc(r)]
        if vals:
            rows.append({"configuration": label, "n seeds": len(vals),
                         "runs (%)": ", ".join(f"{v:.2f}" for v in vals),
                         "mean (%)": round(float(np.mean(vals)), 2),
                         "sd (%)": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        a = [acc(r) for r in seeds["k=2 learned"] if acc(r)]
        b = [acc(r) for r in seeds["RGB (3 bands)"] if acc(r)]
        if len(a) > 1 and len(b) > 1:
            st.info(f"Lowest k=2 run {min(a):.2f}% against highest RGB run {max(b):.2f}%: "
                    f"{'no overlap' if min(a) > max(b) else 'ranges overlap'}.")

    st.subheader("Paired significance, McNemar")
    p = REPORTS / "mcnemar.json"
    if p.exists():
        d = pd.DataFrame(json.load(open(p)))
        d.columns = ["A", "B", "A only correct", "B only correct", "p"]
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.caption("All runs share one fixed test split, so the comparisons are paired.")
    else:
        st.write("Run `python scripts/mcnemar.py` to generate these.")

    st.subheader("Training setup")
    st.write("40 epochs at 64x64, batch 64, AdamW at 1e-3 with weight decay 1e-4, "
             "cosine annealing to zero. Pretrained backbones: 15 epochs at 224x224, "
             "batch 32, 3e-4. Checkpoint selected on lowest validation cross-entropy.")
