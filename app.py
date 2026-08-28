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
MS_ROOT = ROOT / "data" / "raw" / "EuroSATMS"

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

st.set_page_config(page_title="EuroSAT land-cover experiments", layout="wide",
                   initial_sidebar_state="collapsed")


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

CONFIGS = {
    "RGB (3 bands)": "configs/scratch_cnn_ms_rgb.yaml",
    "All bands (12)": "configs/scratch_cnn_ms_all.yaml",
    "Surface bands (10)": "configs/scratch_cnn_ms_surface.yaml",
    "k=2 learned": "configs/scratch_cnn_ms_proj2.yaml",
    "k=3 learned": "configs/scratch_cnn_ms_proj3.yaml",
    "k=6 learned": "configs/scratch_cnn_ms_proj6.yaml",
}


@st.cache_resource(show_spinner="Loading patches...")
def raw_dataset(split="test"):
    from src.data.eurosat_ms import _base_dataset
    return _base_dataset(str(MS_ROOT), split)


@st.cache_resource(show_spinner="Loading model...")
def model_for(label):
    from src.visualization.interpret import load_model
    run = RUNS[label]
    return load_model(str(ROOT / CONFIGS[label]),
                      str(CKPTS / run / "best_model.pth"))


@st.cache_resource(show_spinner="Preparing inputs...")
def dataset_for(label, split="test"):
    from src.visualization.interpret import make_dataset
    return make_dataset(str(ROOT / CONFIGS[label]), split)


@st.cache_data(show_spinner="Sampling patches...")
def indices_by_class(per_class, split="test", seed=0):
    from src.visualization.interpret import sample_indices
    return {k: list(v) for k, v in
            sample_indices(str(MS_ROOT), split, per_class, seed).items()}


@st.cache_data(show_spinner="Scoring patches...")
def predictions(label, idxs, split="test"):
    """(true, predicted) for one model over the given dataset indices."""
    import torch
    model, ds = model_for(label), dataset_for(label, split)
    out = []
    for i in idxs:
        x, y = ds[i]
        with torch.no_grad():
            out.append((int(y), int(model(x.unsqueeze(0)).argmax(dim=1))))
    return out


@st.cache_data(show_spinner="Reading reflectance...")
def spectra(per_class, split="test", seed=0):
    from src.visualization.interpret import class_spectra
    return class_spectra(str(MS_ROOT), split, per_class, seed)


# Navigation is a timeline: one bulb per step, always visible, no menu to open.
# Real buttons carry the click; the keys give the CSS below a scoped hook.
STEPS = [
    ("Overview", "Overview"),
    ("See the data", "Data"),
    ("1. Capacity and distillation", "Capacity"),
    ("2. Input bands", "Bands"),
    ("Spectral signatures", "Spectra"),
    ("3. Learned spectral projection", "Projection"),
    ("Where the model looks", "Grad-CAM"),
    ("Where the errors are", "Errors"),
    ("Reproducibility", "Seeds"),
    ("Report figures", "Figures"),
]

st.markdown("""
<style>
/* help= wraps the button in tooltip elements that shrink to fit, which would
   left align the bulb in its column away from its label. Force the chain to
   the column width so each bulb and its label share one centre. */
div[class*="st-key-nav"],
div[class*="st-key-nav"] [data-testid="stButton"],
div[class*="st-key-nav"] [data-testid="stTooltipIcon"],
div[class*="st-key-nav"] [data-testid="stTooltipHoverTarget"] { width: 100%; }
div[class*="st-key-nav"] [data-testid="stTooltipHoverTarget"] { justify-content: center; }
div[class*="st-key-nav"] button {
    display: block; margin: 0 auto;
    width: 38px; height: 38px; min-height: 38px;
    padding: 0; border-radius: 50%;
    font-size: 12px; font-variant-numeric: tabular-nums;
}
.tl-label {
    text-align: center; font-size: 10.5px; line-height: 1.15;
    margin-top: 5px; opacity: 0.55;
}
.tl-label.tl-on { opacity: 1; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = STEPS[0][0]

for col, (full, short), n in zip(st.columns(len(STEPS), gap="small"), STEPS,
                                 range(1, len(STEPS) + 1)):
    on = st.session_state.page == full
    with col:
        if st.button(str(n), key=f"nav{n}", help=full,
                     type="primary" if on else "secondary"):
            st.session_state.page = full
            st.rerun()
        st.markdown(f"<div class='tl-label{' tl-on' if on else ''}'>{short}</div>",
                    unsafe_allow_html=True)
page = st.session_state.page
st.divider()


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
    # Dot plot rather than bars: the axis is zoomed to make sub-percent
    # differences visible, and a bar truncated well above zero would overstate
    # them. Position carries the value; the stem is only a reading aid.
    cap_base = alt.Chart(m).encode(
        x=alt.X("model:N", sort=None, title="",
                axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        xOffset="training:N",
        y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[90, 98]),
                title="test accuracy (%)"),
        color=alt.Color("training:N", title="",
                        scale=alt.Scale(range=["#e8833a", "#4c8fd4"])),
        tooltip=["model", "training", "accuracy"])
    st.altair_chart(
        (cap_base.mark_rule(size=2, opacity=0.35).encode(y2=alt.datum(90))
         + cap_base.mark_circle(size=150)).properties(height=340),
        use_container_width=True)
    st.caption("The vertical axis starts at 90%, not zero, so the marks show "
               "position rather than magnitude.")
    t = acc("efficientnet_v2_s")
    st.info(
        f"The teacher reaches {t:.2f}%, so the soft targets are accurate. "
        "Distillation still changes accuracy by less than run-to-run variation at the "
        "two larger sizes, and lowers the smallest student by 0.81%."
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
    # Horizontal so the long configuration names stay readable in a narrow
    # pane, and dots rather than bars because the axis does not start at zero.
    band_base = alt.Chart(df).encode(
        y=alt.Y("input:N", sort=None, title="",
                axis=alt.Axis(labelFontSize=13)),
        x=alt.X("accuracy (%):Q", scale=alt.Scale(domain=[lo, hi]),
                title="test accuracy (%)"),
        tooltip=["input", "accuracy (%)"])
    st.altair_chart(
        (band_base.mark_rule(size=2, opacity=0.35).encode(x2=alt.datum(lo))
         + band_base.mark_circle(size=170)).properties(height=200),
        use_container_width=True)
    st.caption(f"The axis spans {lo:.1f}-{hi:.1f}%, not zero to 100, so the "
               "marks show position rather than magnitude.")
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
    line = alt.Chart(df).mark_line(point=True, color="#4c8fd4", size=2.5).encode(
        x=alt.X("k:O", title="channel budget k",
                axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        y=alt.Y("accuracy (%):Q", scale=alt.Scale(domain=[lo, hi]),
                title="test accuracy (%)"),
        tooltip=["k", "accuracy (%)"])
    # The two references sat on the same dash pattern and adjacent default hues,
    # so the legend could not separate them. Vary both channels.
    names = ["RGB, 3 channels", "all 12 bands"]
    rules = alt.Chart(pd.DataFrame({"y": [rgb, full], "reference": names})).mark_rule(
        size=2).encode(
        y="y:Q",
        color=alt.Color("reference:N", title="reference",
                        scale=alt.Scale(domain=names,
                                        range=["#e8833a", "#57a773"])),
        strokeDash=alt.StrokeDash("reference:N", title="reference",
                                  scale=alt.Scale(domain=names,
                                                  range=[[2, 3], [10, 4]])),
        tooltip=["reference", "y"])
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
        # The 12-band matrix does not fit a narrow pane, and the reading of it
        # is the regional share below, so the raw weights sit behind a toggle.
        top = []
        for i, row in enumerate(W):
            rank = np.argsort(-np.abs(row))[:3]
            top.append({"channel": f"channel {i}",
                        "strongest bands": ", ".join(
                            f"{BANDS[j]} ({row[j]:+.2f})" for j in rank)})
        st.write("Bands each channel weights most")
        st.dataframe(pd.DataFrame(top), use_container_width=True, hide_index=True)
        with st.expander("Full weight matrix, all 12 bands"):
            st.dataframe(pd.DataFrame(W, columns=BANDS,
                                      index=[f"channel {i}" for i in range(len(W))]).round(3),
                         use_container_width=True)
            st.caption("Rows are virtual channels, columns the standardised input "
                       "bands. The sign of a row is arbitrary and is fixed here so "
                       "the largest weight is positive.")
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
elif page == "Where the errors are":
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
        pairs = [{"true class": GROUPED[i], "predicted as": GROUPED[j],
                  "patches": int(off[i][j])}
                 for i in range(len(GROUPED)) for j in range(len(GROUPED))
                 if off[i][j] > 0]
        pairs.sort(key=lambda r: -r["patches"])
        st.write("Most frequent confusions")
        st.dataframe(pd.DataFrame(pairs[:8]), use_container_width=True,
                     hide_index=True)
        with st.expander("Full confusion matrix, all 10 classes"):
            st.dataframe(pd.DataFrame(off, index=GROUPED, columns=GROUPED),
                         use_container_width=True)
            st.caption("Off-diagonal counts only. Rows are the true class, columns "
                       "the prediction, ordered so members of a group sit together.")

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
elif page == "Reproducibility":
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


# ------------------------------------------------------------ see the data
elif page == "See the data":
    st.title("See the data")
    st.write(
        "Every patch is 64x64 at 10 m ground sampling. The same patch is shown "
        "under four band combinations. Only the first is what an RGB model sees."
    )
    from src.visualization.interpret import COMPOSITES, composite

    cls = st.selectbox("class", CLASSES, index=CLASSES.index("River"))
    n = st.slider("patches", 2, 6, 4)
    per_class = indices_by_class(8)
    base = raw_dataset()
    idxs = per_class[CLASSES.index(cls)][:n]

    for name in COMPOSITES:
        st.subheader(name)
        cols = st.columns(n)
        for col, i in zip(cols, idxs):
            img = base[i]["image"].float()
            col.image(composite(img, name), use_container_width=True)
    st.caption(
        "River and the crop classes are the ones the 12-band model improves most. "
        "In true colour a river channel and a bare field can both read as a pale "
        "strip; the infrared combinations separate them because water absorbs "
        "near-infrared while vegetation and soil reflect it."
    )


# --------------------------------------------------- spectral signatures
elif page == "Spectral signatures":
    st.title("Spectral signatures")
    st.write(
        "Mean surface reflectance per class, averaged over sampled test patches "
        "and plotted against wavelength. This is the information an RGB model "
        "does not receive."
    )
    from src.visualization.interpret import WAVELENGTH_NM, RAW_INDEX

    per_class = st.slider("patches averaged per class", 6, 40, 16, step=2)
    picks = st.multiselect("classes", CLASSES,
                           default=["River", "AnnualCrop", "Forest", "Residential"])
    if not picks:
        st.info("Select at least one class.")
    else:
        mu, sd = spectra(per_class)
        order = sorted(BANDS, key=lambda b: WAVELENGTH_NM[b])
        rows = []
        for c in picks:
            li = CLASSES.index(c)
            for b in order:
                rows.append({"class": c, "band": b,
                             "wavelength (nm)": WAVELENGTH_NM[b],
                             "reflectance": float(mu[li][RAW_INDEX[b]])})
        df = pd.DataFrame(rows)
        chart = (alt.Chart(df)
                 .mark_line(point=True)
                 .encode(x=alt.X("wavelength (nm):Q",
                                 scale=alt.Scale(type="log", nice=False)),
                         y=alt.Y("reflectance:Q", title="mean reflectance"),
                         color=alt.Color("class:N",
                                         scale=alt.Scale(scheme="tableau10")),
                         tooltip=["class", "band", "wavelength (nm)", "reflectance"])
                 .properties(height=380))
        st.altair_chart(chart, use_container_width=True)

        vis = [b for b in order if WAVELENGTH_NM[b] <= 700]
        rest = [b for b in order if WAVELENGTH_NM[b] > 700]
        st.caption(
            f"The four visible bands ({', '.join(vis)}) span {WAVELENGTH_NM[vis[0]]}"
            f"-{WAVELENGTH_NM[vis[-1]]} nm. The remaining {len(rest)} bands reach "
            f"{WAVELENGTH_NM[rest[-1]]} nm, where vegetation, bare soil and water "
            "separate most clearly."
        )
        st.subheader("Separation gained beyond the visible")
        pairs = []
        for i, a in enumerate(picks):
            for b_ in picks[i + 1:]:
                ia, ib = CLASSES.index(a), CLASSES.index(b_)
                v = np.array([abs(mu[ia][RAW_INDEX[x]] - mu[ib][RAW_INDEX[x]]) for x in vis])
                r = np.array([abs(mu[ia][RAW_INDEX[x]] - mu[ib][RAW_INDEX[x]]) for x in rest])
                pairs.append({"pair": f"{a} vs {b_}",
                              "mean gap, visible": round(float(v.mean()), 1),
                              "mean gap, beyond visible": round(float(r.mean()), 1)})
        if pairs:
            st.dataframe(pd.DataFrame(pairs), hide_index=True,
                         use_container_width=True)
            st.caption("Absolute difference in mean reflectance, averaged over the "
                       "bands in each group.")


# ------------------------------------------------------ where the model looks
elif page == "Where the model looks":
    st.title("Where the model looks")
    st.write(
        "Grad-CAM over the selected conv block. The map is the class score's "
        "gradient-weighted activation, upsampled to the patch. Both models see "
        "the same patch, so the comparison is like for like."
    )
    from src.visualization.interpret import (grad_cam, upsample, composite,
                                             n_blocks)
    import matplotlib.cm as cm

    c1, c2, c3 = st.columns(3)
    left = c1.selectbox("model A", list(CONFIGS), index=0)
    right = c2.selectbox("model B", list(CONFIGS), index=1)
    cls = c3.selectbox("class", CLASSES, index=CLASSES.index("River"))

    per_class = indices_by_class(16)
    candidates = per_class[CLASSES.index(cls)]
    pa = predictions(left, candidates)
    pb = predictions(right, candidates)
    outcome = st.radio(
        "show patches where",
        ["any outcome", "A wrong, B correct", "A correct, B wrong",
         "both wrong", "both correct"], horizontal=True)
    keep = []
    for i, (ya, a), (_, b) in zip(candidates, pa, pb):
        ok_a, ok_b = a == ya, b == ya
        if (outcome == "any outcome"
                or (outcome == "A wrong, B correct" and not ok_a and ok_b)
                or (outcome == "A correct, B wrong" and ok_a and not ok_b)
                or (outcome == "both wrong" and not ok_a and not ok_b)
                or (outcome == "both correct" and ok_a and ok_b)):
            keep.append(i)
    st.caption(f"{len(keep)} of {len(candidates)} sampled {cls} patches match. "
               "The filter selects deliberately, so treat any single patch as an "
               "illustration rather than evidence.")
    if not keep:
        st.info("No sampled patch matches that combination for this class.")
        st.stop()
    pick = st.select_slider("patch", options=list(range(len(keep))), value=0)
    idx = keep[pick]

    depth = st.slider("conv block (1 is earliest, higher is more abstract)",
                      1, 4, 4)
    base_img = raw_dataset()[idx]["image"].float()
    view = composite(base_img, "True colour (B04/B03/B02)")

    cols = st.columns(3)
    cols[0].subheader("patch")
    cols[0].image(view, use_container_width=True)
    cols[0].caption(f"true colour, test index {idx}")

    import torch
    for col, label in zip(cols[1:], [left, right]):
        model = model_for(label)
        ds = dataset_for(label)
        x, y = ds[idx]
        blk = min(depth, n_blocks(model)) - 1
        cam, pred, probs = grad_cam(model, x.unsqueeze(0), block=blk)
        heat = (cm.inferno(upsample(cam, view.shape[0]))[:, :, :3] * 255)
        blend = (0.55 * view + 0.45 * heat).astype(np.uint8)
        col.subheader(label)
        col.image(blend, use_container_width=True)
        ok = "correct" if pred == y else "wrong"
        col.caption(f"predicts **{CLASSES[pred]}** at {probs[pred]:.1%} ({ok}); "
                    f"map is {cam.shape[0]}x{cam.shape[1]} before upsampling")
    st.caption(
        "The map resolution is set by the block: the deepest block of this "
        "architecture is 8x8 for a 64x64 patch, so the overlay shows which "
        "region carried the evidence, not a per-pixel segmentation."
    )


# --------------------------------------------------------- report figures
elif page == "Report figures":
    st.title("Report figures")
    st.write("The figures as they appear in the write-up, generated by "
             "`scripts/make_figures.py`.")
    figs = [
        ("pareto_accuracy_vs_params.png", "Accuracy against model size. The star "
         "is the 12-band run, evaluated on a different split and shown for "
         "orientation rather than as a point on the same curve."),
        ("multispectral_per_class_f1.png", "Per-class F1, RGB against 12 bands. "
         "The gain concentrates on River and the crop classes."),
        ("virtual_sensor_rate_distortion.png", "Accuracy against the number of "
         "learned channels k."),
        ("virtual_sensor_response_functions.png", "Learned weight matrix per k: "
         "each row is one virtual channel's response over the 12 bands."),
        ("confusion_scratch_cnn_ms_rgb.png", "Confusion matrix, RGB."),
        ("confusion_scratch_cnn_ms_all.png", "Confusion matrix, 12 bands."),
        ("confusion_scratch_cnn_ms_indices.png", "Confusion matrix, bands plus indices."),
    ]
    for fname, cap in figs:
        path = REPORTS / "figures" / fname
        if path.exists():
            st.image(str(path), caption=cap, use_container_width=True)
        else:
            st.warning(f"missing: {fname}")
