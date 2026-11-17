from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# IoV SENTINEL — Streamlit CAN intrusion detection interface
# Existing model is loaded from models/final_iov_intrusion_model.pkl
# Inputs: DATA_0 ... DATA_7, each an integer from 0 to 255.
# ============================================================

st.set_page_config(
    page_title="IoV Sentinel | Vehicle Network Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "final_iov_intrusion_model.pkl"
FEATURES = [f"DATA_{i}" for i in range(8)]

# Previously measured frequency-weighted held-out test results.
# These are fixed evaluation results, not live session metrics.
METRICS = {
    "Accuracy": 96.97,
    "Precision": 95.29,
    "Recall": 87.49,
    "F1-score": 91.22,
    "Specificity": 99.05,
    "ROC-AUC": 99.41,
    "Average Precision": 94.88,
}
CONFUSION = np.array([[180429, 1728], [4996, 34931]])

# ---------- Cyber-themed styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp {
  background: radial-gradient(circle at 85% 0%, rgba(0,160,255,.13), transparent 28%), #07111f;
  color: #e8f1fb;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0b1b2d,#081321);
  border-right: 1px solid #174565;
}
[data-testid="stHeader"] { background: rgba(7,17,31,.8); }
h1,h2,h3 { color:#f2f8ff !important; letter-spacing:-.03em; }
p,label,.stMarkdown { color:#c5d5e6; }
div[data-testid="stMetric"] {
  background:linear-gradient(145deg,#112b44,#0a1c2f);
  border:1px solid #24516d; padding:15px 17px; border-radius:14px;
}
div[data-testid="stMetricLabel"] { color:#9db6cc !important; }
div[data-testid="stMetricValue"] { color:#f4f9ff !important; }
.hero {
  padding:27px 30px; border-radius:19px; margin-bottom:18px;
  background:linear-gradient(115deg,#0877b3,#113b63 58%,#0d2645);
  border:1px solid #397da4; box-shadow:0 14px 40px #0005;
}
.hero-kicker { color:#a8e4ff; font-size:11px; font-weight:800; letter-spacing:.17em; }
.hero h1 { margin:8px 0 6px; font-size:32px; }
.hero p { margin:0; color:#d5edff; }
.panel {
  background:linear-gradient(145deg,#11283fdd,#0a1b2dee);
  border:1px solid #28516c; border-radius:15px; padding:17px 19px; margin-bottom:12px;
}
.muted { color:#8fa9c0; font-size:13px; }
.status-good { color:#5ee6a8; font-weight:800; }
.status-bad { color:#ff737d; font-weight:800; }
div.stButton>button {
  border-radius:10px; border:1px solid #287da8;
  background:linear-gradient(100deg,#087eb7,#075b91);
  color:white; font-weight:700; min-height:42px;
}
div.stButton>button:hover { border-color:#72d5ff; color:white; box-shadow:0 0 18px #22acef40; }
div[data-baseweb="input"] input {
  background:#102a43 !important; color:#f4f9ff !important;
  border:1px solid #285675 !important; border-radius:8px !important;
  font-family:'JetBrains Mono',monospace !important;
}
[data-testid="stFileUploader"] {
  background:#0c2237aa; border:1px dashed #326b8d; border-radius:12px; padding:10px;
}
hr { border-color:#31516a66; }
/* Vertical sidebar navigation: clear, clickable module sections */
[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
  display:flex; flex-direction:column; gap:7px;
  width:100%;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {
  background:linear-gradient(145deg,#102a42,#0b1d30);
  border:1px solid #214b68; border-radius:10px;
  padding:10px 12px; margin:0; cursor:pointer;
  transition:all .18s ease;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
  background:#16466a; border-color:#55bce9;
}
[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
  background:linear-gradient(100deg,#087eb7,#075b91);
  border-color:#72d5ff; box-shadow:0 0 14px #22acef35;
}
</style>
""", unsafe_allow_html=True)


# ---------- Model loading ----------
@st.cache_resource
def load_model_package(path):
    package = joblib.load(path)
    if isinstance(package, dict) and "model" in package:
        estimator = package["model"]
        features = package.get("features", FEATURES)
        mapping = package.get("label_mapping", {0: "BENIGN", 1: "ATTACK"})
    else:
        estimator = package
        features = FEATURES
        mapping = {0: "BENIGN", 1: "ATTACK"}

    if list(features) != FEATURES:
        raise ValueError(
            f"Model feature order mismatch. Expected {FEATURES}; found {list(features)}"
        )
    return estimator, list(features), mapping


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Check that final_iov_intrusion_model.pkl is inside the project's models folder.")
    st.stop()

try:
    model, feature_names, label_mapping = load_model_package(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


def class_name(value):
    try:
        return str(label_mapping[value])
    except (KeyError, TypeError):
        return str(value)


def predict_payload(values):
    if len(values) != 8:
        raise ValueError("Exactly eight CAN payload bytes are required.")
    cleaned = []
    for i, value in enumerate(values):
        number = float(value)
        if not number.is_integer() or not 0 <= number <= 255:
            raise ValueError(f"DATA_{i} must be an integer from 0 to 255.")
        cleaned.append(int(number))

    frame = pd.DataFrame([cleaned], columns=feature_names)
    prediction = model.predict(frame)[0]
    probabilities = model.predict_proba(frame)[0]
    probs = {
        class_name(cls): float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }
    return cleaned, class_name(prediction), probs


def explain_decision_tree(payload):
    """Return the exact Decision Tree path taken by one eight-byte payload."""
    if not hasattr(model, "tree_") or not hasattr(model, "decision_path"):
        return None, "The loaded estimator does not expose a scikit-learn Decision Tree path."

    frame = pd.DataFrame([payload], columns=feature_names)
    tree = model.tree_
    node_indicator = model.decision_path(frame)
    leaf_id = int(model.apply(frame)[0])
    node_ids = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    rules = []
    for node_id in node_ids:
        node_id = int(node_id)
        if node_id == leaf_id:
            continue
        feature_index = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        feature_value = int(payload[feature_index])
        went_left = feature_value <= threshold
        rules.append({
            "Node": node_id,
            "Decision rule": (
                f"{feature_names[feature_index]} <= {threshold:.4f}"
                if went_left else f"{feature_names[feature_index]} > {threshold:.4f}"
            ),
            "Payload value": feature_value,
            "Branch taken": "Left" if went_left else "Right",
        })

    leaf_values = np.asarray(tree.value[leaf_id][0], dtype=float)
    total = float(leaf_values.sum())
    if total > 0:
        leaf_percentages = leaf_values / total * 100.0
    else:
        leaf_percentages = np.zeros_like(leaf_values)
    leaf_summary = pd.DataFrame({
        "Class": [class_name(cls) for cls in model.classes_],
        "Leaf distribution (%)": np.round(leaf_percentages, 2),
    })
    return {
        "rules": pd.DataFrame(rules, columns=["Node", "Decision rule", "Payload value", "Branch taken"]),
        "leaf_id": leaf_id,
        "leaf_summary": leaf_summary,
        "leaf_samples": float(tree.n_node_samples[leaf_id]),
        "leaf_weighted_samples": float(tree.weighted_n_node_samples[leaf_id]),
    }, None


def render_xai(payload):
    """Render a transparent, model-specific explanation for a payload."""
    explanation, error = explain_decision_tree(payload)
    if error:
        st.warning(error)
        return

    st.markdown("### Decision path explanation")
    st.caption(
        "The rules below are the actual branch decisions followed by this payload "
        "inside the trained Decision Tree."
    )
    if explanation["rules"].empty:
        st.info("This payload reaches the root leaf; the trained tree has no split rules for this path.")
    else:
        st.dataframe(explanation["rules"], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.metric("Reached leaf node", explanation["leaf_id"])
    c2.metric("Training samples at leaf", f"{explanation['leaf_samples']:,.0f}")
    st.markdown("**Class distribution at reached leaf**")
    st.dataframe(explanation["leaf_summary"], use_container_width=True, hide_index=True)
    st.caption(
        "This is a faithful path explanation for the fitted tree, not a causal explanation. "
        "Leaf distribution describes training samples reaching that leaf; it is not a guarantee "
        "that the prediction is correct or calibrated."
    )


def add_history(payload, result, probs, source="Manual"):
    st.session_state.history.insert(0, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        **{f"DATA_{i}": payload[i] for i in range(8)},
        "Prediction": result,
        "P(BENIGN) %": round(probs.get("BENIGN", 0) * 100, 2),
        "P(ATTACK) %": round(probs.get("ATTACK", 0) * 100, 2),
    })


def hero(title, subtitle, kicker="VEHICLE NETWORK SECURITY"):
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{kicker}</div>'
        f'<h1>🛡️ {title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_result(payload, result, probs):
    benign = probs.get("BENIGN", 0) * 100
    attack = probs.get("ATTACK", 0) * 100
    if result == "ATTACK":
        st.error("🚨 POTENTIAL ATTACK DETECTED")
        st.markdown(
            '<div class="panel"><span class="status-bad">ATTACK CLASSIFICATION</span>'
            '<p class="muted">The trained model classified this payload as potentially malicious.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.success("✓ BENIGN PAYLOAD CLASSIFICATION")
        st.markdown(
            '<div class="panel"><span class="status-good">BENIGN CLASSIFICATION</span>'
            '<p class="muted">The trained model classified this payload as benign.</p></div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    c1.metric("Model output: BENIGN", f"{benign:.2f}%")
    c2.metric("Model output: ATTACK", f"{attack:.2f}%")
    fig = go.Figure(go.Bar(
        x=["BENIGN", "ATTACK"], y=[benign, attack],
        marker_color=["#22c987", "#ff5968"],
        text=[f"{benign:.2f}%", f"{attack:.2f}%"], textposition="outside",
    ))
    fig.update_layout(
        height=250, margin=dict(l=10, r=10, t=20, b=5),
        yaxis=dict(range=[0, 110], title="Model output (%)", gridcolor="rgba(255,255,255,0.07)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dceafa",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Decision Tree leaf probabilities are model outputs, not guaranteed correctness "
        "or calibrated confidence."
    )
    st.markdown("**Analysed CAN payload**")
    st.code("   ".join(f"DATA_{i}: {v:3d}" for i, v in enumerate(payload)))


# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "payload_values" not in st.session_state:
    st.session_state.payload_values = [0] * 8
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

# Initialize widget keys before the number_input widgets are created.
for i in range(8):
    widget_key = f"payload_input_{i}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = int(st.session_state.payload_values[i])


def load_example_payload():
    """Callback: runs before Streamlit recreates the widgets."""
    example_values = [160, 15, 3, 37, 168, 53, 148, 34]
    st.session_state.payload_values = example_values
    for i, value in enumerate(example_values):
        st.session_state[f"payload_input_{i}"] = value


def clear_payload_inputs():
    """Callback: reset all eight payload input widgets safely."""
    cleared_values = [0] * 8
    st.session_state.payload_values = cleared_values
    for i, value in enumerate(cleared_values):
        st.session_state[f"payload_input_{i}"] = value


# ---------- Vertical sidebar navigation ----------
NAVIGATION_PAGES = [
    "Overview",
    "CAN Payload Analyzer",
    "Payload Comparison",
    "Explainable AI (XAI)",
    "Attack Analysis",
    "Detection History",
    "Model Performance",
    "Model Information",
]
with st.sidebar:
    st.markdown("## 🛡️ IoV **SENTINEL**")
    st.caption("INTELLIGENT VEHICLE SECURITY")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        NAVIGATION_PAGES,
        horizontal=False,
        label_visibility="collapsed",
        key="sidebar_navigation",
    )
    st.divider()
    st.caption("UNIVERSITY PROJECT · IoV INTRUSION DETECTION")
    st.caption("Manual payload analysis — no live vehicle connection.")

# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    hero(
        "Vehicle Network Security",
        "Machine-learning-based intrusion detection for Internet of Vehicles. "
        "Inspect CAN payloads and classify potentially malicious traffic.",
    )
    st.markdown("### System overview")
    c = st.columns(4)
    c[0].metric("Detection algorithm", "Decision Tree")
    c[1].metric("Input features", "8 bytes")
    c[2].metric("Classification classes", "2")
    c[3].metric(
        "Tree leaves",
        str(model.get_n_leaves()) if hasattr(model, "get_n_leaves") else "Loaded",
    )

    st.markdown("### Held-out evaluation results")
    c = st.columns(4)
    for col, key in zip(c, ["Accuracy", "Precision", "Recall", "F1-score"]):
        col.metric(key, f"{METRICS[key]:.2f}%")
    st.caption("Previously measured frequency-weighted held-out test results; not live session metrics.")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Detection workflow")
        st.markdown("""
        <div class="panel">
        <b>01 · INPUT</b><p class="muted">Enter eight decimal CAN payload bytes (0–255).</p>
        <b>02 · INFERENCE</b><p class="muted">The saved Decision Tree evaluates the payload in the trained feature order.</p>
        <b>03 · CLASSIFICATION</b><p class="muted">The model returns BENIGN or ATTACK and class probability outputs.</p>
        <b>04 · AUDIT</b><p class="muted">Review predictions in session history and export a CSV log.</p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("### Test-set class distribution")
        distribution = pd.DataFrame({
            "Class": ["BENIGN", "ATTACK"],
            "Messages": [int(CONFUSION[0].sum()), int(CONFUSION[1].sum())],
        })
        fig = px.pie(
            distribution, names="Class", values="Messages", hole=.62,
            color="Class",
            color_discrete_map={"BENIGN": "#22c987", "ATTACK": "#ff5968"},
        )
        fig.update_layout(
            height=270, margin=dict(l=0, r=0, t=5, b=5),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#dceafa",
            legend=dict(orientation="h", y=-.08),
        )
        st.plotly_chart(fig, use_container_width=True)
    st.info("This interface analyses manually entered or uploaded CAN payloads. It does not capture live vehicle traffic.")


# ============================================================
# CAN PAYLOAD ANALYZER
# ============================================================
elif page == "CAN Payload Analyzer":
    hero(
        "CAN Payload Analyzer",
        "Enter the eight bytes of a CAN message and run the trained intrusion detector.",
        "MANUAL CAN MESSAGE INSPECTION",
    )
    st.markdown("### Payload configuration")
    st.caption("Each field is one byte. Enter an integer from 0 to 255.")

    # These buttons are outside the form and use callbacks.
    # Callbacks update widget state before the next script run.
    button_col1, button_col2 = st.columns(2)
    button_col1.button(
        "Load example",
        on_click=load_example_payload,
        use_container_width=True,
    )
    button_col2.button(
        "Clear fields",
        on_click=clear_payload_inputs,
        use_container_width=True,
    )

    with st.form("payload_form"):
        cols = st.columns(4)
        values = []
        for i in range(8):
            with cols[i % 4]:
                values.append(st.number_input(
                    f"DATA_{i}",
                    min_value=0,
                    max_value=255,
                    step=1,
                    key=f"payload_input_{i}",
                ))

        analyse = st.form_submit_button(
            "🔍 ANALYZE CAN TRAFFIC",
            use_container_width=True,
        )

    if analyse:
        try:
            payload, result, probs = predict_payload(values)
            st.session_state.payload_values = payload
            st.session_state.last_result = {
                "payload": payload, "result": result, "probs": probs
            }
            add_history(payload, result, probs, "Manual")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.last_result:
        st.divider()
        st.markdown("### Analysis result")
        r = st.session_state.last_result
        render_result(r["payload"], r["result"], r["probs"])
    else:
        st.markdown(
            '<div class="panel"><b>Awaiting payload</b><p class="muted">'
            'Enter eight bytes and select Analyze CAN Traffic.</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Batch CSV analysis")
    st.caption("CSV must contain DATA_0 through DATA_7. Additional columns are ignored.")
    uploaded = st.file_uploader("Choose CAN payload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                x = batch[FEATURES].apply(pd.to_numeric, errors="coerce")
                invalid = (
                    x.isna().any(axis=1)
                    | ((x < 0) | (x > 255)).any(axis=1)
                    | (x % 1 != 0).any(axis=1)
                )
                valid = x.loc[~invalid].astype(int)
                if invalid.any():
                    st.warning(f"{int(invalid.sum())} invalid row(s) will be skipped.")
                if len(valid) == 0:
                    st.error("No valid payload rows found.")
                elif st.button("Run batch detection", key="batch_run"):
                    predictions = model.predict(valid[FEATURES])
                    probabilities = model.predict_proba(valid[FEATURES])
                    output_df = valid.copy()
                    output_df["Prediction"] = [class_name(p) for p in predictions]
                    class_indices = {
                        class_name(cls): i for i, cls in enumerate(model.classes_)
                    }
                    output_df["P(BENIGN) %"] = (
                        probabilities[:, class_indices["BENIGN"]] * 100
                        if "BENIGN" in class_indices else 0
                    )
                    output_df["P(ATTACK) %"] = (
                        probabilities[:, class_indices["ATTACK"]] * 100
                        if "ATTACK" in class_indices else 0
                    )
                    st.session_state.batch_results = output_df
                    for _, row in output_df.iterrows():
                        add_history(
                            row[FEATURES].astype(int).tolist(),
                            row["Prediction"],
                            {
                                "BENIGN": float(row["P(BENIGN) %"]) / 100,
                                "ATTACK": float(row["P(ATTACK) %"]) / 100,
                            },
                            "CSV batch",
                        )
                    st.success(f"Analysed {len(output_df):,} valid payload rows.")

                if "batch_results" in st.session_state:
                    st.dataframe(st.session_state.batch_results, use_container_width=True)
                    st.download_button(
                        "Download batch results CSV",
                        st.session_state.batch_results.to_csv(index=False).encode(),
                        "iov_batch_detection_results.csv",
                        "text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")


# ============================================================
# PAYLOAD COMPARISON
# ============================================================
elif page == "Payload Comparison":
    hero(
        "Payload Comparison",
        "Compare two CAN payloads side by side and inspect how their model outputs differ.",
        "SIDE-BY-SIDE PAYLOAD INSPECTION",
    )
    st.caption("Enter two eight-byte payloads. Each byte must be an integer from 0 to 255.")

    example_a = [160, 15, 3, 37, 168, 53, 148, 34]
    example_b = [160, 15, 3, 37, 168, 53, 148, 35]
    with st.form("payload_comparison_form"):
        col_a, col_b = st.columns(2)
        values_a, values_b = [], []
        with col_a:
            st.markdown("#### Payload A")
            for i in range(8):
                values_a.append(st.number_input(
                    f"A · DATA_{i}", min_value=0, max_value=255, step=1,
                    value=example_a[i], key=f"compare_a_{i}"
                ))
        with col_b:
            st.markdown("#### Payload B")
            for i in range(8):
                values_b.append(st.number_input(
                    f"B · DATA_{i}", min_value=0, max_value=255, step=1,
                    value=example_b[i], key=f"compare_b_{i}"
                ))
        compare_clicked = st.form_submit_button(
            "⚖️ COMPARE PAYLOADS", use_container_width=True
        )

    if compare_clicked:
        try:
            payload_a, result_a, probs_a = predict_payload(values_a)
            payload_b, result_b, probs_b = predict_payload(values_b)
            changed = [i for i, (a, b) in enumerate(zip(payload_a, payload_b)) if a != b]
            st.session_state.comparison_result = {
                "payload_a": payload_a, "result_a": result_a, "probs_a": probs_a,
                "payload_b": payload_b, "result_b": result_b, "probs_b": probs_b,
                "changed": changed,
            }
            add_history(payload_a, result_a, probs_a, "Comparison A")
            add_history(payload_b, result_b, probs_b, "Comparison B")
        except Exception as exc:
            st.error(str(exc))

    comparison = st.session_state.comparison_result
    if comparison:
        st.divider()
        st.markdown("### Classification comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Payload A**")
            if comparison["result_a"] == "ATTACK":
                st.error("🚨 ATTACK")
            else:
                st.success("✓ BENIGN")
            st.metric("P(ATTACK)", f"{comparison['probs_a'].get('ATTACK', 0) * 100:.2f}%")
        with c2:
            st.markdown("**Payload B**")
            if comparison["result_b"] == "ATTACK":
                st.error("🚨 ATTACK")
            else:
                st.success("✓ BENIGN")
            st.metric("P(ATTACK)", f"{comparison['probs_b'].get('ATTACK', 0) * 100:.2f}%")

        if comparison["result_a"] == comparison["result_b"]:
            st.info(f"Both payloads received the same classification: **{comparison['result_a']}**.")
        else:
            st.warning("The two payloads received different classifications.")

        st.markdown("### Byte-by-byte comparison")
        comparison_table = pd.DataFrame({
            "Feature": FEATURES,
            "Payload A": comparison["payload_a"],
            "Payload B": comparison["payload_b"],
            "Difference (B − A)": [b - a for a, b in zip(comparison["payload_a"], comparison["payload_b"])],
            "Changed": [
                "Yes" if a != b else "No"
                for a, b in zip(comparison["payload_a"], comparison["payload_b"])
            ],
        })
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)
        st.metric("Bytes changed", len(comparison["changed"]))

        probability_comparison = pd.DataFrame({
            "Class": ["BENIGN", "ATTACK"],
            "Payload A (%)": [comparison["probs_a"].get("BENIGN", 0) * 100, comparison["probs_a"].get("ATTACK", 0) * 100],
            "Payload B (%)": [comparison["probs_b"].get("BENIGN", 0) * 100, comparison["probs_b"].get("ATTACK", 0) * 100],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Payload A", x=probability_comparison["Class"], y=probability_comparison["Payload A (%)"], marker_color="#38bdf8"))
        fig.add_trace(go.Bar(name="Payload B", x=probability_comparison["Class"], y=probability_comparison["Payload B (%)"], marker_color="#22c987"))
        fig.update_layout(
            barmode="group", height=320, yaxis=dict(range=[0, 100], title="Model output (%)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dceafa", legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The comparison shows model outputs for these two inputs; it does not establish that a changed byte caused a classification change.")


# ============================================================
# EXPLAINABLE AI (XAI)
# ============================================================
elif page == "Explainable AI (XAI)":
    hero(
        "Explainable AI",
        "Inspect the exact Decision Tree rules used to classify a CAN payload.",
        "MODEL TRANSPARENCY & INTERPRETABILITY",
    )
    st.markdown("### Select a payload to explain")
    latest = st.session_state.last_result
    use_latest = latest is not None and st.radio(
        "Payload source", ["Latest analysed payload", "Enter a payload manually"],
        horizontal=True, key="xai_payload_source"
    ) == "Latest analysed payload"

    if use_latest:
        xai_payload = latest["payload"]
        st.info(f"Using the latest analysed payload — model prediction: **{latest['result']}**.")
        st.code("   ".join(f"DATA_{i}: {v:3d}" for i, v in enumerate(xai_payload)))
        render_xai(xai_payload)
    else:
        with st.form("xai_payload_form"):
            xai_cols = st.columns(4)
            xai_values = []
            defaults = latest["payload"] if latest else [160, 15, 3, 37, 168, 53, 148, 34]
            for i in range(8):
                with xai_cols[i % 4]:
                    xai_values.append(st.number_input(
                        f"DATA_{i}", min_value=0, max_value=255, step=1,
                        value=int(defaults[i]), key=f"xai_payload_{i}"
                    ))
            explain_clicked = st.form_submit_button("EXPLAIN THIS PAYLOAD", use_container_width=True)
        if explain_clicked:
            try:
                xai_payload, xai_prediction, xai_probs = predict_payload(xai_values)
                st.session_state.xai_result = {
                    "payload": xai_payload, "prediction": xai_prediction, "probs": xai_probs
                }
            except Exception as exc:
                st.error(str(exc))
        if st.session_state.get("xai_result"):
            xai_result = st.session_state.xai_result
            st.success(f"Model prediction: {xai_result['prediction']}")
            render_xai(xai_result["payload"])

    st.divider()
    st.markdown("### Global feature importance")
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names, "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig = px.bar(
            importance, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#16496d", "#29c6e8"]
        )
        fig.update_layout(
            height=340, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#dceafa",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Global feature importance summarizes how the fitted tree used each feature across the model. It is not a per-payload causal explanation.")
    else:
        st.info("The loaded estimator does not provide feature_importances_.")


# ============================================================
# ATTACK ANALYSIS
# ============================================================
elif page == "Attack Analysis":
    hero(
        "Attack Analysis",
        "Review classification output and known evaluation limitations.",
        "THREAT CLASSIFICATION",
    )
    if st.session_state.last_result:
        r = st.session_state.last_result
        render_result(r["payload"], r["result"], r["probs"])
    else:
        st.info("Run a payload analysis first to see its classification here.")

    st.markdown("### Dataset attack-category evaluation")
    categories = pd.DataFrame({
        "Attack category": [
            "DoS", "GAS spoofing", "RPM spoofing",
            "SPEED spoofing", "STEERING_WHEEL spoofing",
        ],
        "Detection rate": [99.92, 0, 100, 100, 100],
    })
    fig = px.bar(
        categories, x="Attack category", y="Detection rate",
        range_y=[0, 105], color="Detection rate",
        color_continuous_scale=["#ff6875", "#19c995"],
        text=categories["Detection rate"].map(lambda v: f"{v:.2f}%"),
    )
    fig.update_layout(
        height=340, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#dceafa",
        yaxis_title="Detection rate (%)", coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.warning(
        "GAS spoofing was not detected in the held-out test set. The dataset contains "
        "only two unique GAS payload patterns. This is a significant limitation and should be reported."
    )
    st.caption("Category rates are from the earlier held-out evaluation, not current session results.")


# ============================================================
# DETECTION HISTORY
# ============================================================
elif page == "Detection History":
    hero(
        "Detection History",
        "Review predictions from this running session and export the event log.",
        "SESSION AUDIT LOG",
    )
    history = pd.DataFrame(st.session_state.history)
    c = st.columns(3)
    c[0].metric("Total analysed", len(history))
    c[1].metric(
        "Benign",
        int((history["Prediction"] == "BENIGN").sum()) if not history.empty else 0,
    )
    c[2].metric(
        "Attacks",
        int((history["Prediction"] == "ATTACK").sum()) if not history.empty else 0,
    )

    if history.empty:
        st.info("No payloads have been analysed in this session.")
    else:
        filter_value = st.selectbox("Filter predictions", ["All", "BENIGN", "ATTACK"])
        visible = history if filter_value == "All" else history[history["Prediction"] == filter_value]
        st.dataframe(visible, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download detection history CSV",
            history.to_csv(index=False).encode(),
            "iov_sentinel_detection_history.csv",
            "text/csv",
            use_container_width=True,
        )
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    st.caption("History exists in Streamlit session memory and is cleared when the session/runtime ends unless exported.")


# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    hero(
        "Model Performance",
        "Inspect saved model configuration and previously measured test-set performance.",
        "MODEL EVALUATION",
    )
    c = st.columns(4)
    for col, key in zip(c, ["Accuracy", "Precision", "Recall", "F1-score"]):
        col.metric(key, f"{METRICS[key]:.2f}%")
    c = st.columns(3)
    for col, key in zip(c, ["Specificity", "ROC-AUC", "Average Precision"]):
        col.metric(key, f"{METRICS[key]:.2f}%")

    st.markdown("### Frequency-weighted confusion matrix")
    fig = go.Figure(go.Heatmap(
        z=CONFUSION,
        x=["Predicted BENIGN", "Predicted ATTACK"],
        y=["Actual BENIGN", "Actual ATTACK"],
        text=CONFUSION,
        texttemplate="%{text:,}",
        colorscale=[[0, "#102a43"], [.5, "#087eb7"], [1, "#21c995"]],
        showscale=False,
    ))
    fig.update_layout(
        height=340, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font_color="#dceafa",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rows are actual classes; columns are predicted classes. Counts are frequency-weighted message totals, not unique-pattern counts.")

    st.markdown("### Feature importance")
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance")
        fig = px.bar(
            importance, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#16496d", "#29c6e8"],
        )
        fig.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#dceafa",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Impurity-based importance describes how the fitted tree used features; it is not a causal explanation.")

    with st.expander("Show decision tree visualization"):
        try:
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree

            fig, ax = plt.subplots(figsize=(24, 11))
            plot_tree(
                model, feature_names=feature_names,
                class_names=["BENIGN", "ATTACK"],
                filled=True, rounded=True, fontsize=8, ax=ax,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Could not render tree: {exc}")

    st.warning(
        "Cross-validation showed substantial fold-to-fold variation. The dataset is highly "
        "repetitive and contains very few unique attack patterns. Held-out metrics do not prove "
        "generalization to unseen vehicles, recording sessions, or future traffic."
    )


# ============================================================
# MODEL INFORMATION — FOUR RESEARCH TABS
# ============================================================
elif page == "Model Information":
    hero(
        "Model Information",
        "Explore the dataset, preprocessing methodology, machine learning algorithm and evaluation results.",
        "RESEARCH & TECHNICAL DETAILS",
    )
    tab_dataset, tab_preprocess, tab_algorithm, tab_evaluation = st.tabs([
        "📁 Dataset", "⚙️ Preprocessing", "🌳 Algorithm", "📊 Evaluation"
    ])

    # Dataset tab
    with tab_dataset:
        st.subheader("CICIoV2024 Dataset")
        st.write(
            "This project uses the Decimal CAN Bus dataset from CICIoV2024 "
            "for Internet of Vehicles intrusion detection."
        )
        c = st.columns(4)
        c[0].metric("Total records", "1,408,219")
        c[1].metric("Original columns", "12")
        c[2].metric("Unique patterns", "3,588")
        c[3].metric("Missing values", "0")

        st.markdown("### Dataset characteristics")
        st.dataframe(pd.DataFrame({
            "Property": [
                "Dataset name", "Data format", "Input features",
                "Target variable", "Classification", "Total records",
            ],
            "Description": [
                "CICIoV2024", "Decimal CAN bus data",
                "DATA_0 to DATA_7", "label",
                "BENIGN / ATTACK", "1,408,219",
            ],
        }), use_container_width=True, hide_index=True)

        st.markdown("### Class and attack-category records")
        attack_data = pd.DataFrame({
            "Category": [
                "BENIGN", "DoS", "GAS spoofing",
                "RPM spoofing", "SPEED spoofing", "STEERING_WHEEL spoofing",
            ],
            "Records": [1223737, 74663, 9991, 54900, 24951, 19977],
        })
        st.dataframe(attack_data, use_container_width=True, hide_index=True)
        fig = px.bar(
            attack_data, x="Category", y="Records", color="Category",
            color_discrete_sequence=["#22c987", "#ff5968", "#f59e0b", "#38bdf8", "#a78bfa", "#14b8a6"],
        )
        fig.update_layout(
            height=350, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dceafa",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "The model performs binary classification. The original dataset also contains "
            "attack-category information, but target columns are not used as model input features."
        )
        st.markdown("Dataset source: [Canadian Institute for Cybersecurity — CICIoV2024](https://www.unb.ca/cic/datasets/iov-dataset-2024.html)")

    # Preprocessing tab
    with tab_preprocess:
        st.subheader("Data Preprocessing Pipeline")
        steps = [
            ("01 · Dataset loading", "Six Decimal CAN CSV files are loaded and combined."),
            ("02 · Data inspection", "Missing values, columns and duplicate records are examined."),
            ("03 · Duplicate pattern handling",
             "1,408,219 records are reduced to 3,588 unique payload patterns. "
             "Original occurrence counts are retained as frequency weights."),
            ("04 · Feature selection",
             "DATA_0 to DATA_7 are selected. CAN ID and target columns are excluded."),
            ("05 · Train-test splitting",
             "Unique patterns are divided using a stratified 80:20 split."),
            ("06 · Model training",
             "The Decision Tree is trained using frequency weights."),
        ]
        for title, description in steps:
            st.markdown(f"**{title}**")
            st.write(description)
            st.divider()

        st.markdown("### Final dataset distribution")
        c = st.columns(3)
        c[0].metric("Unique patterns", "3,588")
        c[1].metric("Training patterns", "2,870")
        c[2].metric("Testing patterns", "718")
        st.success("No exact payload pattern overlaps between the training and test sets in the selected split.")
        st.warning(
            "Frequency weighting preserves the influence of repeated CAN messages. "
            "It does not create additional independent attack patterns."
        )

    # Algorithm tab
    with tab_algorithm:
        st.subheader("Decision Tree Classifier")
        st.write(
            "A supervised machine learning algorithm that learns decision rules "
            "from labelled CAN payload data."
        )
        st.markdown("### Model configuration")
        st.dataframe(pd.DataFrame({
            "Parameter": [
                "Algorithm", "Splitting criterion", "Maximum leaf nodes",
                "Minimum samples per leaf", "Random state", "Input features",
                "Actual tree leaves", "Actual tree depth",
            ],
            "Value": [
                "DecisionTreeClassifier", "Gini impurity", "100", "1", "42",
                "8", str(model.get_n_leaves()), str(model.get_depth()),
            ],
        }), use_container_width=True, hide_index=True)

        st.markdown("### How the algorithm works")
        for i, description in enumerate([
            "Receives eight CAN payload bytes.",
            "Evaluates feature values using learned decision rules.",
            "Follows the appropriate branches of the decision tree.",
            "Reaches a terminal leaf node.",
            "Predicts BENIGN or ATTACK.",
        ], 1):
            st.markdown(f"**Step {i}.** {description}")

        if hasattr(model, "feature_importances_"):
            st.markdown("### Feature importance")
            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=False)
            st.dataframe(importance, use_container_width=True, hide_index=True)

        with st.expander("View trained decision tree"):
            try:
                import matplotlib.pyplot as plt
                from sklearn.tree import plot_tree

                fig, ax = plt.subplots(figsize=(24, 11))
                plot_tree(
                    model, feature_names=feature_names,
                    class_names=["BENIGN", "ATTACK"],
                    filled=True, rounded=True, fontsize=8, ax=ax,
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.warning(f"Could not render decision tree: {exc}")

        st.info(
            "This is a scikit-learn DecisionTreeClassifier using Gini impurity. "
            "It is a Fine Tree-style implementation, not an exact reproduction "
            "of MATLAB's Fine Tree preset."
        )

    # Evaluation tab
    with tab_evaluation:
        st.subheader("Model Evaluation")
        st.caption("Previously measured frequency-weighted held-out test results.")
        c = st.columns(4)
        for col, key in zip(c, ["Accuracy", "Precision", "Recall", "F1-score"]):
            col.metric(key, f"{METRICS[key]:.2f}%")
        c = st.columns(3)
        for col, key in zip(c, ["Specificity", "ROC-AUC", "Average Precision"]):
            col.metric(key, f"{METRICS[key]:.2f}%")

        st.markdown("### Frequency-weighted confusion matrix")
        fig = go.Figure(go.Heatmap(
            z=CONFUSION,
            x=["Predicted BENIGN", "Predicted ATTACK"],
            y=["Actual BENIGN", "Actual ATTACK"],
            text=CONFUSION,
            texttemplate="%{text:,}",
            colorscale="Blues",
            showscale=False,
        ))
        fig.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#dceafa",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Rows are actual classes; columns are predicted classes. Counts are "
            "frequency-weighted message totals, not unique-pattern counts."
        )

        st.markdown("### Known limitations")
        st.warning(
            "GAS spoofing attacks were not detected in the held-out test evaluation. "
            "Only two unique GAS payload patterns were available in the complete dataset."
        )
        st.warning(
            "Cross-validation showed substantial fold-to-fold performance variation. "
            "These results do not establish generalization to unseen vehicles or future traffic."
        )
        st.info(
            "This implementation uses CICIoV2024 Decimal CAN data. It adapts a Fine Tree-style "
            "classification approach; it is not an exact reproduction of the original paper's "
            "simulated dataset or MATLAB preset."
        )


# ---------- Footer ----------
st.markdown("---")
st.caption(
    "IoV SENTINEL · Academic prototype · Not a safety-certified vehicle security system."
)
