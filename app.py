"""
Fake News Detector - Streamlit frontend.

This file ONLY handles the user interface and model inference. Training happens
separately in train_model.py. The fine-tuned model is loaded from saved_model/
(or from the Hugging Face Hub if MODEL_REPO_ID is configured).

Run with:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "saved_model"
METRICS_JSON = PROJECT_ROOT / "outputs" / "evaluation_metrics.json"
CONFUSION_MATRIX_PNG = PROJECT_ROOT / "outputs" / "confusion_matrix.png"

MAX_LENGTH = 256
MIN_CHARACTERS = 20
MAX_CHARACTERS = 10_000
LABEL_NAMES = {0: "FAKE", 1: "REAL"}

# These examples are only for trying the interface. They are not claims about
# real people, organizations, or events.
SAMPLES = {
    "Technology & science": (
        "Researchers have published a peer-reviewed study describing a new "
        "battery material. The team reports that the material retained most of "
        "its storage capacity after repeated charging cycles, though independent "
        "replication is still needed."
    ),
    "Public health": (
        "The city health department announced a new vaccination schedule on "
        "Tuesday. Officials said the programme will be offered at community "
        "clinics, and the full schedule is available on the department's website."
    ),
    "Sensational claim": (
        "BREAKING: An anonymous online post claims that a secret laboratory has "
        "confirmed the moon is made of cheese. The post urges readers to share "
        "the story before it is deleted and offers no verifiable evidence."
    ),
    "Political claim": (
        "A viral message says a new law will ban all public transport next week. "
        "No government website or official announcement confirms the claim, and "
        "the message does not name the law or the agency responsible."
    ),
}

NEWSPAPER_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Z"/>
  <path d="M4 22a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/>
  <path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/>
</svg>
"""

ALERT_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/>
  <path d="M12 9v4"/><path d="M12 17h.01"/>
</svg>
"""

CHECK_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/>
  <path d="m9 11 3 3L22 4"/>
</svg>
"""


def get_setting(name: str) -> str:
    """
    Read a setting from Streamlit secrets first, then from environment
    variables. Secrets keep tokens out of the source code.
    """
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:
        # Streamlit raises an exception when no secrets file exists locally.
        pass

    if value:
        return str(value).strip()
    return os.environ.get(name, "").strip()


def resolve_model_source() -> str:
    """Prefer a local trained model; otherwise use the Hugging Face Hub."""
    if (MODEL_DIR / "config.json").exists():
        return str(MODEL_DIR)
    return get_setting("MODEL_REPO_ID")


def is_model_available() -> bool:
    return (MODEL_DIR / "config.json").exists() or bool(get_setting("MODEL_REPO_ID"))


@st.cache_resource(show_spinner=False)
def load_classifier():
    """
    Load the tokenizer and model only once per app process/session.

    The model is loaded from saved_model/ by default. If the local folder does
    not exist and MODEL_REPO_ID is set, it is loaded from the Hugging Face Hub.
    """
    source = resolve_model_source()
    token = get_setting("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(source, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(source, token=token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_news(text: str) -> dict:
    """Tokenize text, run DistilBERT, and convert logits to probabilities."""
    tokenizer, model, device = load_classifier()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    predicted_index = int(torch.argmax(logits, dim=-1).item())

    return {
        "text": text,
        "label_index": predicted_index,
        "label": LABEL_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": probabilities,
    }


def inject_css():
    """Add a custom visual theme on top of Streamlit's dark theme."""
    st.markdown(
        """
        <style>
        :root {
            --navy-950: #070d19;
            --navy-900: #0b1220;
            --navy-800: #101b30;
            --blue-400: #60a5fa;
            --blue-500: #3b82f6;
            --blue-600: #2563eb;
            --text-main: #e8eefc;
            --text-muted: #9fb1cd;
            --line: rgba(148, 163, 184, 0.18);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% -5%, rgba(37, 99, 235, 0.20), transparent 32rem),
                radial-gradient(circle at 95% 5%, rgba(14, 165, 233, 0.10), transparent 28rem),
                linear-gradient(160deg, #070d19 0%, #0b1220 45%, #09101e 100%);
            color: var(--text-main);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1160px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        html, body, [class*="css"] {
            font-family: "Segoe UI", "Aptos", "Helvetica Neue", Arial, sans-serif;
        }

        h1, h2, h3, h4 {
            letter-spacing: -0.025em;
            color: var(--text-main);
        }

        p, li, .stMarkdown {
            color: #cbd8ec;
        }

        /* ---------- Header ---------- */
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.1rem 2.2rem 1.9rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(96, 165, 250, 0.20);
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(30, 64, 175, 0.28), rgba(10, 18, 33, 0.80) 58%),
                rgba(8, 15, 29, 0.85);
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 21rem;
            height: 21rem;
            right: -7rem;
            top: -10rem;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.28), transparent 68%);
            pointer-events: none;
        }

        .hero-top {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 1.25rem;
        }

        .brand-icon {
            display: grid;
            place-items: center;
            width: 3rem;
            height: 3rem;
            border-radius: 14px;
            color: #bfdbfe;
            background: linear-gradient(145deg, rgba(37, 99, 235, 0.42), rgba(14, 165, 233, 0.14));
            border: 1px solid rgba(147, 197, 253, 0.28);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.10);
        }

        .brand-icon svg {
            width: 1.6rem;
            height: 1.6rem;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.45rem 0.8rem;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.62);
            color: #c7d7ef;
            font-size: 0.77rem;
            font-weight: 650;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .status-dot {
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 50%;
            background: #34d399;
            box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.12);
        }

        .hero-kicker {
            color: var(--blue-400);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .hero h1 {
            margin: 0;
            max-width: 760px;
            font-size: clamp(2.35rem, 5vw, 4rem);
            line-height: 1.02;
            font-weight: 800;
            color: #f8fbff;
        }

        .hero h1 span {
            color: #93c5fd;
        }

        .hero-subtitle {
            max-width: 700px;
            margin: 1rem 0 0;
            color: #b7c8e0;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.35rem;
        }

        .chip {
            padding: 0.38rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(96, 165, 250, 0.18);
            background: rgba(15, 23, 42, 0.52);
            color: #c6d6ee;
            font-size: 0.78rem;
        }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            padding: 0.35rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(10, 18, 33, 0.62);
        }

        .stTabs [data-baseweb="tab"] {
            height: 2.7rem;
            padding: 0 1.05rem;
            border-radius: 10px;
            color: #9fb1cd;
            font-weight: 650;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #e8eefc;
            background: rgba(59, 130, 246, 0.08);
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.90), rgba(59, 130, 246, 0.78)) !important;
            box-shadow: 0 7px 20px rgba(37, 99, 235, 0.24);
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        /* ---------- Main input ---------- */
        .section-kicker {
            margin: 1.8rem 0 0.4rem;
            color: var(--blue-400);
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .input-title {
            margin: 0 0 0.35rem;
            font-size: 1.55rem;
            font-weight: 750;
            color: #f3f7ff;
        }

        .input-help {
            margin: 0 0 0.8rem;
            color: var(--text-muted);
            font-size: 0.92rem;
        }

        div[data-testid="stTextArea"] textarea {
            min-height: 235px;
            padding: 1.1rem 1.15rem;
            border: 1px solid rgba(96, 165, 250, 0.23);
            border-radius: 16px;
            background: rgba(7, 13, 25, 0.72);
            color: #eef4ff;
            caret-color: #60a5fa;
            font-size: 1rem;
            line-height: 1.65;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
            transition: border-color 160ms ease, box-shadow 160ms ease;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: rgba(96, 165, 250, 0.68);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
        }

        div[data-testid="stTextArea"] textarea::placeholder {
            color: #6f86a6;
        }

        div[data-testid="stTextArea"] label {
            color: #dbeafe;
        }

        .char-count {
            margin: -0.15rem 0 0.95rem;
            color: #7f93b0;
            font-size: 0.8rem;
            text-align: right;
        }

        /* ---------- Buttons ---------- */
        .stButton button {
            min-height: 2.85rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.68);
            color: #dbeafe;
            font-weight: 650;
            transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
        }

        .stButton button:hover {
            border-color: rgba(96, 165, 250, 0.55);
            background: rgba(30, 58, 95, 0.58);
            color: #ffffff;
            transform: translateY(-1px);
        }

        .stButton button[kind="primary"] {
            border: none;
            background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 55%, #60a5fa 100%);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 750;
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.28);
        }

        .stButton button[kind="primary"]:hover {
            background: linear-gradient(135deg, #2563eb 0%, #60a5fa 100%);
            box-shadow: 0 16px 36px rgba(37, 99, 235, 0.35);
        }

        .stButton button:disabled {
            opacity: 0.45;
            transform: none;
        }

        /* ---------- Side cards ---------- */
        .side-heading {
            margin: 1.8rem 0 0.65rem;
            color: #f3f7ff;
            font-size: 1rem;
            font-weight: 720;
        }

        .info-card {
            margin-top: 1.2rem;
            padding: 1.2rem 1.25rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: linear-gradient(160deg, rgba(17, 28, 51, 0.86), rgba(9, 16, 30, 0.80));
        }

        .card-kicker {
            margin-bottom: 0.8rem;
            color: var(--blue-400);
            font-size: 0.68rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .step {
            display: flex;
            gap: 0.7rem;
            margin: 0.75rem 0;
            color: #bfd0e6;
            font-size: 0.87rem;
            line-height: 1.5;
        }

        .step-num {
            flex: 0 0 auto;
            display: grid;
            place-items: center;
            width: 1.55rem;
            height: 1.55rem;
            border: 1px solid rgba(96, 165, 250, 0.25);
            border-radius: 8px;
            background: rgba(37, 99, 235, 0.13);
            color: #93c5fd;
            font-size: 0.68rem;
            font-weight: 750;
        }

        .small-note {
            color: #8ea3c1;
            font-size: 0.78rem;
            line-height: 1.55;
        }

        .setup-warning {
            margin: 1.1rem 0 0;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(251, 191, 36, 0.28);
            border-radius: 12px;
            background: rgba(120, 53, 15, 0.24);
            color: #fde68a;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        /* ---------- Prediction result ---------- */
        .result-card {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            margin-top: 1.35rem;
            padding: 1.35rem 1.4rem;
            border-radius: 18px;
            animation: result-in 260ms ease-out;
        }

        @keyframes result-in {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card.fake {
            border: 1px solid rgba(248, 113, 113, 0.32);
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.28), rgba(15, 23, 42, 0.70));
        }

        .result-card.real {
            border: 1px solid rgba(52, 211, 153, 0.30);
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.26), rgba(15, 23, 42, 0.70));
        }

        .result-icon {
            display: grid;
            place-items: center;
            width: 2.9rem;
            height: 2.9rem;
            flex: 0 0 auto;
            border-radius: 13px;
        }

        .result-card.fake .result-icon {
            color: #fca5a5;
            background: rgba(239, 68, 68, 0.13);
            border: 1px solid rgba(248, 113, 113, 0.20);
        }

        .result-card.real .result-icon {
            color: #6ee7b7;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(52, 211, 153, 0.20);
        }

        .result-icon svg {
            width: 1.5rem;
            height: 1.5rem;
        }

        .result-kicker {
            margin-bottom: 0.22rem;
            color: #93a9c7;
            font-size: 0.68rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .result-title {
            font-size: 1.7rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }

        .result-card.fake .result-title { color: #fecaca; }
        .result-card.real .result-title { color: #a7f3d0; }

        .result-copy {
            margin-top: 0.45rem;
            color: #c1d0e5;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .result-copy strong { color: #f8fbff; }

        .probability-panel {
            margin-top: 1rem;
            padding: 1.15rem 1.25rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: rgba(9, 16, 30, 0.64);
        }

        .prob-title {
            margin-bottom: 0.95rem;
            color: #dbeafe;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .prob-row + .prob-row {
            margin-top: 0.9rem;
        }

        .prob-head {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.38rem;
            color: #b9cbe3;
            font-size: 0.82rem;
            font-weight: 650;
        }

        .prob-track {
            height: 0.55rem;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.12);
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
        }

        .result-note {
            margin-top: 0.8rem;
            color: #8ea3c1;
            font-size: 0.79rem;
            line-height: 1.55;
        }

        /* ---------- Evaluation ---------- */
        .metric-card {
            padding: 1.05rem 1.05rem 0.95rem;
            border: 1px solid var(--line);
            border-radius: 15px;
            background: linear-gradient(160deg, rgba(17, 28, 51, 0.90), rgba(9, 16, 30, 0.72));
        }

        .metric-label {
            color: #93a9c7;
            font-size: 0.71rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .metric-value {
            margin-top: 0.35rem;
            color: #eaf2ff;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .metric-foot {
            margin-top: 0.2rem;
            color: #7187a5;
            font-size: 0.72rem;
        }

        .perf-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0.35rem;
            font-size: 0.85rem;
        }

        .perf-table th {
            padding: 0.58rem 0.55rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.22);
            color: #93a9c7;
            text-align: left;
            font-size: 0.69rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .perf-table td {
            padding: 0.66rem 0.55rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.09);
            color: #cedaeb;
        }

        .perf-table tr:last-child td {
            border-bottom: none;
        }

        .perf-table td:first-child {
            color: #eef4ff;
            font-weight: 700;
        }

        /* ---------- About ---------- */
        .about-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1.15rem;
        }

        .about-card {
            padding: 1.25rem 1.3rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: linear-gradient(160deg, rgba(17, 28, 51, 0.84), rgba(9, 16, 30, 0.72));
        }

        .about-card h3 {
            margin: 0 0 0.65rem;
            color: #f1f6ff;
            font-size: 1.03rem;
        }

        .about-card p {
            margin: 0;
            color: #b4c5dc;
            font-size: 0.88rem;
            line-height: 1.65;
        }

        .about-card code {
            padding: 0.1rem 0.3rem;
            border-radius: 5px;
            background: rgba(37, 99, 235, 0.15);
            color: #bfdbfe;
        }

        .callout {
            margin-top: 1.1rem;
            padding: 1rem 1.15rem;
            border-left: 3px solid #3b82f6;
            border-radius: 0 12px 12px 0;
            background: rgba(37, 99, 235, 0.08);
            color: #bfd0e6;
            font-size: 0.88rem;
            line-height: 1.65;
        }

        .footer-note {
            margin-top: 2.3rem;
            padding-top: 1.1rem;
            border-top: 1px solid var(--line);
            color: #6f86a6;
            font-size: 0.75rem;
            line-height: 1.6;
            text-align: center;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 13px;
            background: rgba(9, 16, 30, 0.50);
        }

        .stAlert {
            border-radius: 12px;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }

            .hero {
                padding: 1.5rem 1.25rem 1.4rem;
                border-radius: 19px;
            }

            .hero h1 {
                font-size: 2.35rem;
            }

            .hero-subtitle {
                font-size: 0.92rem;
            }

            .about-grid {
                grid-template-columns: 1fr;
            }

            .result-title {
                font-size: 1.45rem;
            }

            .result-card {
                padding: 1.1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-top">
                <div class="brand-icon">{NEWSPAPER_ICON}</div>
                <div class="hero-badge">
                    <span class="status-dot"></span>
                    DistilBERT · NLP classifier
                </div>
            </div>
            <div class="hero-kicker">Transformer-powered news analysis</div>
            <h1>Fake News <span>Detector</span></h1>
            <p class="hero-subtitle">
                Paste a news headline or article below. A DistilBERT model fine-tuned
                on labelled news will estimate whether the writing looks more like
                fake-news or real-news examples. The result is a language-pattern
                estimate, not a factual verification.
            </p>
            <div class="chips">
                <span class="chip">Fine-tuned transformer</span>
                <span class="chip">Binary classification</span>
                <span class="chip">Confidence score</span>
                <span class="chip">Test-set evaluation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_selected_sample():
    """Button callback: put the selected example into the text area."""
    selected_name = st.session_state.get("sample_choice")
    if selected_name in SAMPLES:
        st.session_state["news_text"] = SAMPLES[selected_name]
        st.session_state.pop("prediction", None)


def render_probability_bars(probabilities):
    fake_percent = probabilities[0] * 100
    real_percent = probabilities[1] * 100

    st.markdown(
        f"""
        <div class="probability-panel">
            <div class="prob-title">Class probabilities</div>
            <div class="prob-row">
                <div class="prob-head"><span>FAKE</span><span>{fake_percent:.1f}%</span></div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{fake_percent:.2f}%; background:linear-gradient(90deg,#dc2626,#f87171);"></div>
                </div>
            </div>
            <div class="prob-row">
                <div class="prob-head"><span>REAL</span><span>{real_percent:.1f}%</span></div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{real_percent:.2f}%; background:linear-gradient(90deg,#059669,#34d399);"></div>
                </div>
            </div>
            <div class="result-note">
                Softmax turns the model's raw scores into probabilities. A high score
                means the text resembles one class more strongly in this model's
                training data; it is not proof that the article is true or false.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(prediction):
    is_fake = prediction["label"] == "FAKE"
    tone = "fake" if is_fake else "real"
    icon = ALERT_ICON if is_fake else CHECK_ICON
    title = "FAKE NEWS" if is_fake else "REAL NEWS"
    confidence = prediction["confidence"] * 100

    explanation = (
        "The model found language patterns that are more similar to the fake-news "
        "examples it was trained on."
        if is_fake
        else "The model found language patterns that are more similar to the real-news "
        "examples it was trained on."
    )

    st.markdown(
        f"""
        <div class="result-card {tone}">
            <div class="result-icon">{icon}</div>
            <div>
                <div class="result-kicker">Model prediction</div>
                <div class="result-title">{title}</div>
                <div class="result-copy">
                    {explanation} Estimated confidence for this class:
                    <strong>{confidence:.1f}%</strong>.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_probability_bars(prediction["probabilities"])

    st.caption(
        "Confidence is not evidence. The model does not browse the web, check "
        "sources, or verify facts. Always confirm important news with reliable outlets."
    )


def render_detector_tab():
    model_ready = is_model_available()

    input_column, side_column = st.columns([1.65, 1], gap="large")

    with input_column:
        st.markdown('<div class="section-kicker">News input</div>', unsafe_allow_html=True)
        st.markdown(
            '<h2 class="input-title">Paste a headline or article</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="input-help">Longer text usually gives the model more context. '
            "The first 256 tokens are used.</p>",
            unsafe_allow_html=True,
        )

        if not model_ready:
            st.markdown(
                '<div class="setup-warning">'
                "No trained model was found. Run <code>python train_model.py</code> "
                "to create <code>saved_model/</code>, or set "
                "<code>MODEL_REPO_ID</code> in Streamlit secrets to load a model "
                "from the Hugging Face Hub."
                "</div>",
                unsafe_allow_html=True,
            )

        text = st.text_area(
            "News text",
            key="news_text",
            height=250,
            max_chars=MAX_CHARACTERS,
            label_visibility="collapsed",
            placeholder=(
                "Paste a news headline or article here...\n\n"
                "Example: Officials announced a new policy on Tuesday..."
            ),
        )

        st.markdown(
            f'<div class="char-count">{len(text):,} / {MAX_CHARACTERS:,} characters</div>',
            unsafe_allow_html=True,
        )

        analyze_clicked = st.button(
            "Analyze News",
            type="primary",
            use_container_width=True,
            disabled=not model_ready,
        )

        clean_text = text.strip()

        if analyze_clicked:
            if len(clean_text) < MIN_CHARACTERS:
                st.warning(
                    f"Please enter at least {MIN_CHARACTERS} characters so the "
                    "model has enough context to analyze."
                )
                st.session_state.pop("prediction", None)
            else:
                try:
                    with st.spinner("Reading context and estimating probabilities..."):
                        prediction = predict_news(clean_text)
                    st.session_state["prediction"] = prediction
                except Exception as error:
                    st.error(
                        "The model could not analyze this text. Check that the model "
                        "files are present or that MODEL_REPO_ID is configured correctly."
                    )
                    with st.expander("Technical details"):
                        st.exception(error)
                    st.session_state.pop("prediction", None)

        saved_prediction = st.session_state.get("prediction")
        if saved_prediction and saved_prediction["text"] == clean_text:
            render_prediction(saved_prediction)

    with side_column:
        st.markdown('<div class="side-heading">Try a sample</div>', unsafe_allow_html=True)
        st.selectbox(
            "Sample text",
            options=list(SAMPLES.keys()),
            key="sample_choice",
            label_visibility="collapsed",
        )
        st.button(
            "Load example",
            use_container_width=True,
            on_click=load_selected_sample,
        )
        st.caption("Examples are for testing the interface, not verified facts.")

        st.markdown(
            """
            <div class="info-card">
                <div class="card-kicker">How it works</div>
                <div class="step">
                    <span class="step-num">01</span>
                    <span>Your text is split into word pieces by the DistilBERT tokenizer.</span>
                </div>
                <div class="step">
                    <span class="step-num">02</span>
                    <span>The fine-tuned model reads the whole sequence and its context.</span>
                </div>
                <div class="step">
                    <span class="step-num">03</span>
                    <span>Softmax converts the two class scores into FAKE and REAL probabilities.</span>
                </div>
                <div class="step">
                    <span class="step-num">04</span>
                    <span>The higher probability becomes the displayed prediction.</span>
                </div>
            </div>
            <div class="info-card">
                <div class="card-kicker">What the score means</div>
                <p class="small-note">
                    The score reflects similarity to patterns learned from labelled
                    training data. It can be wrong, especially for satire, new topics,
                    or articles written in an unusual style.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="callout"><strong>Disclaimer:</strong> This is an educational '
        "machine-learning project, not a fact-checking service. Predictions may be "
        "incorrect. Verify important information with primary sources and trusted "
        "news organizations.</div>",
        unsafe_allow_html=True,
    )


def render_performance_tab():
    st.markdown('<div class="section-kicker">Held-out test results</div>', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="input-title">Model performance</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="input-help">These numbers are read from '
        "<code>outputs/evaluation_metrics.json</code>, which is created by "
        "<code>evaluate_model.py</code> after training.</p>",
        unsafe_allow_html=True,
    )

    if not METRICS_JSON.exists():
        st.info(
            "No evaluation file is available yet. Train the model, then run "
            "`python evaluate_model.py` to generate real test-set metrics."
        )
        return

    try:
        metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.error("The evaluation file could not be read.")
        with st.expander("Technical details"):
            st.exception(error)
        return

    metric_cards = [
        ("Accuracy", metrics.get("accuracy"), "All test articles"),
        ("Precision", metrics.get("precision"), "Positive class: REAL"),
        ("Recall", metrics.get("recall"), "Positive class: REAL"),
        ("F1-score", metrics.get("f1_score"), "Balance of P and R"),
    ]

    columns = st.columns(4)
    for column, (label, value, footer) in zip(columns, metric_cards):
        with column:
            display_value = f"{float(value) * 100:.2f}%" if value is not None else "N/A"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{display_value}</div>
                    <div class="metric-foot">{footer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    test_samples = metrics.get("test_samples")
    class_counts = metrics.get("class_counts", {})
    if test_samples:
        st.caption(
            f"Measured on {test_samples:,} unseen test articles "
            f"({class_counts.get('FAKE', 0):,} FAKE, {class_counts.get('REAL', 0):,} REAL). "
            "The test split was never used to update the model."
        )

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        if CONFUSION_MATRIX_PNG.exists():
            st.image(
                str(CONFUSION_MATRIX_PNG),
                caption="Predicted vs. true labels on the held-out test set",
                use_container_width=True,
            )
        else:
            st.info("Confusion-matrix image not found. Run `python evaluate_model.py`.")

    with right:
        per_class = metrics.get("per_class", {})
        if per_class:
            rows = ""
            for label in ("FAKE", "REAL"):
                details = per_class.get(label, {})
                rows += (
                    "<tr>"
                    f"<td>{label}</td>"
                    f"<td>{float(details.get('precision', 0)) * 100:.2f}%</td>"
                    f"<td>{float(details.get('recall', 0)) * 100:.2f}%</td>"
                    f"<td>{float(details.get('f1_score', 0)) * 100:.2f}%</td>"
                    f"<td>{int(details.get('support', 0)):,}</td>"
                    "</tr>"
                )

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-kicker">Per-class breakdown</div>
                    <table class="perf-table">
                        <thead>
                            <tr>
                                <th>Class</th><th>Precision</th><th>Recall</th>
                                <th>F1</th><th>Support</th>
                            </tr>
                        </thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("What do these metrics mean?"):
            st.markdown(
                """
                **Accuracy** is the percentage of all test articles classified correctly.

                **Precision (REAL)** answers: of all articles the model labelled REAL,
                how many really were REAL? High precision means fewer false alarms.

                **Recall (REAL)** answers: of all truly REAL articles, how many did the
                model find? High recall means fewer REAL articles were missed.

                **F1-score** combines precision and recall into one balanced number.
                It is useful when the classes are not perfectly balanced.

                **Confusion matrix:** the top-left cell is correctly detected FAKE,
                bottom-right is correctly detected REAL. The off-diagonal cells are
                mistakes (false positives and false negatives).
                """
            )

        st.caption(
            "The positive class for the headline precision/recall values is REAL. "
            "The per-class table shows both classes separately."
        )


def render_about_tab():
    st.markdown('<div class="section-kicker">Under the hood</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="input-title">About the model</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="about-grid">
            <div class="about-card">
                <h3>What is DistilBERT?</h3>
                <p>
                    DistilBERT is a smaller, faster version of BERT, a transformer
                    language model. It reads words together with their context instead
                    of treating each word alone. Here it is fine-tuned on thousands of
                    labelled fake and real news examples to perform binary
                    classification.
                </p>
            </div>
            <div class="about-card">
                <h3>Why keywords are not enough</h3>
                <p>
                    The word <code>report</code> can appear in a real article
                    (&ldquo;The report describes the new policy&rdquo;) and in a fake
                    article (&ldquo;A secret report proves everything&rdquo;). A keyword
                    rule sees the same word in both. DistilBERT learns from the
                    surrounding words, phrase patterns, and the whole sequence.
                </p>
            </div>
            <div class="about-card">
                <h3>What it learns</h3>
                <p>
                    Fine-tuning adjusts the model's internal weights to separate the
                    two labelled classes. The output is a statistical pattern match,
                    not a database lookup. The model does not know whether a claim is
                    actually true unless that pattern was represented in its training
                    data.
                </p>
            </div>
            <div class="about-card">
                <h3>Important limitations</h3>
                <p>
                    It does not browse the internet, inspect sources, or verify facts.
                    It can learn dataset shortcuts such as topic or writing style.
                    Near-duplicate stories and source overlap can also inflate test
                    scores. Always confirm important news independently.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="callout"><strong>For the viva:</strong> this project uses '
        "supervised NLP. The model is fine-tuned on text labelled 0 (FAKE) and 1 "
        "(REAL), then evaluated on a separate held-out test set using accuracy, "
        "precision, recall, F1-score, and a confusion matrix. The probabilities "
        "describe model confidence, not factual truth.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <div class="card-kicker">Project stack</div>
            <p class="small-note">
                Python · PyTorch · Hugging Face Transformers · DistilBERT ·
                scikit-learn · pandas · Matplotlib/Seaborn · Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_css()
    render_header()

    detector_tab, performance_tab, about_tab = st.tabs(
        ["Detector", "Model performance", "About the model"]
    )

    with detector_tab:
        render_detector_tab()
    with performance_tab:
        render_performance_tab()
    with about_tab:
        render_about_tab()

    st.markdown(
        '<div class="footer-note">'
        "Fake News Detector · Educational NLP project · DistilBERT fine-tuning "
        "with Hugging Face Transformers and PyTorch<br>"
        "Predictions can be wrong. This app does not provide fact-checking or "
        "professional advice."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
