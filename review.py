"""
ReelSense — IMDb Movie Review Sentiment Analyzer
==================================================
A Streamlit front end for the SimpleRNN sentiment model trained in
`IMDb_Sentiment_Analysis_RNN.ipynb`.

Expected files in the same folder as this script:
    - sentiment_rnn_model.keras   (trained model, saved by the notebook)
    - word_index.pkl              (IMDb word index, saved by the notebook)

Run with:
    streamlit run app.py
"""

import os
import re
import pickle
import datetime as dt

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------------------------------
# 1. PAGE CONFIG & CONSTANTS
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ReelSense · IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

VOCAB_SIZE = 10000
MAX_LENGTH = 500
MODEL_PATH = "sentiment_rnn_model.keras"
WORD_INDEX_PATH = "word_index.pkl"

# Reported in the training notebook (Steps 8 & 10)
TEST_ACCURACY = 0.7866
TEST_PRECISION_RECALL_F1 = 0.79

POSITIVE_COLOR = ("#33D6A6", "#1FA98A")   # emerald / teal gradient
NEGATIVE_COLOR = ("#FF5C7A", "#C6335E")   # crimson / pink gradient
GOLD = "#E8B84B"

SAMPLE_REVIEWS = {
    "🌟 Glowing praise": (
        "An absolute masterpiece from start to finish. The performances were "
        "captivating, the cinematography breathtaking, and the story kept me "
        "glued to my seat the entire time. I walked out of the theater "
        "speechless."
    ),
    "🎻 Emotional score": (
        "The soundtrack alone gives me chills. Every scene was crafted with "
        "so much heart and detail, easily one of the best films I have seen "
        "this year. I would watch it again in a heartbeat."
    ),
    "🏆 Awards-worthy": (
        "A bold, beautifully directed film with a script that respects its "
        "audience. The lead actor delivers a career-best performance. "
        "Genuinely moving and endlessly rewatchable."
    ),
    "💤 Dragged on": (
        "I really wanted to like this movie, but the plot dragged on forever "
        "and the acting felt wooden. Two hours I will never get back."
    ),
    "🎯 Missed the mark": (
        "Predictable, poorly paced, and the dialogue was cringe worthy "
        "throughout. The trailer promised so much more than this delivered."
    ),
    "🧨 Style over substance": (
        "The special effects looked cheap and the storyline made no sense. "
        "A frustrating, forgettable experience and a complete waste of an "
        "evening."
    ),
}

FLOAT_EMOJIS = ["🎬", "🍿", "🎥", "🎞️", "⭐", "🎭", "🎦", "🎟️"]


# --------------------------------------------------------------------------
# 2. CACHED RESOURCE LOADERS
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_word_index():
    if not os.path.exists(WORD_INDEX_PATH):
        return None
    with open(WORD_INDEX_PATH, "rb") as f:
        return pickle.load(f)


model = load_sentiment_model()
word_index = load_word_index()


# --------------------------------------------------------------------------
# 3. TEXT PREPROCESSING (mirrors the notebook's training pipeline exactly)
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip HTML/punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_review(text: str, index: dict, vocab_size: int = VOCAB_SIZE,
                   max_length: int = MAX_LENGTH) -> np.ndarray:
    """
    Convert raw text into the padded integer sequence the SimpleRNN expects.
    Keras' IMDb dataset reserves index 0 for padding, 1 for the start token,
    and 2 for out-of-vocabulary words; real word indices are offset by +3.
    """
    tokens = clean_text(text).split()
    encoded = [1]  # <START>
    for word in tokens:
        idx = index.get(word)
        if idx is None or idx + 3 >= vocab_size:
            encoded.append(2)  # <OOV>
        else:
            encoded.append(idx + 3)
    return pad_sequences([encoded], maxlen=max_length, padding="pre", truncating="pre")


def predict_sentiment(text: str):
    seq = encode_review(text, word_index)
    prob = float(model.predict(seq, verbose=0)[0][0])
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob
    return sentiment, confidence, prob


# --------------------------------------------------------------------------
# 4. GLOBAL STYLE — fonts, glassmorphism, floating background emojis
# --------------------------------------------------------------------------
def inject_global_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

        html, body, [class*="css"]  { font-family: 'Manrope', sans-serif; }

        .stApp {
            background: radial-gradient(circle at 15% 0%, #1c1636 0%, transparent 45%),
                        radial-gradient(circle at 85% 10%, #17233c 0%, transparent 40%),
                        linear-gradient(160deg, #0a0b1a 0%, #12132b 45%, #1a1030 100%);
            background-attachment: fixed;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .block-container { position: relative; z-index: 1; padding-top: 2rem; max-width: 1200px; }

        /* ---------- Floating cinema emoji backdrop ---------- */
        .bg-emoji-container {
            position: fixed; inset: 0; overflow: hidden;
            z-index: 0; pointer-events: none;
        }
        .bg-emoji {
            position: absolute; bottom: -12%;
            opacity: 0; filter: blur(0.2px);
            animation-name: floatUp;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
        }
        @keyframes floatUp {
            0%   { transform: translateY(0) translateX(0) rotate(0deg);   opacity: 0; }
            8%   { opacity: 0.16; }
            50%  { transform: translateY(-55vh) translateX(24px) rotate(18deg); }
            92%  { opacity: 0.16; }
            100% { transform: translateY(-118vh) translateX(-18px) rotate(-14deg); opacity: 0; }
        }

        /* ---------- Typography ---------- */
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-weight: 800;
            font-size: 3.1rem;
            line-height: 1.1;
            background: linear-gradient(100deg, #F5F3FF 20%, #E8B84B 55%, #F5F3FF 80%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #A8A3C9; font-size: 1.05rem; font-weight: 500;
            margin-bottom: 1.6rem;
        }
        .section-label {
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.14em; text-transform: uppercase;
            font-size: 0.72rem; color: #E8B84B; font-weight: 600;
            margin-bottom: 0.4rem;
        }

        /* ---------- Glass cards ---------- */
        .glass-card {
            background: rgba(255,255,255,0.055);
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
            border-radius: 20px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        }

        /* ---------- Stat chips ---------- */
        .stat-row { display: flex; gap: 0.9rem; flex-wrap: wrap; margin-bottom: 1.8rem; }
        .stat-chip {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px; padding: 0.6rem 1.1rem;
            font-family: 'JetBrains Mono', monospace;
            color: #E8E6F5; font-size: 0.82rem;
            display: flex; align-items: center; gap: 0.5rem;
        }
        .stat-chip b { color: #E8B84B; font-size: 0.95rem; }

        /* ---------- Streamlit widget overrides ---------- */
        .stTextArea textarea {
            background: rgba(255,255,255,0.045) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 16px !important;
            color: #F5F3FF !important;
            font-size: 1rem !important;
            padding: 1rem !important;
        }
        .stTextArea textarea:focus {
            border: 1px solid #E8B84B !important;
            box-shadow: 0 0 0 1px #E8B84B33 !important;
        }
        .stButton>button {
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            background: rgba(255,255,255,0.06) !important;
            color: #F5F3FF !important;
            font-weight: 600 !important;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            border: 1px solid #E8B84B !important;
            color: #E8B84B !important;
            transform: translateY(-1px);
        }
        div[data-testid="stFormSubmitButton"] button,
        .stButton>button[kind="primary"] {
            background: linear-gradient(100deg, #E8B84B, #C6923A) !important;
            color: #14152E !important; border: none !important;
        }
        .stButton>button[kind="primary"]:hover {
            filter: brightness(1.08);
            color: #14152E !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0e22 0%, #14122a 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * { color: #E8E6F5; }

        .disclaimer {
            color: #A8A3C9; font-size: 0.82rem; text-align: center;
            margin-top: 2.5rem; padding-top: 1.2rem;
            border-top: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Floating emoji layer (rendered once, purely decorative)
    positions = [4, 12, 20, 30, 38, 47, 55, 63, 71, 79, 86, 93, 9, 60]
    spans = []
    for i, left in enumerate(positions):
        emoji = FLOAT_EMOJIS[i % len(FLOAT_EMOJIS)]
        size = 1.3 + (i % 4) * 0.35
        duration = 18 + (i * 3) % 20
        delay = -(i * 2.7)
        spans.append(
            f'<span class="bg-emoji" style="left:{left}%; font-size:{size:.2f}rem; '
            f'animation-duration:{duration}s; animation-delay:{delay:.1f}s;">{emoji}</span>'
        )
    st.markdown(f'<div class="bg-emoji-container">{"".join(spans)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 5. RESULT COMPONENT — animated gauge + cinema marquee verdict
# --------------------------------------------------------------------------
def render_result_component(sentiment: str, confidence: float, prob: float):
    is_positive = sentiment == "Positive"
    c1, c2 = POSITIVE_COLOR if is_positive else NEGATIVE_COLOR
    emoji = "✅" if is_positive else "🛑"
    verdict_word = "POSITIVE" if is_positive else "NEGATIVE"
    pct = round(confidence * 100, 1)

    radius = 80
    arc_len = round(3.14159265 * radius, 2)          # semicircle length
    target_offset = round(arc_len * (1 - confidence), 2)

    html = f"""
    <div style="font-family:'Manrope',sans-serif; display:flex; flex-direction:column; align-items:center;">
      <style>
        @keyframes bulbChase {{
            0%, 100% {{ opacity: 0.25; }}
            50% {{ opacity: 1; }}
        }}
        @keyframes glowPulse {{
            0%, 100% {{ box-shadow: 0 0 18px 2px {c1}55; }}
            50% {{ box-shadow: 0 0 30px 6px {c1}88; }}
        }}
        .marquee {{
            position: relative;
            border: 2px solid {c1};
            border-radius: 999px;
            padding: 0.7rem 2.2rem;
            font-family: 'Playfair Display', serif;
            font-weight: 700;
            font-size: 1.4rem;
            letter-spacing: 0.08em;
            color: {c1};
            background: rgba(255,255,255,0.04);
            animation: glowPulse 2.4s ease-in-out infinite;
            margin-top: 0.6rem;
        }}
        .bulb {{
            position: absolute; width: 6px; height: 6px; border-radius: 50%;
            background: {c1}; animation: bulbChase 1.4s infinite;
        }}
      </style>

      <svg viewBox="0 0 200 110" width="230" height="128">
        <defs>
          <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="{c1}" />
            <stop offset="100%" stop-color="{c2}" />
          </linearGradient>
        </defs>
        <path d="M20,105 A{radius},{radius} 0 0 1 180,105" fill="none"
              stroke="rgba(255,255,255,0.08)" stroke-width="14" stroke-linecap="round" />
        <path id="arcFill" d="M20,105 A{radius},{radius} 0 0 1 180,105" fill="none"
              stroke="url(#arcGrad)" stroke-width="14" stroke-linecap="round"
              stroke-dasharray="{arc_len}" stroke-dashoffset="{arc_len}"
              style="transition: stroke-dashoffset 1.1s cubic-bezier(.34,1.4,.64,1);" />
        <text x="100" y="88" text-anchor="middle" font-family="JetBrains Mono, monospace"
              font-size="26" font-weight="700" fill="#F5F3FF" id="pctText">0%</text>
        <text x="100" y="104" text-anchor="middle" font-family="Manrope, sans-serif"
              font-size="9" letter-spacing="2" fill="#A8A3C9">CONFIDENCE</text>
      </svg>

      <div class="marquee">{emoji}&nbsp; {verdict_word}</div>
      <div style="color:#A8A3C9; font-size:0.78rem; font-family:'JetBrains Mono',monospace; margin-top:0.6rem;">
        raw sigmoid output&nbsp;=&nbsp;{prob:.4f}
      </div>
    </div>

    <script>
      const arcEl = document.getElementById("arcFill");
      const pctEl = document.getElementById("pctText");
      const targetOffset = {target_offset};
      const targetPct = {pct};
      requestAnimationFrame(() => {{
        arcEl.style.strokeDashoffset = targetOffset;
      }});
      let start = null;
      function step(ts) {{
        if (!start) start = ts;
        const progress = Math.min((ts - start) / 1100, 1);
        pctEl.textContent = (progress * targetPct).toFixed(1) + "%";
        if (progress < 1) requestAnimationFrame(step);
        else pctEl.textContent = targetPct.toFixed(1) + "%";
      }}
      requestAnimationFrame(step);
    </script>
    """
    components.html(html, height=280)


# --------------------------------------------------------------------------
# 6. APP BODY
# --------------------------------------------------------------------------
inject_global_css()

if "review_input" not in st.session_state:
    st.session_state.review_input = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None


def set_sample(text: str):
    st.session_state.review_input = text
    st.session_state.result = None


# ---------------------------- Sidebar -------------------------------------
with st.sidebar:
    st.markdown("### 🎬 ReelSense")
    st.caption("SimpleRNN · IMDb Sentiment Engine")
    st.markdown("---")

    st.markdown("**Model status**")
    if model is not None and word_index is not None:
        st.success("Model & word index loaded ✅")
    else:
        st.error("Model files not found ⚠️")
        st.caption(
            f"Place `{MODEL_PATH}` and `{WORD_INDEX_PATH}` in this app's "
            "folder — both are produced by running the training notebook."
        )

    st.markdown("---")
    with st.expander("🧠 Model architecture"):
        st.code(
            "Input(500,)\n"
            "→ Embedding(10000, 128)\n"
            "→ SimpleRNN(64)\n"
            "→ Dropout(0.5)\n"
            "→ Dense(1, activation='sigmoid')",
            language="text",
        )

    with st.expander("📊 Training metrics"):
        st.metric("Test accuracy", f"{TEST_ACCURACY*100:.2f}%")
        st.metric("Precision / Recall / F1", f"≈ {TEST_PRECISION_RECALL_F1:.2f}")
        st.caption("25,000 training reviews · 25,000 test reviews · vocab = 10,000 · max length = 500")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("**🕘 Recent analyses**")
        for item in reversed(st.session_state.history[-5:]):
            badge = "🟢" if item["sentiment"] == "Positive" else "🔴"
            st.caption(f"{badge} {item['sentiment']} · {item['confidence']*100:.1f}% · {item['time']}")

    st.markdown("---")
    st.caption("Built with Streamlit · TensorFlow/Keras SimpleRNN")

# ---------------------------- Hero -----------------------------------------
st.markdown('<div class="hero-title">🎬 ReelSense</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-powered sentiment analysis for movie reviews, '
    'powered by a Recurrent Neural Network trained on 50,000 IMDb reviews.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-chip">🎯 <b>{TEST_ACCURACY*100:.1f}%</b>&nbsp;test accuracy</div>
        <div class="stat-chip">🗂️ <b>50,000</b>&nbsp;reviews trained</div>
        <div class="stat-chip">🔤 <b>10,000</b>&nbsp;word vocabulary</div>
        <div class="stat-chip">🧬 <b>SimpleRNN</b>&nbsp;architecture</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None or word_index is None:
    st.warning(
        "⚠️ The trained model files were not found. Run the training notebook "
        f"first so it saves **{MODEL_PATH}** and **{WORD_INDEX_PATH}** next to "
        "this app, then relaunch Streamlit."
    )
    st.stop()

# ---------------------------- Main layout -----------------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Step 1 · Write or pick a review</div>', unsafe_allow_html=True)
    st.text_area(
        "Movie review",
        key="review_input",
        height=180,
        placeholder="Type a movie review here, e.g. \"The pacing was slow but the ending completely won me over...\"",
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label" style="margin-top:0.9rem;">Or try a sample</div>', unsafe_allow_html=True)
    sample_cols = st.columns(3)
    for i, (label, text) in enumerate(SAMPLE_REVIEWS.items()):
        sample_cols[i % 3].button(label, on_click=set_sample, args=(text,), use_container_width=True)

    btn_col1, btn_col2 = st.columns([1, 1])
    analyze_clicked = btn_col1.button("🎬 Analyze Sentiment", type="primary", use_container_width=True)
    if btn_col2.button("🧹 Clear", use_container_width=True):
        st.session_state.review_input = ""
        st.session_state.result = None
        st.rerun()

    if analyze_clicked:
        text = st.session_state.review_input.strip()
        if not text:
            st.error("Please enter a review or choose a sample first.")
        else:
            with st.spinner("🎞️ Rolling the film... analyzing sentiment"):
                sentiment, confidence, prob = predict_sentiment(text)
            st.session_state.result = {"sentiment": sentiment, "confidence": confidence, "prob": prob}
            st.session_state.history.append(
                {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "time": dt.datetime.now().strftime("%H:%M:%S"),
                }
            )

    word_count = len(st.session_state.review_input.split())
    truncated_note = " · truncated to first 500 tokens" if word_count > MAX_LENGTH else ""
    st.caption(f"📝 {word_count} words{truncated_note}")

with right:
    st.markdown('<div class="section-label">Step 2 · Verdict</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if st.session_state.result is None:
        st.markdown(
            '<div style="text-align:center; padding:2.4rem 0; color:#A8A3C9;">'
            '🎥<br><br>Your sentiment verdict will light up here.</div>',
            unsafe_allow_html=True,
        )
    else:
        r = st.session_state.result
        render_result_component(r["sentiment"], r["confidence"], r["prob"])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">ReelSense uses a SimpleRNN trained for educational purposes and can '
    'misjudge sarcasm, mixed reviews, or very long text. Not intended for production moderation use.</div>',
    unsafe_allow_html=True,
)