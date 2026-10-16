import os
import sys

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make our src/ modules importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from metrics import match_sources, si_sdr   # noqa: E402
from pipeline import separate               # noqa: E402

SR = 16000
MAX_SECONDS = 30
NAMES = ["Speech", "Music", "Ambient"]
COLORS = ["#4F8CFF", "#B07CFF", "#3DDC97"]

st.set_page_config(page_title="Unmix the Signal", page_icon="🎛️", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.hero {padding: 2.2rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, #1b2a4a 0%, #3a2766 55%, #0f3d3e 100%);}
.hero h1 {margin: 0; font-size: 2.6rem; color: #fff;}
.hero p {margin: .4rem 0 0 0; font-size: 1.05rem; color: #cfd8ee;}
div[data-testid="stMetric"] {background: #1A1F2B; border: 1px solid #2a3142;
       padding: 1rem 1.2rem; border-radius: 14px;}
.stButton > button {border-radius: 12px; padding: .6rem 1.4rem; font-weight: 600;}
.block-container {padding-top: 2rem;}
</style>
<div class="hero">
  <h1>🎛️ Unmix the Signal</h1>
  <p>Blind source separation with Independent Component Analysis,
     PCA whitening and FastICA written from scratch in NumPy.</p>
</div>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
def to_audio(x):
    x = x - x.mean()
    return (x / (np.max(np.abs(x)) + 1e-12) * 0.9).astype(np.float32)


def audio_row(signals, labels):
    cols = st.columns(len(signals))
    for col, s, label in zip(cols, signals, labels):
        with col:
            st.caption(label)
            st.audio(to_audio(s), sample_rate=SR)


def waveform_fig(signals, labels, title, colors):
    n, T = signals.shape
    step = max(1, T // 2500)                      # downsample for a fast plot
    t = np.arange(T)[::step] / SR
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=0.07, subplot_titles=labels)
    for i, s in enumerate(signals):
        s = s - s.mean()
        s = s / (np.max(np.abs(s)) + 1e-12)
        fig.add_trace(go.Scatter(x=t, y=s[::step], mode="lines",
                                 line=dict(width=1, color=colors[i % len(colors)]),
                                 showlegend=False), row=i + 1, col=1)
    fig.update_layout(height=150 * n + 70, title=title, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(title_text="time (s)", row=n, col=1)
    return fig


@st.cache_data(show_spinner=False)
def run_ica(X):
    return separate(X)


tab_demo, tab_upload, tab_how = st.tabs(["🎧 Demo", "⬆️ Upload your own", "🧠 How it works"])

# ---------- Demo tab ----------
with tab_demo:
    st.subheader("Three microphones, three sources")
    st.write("We mixed a **speech** clip, a **music** clip and **rain/wind** with our own "
             "mixing matrix. Press the button to recover them with ICA.")

    S = np.load(os.path.join(ROOT, "data", "mixed", "S_true.npy"))
    X = np.load(os.path.join(ROOT, "data", "mixed", "X_mixed.npy"))

    st.markdown("##### What the microphones recorded")
    audio_row(X, [f"Mixture {i + 1}" for i in range(len(X))])

    if st.button("✨ Separate the sources", type="primary", key="demo_btn"):
        st.session_state["demo_done"] = True

    if st.session_state.get("demo_done"):
        with st.spinner("Running FastICA..."):
            res = run_ica(X)
        S_est = res["S_est"]
        order = match_sources(S_est, S)           # resolve permutation
        est = np.array([S_est[order[j]] for j in range(3)])

        st.markdown("##### Recovered by ICA")
        audio_row(est, NAMES)

        st.markdown("##### Quality (higher is better)")
        cols = st.columns(3)
        for j, col in enumerate(cols):
            sdr_ica = si_sdr(est[j], S[j])
            sdr_mix = max(si_sdr(x, S[j]) for x in X)
            corr = abs(np.corrcoef(est[j], S[j])[0, 1])
            col.metric(f"{NAMES[j]}  (|corr| {corr:.4f})", f"{sdr_ica:.1f} dB SI-SDR",
                       delta=f"{sdr_ica - sdr_mix:+.1f} dB vs best mixture")

        c1, c2 = st.columns(2)
        c1.plotly_chart(waveform_fig(X, [f"Mixture {i + 1}" for i in range(3)],
                                     "Mixtures", ["#8a93a6"]), use_container_width=True)
        c2.plotly_chart(waveform_fig(est, NAMES, "Separated", COLORS),
                        use_container_width=True)

# ---------- Upload tab ----------
with tab_upload:
    st.subheader("Bring your own mixtures")
    st.write("Upload **at least 2** recordings in which several sounds overlap, with **as many "
             "recordings as sources** (for example 3 microphones for 3 sources). The recordings "
             "should capture the same sounds at the same time. WAV files work best.")

    files = st.file_uploader("Mixed recordings", type=["wav", "flac", "ogg", "mp3"],
                             accept_multiple_files=True)

    if files and len(files) < 2:
        st.warning("Upload at least 2 files.")

    if files and len(files) >= 2:
        try:
            sigs = []
            for f in files:
                f.seek(0)
                y, _ = librosa.load(f, sr=SR, mono=True, duration=MAX_SECONDS)
                sigs.append(y)
            T = min(len(y) for y in sigs)
            Xu = np.vstack([y[:T] for y in sigs])
            st.caption(f"{len(files)} signals, {T / SR:.1f} s each (trimmed to the shortest, "
                       f"max {MAX_SECONDS} s)")
            audio_row(Xu, [f.name for f in files])

            if st.button("✨ Separate", type="primary", key="up_btn"):
                st.session_state["up_done"] = True

            if st.session_state.get("up_done"):
                with st.spinner("Running FastICA..."):
                    res = run_ica(Xu)
                if res["eigvals"].min() / res["eigvals"].max() < 1e-6:
                    st.warning("Some recordings are almost identical, so the problem is "
                               "ill-conditioned and separation may be poor.")
                est = res["S_est"]
                labels = [f"Source {i + 1}" for i in range(len(est))]
                st.markdown("##### Recovered signals")
                audio_row(est, labels)
                st.caption("ICA cannot know the order, sign or loudness of the sources, so "
                           "'Source 1' may be any of them.")
                st.plotly_chart(waveform_fig(est, labels, "Separated signals", COLORS),
                                use_container_width=True)
        except Exception as e:
            st.error(f"Could not process these files: {e}")

# ---------- How it works tab ----------
with tab_how:
    st.subheader("The pipeline")
    st.markdown("""
1. **Mix:** each microphone records a weighted sum of all sources.
2. **Center:** subtract each signal's mean so the data has zero mean.
3. **Whiten (PCA):** rotate and rescale so the signals are uncorrelated with unit variance.
4. **FastICA:** find the rotation that makes each output as **non-Gaussian** as possible.
5. **Evaluate:** match outputs to sources (permutation), then score with correlation and SI-SDR.
""")
    st.latex(r"X = AS \quad\Rightarrow\quad Z = VX_c \quad\Rightarrow\quad \hat S = WZ")
    st.markdown("""
**Why non-Gaussianity?** By the Central Limit Theorem, a sum of independent signals is more
Gaussian than the signals themselves, so the most non-Gaussian projections are the original sources.

**Why not just PCA?** PCA only uses second-order statistics (covariance). Any rotation of white data
is still white, so covariance cannot tell which rotation is right. ICA uses higher-order statistics.
""")