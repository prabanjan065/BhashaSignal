
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sarvamai import SarvamAI
except Exception:
    SarvamAI = None

st.set_page_config(
    page_title="BhashaSignal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background: #08111f; color: #e8eef7; }
[data-testid="stSidebar"] { background: #0d1728; }
.block-container { padding-top: 1.2rem; max-width: 1450px; }
.hero { padding: 24px 26px; border: 1px solid #243653; border-radius: 18px;
        background: linear-gradient(135deg,#101e34,#0b1627); margin-bottom: 18px; }
.hero h1 { margin: 0; font-size: 38px; }
.hero p { margin: 7px 0 0; color: #a9b8cd; font-size: 16px; }
.card { background:#0f1b2d; border:1px solid #243653; border-radius:16px;
        padding:18px; min-height:120px; }
.metric { font-size:30px; font-weight:700; margin-top:5px; }
.muted { color:#94a8c2; font-size:13px; }
.signal { border-left:5px solid #ffb020; background:#121f31; padding:18px 20px;
          border-radius:12px; border-top:1px solid #2b3e5b;
          border-right:1px solid #2b3e5b; border-bottom:1px solid #2b3e5b; }
.signal h2 { margin:0 0 5px; }
.pill { display:inline-block; padding:4px 9px; border-radius:20px;
        background:#18324d; color:#b9dcff; margin-right:5px; font-size:12px; }
.evidence { background:#0c1727; border:1px solid #20334e; border-radius:12px;
            padding:12px 14px; margin:8px 0; }
.smalltitle { color:#b7c7dc; font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "observations.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def extract_observation(text, location="Unknown", language="Auto"):
    t = text.lower()
    issue = "General observation"
    category = "Other"
    severity = "Medium"

    rules = [
        (["water", "தண்ணி", "தண்ணீர்", "pressure", "smell", "வாசனை"], "Water service", "Water"),
        (["power", "electricity", "current", "மின்சாரம்", "லைட்", "light"], "Power service", "Utilities"),
        (["road", "pothole", "சாலை", "குழி", "traffic", "போக்குவரத்து"], "Road / traffic", "Infrastructure"),
        (["garbage", "waste", "குப்பை", "smell", "துர்நாற்றம்"], "Waste / sanitation", "Sanitation"),
        (["internet", "wifi", "network", "இணையம்"], "Connectivity", "Connectivity"),
    ]
    for keywords, label, cat in rules:
        if any(k in t for k in keywords):
            issue, category = label, cat
            break

    if any(k in t for k in ["very", "severe", "danger", "critical", "மிகவும்", "அவசரம்"]):
        severity = "High"
    elif any(k in t for k in ["slight", "little", "minor", "கொஞ்சம்"]):
        severity = "Low"

    frequency = "Single"
    if any(k in t for k in ["again", "repeated", "every day", "daily", "yesterday", "since", "மீண்டும்", "தினமும்", "நேற்று"]):
        frequency = "Repeated"

    return {
        "issue": issue,
        "category": category,
        "location": location if location != "Unknown" else "Unspecified",
        "severity": severity,
        "frequency": frequency,
        "language": language,
        "confidence": 0.90 if issue != "General observation" else 0.64,
    }

def find_signal(df):
    if len(df) < 2:
        return None

    texts = df["text"].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    threshold = 0.30
    related = set()
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if sim[i,j] >= threshold:
                related.add(i); related.add(j)

    if not related:
        # Demo fallback: use category frequency.
        counts = df["category"].value_counts()
        if counts.empty:
            return None
        cat = counts.index[0]
        members = df[df["category"] == cat]
    else:
        members = df.iloc[sorted(related)]

    counts = members["category"].value_counts()
    if counts.empty:
        return None
    cat = counts.index[0]
    members = members[members["category"] == cat]

    if len(members) >= 3:
        signal_name = members["issue"].iloc[0]
        score = min(98, 55 + len(members)*5 + members["location"].nunique()*4)
        return {
            "name": signal_name,
            "category": cat,
            "count": len(members),
            "locations": int(members["location"].nunique()),
            "score": int(score),
            "members": members,
        }
    return None

def sarvam_transcribe(uploaded_file, mode="transcribe"):
    key = os.getenv("SARVAM_API_KEY", "")
    if not key or SarvamAI is None:
        return None, "Sarvam API key not configured. Demo mode is available."
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        client = SarvamAI(api_subscription_key=key)
        response = client.speech_to_text.transcribe(
            file=open(tmp_path, "rb"),
            model="saaras:v4",
            mode=mode,
        )
        return response.transcript, None
    except Exception as e:
        return None, f"Sarvam transcription failed: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

df = load_data()
signal = find_signal(df)

with st.sidebar:
    st.markdown("## 📡 BhashaSignal")
    st.caption("AI Early-Signal Detection from Multilingual Voice Observations")
    page = st.radio(
        "Navigate",
        ["Command Center", "New Observation", "Signal Explorer", "Evidence"],
        index=0,
    )
    st.divider()
    st.markdown("### Demo mode")
    st.caption("The included dataset lets you demonstrate the full pipeline without an API key.")
    if os.getenv("SARVAM_API_KEY"):
        st.success("Sarvam API connected")
    else:
        st.info("Sarvam API not connected")
    st.divider()
    st.caption("PS41 • Sarvam API")

if page == "Command Center":
    st.markdown("""
    <div class="hero">
      <h1>📡 BhashaSignal</h1>
      <p>From multilingual voice observations to explainable early signals.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    metrics = [
        ("Observations", len(df), "Across the demo window"),
        ("Languages", df["language"].nunique(), "Multilingual inputs"),
        ("Locations", df["location"].nunique(), "Observed areas"),
        ("Active signals", 1 if signal else 0, "Pattern requiring attention"),
    ]
    for col,(label,val,sub) in zip([c1,c2,c3,c4],metrics):
        with col:
            st.markdown(f'<div class="card"><div class="smalltitle">{label}</div><div class="metric">{val}</div><div class="muted">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("### 🚨 AI Signal")
    if signal:
        st.markdown(f"""
        <div class="signal">
          <div class="smalltitle">Emerging pattern</div>
          <h2>{signal["name"]}</h2>
          <span class="pill">{signal["category"]}</span>
          <span class="pill">{signal["count"]} related observations</span>
          <span class="pill">{signal["locations"]} locations</span>
          <p style="margin-top:14px;">Signal confidence <b>{signal["score"]}%</b> — the system found repeated, semantically related observations across the dataset.</p>
        </div>
        """, unsafe_allow_html=True)

        left,right = st.columns([1,1])
        with left:
            st.markdown("#### Why did AI detect this?")
            st.write(
                f"The pattern combines {signal['count']} related observations from "
                f"{signal['locations']} distinct locations. Similar wording is not required; "
                "the detector uses text similarity plus category/frequency evidence."
            )
        with right:
            counts = df["category"].value_counts().reset_index()
            counts.columns = ["category","count"]
            fig = px.bar(counts, x="count", y="category", orientation="h",
                         title="Observation categories")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=45,b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active signal detected in the current observation window.")

    st.markdown("### Latest observations")
    show = df.sort_values("timestamp", ascending=False).head(6).copy()
    st.dataframe(show[["timestamp","language","location","category","issue","severity","text"]],
                 use_container_width=True, hide_index=True)

elif page == "New Observation":
    st.markdown("## 🎙️ New Observation")
    st.caption("Speak naturally. BhashaSignal turns the observation into structured intelligence.")

    uploaded = st.file_uploader(
        "Upload a short voice observation",
        type=["wav","mp3","m4a","ogg","webm","flac"],
        help="Sarvam REST speech-to-text is designed for short clips up to 30 seconds.",
    )
    col1,col2 = st.columns(2)
    with col1:
        location = st.text_input("Location", "Block B")
    with col2:
        language = st.selectbox("Language", ["Auto","Tamil","English","Tamil-English","Kannada","Hindi","Telugu"])

    if uploaded:
        st.audio(uploaded)
        if st.button("🧠 Transcribe with Sarvam", type="primary"):
            transcript, err = sarvam_transcribe(uploaded, "transcribe")
            if transcript:
                st.session_state["transcript"] = transcript
                st.success("Transcription completed.")
            else:
                st.warning(err)

    transcript = st.text_area(
        "Observation text",
        value=st.session_state.get("transcript",""),
        height=130,
        placeholder="Example: Water pressure has been very low since yesterday...",
    )

    if st.button("🔎 Analyse Observation"):
        if not transcript.strip():
            st.error("Add an observation first.")
        else:
            result = extract_observation(transcript, location, language)
            st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.markdown("### AI-Extracted Observation")
        a,b,c = st.columns(3)
        for col,title,value in [(a,"Issue",r["issue"]),(b,"Category",r["category"]),(c,"Severity",r["severity"])]:
            with col:
                st.markdown(f'<div class="card"><div class="smalltitle">{title}</div><div class="metric" style="font-size:22px">{value}</div></div>', unsafe_allow_html=True)
        st.write("")
        st.json(r)
        st.success("Observation is ready to enter the signal engine.")

elif page == "Signal Explorer":
    st.markdown("## 🔎 Signal Explorer")
    st.caption("Explore the evidence behind an emerging pattern.")

    cat = st.selectbox("Filter category", ["All"] + sorted(df["category"].unique().tolist()))
    filtered = df if cat == "All" else df[df["category"] == cat]

    fig = px.scatter(
        filtered, x="timestamp", y="location",
        size="severity_score", color="category",
        hover_data=["language","issue","text"],
        title="Observations over time",
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    if signal:
        st.markdown("### Signal evidence")
        for _, row in signal["members"].sort_values("timestamp").iterrows():
            st.markdown(
                f'<div class="evidence"><b>{row["location"]}</b> · {row["language"]} · '
                f'{row["timestamp"].strftime("%d %b %H:%M")}<br>{row["text"]}</div>',
                unsafe_allow_html=True
            )

elif page == "Evidence":
    st.markdown("## 🧾 Evidence & Explainability")
    st.caption("Every signal is traceable to the observations that contributed to it.")

    if signal:
        st.markdown(f"### {signal['name']}")
        st.write(
            f"**Signal confidence:** {signal['score']}%  \n"
            f"**Supporting observations:** {signal['count']}  \n"
            f"**Distinct locations:** {signal['locations']}"
        )
        st.markdown("#### Contributing observations")
        for _, row in signal["members"].iterrows():
            st.markdown(f"""
            <div class="evidence">
              <span class="pill">{row["language"]}</span>
              <span class="pill">{row["location"]}</span>
              <span class="pill">{row["severity"]} severity</span>
              <p style="margin:10px 0 0;">“{row["text"]}”</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No signal evidence is currently available.")
