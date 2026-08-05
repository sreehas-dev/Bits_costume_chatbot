import streamlit as st
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval
import faiss
import json
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import requests
import assemblyai as aai
import difflib
import re
from pathlib import Path
from typing import Any
from uuid import uuid4
from sentence_transformers import SentenceTransformer

from feedback import CsvFeedbackStore, FeedbackRenderer

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "finetuned_model"
EMBEDDINGS_PATH = BASE_DIR / "embeddings"
DATA_PATH = BASE_DIR / "data"
INDEX_PATH = EMBEDDINGS_PATH / "faiss_index.bin"
FAQ_DATA_PATH = DATA_PATH / "faq_merged_v2.json"
LOGO_PATH  = "bits_logo.png"
FEEDBACK_CSV_PATH = BASE_DIR / "feedback_logs.csv"

try:
    ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]
except Exception:
    ASSEMBLYAI_API_KEY = "YOUR_ASSEMBLYAI_API_KEY_HERE"

THRESHOLD = 0.60
TOP_K = 1
CHAT_INPUT_PLACEHOLDER = "Ask me anything about courses, registration, or campus life."
BASE_ASSISTANT_INTRO = "I am BITS Pilani AI Assistant."
FALLBACK_RESPONSE = (
    "I'm sorry, I couldn't find specific information regarding that "
    "in the student handbook. Please check with the Academic Division directly."
)
GREETING_PROMPTS = {
    "hi",
    "hello",
    "hey",
    "hii",
    "hiii",
    "helo",
    "helloo",
    "good morning",
    "good afternoon",
    "good evening",
}

def _as_valid_hour(value: Any) -> int | None:
    try:
        if value is None:
            return None
        hour = int(value)
        return hour if 0 <= hour <= 23 else None
    except Exception:
        return None


def read_browser_time() -> dict[str, Any]:
    """Read browser-local time directly from the frontend.

    This avoids depending on Streamlit Cloud/server timezone or URL rewriting,
    which can be inconsistent on deployed desktop browsers.
    """
    value = streamlit_js_eval(
        js_expressions="""
        (() => {
          const now = new Date();
          return {
            hour: now.getHours(),
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || "",
            timezoneOffset: now.getTimezoneOffset(),
            timestamp: now.getTime()
          };
        })()
        """,
        key="browser_time_bridge",
        default=None,
    )
    if isinstance(value, dict):
        hour = _as_valid_hour(value.get("hour"))
        if hour is not None:
            browser_time = {
                "hour": hour,
                "timezone": str(value.get("timezone") or ""),
                "timezone_offset": int(value.get("timezoneOffset") or 0),
                "timestamp": int(value.get("timestamp") or 0),
            }
            st.session_state.browser_time = browser_time
            return browser_time
    existing = st.session_state.get("browser_time", {})
    return existing if isinstance(existing, dict) else {}


def get_browser_hour() -> int | None:
    browser_time = st.session_state.get("browser_time", {})
    if isinstance(browser_time, dict):
        hour = _as_valid_hour(browser_time.get("hour"))
        if hour is not None:
            return hour
    return None


def time_greeting_for_hour(hour: int | None) -> str:
    if hour is None:
        return "Hello"
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    return "Good evening"


def get_time_greeting() -> str:
    return time_greeting_for_hour(get_browser_hour())


def get_default_assistant_greeting() -> str:
    return f"{get_time_greeting()}! {BASE_ASSISTANT_INTRO}"


def make_initial_message() -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": get_default_assistant_greeting(),
        "feedback_enabled": False,
        "is_initial_greeting": True,
    }


def is_initial_landing(messages_list: list[dict[str, Any]]) -> bool:
    return (
        len(messages_list) == 1
        and messages_list[0].get("role") == "assistant"
        and bool(messages_list[0].get("is_initial_greeting"))
    )


aai.settings.api_key = ASSEMBLYAI_API_KEY

# ──────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="BITS Pilani Assistant",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }
        .landing-greeting-client { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

read_browser_time()

components.html(
    """
    <script>
    (function () {
      function greetingForHour(hour) {
        if (hour >= 5 && hour < 12) return "Good morning";
        if (hour >= 12 && hour < 17) return "Good afternoon";
        return "Good evening";
      }
      function updateGreeting() {
        const text = greetingForHour(new Date().getHours()) + "! I am BITS Pilani AI Assistant.";
        const doc = window.parent.document;
        doc.querySelectorAll(".landing-greeting-client").forEach(function (el) {
          el.textContent = text;
          el.style.visibility = "visible";
        });
      }
      try {
        updateGreeting();
        window.setInterval(updateGreeting, 60000);
      } catch (error) {
        console.warn("Unable to update browser-local greeting", error);
      }
    })();
    </script>
    """,
    height=0,
    scrolling=False,
)

# ──────────────────────────────────────────────
# LOAD RESOURCES
# ──────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = SentenceTransformer(str(MODEL_PATH))
    index = faiss.read_index(str(INDEX_PATH))
    with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model, index, data

try:
    model, index, data = load_resources()
except Exception:
    st.error("System is initialising… Please wait or check resource paths.")
    st.stop()

# ──────────────────────────────────────────────
# STREAMING TOKEN
# ──────────────────────────────────────────────
@st.cache_data(ttl=540)
def get_streaming_token(api_key: str) -> str:
    try:
        resp = requests.get(
            "https://streaming.assemblyai.com/v3/token",
            params={"expires_in_seconds": 600},
            headers={"Authorization": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("token", "")
    except Exception as e:
        st.error(f"AssemblyAI token error: {e}")
        return ""

STREAMING_TOKEN = get_streaming_token(ASSEMBLYAI_API_KEY)

@st.cache_resource
def get_feedback_renderer() -> FeedbackRenderer:
    return FeedbackRenderer(CsvFeedbackStore(FEEDBACK_CSV_PATH))


feedback_renderer = get_feedback_renderer()

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [make_initial_message()]

messages: list[dict[str, Any]] = st.session_state.messages

current_initial_greeting = get_default_assistant_greeting()
if is_initial_landing(messages) and messages[0].get("content") != current_initial_greeting:
    messages[0]["content"] = current_initial_greeting

if "feedback_state_loaded" not in st.session_state:
    for rated_message_id in feedback_renderer.store.get_rated_message_ids():
        st.session_state[f"feedback_{rated_message_id}"] = "saved"
    st.session_state.feedback_state_loaded = True

# This is kept only if you later want to use it; the current code writes directly into the chat input.
if "voice_prompt" not in st.session_state:
    st.session_state.voice_prompt = None

if "pending_landing_prompt" not in st.session_state:
    st.session_state.pending_landing_prompt = None

# ──────────────────────────────────────────────
# ANSWER LOGIC
# ──────────────────────────────────────────────
def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def is_greeting_prompt(prompt: str) -> bool:
    normalized = normalize_text(prompt)
    return normalized in GREETING_PROMPTS


@st.cache_data(show_spinner=False)
def build_faq_vocabulary(faq_data: list[dict[str, Any]]) -> list[str]:
    vocabulary: set[str] = set()
    for record in faq_data:
        for field in ("question", "answer", "category"):
            value = str(record.get(field, ""))
            for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", value.lower()):
                vocabulary.add(token)
    return sorted(vocabulary)


def autocorrect_prompt(prompt: str, faq_data: list[dict[str, Any]]) -> str:
    vocabulary = build_faq_vocabulary(faq_data)
    if not vocabulary:
        return prompt

    def correct_token(match: re.Match[str]) -> str:
        token = match.group(0)
        lower_token = token.lower()
        if len(lower_token) < 4 or lower_token in vocabulary or token.isupper():
            return token
        matches = difflib.get_close_matches(lower_token, vocabulary, n=1, cutoff=0.84)
        if not matches:
            return token
        replacement = matches[0]
        if token[:1].isupper():
            replacement = replacement.capitalize()
        return replacement

    return re.sub(r"\b[A-Za-z][A-Za-z0-9-]*\b", correct_token, prompt)


def stream_markdown_response(text: str, placeholder: Any) -> None:
    """Stream Markdown text without collapsing newlines or list formatting."""
    displayed = ""
    for token in re.findall(r"\S+\s*|\n+", text):
        displayed += token
        placeholder.markdown(displayed + "▌")
        time.sleep(0.02)
    placeholder.markdown(displayed.strip())


def format_faq_answer(answer: str) -> str:
    """Convert raw FAQ answer text into readable Markdown using generic patterns."""
    text = re.sub(r"\s+", " ", str(answer).strip())
    if not text:
        return text

    text = re.sub(r"\s*●\s*", "\n- ", text)
    text = re.sub(r"(?<![\w.])([a-g])\.\s+(?=[A-Z])", "\n- ", text)
    text = re.sub(r"\s+o\s+(?=[A-Z])", "\n  - ", text)
    text = re.sub(r"\s*(Option)[-\s]*(\d+)\s*:", r"\n\n**Option \2:** ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<![\w.])(\d+)\.\s+", r"\n\1. ", text)
    text = re.sub(r"\s*PS\s*:", "\n\n**Important note:**", text, flags=re.IGNORECASE)

    heading_patterns = [
        r"NOC Submission Process",
        r"Qualification requirements",
        r"Selection process",
        r"Points to cover in each VIVA",
        r"Common links to be used by all students",
        r"Important Format Links for General/Engineering Students",
        r"Important Format Links for Management Students",
        r"Restriction",
        r"Important",
        r"Note",
    ]
    for pattern in heading_patterns:
        text = re.sub(
            rf"\s*({pattern})\s*:",
            lambda match: f"\n\n**{match.group(1)}:**",
            text,
            flags=re.IGNORECASE,
        )

    text = re.sub(
        r"\s+(For\s+(?:MBA|M\.Tech|B\.Tech|Dissertation|Project work)[^:]{0,80}?Students)\s*:",
        lambda match: f"\n\n**{match.group(1)}:**",
        text,
        flags=re.IGNORECASE,
    )

    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and cleaned_lines:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def process_prompt(prompt: str, append_user_message: bool = True, render_user_message: bool = True) -> None:
    prompt = prompt.strip()
    if not prompt:
        return

    if append_user_message:
        messages.append({"role": "user", "content": prompt})
    if render_user_message:
        st.chat_message("user").write(prompt)

    try:
        assistant_ctx = st.chat_message("assistant", avatar=LOGO_PATH)
    except Exception:
        assistant_ctx = st.chat_message("assistant")

    with assistant_ctx:
        placeholder = st.empty()

        matched_question = ""
        faq_id = ""
        category = ""
        top_score = 1.0

        if is_greeting_prompt(prompt):
            full_response = get_default_assistant_greeting()
            st.session_state.last_confidence = top_score
        else:
            search_prompt = autocorrect_prompt(prompt, data)
            embedding = np.asarray(model.encode(search_prompt, normalize_embeddings=True), dtype="float32")
            scores, indices = index.search(np.array([embedding]), TOP_K)

            top_score = float(scores[0][0])
            top_idx = int(indices[0][0])
            st.session_state.last_confidence = top_score

            if top_score < THRESHOLD:
                full_response = FALLBACK_RESPONSE
            else:
                matched_record = data[top_idx]
                full_response = format_faq_answer(matched_record["answer"])
                matched_question = matched_record.get("question", "")
                faq_id = matched_record.get("id", "")
                category = matched_record.get("category", "")

        stream_markdown_response(full_response, placeholder)

    assistant_message = {
        "role": "assistant",
        "content": full_response,
        "message_id": str(uuid4()),
        "query": prompt,
        "confidence": top_score,
        "matched_question": matched_question,
        "faq_id": faq_id,
        "category": category,
        "feedback_enabled": not is_greeting_prompt(prompt),
    }
    messages.append(assistant_message)
    feedback_renderer.render(assistant_message, st.session_state)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    try:
        st.image(LOGO_PATH, width="stretch")
    except Exception:
        st.header("BITS Pilani")


    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [make_initial_message()]
        st.session_state.voice_prompt = None
        st.session_state.pending_landing_prompt = None
        st.rerun()

    with st.expander("🛠️ Admin Tools"):
        if "last_confidence" in st.session_state:
            st.metric("Last Confidence", f"{st.session_state.last_confidence:.3f}")
        st.metric("Feedback Records", feedback_renderer.store.count())
        st.caption(f"CSV: {FEEDBACK_CSV_PATH}")
        browser_time = st.session_state.get("browser_time", {})
        st.caption(
            "Browser time debug: "
            f"hour={browser_time.get('hour', '') if isinstance(browser_time, dict) else ''}, "
            f"tz={browser_time.get('timezone', '') if isinstance(browser_time, dict) else ''}, "
            f"offset={browser_time.get('timezone_offset', '') if isinstance(browser_time, dict) else ''}, "
            f"computed={get_browser_hour()}"
        )

# ──────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ──────────────────────────────────────────────
landing_mode = is_initial_landing(messages)

if landing_mode:
    st.markdown(
        f"""
        <style>
            .block-container {{ padding-top: 28vh !important; }}
            div[data-testid="stChatInput"] {{
                position: relative !important;
                bottom: auto !important;
                max-width: 720px;
                margin: 1.25rem auto 0 auto;
            }}
            div[data-testid="stBottom"] {{
                position: relative !important;
                bottom: auto !important;
                background: transparent !important;
                box-shadow: none !important;
            }}
            .landing-greeting {{
                text-align: center;
                font-size: 1.9rem;
                font-weight: 650;
                line-height: 1.35;
                margin: 0 auto;
            }}
        </style>
        <div class="landing-greeting landing-greeting-client">{get_default_assistant_greeting()}</div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in messages:
        if msg["role"] == "assistant":
            try:
                with st.chat_message("assistant", avatar=LOGO_PATH):
                    st.markdown(msg["content"])
                    feedback_renderer.render(msg, st.session_state)
            except Exception:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    feedback_renderer.render(msg, st.session_state)
        else:
            st.chat_message("user").write(msg["content"])

# ──────────────────────────────────────────────
# OPTIONAL: if you later set voice_prompt from JS bridge
# ──────────────────────────────────────────────
if st.session_state.voice_prompt:
    pending = st.session_state.voice_prompt
    st.session_state.voice_prompt = None
    if landing_mode:
        messages.append({"role": "user", "content": str(pending)})
        st.session_state.pending_landing_prompt = str(pending)
        st.rerun()
    else:
        process_prompt(pending)

pending_landing_prompt = st.session_state.pending_landing_prompt
if pending_landing_prompt and not landing_mode:
    st.session_state.pending_landing_prompt = None
    process_prompt(str(pending_landing_prompt), append_user_message=False, render_user_message=False)

# ──────────────────────────────────────────────
# TYPED INPUT
# ──────────────────────────────────────────────
if typed_prompt := st.chat_input(CHAT_INPUT_PLACEHOLDER, key="landing_chat_input" if landing_mode else "chat_input"):
    if landing_mode:
        prompt_text = str(typed_prompt)
        messages.append({"role": "user", "content": prompt_text})
        st.session_state.pending_landing_prompt = prompt_text
        st.rerun()
    else:
        process_prompt(str(typed_prompt))

# ──────────────────────────────────────────────
# VOICE COMPONENT
# ──────────────────────────────────────────────
VOICE_HTML = r"""
<script>
(function () {
  const P = window.parent;
  const TOKEN = "__TOKEN__";
  const WS_URL = "wss://streaming.assemblyai.com/v3/ws";
  const SAMPLE_RATE = 16000;
  const BTN_ID = "aai-mic-btn";
  const RECORDRTC_ID = "aai-recordrtc-script";
  const PULSE_STYLE_ID = "aai-pulse-css";

  function getDoc() {
    return P.document;
  }

  function loadScriptIntoParent(src, id, cb) {
    const pd = getDoc();
    if (!pd || !pd.head) return;

    if (id && pd.getElementById(id)) {
      cb();
      return;
    }

    const s = pd.createElement("script");
    if (id) s.id = id;
    s.src = src;
    s.onload = cb;
    s.onerror = () => console.error("Failed to load script:", src);
    pd.head.appendChild(s);
  }

  function injectCss() {
    const pd = getDoc();
    if (!pd || pd.getElementById(PULSE_STYLE_ID)) return;

    const style = pd.createElement("style");
    style.id = PULSE_STYLE_ID;
    style.textContent = `
      @keyframes aaiPulse {
        0%   { box-shadow: 0 0 0 0px rgba(168,85,247,0.7); }
        70%  { box-shadow: 0 0 0 10px rgba(168,85,247,0); }
        100% { box-shadow: 0 0 0 0px rgba(168,85,247,0); }
      }
      #${BTN_ID}.recording {
        animation: aaiPulse 1.2s ease-out infinite !important;
      }
    `;
    pd.head.appendChild(style);
  }

  function getWrapper() {
    return P.document.querySelector('div[data-testid="stChatInput"]');
  }

  function getTextarea() {
    const w = getWrapper();
    return w ? w.querySelector("textarea") : null;
  }

  function setTextarea(text) {
    const ta = getTextarea();
    if (!ta) return;

    const setter = Object.getOwnPropertyDescriptor(
      P.HTMLTextAreaElement.prototype,
      "value"
    ).set;

    setter.call(ta, text);
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    ta.focus();
  }

  function buildMicSvg(active) {
    const stroke = active ? "#a855f7" : "#ffffff";
    return `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
        stroke="${stroke}" stroke-width="2.2"
        stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="2" width="6" height="12" rx="3"></rect>
        <path d="M5 10a7 7 0 0 0 14 0"></path>
        <line x1="12" y1="19" x2="12" y2="22"></line>
        <line x1="8" y1="22" x2="16" y2="22"></line>
      </svg>
    `;
  }

  function ensureState() {
    if (P.__aaiMicState) return P.__aaiMicState;

    P.__aaiMicState = {
      socket: null,
      recorder: null,
      micStream: null,
      isRecording: false,
      committedText: "",
      partialText: "",
      finalizing: false,
      finalizeTimer: null,
      intentionalClose: false,
    };

    return P.__aaiMicState;
  }

  function setButtonVisual(active) {
    const btn = getButton();
    if (!btn) return;

    btn.innerHTML = buildMicSvg(active);

    if (active) {
      btn.classList.add("recording");
      btn.style.background = "rgba(168,85,247,0.18)";
    } else {
      btn.classList.remove("recording");
      btn.style.background = "transparent";
    }
  }

  function getButton() {
    const w = getWrapper();
    return w ? w.querySelector(`#${BTN_ID}`) : null;
  }

  function removeOldButton() {
    const old = P.document.getElementById(BTN_ID);
    if (old) old.remove();
  }

  function silentCleanup() {
    const s = ensureState();

    if (s.finalizeTimer) {
      clearTimeout(s.finalizeTimer);
      s.finalizeTimer = null;
    }

    if (s.recorder) {
      try { s.recorder.stopRecording(); } catch (e) {}
      s.recorder = null;
    }

    if (s.micStream) {
      try { s.micStream.getTracks().forEach(t => t.stop()); } catch (e) {}
      s.micStream = null;
    }

    if (s.socket) {
      try {
        s.socket.onopen = null;
        s.socket.onmessage = null;
        s.socket.onerror = null;
        s.socket.onclose = null;
        s.socket.close();
      } catch (e) {}
      s.socket = null;
    }
  }

  function hardReset() {
    const s = ensureState();
    silentCleanup();
    s.isRecording = false;
    s.committedText = "";
    s.partialText = "";
    s.finalizing = false;
    s.intentionalClose = false;
    setButtonVisual(false);
  }

  function finalizeTranscript() {
    const s = ensureState();
    if (s.finalizing) return;
    s.finalizing = true;

    if (s.finalizeTimer) {
      clearTimeout(s.finalizeTimer);
      s.finalizeTimer = null;
    }

    if (s.socket) {
      try {
        s.socket.onopen = null;
        s.socket.onmessage = null;
        s.socket.onerror = null;
        s.socket.onclose = null;
        s.socket.close();
      } catch (e) {}
      s.socket = null;
    }

    const finalText = (s.committedText || "").trim();

    s.isRecording = false;
    s.committedText = "";
    s.partialText = "";
    s.finalizing = false;
    s.intentionalClose = false;

    setButtonVisual(false);

    if (finalText) setTextarea(finalText);
  }

  function startSession() {
    const s = ensureState();

    if (s.isRecording) return;

    if (!TOKEN || TOKEN === "YOUR_ASSEMBLYAI_API_KEY_HERE") {
      alert("AssemblyAI streaming token is missing.");
      return;
    }

    hardReset();
    s.isRecording = true;
    setButtonVisual(true);

    let ws;
    try {
      ws = new P.WebSocket(
        WS_URL +
          "?sample_rate=" + SAMPLE_RATE +
          "&encoding=pcm_s16le" +
          "&token=" + encodeURIComponent(TOKEN)
      );
    } catch (e) {
      alert("WebSocket could not be created: " + e.message);
      hardReset();
      return;
    }

    s.socket = ws;

    ws.onopen = async function () {
      if (s.socket !== ws) {
        try { ws.close(); } catch (e) {}
        return;
      }

      try {
        const stream = await P.navigator.mediaDevices.getUserMedia({ audio: true });
        if (s.socket !== ws) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }

        s.micStream = stream;

        if (!P.RecordRTC || !P.StereoAudioRecorder) {
          alert("RecordRTC did not load correctly.");
          hardReset();
          return;
        }

        s.recorder = new P.RecordRTC(stream, {
          type: "audio",
          mimeType: "audio/webm;codecs=pcm",
          recorderType: P.StereoAudioRecorder,
          desiredSampRate: SAMPLE_RATE,
          numberOfAudioChannels: 1,
          bufferSize: 4096,
          timeSlice: 250,
          ondataavailable: function (blob) {
            if (s.socket === ws && ws.readyState === 1) {
              blob.arrayBuffer().then(function (buf) {
                if (s.socket === ws && ws.readyState === 1) {
                  ws.send(buf);
                }
              });
            }
          }
        });

        s.recorder.startRecording();
      } catch (err) {
        alert("Microphone access denied: " + err.message);
        hardReset();
      }
    };

    ws.onmessage = function (ev) {
      if (s.socket !== ws) return;

      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (e) {
        return;
      }

      if (msg.type === "Turn") {
        s.partialText = msg.transcript || "";
        const live = (s.committedText + " " + s.partialText).trim();
        if (live) setTextarea(live);

        if (msg.end_of_turn) {
          s.committedText = live;
          s.partialText = "";
        }
      }
    };

    ws.onerror = function () {
      if (s.socket === ws) hardReset();
    };

    ws.onclose = function () {
      if (s.socket === ws && s.isRecording && !s.intentionalClose) {
        hardReset();
      }
    };
  }

  function stopSession() {
    const s = ensureState();
    if (!s.isRecording) return;

    const ws = s.socket;
    s.isRecording = false;
    s.intentionalClose = true;
    setButtonVisual(false);

    if (s.recorder) {
      try { s.recorder.stopRecording(); } catch (e) {}
      s.recorder = null;
    }

    if (s.micStream) {
      try { s.micStream.getTracks().forEach(t => t.stop()); } catch (e) {}
      s.micStream = null;
    }

    if (ws && ws.readyState === 1) {
      ws.onmessage = function (ev) {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch (e) {
          return;
        }

        if (msg.type === "Turn" && msg.transcript) {
          s.committedText = (s.committedText + " " + msg.transcript).trim();
          setTextarea(s.committedText);
        }

        if (msg.type === "Termination") {
          finalizeTranscript();
        }
      };

      ws.onclose = null;

      try {
        ws.send(JSON.stringify({ type: "Terminate" }));
      } catch (e) {}

      s.finalizeTimer = setTimeout(finalizeTranscript, 1500);
    } else {
      finalizeTranscript();
    }
  }

  function ensureButton() {
    const w = getWrapper();
    if (!w) return;

    removeOldButton();

    const ta = getTextarea();
    if (ta) ta.style.paddingRight = "92px";
    w.style.position = "relative";

    const sendButton = w.querySelector('button[kind="icon"], button[data-testid="stChatInputSubmitButton"], button[aria-label="Send"]');
    if (sendButton) {
      sendButton.style.position = "absolute";
      sendButton.style.right = "12px";
      sendButton.style.bottom = "50%";
      sendButton.style.transform = "translateY(50%)";
      sendButton.style.zIndex = "10000";
    }

    const extraButtons = Array.from(w.querySelectorAll('button')).filter(function (button) {
      return button.id !== BTN_ID && button !== sendButton && button.getAttribute("aria-label") !== "Send";
    });
    extraButtons.forEach(function (button) {
      const label = (button.getAttribute("aria-label") || button.title || "").toLowerCase();
      if (label.includes("attach") || label.includes("file") || label.includes("add")) {
        button.style.display = "none";
      }
    });

    const s = ensureState();

    const btn = P.document.createElement("button");
    btn.id = BTN_ID;
    btn.type = "button";
    btn.title = "Voice input";
    btn.innerHTML = buildMicSvg(s.isRecording);

    btn.style.position = "absolute";
    btn.style.right = "48px";
    btn.style.bottom = "50%";
    btn.style.transform = "translateY(50%)";
    btn.style.zIndex = "9999";
    btn.style.background = "transparent";
    btn.style.border = "none";
    btn.style.cursor = "pointer";
    btn.style.padding = "6px";
    btn.style.borderRadius = "50%";
    btn.style.lineHeight = "0";
    btn.style.display = "flex";
    btn.style.alignItems = "center";
    btn.style.justifyContent = "center";
    btn.style.transition = "background 0.2s";
    btn.style.outline = "none";

    if (s.isRecording) btn.classList.add("recording");

    btn.addEventListener("mouseenter", function () {
      if (!s.isRecording) btn.style.background = "rgba(168,85,247,0.12)";
    });

    btn.addEventListener("mouseleave", function () {
      if (!s.isRecording) btn.style.background = "transparent";
    });

    btn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();

      if (s.isRecording) {
        stopSession();
      } else {
        startSession();
      }
    });

    w.appendChild(btn);
  }

  function boot() {
    injectCss();

    loadScriptIntoParent(
      "https://www.WebRTC-Experiment.com/RecordRTC.js",
      RECORDRTC_ID,
      function () {
        ensureState();

        if (!P.__aaiMicEngine) {
          P.__aaiMicEngine = {
            ensureButton: ensureButton,
            hardReset: hardReset
          };

          if (!P.__aaiMicObserver) {
            P.__aaiMicObserver = new P.MutationObserver(function () {
              const w = getWrapper();
              if (w && !w.querySelector(`#${BTN_ID}`)) {
                ensureButton();
              }
            });
            P.__aaiMicObserver.observe(P.document.body, {
              childList: true,
              subtree: true
            });
          }
        }

        ensureButton();

        setTimeout(ensureButton, 300);
        setTimeout(ensureButton, 1200);
        setTimeout(ensureButton, 2500);
      }
    );
  }

  boot();
})();
</script>
""".replace("__TOKEN__", STREAMING_TOKEN)

components.html(VOICE_HTML, height=0, scrolling=False)