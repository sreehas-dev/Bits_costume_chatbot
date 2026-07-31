import streamlit as st
import faiss
import json
import os
import numpy as np
import time
from sentence_transformers import SentenceTransformer

# ================= CONFIGURATION =================
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(str(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "finetuned_model")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings")
DATA_PATH = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(EMBEDDINGS_PATH, "faiss_index.bin")
FAQ_DATA_PATH = os.path.join(DATA_PATH, "faq_merged_v2.json")
LOGO_PATH = "bits_logo.png"  # Ensure this file is in your folder

THRESHOLD = 0.60
TOP_K = 1

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="BITS Pilani Assistant",
    page_icon="🎓",
    layout="centered"
)


# ================= CUSTOM CSS =================
# This removes the top header padding to make it look more like a native app
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Hide the default streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ================= LOAD RESOURCES =================
@st.cache_resource
def load_resources():
    model = SentenceTransformer(str(MODEL_PATH))
    index = faiss.read_index(INDEX_PATH)
    with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return model, index, data


try:
    model, index, data = load_resources()
except Exception as e:
    st.error("System is initializing... Please wait or check resource paths.")
    st.stop()

# ================= SESSION STATE (CHAT MEMORY) =================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I am BITS Pilani AI Assistant. Ask me anything about courses, registration, or campus life."}
    ]

# ================= SIDEBAR (LOGO & CONTROLS) =================
with st.sidebar:
    # 1. Logo Area
    try:
        st.image(LOGO_PATH, use_container_width=True)
    except:
        st.header("BITS Pilani")

    st.markdown("---")

    # 2. Reset Button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello! I am BITS Pilani AI Assistant. Ask me anything about courses, registration, or campus life."}
        ]
        st.rerun()

    # 3. Admin Debug (Hidden from students)
    with st.expander("🛠️ Admin Tools"):
        st.write("Debug Mode")
        if "last_confidence" in st.session_state:
            st.metric("Last Confidence", f"{st.session_state.last_confidence:.3f}")

# ================= CHAT INTERFACE =================

# 1. Display Chat History
for msg in st.session_state.messages:
    # Use the logo for the assistant avatar if available
    if msg["role"] == "assistant":
        try:
            st.chat_message(msg["role"], avatar=LOGO_PATH).write(msg["content"])
        except:
            st.chat_message(msg["role"]).write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# 2. Chat Input Handler
if prompt := st.chat_input("Type your question here..."):

    # A. Display User Message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # B. Process Response
    with st.chat_message("assistant", avatar=LOGO_PATH if "logo" not in str(LOGO_PATH) else None):
        # Create a placeholder to stream the thinking or text
        response_placeholder = st.empty()

        # Search Logic
        embedding = model.encode(prompt, normalize_embeddings=True).astype("float32")
        scores, indices = index.search(np.array([embedding]), TOP_K)

        top_score = float(scores[0][0])
        top_idx = int(indices[0][0])

        # Save confidence for debug sidebar
        st.session_state.last_confidence = top_score

        # Determine Answer
        if top_score < THRESHOLD:
            full_response = "I'm sorry, I couldn't find specific information regarding that in the student handbook. Please check with the Academic Division directly."
        else:
            full_response = data[top_idx]['answer']

        # C. Simulate "Typing" effect (makes it feel more natural)
        displayed_response = ""
        # Split by words for a smoother typing effect
        for word in full_response.split():
            displayed_response += word + " "
            response_placeholder.markdown(displayed_response + "▌")
            time.sleep(0.02)  # Adjust speed of typing here

        response_placeholder.markdown(displayed_response)

    # D. Save Assistant Response to History
    st.session_state.messages.append({"role": "assistant", "content": full_response})