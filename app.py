"""LocalGPT - Modern ChatGPT-Style Local AI Assistant with RAG & LLM X-Ray.

Phase 2 UI/UX upgrade with modern styling, responsive cards,
interactive starter prompts, citation badges, and on-demand X-Ray inspection.
"""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import os
import streamlit as st
from tokenizer import load_tokenizer, DEFAULT_MODEL_ID
from model import load_model, get_model_metadata, get_device
from database import (
    init_db,
    create_conversation,
    get_conversations,
    get_conversation,
    rename_conversation,
    delete_conversation,
    update_system_prompt,
    get_messages,
    add_message,
    delete_messages_from,
    DEFAULT_SYSTEM_PROMPT,
)
from rag import RAGPipeline
from chat import ChatCoordinator
from xray import render_xray_panel

# ------------------------------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="LocalGPT — Local AI with RAG & X-Ray",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Top Status Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .dot-pulse {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.4);
    }

    /* Welcome Hero Card */
    .hero-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e293b, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #64748b;
        max-width: 620px;
        margin: 0 auto 16px auto;
        line-height: 1.5;
    }

    /* Starter Prompt Cards */
    .prompt-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        height: 100%;
        text-align: left;
    }
    .prompt-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.12);
        transform: translateY(-2px);
    }
    .prompt-card-icon {
        font-size: 1.3rem;
        margin-bottom: 6px;
    }
    .prompt-card-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .prompt-card-desc {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Citation Source Box */
    .citation-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3.5px solid #6366f1;
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 8px;
        font-size: 0.86rem;
    }

    /* Sidebar Tweaks */
    .sidebar-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-top: 14px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Qwen/Qwen2.5-1.5B-Instruct Tokenizer...")
def get_cached_tokenizer():
    return load_tokenizer()


@st.cache_resource(show_spinner="Loading Qwen/Qwen2.5-1.5B-Instruct Model (Local)...")
def get_cached_model():
    return load_model()


@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()


# Initialize Database & State
init_db()

tokenizer = get_cached_tokenizer()
model, device = get_cached_model()
metadata = get_model_metadata(model)
rag_pipeline = get_rag_pipeline()
chat_coordinator = ChatCoordinator(model=model, tokenizer=tokenizer, rag_pipeline=rag_pipeline)

# Ensure active conversation ID
if "active_conv_id" not in st.session_state:
    convs = get_conversations()
    if convs:
        st.session_state["active_conv_id"] = convs[0]["id"]
    else:
        st.session_state["active_conv_id"] = create_conversation(title="New Chat")

if "inspect_xray_msg_id" not in st.session_state:
    st.session_state["inspect_xray_msg_id"] = None

current_conv = get_conversation(st.session_state["active_conv_id"])
if current_conv is None:
    st.session_state["active_conv_id"] = create_conversation(title="New Chat")
    current_conv = get_conversation(st.session_state["active_conv_id"])


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="font-size:1.8rem;">🤖</div>
            <div>
                <div style="font-size:1.25rem;font-weight:700;color:#0f172a;line-height:1.2;">LocalGPT</div>
                <div style="font-size:0.75rem;color:#64748b;font-weight:500;">Private Local AI &bull; RAG &bull; X-Ray</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("➕  New Chat", type="primary", use_container_width=True):
        new_id = create_conversation(title="New Chat")
        st.session_state["active_conv_id"] = new_id
        st.session_state["inspect_xray_msg_id"] = None
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Recent Chats</div>', unsafe_allow_html=True)
    conversations = get_conversations()

    for conv in conversations:
        conv_id = conv["id"]
        is_active = conv_id == st.session_state["active_conv_id"]

        col_title, col_action = st.columns([4.2, 1])
        with col_title:
            btn_label = f"💬 {conv['title']}" if not is_active else f"👉 {conv['title']}"
            if st.button(
                btn_label,
                key=f"conv_btn_{conv_id}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state["active_conv_id"] = conv_id
                st.session_state["inspect_xray_msg_id"] = None
                st.rerun()

        with col_action:
            with st.popover("⚙️", help="Chat actions"):
                new_title = st.text_input("Rename Chat:", value=conv["title"], key=f"rename_input_{conv_id}")
                if st.button("Save", key=f"save_rename_{conv_id}", use_container_width=True):
                    if new_title.strip():
                        rename_conversation(conv_id, new_title.strip())
                        st.rerun()
                if st.button("🗑️ Delete", key=f"del_{conv_id}", type="primary", use_container_width=True):
                    delete_conversation(conv_id)
                    remaining = get_conversations()
                    st.session_state["active_conv_id"] = remaining[0]["id"] if remaining else create_conversation()
                    st.rerun()

    st.markdown('<div class="sidebar-section-title">Knowledge Base (RAG)</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT documents:",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Documents are chunked and indexed into a local FAISS vector store for semantic retrieval.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        with st.spinner("Indexing documents into FAISS..."):
            indexed_count = 0
            for up_file in uploaded_files:
                doc_name, chunks = rag_pipeline.ingest_uploaded_file(up_file)
                indexed_count += chunks
            if indexed_count > 0:
                st.success(f"Indexed {len(uploaded_files)} file(s) ({indexed_count} chunks)!")

    rag_stats = rag_pipeline.get_stats()
    st.caption(f"📚 **Indexed:** {rag_stats['total_documents']} docs ({rag_stats['total_chunks']} chunks in FAISS)")

    if rag_stats["documents"]:
        with st.expander("View Uploaded Documents"):
            for doc in rag_stats["documents"]:
                st.markdown(f"- 📄 `{doc}`")
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                rag_pipeline.clear_index()
                st.success("Knowledge base cleared.")
                st.rerun()

    st.markdown('<div class="sidebar-section-title">Settings & Model</div>', unsafe_allow_html=True)
    with st.expander("⚙️ System Prompt & Hyperparameters"):
        use_rag_toggle = st.checkbox("Enable RAG Context Retrieval", value=True)

        prompt_preset = st.selectbox(
            "Persona Preset:",
            options=["Custom", "Helpful Assistant", "Technical AI Tutor", "Concise Scientist", "Code Specialist"],
        )
        preset_prompts = {
            "Helpful Assistant": "You are LocalGPT, a helpful, intelligent, and concise AI assistant running 100% locally.",
            "Technical AI Tutor": "You are an expert AI tutor. Explain deep technical concepts with intuitive analogies and clear examples.",
            "Concise Scientist": "You are a research scientist. Provide factual, concise, and structured answers with bullet points.",
            "Code Specialist": "You are a senior software architect. Provide clean, secure, and production-ready code with explanations.",
        }

        default_prompt_val = preset_prompts.get(prompt_preset, current_conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        current_system_prompt = st.text_area("System Prompt:", value=default_prompt_val, height=90)
        if current_system_prompt != current_conv.get("system_prompt"):
            update_system_prompt(current_conv["id"], current_system_prompt)

        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, help="0 = greedy deterministic, >0 = creative")
        top_k = st.slider("Top-K", 1, 100, 50, 1)
        top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)
        max_new_tokens = st.slider("Max Tokens", 32, 1024, 512, 32)


# ==============================================================================
# MAIN CHAT AREA
# ==============================================================================

# Top Bar
device_label = "⚡ GPU (CUDA)" if device == "cuda" else "💻 CPU"
st.markdown(
    f"""
    <div class="top-bar">
        <div style="font-size:1.15rem;font-weight:700;color:#0f172a;display:flex;align-items:center;gap:8px;">
            <span>💬</span> <span>{current_conv['title']}</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
            <div class="status-pill">
                <div class="dot-pulse"></div>
                <span>{DEFAULT_MODEL_ID.split('/')[-1]} &bull; {device_label}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch conversation messages
messages = get_messages(st.session_state["active_conv_id"])

# ------------------------------------------------------------------------------
# Empty State / Starter Prompt Cards
# ------------------------------------------------------------------------------
if not messages:
    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size:2.2rem;margin-bottom:4px;">🤖</div>
            <div class="hero-title">Welcome to LocalGPT</div>
            <div class="hero-subtitle">
                A 100% private, local ChatGPT-style assistant powered by <b>Qwen2.5-1.5B</b> with
                semantic Document RAG and deep internal <b>LLM X-Ray</b> inspection.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 💡 Try an Example Prompt:")
    c_p1, c_p2 = st.columns(2)
    c_p3, c_p4 = st.columns(2)

    quick_prompt_clicked = None

    with c_p1:
        if st.button("🧠 Explain Transformer Architecture\nHow Multi-Head Attention works", use_container_width=True):
            quick_prompt_clicked = "Explain the Transformer architecture and how Multi-Head Attention works in simple terms."
    with c_p2:
        if st.button("📄 What is Retrieval-Augmented Generation?\nHow RAG helps local LLMs", use_container_width=True):
            quick_prompt_clicked = "What is Retrieval-Augmented Generation (RAG) and how does it prevent LLM hallucinations?"
    with c_p3:
        if st.button("⚡ Quantum Computing Basics\nExplain Qubits and Superposition", use_container_width=True):
            quick_prompt_clicked = "Explain Quantum Computing in simple terms: What are qubits and superposition?"
    with c_p4:
        if st.button("🔍 Explain LLM Embedding Space\nHow tokens turn into vectors", use_container_width=True):
            quick_prompt_clicked = "How does an LLM embedding layer convert token IDs into high-dimensional vectors?"

    if quick_prompt_clicked:
        st.session_state["trigger_prompt"] = quick_prompt_clicked
        st.rerun()


# ------------------------------------------------------------------------------
# Render Conversation Turns
# ------------------------------------------------------------------------------
for idx, msg in enumerate(messages):
    msg_id = msg["id"]
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        st.markdown(content)

        # RAG Source References
        if role == "assistant" and sources:
            with st.expander(f"📚 {len(sources)} Source Reference(s) from Knowledge Base"):
                for s_idx, s in enumerate(sources):
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <b>Source {s_idx + 1}:</b> 📄 <code>{s['source']}</code> (Page {s['page']}) &bull; <b>Match Similarity:</b> <code>{s['similarity']}</code><br>
                            <span style="color:#475569;font-style:italic;">"{s['snippet']}"</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Assistant Actions Bar
        if role == "assistant":
            col_b1, col_b2, col_b3 = st.columns([1, 1.2, 2.2])
            with col_b1:
                with st.popover("📋 Copy"):
                    st.code(content, language="markdown")
            with col_b2:
                if st.button("🔄 Regenerate", key=f"regen_{msg_id}"):
                    prompt_to_regen = chat_coordinator.regenerate_turn(st.session_state["active_conv_id"], msg_id)
                    st.session_state["trigger_prompt"] = prompt_to_regen
                    st.session_state["inspect_xray_msg_id"] = None
                    st.rerun()
            with col_b3:
                is_xray_active = st.session_state["inspect_xray_msg_id"] == msg_id
                xray_btn_label = "🔬 Close X-Ray" if is_xray_active else "🔬 Inspect LLM X-Ray"
                if st.button(xray_btn_label, key=f"xray_btn_{msg_id}", type="primary" if is_xray_active else "secondary"):
                    if is_xray_active:
                        st.session_state["inspect_xray_msg_id"] = None
                    else:
                        st.session_state["inspect_xray_msg_id"] = msg_id
                    st.rerun()

        # Render X-Ray Inspection Panel if opened for this message
        if role == "assistant" and st.session_state["inspect_xray_msg_id"] == msg_id:
            preceding_user_msg = ""
            for prev_idx in range(idx - 1, -1, -1):
                if messages[prev_idx]["role"] == "user":
                    preceding_user_msg = messages[prev_idx]["content"]
                    break

            with st.container(border=True):
                xray_context = preceding_user_msg if preceding_user_msg else content
                render_xray_panel(xray_context, model, tokenizer)


# ------------------------------------------------------------------------------
# Input Handling & Streaming Generation
# ------------------------------------------------------------------------------
user_query = st.chat_input("Message LocalGPT or ask about your documents...")

if "trigger_prompt" in st.session_state and st.session_state["trigger_prompt"]:
    user_query = st.session_state.pop("trigger_prompt")

if user_query:
    active_id = st.session_state["active_conv_id"]

    # 1. Display User Message Immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    # Automatically title the chat if first message
    if len(messages) == 0 or current_conv["title"] == "New Chat":
        auto_title = user_query[:28] + ("..." if len(user_query) > 28 else "")
        rename_conversation(active_id, auto_title)

    # 2. Stream Assistant Response
    with st.chat_message("assistant", avatar="🤖"):
        token_stream, sources, full_prompt_text = chat_coordinator.process_turn_stream(
            conv_id=active_id,
            user_message=user_query,
            system_prompt=current_conv.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_rag=use_rag_toggle,
        )

        response_text = st.write_stream(token_stream)

        # Show source citations
        if sources:
            with st.expander(f"📚 {len(sources)} Source Reference(s) from Knowledge Base"):
                for s_idx, s in enumerate(sources):
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <b>Source {s_idx + 1}:</b> 📄 <code>{s['source']}</code> (Page {s['page']}) &bull; <b>Match Similarity:</b> <code>{s['similarity']}</code><br>
                            <span style="color:#475569;font-style:italic;">"{s['snippet']}"</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # 3. Save completed turn to SQLite database
    chat_coordinator.save_completed_turn(
        conv_id=active_id,
        user_message=user_query,
        assistant_response=response_text,
        sources=sources,
    )

    st.rerun()
