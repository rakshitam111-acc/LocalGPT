"""Modernized X-Ray inspection module and UI bridge for LocalGPT."""

import streamlit as st
import pandas as pd
from typing import Any, Dict
from tokenizer import get_token_breakdown
from model import extract_prompt_internals, get_model_metadata
from visualization import (
    plot_embeddings_pca,
    plot_embedding_stats,
    plot_embedding_slice_heatmap,
    plot_attention_heatmap,
    plot_hidden_states_pca,
    plot_hidden_state_norms,
    plot_logits_distribution,
)


def render_xray_panel(prompt_text: str, model, tokenizer):
    """Render the full interactive X-Ray inspection panel for a conversation turn."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%); border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; border-left: 4px solid #6366f1;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                🔬 <span>LLM X-Ray — Deep Inference Inspection Layer</span>
            </div>
            <div style="font-size: 0.88rem; color: #64748b; margin-top: 2px;">
                Inspecting real-time tensor activations, token embeddings, multi-head attention matrices, and logits.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Extracting model representations..."):
        internals = extract_prompt_internals(prompt_text, model, tokenizer, top_k_preds=12)
        metadata = get_model_metadata(model)

    # Top Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Parameters", metadata["total_params_formatted"])
    m2.metric("Layers", metadata["num_layers"])
    m3.metric("Hidden Size", metadata["hidden_size"])
    m4.metric("Attention Heads", metadata["num_attention_heads"])
    m5.metric("Seq Length", f"{internals['seq_len']} tokens")

    tab_tok, tab_emb, tab_trans, tab_att, tab_hid, tab_log = st.tabs([
        "🔤 Tokens & IDs",
        "🌐 Embeddings PCA",
        "🏗️ Transformer Blocks",
        "🔥 Attention Heatmap",
        "🧠 Hidden States",
        "📊 LM Head Logits",
    ])

    # 1. Tokens Tab
    with tab_tok:
        st.markdown("#### Input Token Breakdown & Vocabulary IDs")
        breakdown = get_token_breakdown(prompt_text, tokenizer)

        badges_html = "".join([
            f'<span style="display:inline-block;padding:4px 8px;margin:3px;background:#eef2ff;color:#3730a3;border:1px solid #c7d2fe;border-radius:6px;font-family:monospace;font-size:0.88rem;font-weight:500;">{d["display_token"]} <span style="color:#6366f1;font-size:0.75rem;">({d["token_id"]})</span></span>'
            for d in breakdown["details"]
        ])
        st.markdown(badges_html, unsafe_allow_html=True)

        st.markdown(f"**Formatted Tokens:** `{breakdown['tokens_formatted']}`")
        st.markdown(f"**Token IDs:** `{breakdown['ids_formatted']}`")

        df_t = pd.DataFrame(breakdown["details"])
        if not df_t.empty:
            st.dataframe(df_t[["index", "display_token", "token_id", "length"]], use_container_width=True)

    # 2. Embeddings Tab
    with tab_emb:
        st.markdown("#### Input Embeddings (1536-Dimensional Representation)")
        st.plotly_chart(
            plot_embeddings_pca(internals["input_embeddings"], internals["tokens"], internals["input_ids"]),
            use_container_width=True,
            key="xray_emb_pca",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_embedding_stats(internals["embedding_stats"]), use_container_width=True, key="xray_emb_stats")
        with c2:
            st.plotly_chart(plot_embedding_slice_heatmap(internals["input_embeddings"], internals["tokens"]), use_container_width=True, key="xray_emb_slice")

    # 3. Layers Tab
    with tab_trans:
        st.markdown(f"#### Transformer Structure ({metadata['num_layers']} Decoder Blocks)")
        c_l1, c_l2 = st.columns([1, 2])
        with c_l1:
            st.markdown(
                f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;font-size:0.9rem;">
                    <b>Block Type:</b> <code>Qwen2DecoderLayer</code><br>
                    <b>Attention Heads:</b> <code>{metadata['num_attention_heads']}</code><br>
                    <b>KV Heads:</b> <code>{metadata['num_key_value_heads']}</code><br>
                    <b>Hidden Dim:</b> <code>{metadata['hidden_size']}</code><br>
                    <b>MLP Intermediate:</b> <code>{metadata['intermediate_size']}</code><br>
                    <b>Activation:</b> <code>SiLU / SwiGLU</code><br>
                    <b>Normalization:</b> <code>RMSNorm</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_l2:
            st.plotly_chart(plot_hidden_state_norms(internals["hidden_states"]), use_container_width=True, key="xray_norm_flow")

    # 4. Attention Tab
    with tab_att:
        st.markdown("#### Multi-Head Self-Attention Matrix Inspection")
        ac1, ac2 = st.columns(2)
        with ac1:
            chosen_layer = st.slider("Transformer Layer:", 0, internals["num_attention_layers"] - 1, 11, key="xray_att_layer")
        with ac2:
            chosen_head = st.slider("Attention Head:", 0, internals["num_heads"] - 1, 4, key="xray_att_head")

        if internals["attentions"]:
            att_mat = internals["attentions"][chosen_layer][chosen_head]
            st.plotly_chart(
                plot_attention_heatmap(att_mat, internals["tokens"], chosen_layer, chosen_head),
                use_container_width=True,
                key="xray_att_heatmap",
            )

    # 5. Hidden States Tab
    with tab_hid:
        st.markdown("#### Hidden State Trajectory & Representation Drift Across Layers")
        hidden_layer_options = list(range(len(internals["hidden_states"])))
        default_layers = [0, 10, 20, 28] if len(internals["hidden_states"]) > 28 else [0, len(internals["hidden_states"]) - 1]
        selected_hl = st.multiselect(
            "Select Layers to Project in Shared PCA Space:",
            options=hidden_layer_options,
            default=default_layers,
            format_func=lambda x: "Embedding Output (L0)" if x == 0 else f"Layer {x}",
            key="xray_hl_select",
        )
        st.plotly_chart(
            plot_hidden_states_pca(internals["hidden_states"], internals["tokens"], selected_hl),
            use_container_width=True,
            key="xray_hid_pca",
        )

    # 6. Logits Tab
    with tab_log:
        st.markdown("#### LM Head Next-Token Prediction & Softmax Probabilities")
        st.plotly_chart(plot_logits_distribution(internals["predictions"]), use_container_width=True, key="xray_logits_chart")
        df_p = pd.DataFrame(internals["predictions"])
        st.dataframe(df_p[["rank", "display_token", "probability_percent", "logit", "token_id"]], use_container_width=True)
