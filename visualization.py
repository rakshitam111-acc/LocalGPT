"""Modernized Visualization module for LocalGPT & LLM X-Ray.

Generates high-aesthetic Plotly charts for Tokenization, Embeddings PCA,
Transformer Architecture, Attention Heatmaps, Hidden State Trajectories,
Logits/Probabilities distributions, and Generation Steppers.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Consistent Modern Theme
MODERN_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
PRIMARY_COLOR = "#6366f1"
SECONDARY_COLOR = "#10b981"
ACCENT_COLOR = "#f59e0b"
DARK_TEXT = "#1e293b"
LIGHT_BG = "#ffffff"


def plot_embeddings_pca(
    embeddings: np.ndarray,
    tokens: List[str],
    token_ids: List[int],
) -> go.Figure:
    """High-aesthetic 2D PCA scatter plot of token embeddings."""
    seq_len, hidden_dim = embeddings.shape

    if seq_len < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 tokens to compute 2D PCA projection.",
            showarrow=False,
            font=dict(size=14, family=MODERN_FONT, color="#64748b"),
        )
        fig.update_layout(template="plotly_white", height=380)
        return fig

    n_components = min(2, seq_len, hidden_dim)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)

    if n_components == 1:
        coords = np.column_stack([coords, np.zeros(seq_len)])

    norms = [float(np.linalg.norm(vec)) for vec in embeddings]
    display_tokens = [repr(t)[1:-1] for t in tokens]

    df = pd.DataFrame({
        "Token": display_tokens,
        "Token_ID": token_ids,
        "Position": list(range(seq_len)),
        "PCA_1": coords[:, 0],
        "PCA_2": coords[:, 1],
        "L2_Norm": norms,
    })

    var_explained = pca.explained_variance_ratio_
    var_text = f"PC1: {var_explained[0]*100:.1f}%" + (
        f", PC2: {var_explained[1]*100:.1f}%" if len(var_explained) > 1 else ""
    )

    fig = go.Figure()

    # Trajectory connecting line
    fig.add_trace(
        go.Scatter(
            x=df["PCA_1"],
            y=df["PCA_2"],
            mode="lines",
            line=dict(color="rgba(99, 102, 241, 0.35)", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Token Scatter points
    fig.add_trace(
        go.Scatter(
            x=df["PCA_1"],
            y=df["PCA_2"],
            mode="markers+text",
            text=df["Token"],
            textposition="top center",
            textfont=dict(size=12, family="monospace", color="#0f172a"),
            marker=dict(
                size=14,
                color=df["Position"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text="Token Index", font=dict(family=MODERN_FONT, size=11)), thickness=14),
                line=dict(width=1.5, color="#ffffff"),
            ),
            customdata=np.stack((df["Token_ID"], df["L2_Norm"], df["Position"]), axis=-1),
            hovertemplate=(
                "<b>Token:</b> %{text}<br>"
                "<b>Position:</b> %{customdata[2]}<br>"
                "<b>Token ID:</b> %{customdata[0]}<br>"
                "<b>L2 Norm:</b> %{customdata[1]:.3f}<br>"
                "<b>PCA:</b> (%{x:.2f}, %{y:.2f})<extra></extra>"
            ),
            name="Tokens",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"2D PCA Projection of Input Token Embeddings <span style='font-size:12px;color:#64748b;'>({var_text})</span>",
            font=dict(family=MODERN_FONT, size=15, color=DARK_TEXT),
        ),
        xaxis_title=dict(text=f"Principal Component 1 ({var_explained[0]*100:.1f}%)", font=dict(family=MODERN_FONT, size=12)),
        yaxis_title=dict(text=f"Principal Component 2 ({var_explained[1]*100:.1f}%)" if len(var_explained) > 1 else "PC2", font=dict(family=MODERN_FONT, size=12)),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        height=480,
    )
    return fig


def plot_embedding_stats(embedding_stats: List[Dict[str, Any]]) -> go.Figure:
    """Grouped bar chart for token embedding metrics."""
    df = pd.DataFrame(embedding_stats)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["display_token"],
            y=df["l2_norm"],
            name="L2 Norm",
            marker_color="#6366f1",
            hovertemplate="<b>%{x}</b><br>L2 Norm: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["display_token"],
            y=df["std"],
            name="Std Dev",
            marker_color="#10b981",
            hovertemplate="<b>%{x}</b><br>Std: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["display_token"],
            y=df["mean"],
            name="Mean",
            marker_color="#f59e0b",
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Embedding Vector Statistics (L2 Norm, Std Dev, Mean)",
            font=dict(family=MODERN_FONT, size=14, color=DARK_TEXT),
        ),
        xaxis_title=dict(text="Tokens", font=dict(family=MODERN_FONT, size=12)),
        yaxis_title=dict(text="Magnitude", font=dict(family=MODERN_FONT, size=12)),
        barmode="group",
        template="plotly_white",
        margin=dict(l=30, r=30, t=45, b=30),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family=MODERN_FONT, size=11)),
    )
    return fig


def plot_embedding_slice_heatmap(
    embeddings: np.ndarray,
    tokens: List[str],
    num_dims: int = 40,
) -> go.Figure:
    """Heatmap showing the first N embedding dimensions per token."""
    seq_len, total_dims = embeddings.shape
    slice_dims = min(num_dims, total_dims)
    matrix = embeddings[:, :slice_dims]
    display_tokens = [repr(t)[1:-1] for t in tokens]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"D{i}" for i in range(slice_dims)],
            y=display_tokens,
            colorscale="RdBu_r",
            colorbar=dict(title=dict(text="Weight", font=dict(family=MODERN_FONT, size=10)), thickness=12),
            hovertemplate="Token: %{y}<br>Dim: %{x}<br>Val: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Embedding Dimensions Heatmap (First {slice_dims} / {total_dims} Dims)",
            font=dict(family=MODERN_FONT, size=14, color=DARK_TEXT),
        ),
        xaxis_title=dict(text="Dimensions", font=dict(family=MODERN_FONT, size=11)),
        yaxis_title=dict(text="Tokens", font=dict(family=MODERN_FONT, size=11)),
        template="plotly_white",
        margin=dict(l=30, r=30, t=45, b=30),
        height=max(320, seq_len * 24),
    )
    return fig


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: List[str],
    layer_idx: int,
    head_idx: int,
) -> go.Figure:
    """Interactive heatmap of attention scores between tokens."""
    display_tokens = [f"[{i}] {repr(t)[1:-1]}" for i, t in enumerate(tokens)]

    fig = go.Figure(
        data=go.Heatmap(
            z=attention_matrix,
            x=display_tokens,
            y=display_tokens,
            colorscale="Plasma",
            zmin=0.0,
            zmax=float(np.max(attention_matrix)) if np.max(attention_matrix) > 0 else 1.0,
            colorbar=dict(title=dict(text="Weight", font=dict(family=MODERN_FONT, size=11)), thickness=14),
            hovertemplate=(
                "<b>Query:</b> %{y}<br>"
                "<b>Key:</b> %{x}<br>"
                "<b>Attention Weight:</b> %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Self-Attention Matrix — Layer {layer_idx + 1} (Index {layer_idx}), Head {head_idx}",
            font=dict(family=MODERN_FONT, size=15, color=DARK_TEXT),
        ),
        xaxis_title=dict(text="Key Tokens (Attended To)", font=dict(family=MODERN_FONT, size=12)),
        yaxis_title=dict(text="Query Tokens (Attending From)", font=dict(family=MODERN_FONT, size=12)),
        yaxis_autorange="reversed",
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=500,
    )
    return fig


def plot_hidden_states_pca(
    hidden_states: List[np.ndarray],
    tokens: List[str],
    selected_layers: List[int],
) -> go.Figure:
    """PCA trajectory of token representations across transformer layers."""
    display_tokens = [repr(t)[1:-1] for t in tokens]
    seq_len = len(tokens)

    all_vectors = []
    layer_labels = []
    token_labels = []
    token_indices = []

    for l_idx in selected_layers:
        if 0 <= l_idx < len(hidden_states):
            layer_name = "Embedding" if l_idx == 0 else f"Layer {l_idx}"
            layer_mat = hidden_states[l_idx]
            for t_idx in range(seq_len):
                all_vectors.append(layer_mat[t_idx])
                layer_labels.append(layer_name)
                token_labels.append(display_tokens[t_idx])
                token_indices.append(t_idx)

    if not all_vectors:
        fig = go.Figure()
        fig.add_annotation(text="No layers selected.", showarrow=False)
        return fig

    all_vectors_mat = np.array(all_vectors)
    if all_vectors_mat.shape[0] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient points for PCA projection.", showarrow=False)
        return fig

    n_comp = min(2, all_vectors_mat.shape[0], all_vectors_mat.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(all_vectors_mat)
    if n_comp == 1:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])

    df = pd.DataFrame({
        "Token": token_labels,
        "Layer": layer_labels,
        "Token_Idx": token_indices,
        "PCA_1": coords[:, 0],
        "PCA_2": coords[:, 1],
    })

    fig = go.Figure()

    # Trajectory lines connecting same token across layers
    for t_idx in range(seq_len):
        token_df = df[df["Token_Idx"] == t_idx]
        fig.add_trace(
            go.Scatter(
                x=token_df["PCA_1"],
                y=token_df["PCA_2"],
                mode="lines",
                line=dict(width=1.5, dash="dash", color="rgba(148, 163, 184, 0.5)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Scatter markers grouped by Layer
    for layer_name in df["Layer"].unique():
        layer_df = df[df["Layer"] == layer_name]
        fig.add_trace(
            go.Scatter(
                x=layer_df["PCA_1"],
                y=layer_df["PCA_2"],
                mode="markers+text",
                name=layer_name,
                text=layer_df["Token"],
                textposition="top center",
                textfont=dict(size=11, family="monospace"),
                marker=dict(size=11, line=dict(width=1, color="#ffffff")),
                hovertemplate=(
                    "<b>Token:</b> %{text}<br>"
                    "<b>Layer:</b> " + layer_name + "<br>"
                    "<b>PCA:</b> (%{x:.2f}, %{y:.2f})<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Hidden State Representation Trajectory Across Transformer Layers",
            font=dict(family=MODERN_FONT, size=15, color=DARK_TEXT),
        ),
        xaxis_title=dict(text="Principal Component 1", font=dict(family=MODERN_FONT, size=12)),
        yaxis_title=dict(text="Principal Component 2", font=dict(family=MODERN_FONT, size=12)),
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family=MODERN_FONT, size=11)),
    )
    return fig


def plot_hidden_state_norms(hidden_states: List[np.ndarray]) -> go.Figure:
    """Progression of hidden state L2 norms across all 28 layers."""
    layer_names = ["Embed"] + [f"L{i}" for i in range(1, len(hidden_states))]
    mean_norms = [float(np.mean(np.linalg.norm(h, axis=-1))) for h in hidden_states]
    max_norms = [float(np.max(np.linalg.norm(h, axis=-1))) for h in hidden_states]
    min_norms = [float(np.min(np.linalg.norm(h, axis=-1))) for h in hidden_states]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=layer_names,
            y=mean_norms,
            mode="lines+markers",
            name="Mean Norm",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(size=6, color="#6366f1"),
            hovertemplate="<b>%{x}</b><br>Mean: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layer_names,
            y=max_norms,
            mode="lines",
            name="Max Norm",
            line=dict(color="#ef4444", width=1.5, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Max: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=layer_names,
            y=min_norms,
            mode="lines",
            name="Min Norm",
            line=dict(color="#10b981", width=1.5, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Min: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Hidden State Activation Norms (Layer 0 → 28)",
            font=dict(family=MODERN_FONT, size=14, color=DARK_TEXT),
        ),
        xaxis_title=dict(text="Transformer Layer", font=dict(family=MODERN_FONT, size=11)),
        yaxis_title=dict(text="L2 Norm Magnitude", font=dict(family=MODERN_FONT, size=11)),
        template="plotly_white",
        margin=dict(l=30, r=30, t=45, b=30),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family=MODERN_FONT, size=10)),
    )
    return fig


def plot_logits_distribution(predictions: List[Dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart for top candidate tokens."""
    df = pd.DataFrame(predictions)
    df = df.iloc[::-1].reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=[f"#{d['rank']} '{d['display_token']}'" for _, d in df.iterrows()],
            x=df["probability"] * 100,
            orientation="h",
            marker=dict(
                color=df["probability"],
                colorscale="Purp",
                showscale=False,
            ),
            text=[f"{p:.1f}%" for p in df["probability"] * 100],
            textposition="outside",
            hovertemplate="<b>Token:</b> %{y}<br><b>Probability:</b> %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Next-Token Probability Distribution (LM Head Output)",
            font=dict(family=MODERN_FONT, size=15, color=DARK_TEXT),
        ),
        xaxis_title=dict(text="Probability (%)", font=dict(family=MODERN_FONT, size=12)),
        yaxis_title=dict(text="Candidate Token", font=dict(family=MODERN_FONT, size=12)),
        xaxis=dict(range=[0, min(105, max(df["probability"] * 100) * 1.25 + 10)]),
        template="plotly_white",
        margin=dict(l=30, r=30, t=50, b=30),
        height=420,
    )
    return fig
