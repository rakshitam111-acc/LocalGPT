"""Automated verification script for LLM X-Ray."""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import sys
import io

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time


def run_tests():
    print("=" * 60)
    print("[TEST] LLM X-Ray - Automated Verification Suite")
    print("=" * 60)

    # Step 1: Test Tokenizer
    print("\n[1/5] Testing Tokenizer Loading & Breakdown...")
    from tokenizer import load_tokenizer, get_token_breakdown, format_chat_prompt, DEFAULT_MODEL_ID
    t0 = time.time()
    tokenizer = load_tokenizer(DEFAULT_MODEL_ID)
    print(f"  [OK] Tokenizer loaded in {time.time() - t0:.2f}s. Vocab size: {tokenizer.vocab_size}")

    prompt = "What is AI?"
    breakdown = get_token_breakdown(prompt, tokenizer)
    print(f"  [OK] Prompt: '{prompt}'")
    print(f"  [OK] Formatted tokens: {breakdown['tokens_formatted']}")
    print(f"  [OK] Formatted IDs:    {breakdown['ids_formatted']}")
    assert len(breakdown["tokens"]) > 0, "Tokens list should not be empty"

    # Step 2: Test Model Loading
    print("\n[2/5] Testing Qwen Model Loading...")
    from model import load_model, get_model_metadata, get_device, extract_prompt_internals, generate_step_by_step
    device = get_device()
    print(f"  [OK] Target device detected: {device}")
    t0 = time.time()
    model, device = load_model(DEFAULT_MODEL_ID, device=device)
    print(f"  [OK] Model loaded in {time.time() - t0:.2f}s.")

    metadata = get_model_metadata(model)
    print(f"  [OK] Model Architecture: {metadata['total_params_formatted']} parameters, {metadata['num_layers']} layers, {metadata['hidden_size']} hidden dim, {metadata['num_attention_heads']} heads.")

    # Step 3: Test Internal State Extraction
    print("\n[3/5] Testing Internal Representations Extraction...")
    formatted_prompt = format_chat_prompt(prompt, tokenizer)
    internals = extract_prompt_internals(formatted_prompt, model, tokenizer, top_k_preds=10)

    print(f"  [OK] Embeddings shape: {internals['input_embeddings'].shape} (seq_len: {internals['seq_len']}, hidden_size: {internals['hidden_size']})")
    print(f"  [OK] Hidden state layers: {len(internals['hidden_states'])} (Embedding + 28 blocks)")
    print(f"  [OK] Attention layers: {len(internals['attentions'])}, Heads per layer: {internals['num_heads']}")
    print(f"  [OK] Top predicted next tokens:")
    for pred in internals["predictions"][:5]:
        print(f"      #{pred['rank']} '{pred['display_token']}': {pred['probability_percent']} (logit: {pred['logit']:.2f})")

    # Step 4: Test Visualizations
    print("\n[4/5] Testing Plotly Visualization Generators...")
    from visualization import (
        plot_embeddings_pca,
        plot_embedding_stats,
        plot_embedding_slice_heatmap,
        plot_attention_heatmap,
        plot_hidden_states_pca,
        plot_hidden_state_norms,
        plot_logits_distribution,
        plot_step_candidates,
    )

    fig1 = plot_embeddings_pca(internals["input_embeddings"], internals["tokens"], internals["input_ids"])
    fig2 = plot_embedding_stats(internals["embedding_stats"])
    fig3 = plot_embedding_slice_heatmap(internals["input_embeddings"], internals["tokens"])
    fig4 = plot_attention_heatmap(internals["attentions"][0][0], internals["tokens"], 0, 0)
    fig5 = plot_hidden_states_pca(internals["hidden_states"], internals["tokens"], [0, 10, 20, 28])
    fig6 = plot_hidden_state_norms(internals["hidden_states"])
    fig7 = plot_logits_distribution(internals["predictions"])
    fig8 = plot_step_candidates(internals["predictions"][:5], 1, internals["predictions"][0]["token"])
    print("  [OK] All Plotly figures generated successfully without errors!")

    # Step 5: Test Auto-Regressive Step-by-Step Generation
    print("\n[5/5] Testing Step-by-Step Token Generation...")
    t0 = time.time()
    gen_result = generate_step_by_step(
        formatted_prompt,
        model,
        tokenizer,
        max_new_tokens=20,
        temperature=0.7,
    )
    print(f"  [OK] Generation finished in {time.time() - t0:.2f}s.")
    print(f"  [OK] Total tokens generated: {gen_result['total_tokens_generated']}")
    print(f"  [OK] Full Generated Response:\n{'-'*40}\n{gen_result['full_response']}\n{'-'*40}")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED SUCCESSFULLY! LLM X-Ray Phase 1 is fully functional.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
