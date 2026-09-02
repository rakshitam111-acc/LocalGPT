"""Automated end-to-end verification test suite for Project Phase 2 - LocalGPT."""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import os
import sys
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def run_phase2_tests():
    print("=" * 65)
    print("🤖 LocalGPT (Phase 2) - Automated End-to-End Verification Suite")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. Test Database CRUD & Persistence
    # -------------------------------------------------------------
    print("\n[1/5] Testing SQLite Database Persistence & CRUD...")
    from database import (
        init_db,
        create_conversation,
        get_conversations,
        get_conversation,
        rename_conversation,
        add_message,
        get_messages,
        delete_conversation,
        delete_messages_from,
    )
    init_db()
    test_conv_id = create_conversation(title="Test Verification Chat")
    print(f"  [OK] Created conversation ID: {test_conv_id}")

    rename_conversation(test_conv_id, "Renamed Verification Chat")
    conv_info = get_conversation(test_conv_id)
    assert conv_info["title"] == "Renamed Verification Chat", "Title should match renamed value"
    print(f"  [OK] Successfully renamed conversation: '{conv_info['title']}'")

    m1_id = add_message(test_conv_id, role="user", content="Hello LocalGPT!")
    m2_id = add_message(test_conv_id, role="assistant", content="Hello! How can I help you today?", sources=[{"source": "doc1.txt", "page": 1}])
    m3_id = add_message(test_conv_id, role="user", content="Tell me about Transformers.")

    messages = get_messages(test_conv_id)
    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
    print(f"  [OK] Message persistence verified: {len(messages)} turns retrieved from SQLite.")

    delete_messages_from(test_conv_id, m3_id)
    messages_after = get_messages(test_conv_id)
    assert len(messages_after) == 2, f"Expected 2 messages after deletion, got {len(messages_after)}"
    print("  [OK] Message regeneration / rollback verified.")

    # -------------------------------------------------------------
    # 2. Test Document Ingestion & Chunking
    # -------------------------------------------------------------
    print("\n[2/5] Testing Document Loader & Chunking...")
    from document_loader import load_document, chunk_text

    test_doc_path = os.path.join(os.path.dirname(__file__), "data", "documents", "test_guide.txt")
    os.makedirs(os.path.dirname(test_doc_path), exist_ok=True)
    with open(test_doc_path, "w", encoding="utf-8") as f:
        f.write(
            "LocalGPT Architecture Guide.\n\n"
            "LocalGPT is a privacy-first, 100% offline AI assistant running Qwen2.5-1.5B-Instruct.\n"
            "It features FAISS vector database for retrieval-augmented generation (RAG) and sentence-transformers.\n"
            "Internal transformer operations such as attention matrices, hidden states, and logits can be inspected in real-time.\n"
        )

    pages = load_document(test_doc_path)
    chunks = chunk_text(pages, chunk_size=100, chunk_overlap=20)
    print(f"  [OK] Document loaded: {len(pages)} page(s), split into {len(chunks)} chunk(s).")
    assert len(chunks) > 0, "Chunks list should not be empty"

    # -------------------------------------------------------------
    # 3. Test Sentence Transformers & FAISS Vector Search
    # -------------------------------------------------------------
    print("\n[3/5] Testing Sentence Transformers & FAISS RAG Retrieval...")
    from rag import RAGPipeline
    rag = RAGPipeline()
    num_indexed = rag.ingest_file(test_doc_path)
    print(f"  [OK] Ingested document into FAISS vector index ({num_indexed} chunks).")

    test_query = "What vector database does LocalGPT use?"
    context, sources = rag.retrieve_context(test_query, top_k=2)
    print(f"  [OK] Query: '{test_query}'")
    print(f"  [OK] Retrieved {len(sources)} source(s):")
    for s in sources:
        print(f"      - {s['source']} (Page {s['page']}) [Similarity: {s['similarity']}]: '{s['snippet']}'")
    assert len(sources) > 0, "Should retrieve at least 1 relevant source chunk"
    assert "FAISS" in context, "Retrieved context should mention FAISS"

    # -------------------------------------------------------------
    # 4. Test Multi-Turn Streaming Generation with Qwen
    # -------------------------------------------------------------
    print("\n[4/5] Testing Multi-Turn Streaming Generation with Qwen...")
    from tokenizer import load_tokenizer, DEFAULT_MODEL_ID
    from model import load_model, get_device
    from chat import ChatCoordinator

    tokenizer = load_tokenizer(DEFAULT_MODEL_ID)
    model, device = load_model(DEFAULT_MODEL_ID)
    coordinator = ChatCoordinator(model=model, tokenizer=tokenizer, rag_pipeline=rag)

    print("  [OK] Model and Tokenizer loaded successfully.")
    t0 = time.time()
    stream_gen, sources, prompt_text = coordinator.process_turn_stream(
        conv_id=test_conv_id,
        user_message="What vector database does LocalGPT use according to the document?",
        system_prompt="You are LocalGPT, an intelligent assistant.",
        max_new_tokens=40,
        temperature=0.7,
        use_rag=True,
    )

    streamed_text = ""
    for token in stream_gen:
        streamed_text += token

    print(f"  [OK] Streamed response generated in {time.time() - t0:.2f}s:")
    print(f"      Response: '{streamed_text.strip()}'")
    assert len(streamed_text) > 0, "Streamed text should not be empty"

    coordinator.save_completed_turn(
        conv_id=test_conv_id,
        user_message="What vector database does LocalGPT use according to the document?",
        assistant_response=streamed_text,
        sources=sources,
    )

    # -------------------------------------------------------------
    # 5. Test Integrated LLM X-Ray Inspection
    # -------------------------------------------------------------
    print("\n[5/5] Testing Integrated LLM X-Ray Inspection on Chat Turn...")
    from model import extract_prompt_internals
    internals = extract_prompt_internals(prompt_text, model, tokenizer, top_k_preds=5)
    print(f"  [OK] X-Ray Embeddings Shape: {internals['input_embeddings'].shape}")
    print(f"  [OK] X-Ray Hidden States: {len(internals['hidden_states'])} layers")
    print(f"  [OK] X-Ray Attention Matrices: {len(internals['attentions'])} layers x {internals['num_heads']} heads")
    print(f"  [OK] X-Ray Top LM Head Logit Candidate: '{internals['predictions'][0]['display_token']}' ({internals['predictions'][0]['probability_percent']})")

    # Clean up test conversation
    delete_conversation(test_conv_id)
    print("\n" + "=" * 65)
    print("🎉 ALL PHASE 2 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_phase2_tests()
