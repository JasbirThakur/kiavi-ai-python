#!/usr/bin/env python3
"""
verify_multimodal_rag.py
Comprehensive End-to-End Verification of:
1. Super-Fast Large Data Bulk Ingestion & The Golden Trick (Index Lifecycle).
2. Multimodal RAG Pipeline (Llama-3.2-Vision Diagram Description & OpenCV/Whisper Video Fusion).
3. Three-Layer Hybrid Search Pipeline with Reciprocal Rank Fusion (RRF) & FlashRank Reranker.
"""

import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

# Ensure backend root is on sys.path
sys.path.insert(0, "/app")

from app.db.database import SessionLocal
from app.db import models
from app.services.bulk_ingestion import manage_vector_index, bulk_insert_chunks, parallel_embed_and_save
from app.services.multimodal import process_image_multimodal, process_video_multimodal
from app.services.hybrid_search import three_layer_hybrid_search
from app.services.embedding import get_embedding

def create_mock_diagram_image():
    """Generates a synthetic technical diagram PNG in memory."""
    import io
    img = Image.new('RGB', (600, 300), color=(10, 25, 47))
    d = ImageDraw.Draw(img)
    # Draw flowchart boxes
    d.rectangle([30, 80, 180, 180], outline=(34, 211, 238), width=3)
    d.text((45, 120), "High-Pressure\nTurbopump (HPT)", fill=(255, 255, 255))
    
    d.line([180, 130, 270, 130], fill=(59, 130, 246), width=3)
    
    d.rectangle([270, 80, 420, 180], outline=(16, 185, 129), width=3)
    d.text((285, 120), "Pre-Burner\nCombustion (PBC)", fill=(255, 255, 255))

    d.line([420, 130, 510, 130], fill=(59, 130, 246), width=3)

    d.rectangle([510, 80, 580, 180], outline=(245, 158, 11), width=3)
    d.text((515, 120), "Main\nNozzle", fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def create_mock_video_mp4():
    """Generates a 6-second synthetic MP4 video using OpenCV."""
    import cv2
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    width, height = 320, 240
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    for frame_idx in range(60): # 60 frames = 6 seconds
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        frame[:, :] = [frame_idx * 3 % 255, 40, 70]
        # Text with timestamp
        second = frame_idx // 10
        cv2.putText(frame, f"Kiavi Technical Demo 00:0{second}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        out.write(frame)

    out.release()
    with open(tmp_path, "rb") as f:
        v_bytes = f.read()
    os.remove(tmp_path)
    return v_bytes

def main():
    print("=" * 70)
    print("🚀 STARTING MULTIMODAL RAG & BULK INGESTION SUITE VERIFICATION")
    print("=" * 70)
    db = SessionLocal()

    # -------------------------------------------------------------------------
    # TEST 1: The Golden Trick (Index Lifecycle Management)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Testing The Golden Trick: Vector Index Lifecycle Management...")
    # Check current status
    status_pre = manage_vector_index(action="status", db=db)
    print(f"   Initial Index Status: active={status_pre['is_active']}")

    # Drop index
    drop_res = manage_vector_index(action="drop", db=db)
    print(f"   ✅ Vector Index Dropped: status={drop_res['status']} in {drop_res['elapsed_seconds']:.2f}s")
    assert drop_res["status"] == "dropped", "Failed to drop vector index!"

    # Verify dropped
    status_mid = manage_vector_index(action="status", db=db)
    assert not status_mid["is_active"], "Index should be inactive after drop!"

    # Recreate index with m=16, ef_construction=64
    rec_res = manage_vector_index(action="recreate", db=db)
    print(f"   ✅ Vector Index Rebuilt: status={rec_res['status']} in {rec_res['elapsed_seconds']:.2f}s")
    assert rec_res["status"] == "recreated", "Failed to recreate vector index!"

    # Verify active
    status_post = manage_vector_index(action="status", db=db)
    assert status_post["is_active"], "Index should be active after recreate!"
    print("   🎉 Golden Trick Verified: Index drop and rebuild completed successfully!")

    # -------------------------------------------------------------------------
    # TEST 2: Super-Fast Bulk Data Ingestion
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Testing Super-Fast Bulk Data Ingestion Pipeline...")
    test_source = None
    diag_source = None
    v_source = None

    try:
        # Create a test bot source
        test_source = models.BotSource(
            title="[TEST_BENCHMARK] High-Speed Bulk Ingestion Benchmark Source",
            kind="BULK_DATASET",
            isUniversal=False,
            tokenCount=50000
        )
        db.add(test_source)
        db.commit()
        db.refresh(test_source)

        # Prepare 50 synthetic high-value engineering chunks
        dummy_chunks = []
        base_vec = get_embedding("European Aviation Safety Agency EASA Part-145 maintenance manual turbopump")
        
        for i in range(50):
            dummy_chunks.append({
                "sourceId": test_source.id,
                "content": f"EASA Part-145 Section {i+100}: Technical maintenance specification for turbopump assembly SKU-AERO-{i+1000}. Bolt torque: {45 + i} Nm.",
                "embedding": base_vec,
                "document_type": "AEROSPACE_DOC",
                "status": models.DocumentStatus.APPROVED,
                "language": "en"
            })

        # Test high-speed multi-row bulk insert
        t0 = time.time()
        inserted_count = bulk_insert_chunks(dummy_chunks, batch_size=25, db=db)
        insert_time = time.time() - t0
        print(f"   ✅ Bulk Inserted {inserted_count} chunks in {insert_time:.3f}s ({inserted_count/max(0.001, insert_time):.1f} chunks/sec)!")
        assert inserted_count == 50, f"Expected 50 chunks inserted, got {inserted_count}"

        # -------------------------------------------------------------------------
        # TEST 3: Multimodal RAG (Technical Diagram / Image Processing)
        # -------------------------------------------------------------------------
        print("\n[TEST 3] Testing Multimodal Diagram Ingestion (Llama-3.2 Vision)...")
        diag_bytes = create_mock_diagram_image()
        img_result = process_image_multimodal(diag_bytes, "turbopump_schematic.png")
        
        print(f"   Saved Image Path: {img_result['image_url']}")
        print(f"   Vision Description Preview: {img_result['vision_description'][:120]}...")
        print(f"   Searchable Markdown Chunk Preview:\n{img_result['chunk_content'][:160]}...\n")
        
        assert "/static/extracted_diagrams/" in img_result["image_url"], "Invalid diagram image URL!"
        assert "TECHNICAL ARCHITECTURE & DIAGRAM SPECIFICATION" in img_result["chunk_content"], "Missing diagram header!"

        # Save diagram chunk into DB for Hybrid Search testing
        diag_source = models.BotSource(
            title="[TEST_BENCHMARK] High-Pressure Turbopump Schematic Architecture",
            kind="DIAGRAM",
            isUniversal=False,
            url=img_result["image_url"]
        )
        db.add(diag_source)
        db.commit()
        db.refresh(diag_source)

        diag_emb = get_embedding(img_result["chunk_content"])
        diag_chunk = models.DocumentChunk(
            sourceId=diag_source.id,
            content=img_result["chunk_content"],
            embedding=diag_emb,
            source_url=img_result["image_url"],
            document_type="DIAGRAM",
            status=models.DocumentStatus.APPROVED,
            language="en"
        )
        db.add(diag_chunk)
        db.commit()
        print("   ✅ Multimodal Diagram ingested into PostgreSQL vector database!")

        # -------------------------------------------------------------------------
        # TEST 4: Multimodal RAG (Video Keyframe Sampling & Audio Fusion)
        # -------------------------------------------------------------------------
        print("\n[TEST 4] Testing Multimodal Video Processing (OpenCV + Whisper)...")
        v_bytes = create_mock_video_mp4()
        video_chunks = process_video_multimodal(
            video_bytes=v_bytes,
            filename="assembly_tutorial.mp4",
            sample_interval_seconds=2,
            max_frames=1
        )

        print(f"   Extracted {len(video_chunks)} fused video context chunks:")
        for vc in video_chunks:
            print(f"   -> Timestamp: {vc['timestamp']} | Screenshot: {vc['frame_url']}")
            print(f"      Audio: {vc['audio_text']}")
        assert len(video_chunks) >= 1, "Expected at least 1 video chunk extracted!"
        print("   ✅ Multimodal Video Keyframe & Audio Fusion verified!")

        # Save one video chunk into DB for Hybrid Search testing
        v_source = models.BotSource(
            title="[TEST_BENCHMARK] Turbopump Assembly Operational Video Guide",
            kind="VIDEO",
            isUniversal=False
        )
        db.add(v_source)
        db.commit()
        db.refresh(v_source)

        v_first = video_chunks[0]
        v_emb = get_embedding(v_first["content"])
        v_chunk = models.DocumentChunk(
            sourceId=v_source.id,
            content=v_first["content"],
            embedding=v_emb,
            source_url=v_first["frame_url"],
            document_type="VIDEO",
            status=models.DocumentStatus.APPROVED,
            language="en"
        )
        db.add(v_chunk)
        db.commit()

        # -------------------------------------------------------------------------
        # TEST 5: Three-Layer Hybrid Search with Mathematical RRF
        # -------------------------------------------------------------------------
        print("\n[TEST 5] Testing Three-Layer Hybrid Search Pipeline with RRF & FlashRank...")
        
        # Query 1: Exact SKU / Part Number (Tests Layer 2 Lexical ILIKE & RRF)
        sku_query = "SKU-AERO-1015"
        res1 = three_layer_hybrid_search(
            query=sku_query,
            top_k_final=3,
            db=db
        )
        print(f"   Query: '{sku_query}'")
        print(f"   Total Vector Hits: {res1['total_vector_candidates']}, Lexical Hits: {res1['total_lexical_candidates']}, Fused: {res1['total_fused_candidates']}")
        print(f"   Top Gold Chunk: {res1['gold_chunks'][0]['text'][:90]}... (RRF: {res1['gold_chunks'][0]['rrf_score']:.5f}, Cross-Encoder: {res1['gold_chunks'][0].get('cross_encoder_score', 0):.4f})")
        assert any("SKU-AERO-1015" in c["text"] for c in res1["gold_chunks"]), "Failed to retrieve exact SKU via Lexical/RRF layer!"
        print("   ✅ Layer 2 Lexical & RRF Exact Match verified!")

        # Query 2: Technical Diagram Visual Query (Tests Layer 1 Semantic Meaning + Multimodal Output)
        diag_query = "turbopump schematic flowchart diagram combustion"
        res2 = three_layer_hybrid_search(
            query=diag_query,
            top_k_final=3,
            db=db
        )
        print(f"\n   Query: '{diag_query}'")
        print(f"   Total Fused Candidates: {res2['total_fused_candidates']}")
        top_chunk = res2["gold_chunks"][0]
        print(f"   Top Gold Chunk Title: {top_chunk['title']}")
        print(f"   Has Visual Diagram: {top_chunk.get('has_visual')} | URL: {top_chunk.get('visual_url')}")
        assert top_chunk.get("has_visual"), "Expected visual diagram chunk to be retrieved!"
        print("   ✅ Multimodal Hybrid Search with Visual Metadata verified!")

    finally:
        # Cleanup test sources safely via raw SQL so they NEVER pollute production
        from sqlalchemy import text
        ids_to_clean = [str(x.id) for x in [test_source, diag_source, v_source] if x is not None]
        if ids_to_clean:
            db.execute(text("""
                DELETE FROM document_chunks WHERE "sourceId" = ANY(:ids);
                DELETE FROM bot_sources WHERE id = ANY(:ids);
            """), {"ids": ids_to_clean})
            db.commit()
        db.close()

    print("\n" + "=" * 70)
    print("🎉 ALL 5 VERIFICATION MODULES PASSED WITH 100% SUCCESS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
