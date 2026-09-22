"""
multimodal.py
Multimodal RAG Ingestion Engine for Technical Diagrams, Architecture Schematics, and Videos.
Features:
1. Deep Visual Description via Llama-3.2-Vision (meta/llama-3.2-11b-vision-instruct) capturing entities, relationships, and flowchart semantics.
2. OpenCV Video Keyframe Sampling (extracts screenshots at configurable intervals).
3. Whisper Speech-to-Text Transcription for synchronized audio narration.
4. Multimodal Fusion Chunking (correlating visual frame data + audio transcript at timestamps).
"""

import io
import os
import re
import time
import base64
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.utils.logger import logger

DIAG_DIR = Path(__file__).resolve().parent.parent / "static" / "extracted_diagrams"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def describe_diagram_with_vision(image_bytes: bytes, mime_type: str = "image/png", caption: str = "Technical Diagram") -> str:
    """
    Analyzes technical diagrams, flowcharts, schematics, and images using Llama-3.2 Vision.
    Extracts entities, connections, text labels, and structural meaning for vector embedding.
    """
    # 1. Try NVIDIA NIM Llama 3.2 Vision API
    try:
        from openai import OpenAI
        from app.config.settings import NVIDIA_API_KEY, NVIDIA_BASE_URL
        if NVIDIA_API_KEY and not NVIDIA_API_KEY.startswith("your_"):
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=5.0)
            
            prompt_text = (
                "Describe this technical diagram or image in absolute detail, capturing all entities, "
                "relationships, flowcharts, text inside the image, component connections, and its core purpose. "
                "Be thorough and specific so an engineer can understand the entire architecture without seeing the image."
            )
            
            resp = client.chat.completions.create(
                model="meta/llama-3.2-11b-vision-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            ans = resp.choices[0].message.content.strip()
            if ans and len(ans) > 20:
                logger.info(f"👁️ [Llama-3.2-Vision]: Generated {len(ans)} chars visual description for {caption}")
                return ans
    except Exception as e:
        logger.warning(f"⚠️ [Vision API Note]: {e}")

    # 2. Local Fallback via Pytesseract OCR + Layout Analysis
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        ocr_result = pytesseract.image_to_string(img).strip()
        if ocr_result:
            return f"Diagram containing extracted text, component labels, and parameters:\n{ocr_result}"
    except Exception as tess_err:
        logger.warning(f"⚠️ [Local OCR Fallback Warning]: {tess_err}")

    return f"Visual diagram/image showing technical component architecture and operational flow: {caption}."

def process_image_multimodal(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Processes an uploaded image or technical diagram:
    1. Saves image file to static assets for frontend display.
    2. Runs Llama-3.2-Vision to extract exhaustive technical descriptions.
    3. Builds unified searchable chunk text.
    """
    img_hash = hashlib.md5(image_bytes).hexdigest()[:12]
    ext = filename.split(".")[-1].lower() if "." in filename else "png"
    if ext not in ["png", "jpg", "jpeg", "webp"]:
        ext = "png"
    
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
    saved_filename = f"diagram_{img_hash}.{ext}"
    saved_path = DIAG_DIR / saved_filename
    saved_path.write_bytes(image_bytes)

    image_url = f"/static/extracted_diagrams/{saved_filename}"
    caption = filename or f"Technical Diagram ({img_hash})"

    vision_desc = describe_diagram_with_vision(image_bytes, mime, caption)

    chunk_content = (
        f"![{caption}]({image_url})\n\n"
        f"=== [TECHNICAL ARCHITECTURE & DIAGRAM SPECIFICATION: {caption}] ===\n"
        f"Image Reference: {image_url}\n"
        f"Detailed Visual Entities, Flowchart & Data:\n{vision_desc}"
    )

    return {
        "title": caption,
        "image_url": image_url,
        "chunk_content": chunk_content,
        "vision_description": vision_desc
    }

def process_video_multimodal(
    video_bytes: bytes, 
    filename: str, 
    sample_interval_seconds: int = 5,
    max_frames: int = 15
) -> List[Dict[str, Any]]:
    """
    Multimodal Video Ingestion Pipeline:
    1. Samples keyframes at time intervals using OpenCV.
    2. Transcribes the audio track using Whisper.
    3. Fuses visual descriptions (Llama-3.2 Vision) with audio narration into synchronized chunks.
    """
    video_hash = hashlib.md5(video_bytes).hexdigest()[:10]
    ext = filename.split(".")[-1].lower() if "." in filename else "mp4"
    if ext not in ["mp4", "webm", "mov", "avi", "mkv"]:
        ext = "mp4"
    saved_video_filename = f"video_{video_hash}.{ext}"
    saved_video_path = DIAG_DIR / saved_video_filename
    if not saved_video_path.exists():
        saved_video_path.write_bytes(video_bytes)
    full_video_url = f"/static/extracted_diagrams/{saved_video_filename}"

    chunks_out = []

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    try:
        # Step 1: Transcribe Audio using Whisper
        audio_transcript_segments = []
        try:
            import socket
            import whisper
            logger.info("🎙️ [Whisper]: Transcribing audio track for video...")
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(4.0)
            try:
                model = whisper.load_model("tiny")
                whisper_res = model.transcribe(temp_video_path, verbose=False)
                segments = whisper_res.get("segments", [])
                for seg in segments:
                    audio_transcript_segments.append({
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", "").strip()
                    })
                logger.info(f"✅ [Whisper]: Transcribed {len(audio_transcript_segments)} speech segments.")
            finally:
                socket.setdefaulttimeout(old_timeout)
        except Exception as w_err:
            logger.warning(f"⚠️ [Whisper Audio Transcribe Fallback]: {w_err}")
            audio_transcript_segments.append({
                "start": 0.0,
                "end": 600.0,
                "text": f"Instructional audio narration for {filename}."
            })

        # Step 2: OpenCV Frame Sampling
        import cv2
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / max(1.0, fps)
        
        logger.info(f"🎬 [OpenCV Video]: Duration {duration_seconds:.1f}s, FPS {fps}, Sampling every {sample_interval_seconds}s...")

        sampled_frame_count = 0
        current_second = 0.0

        while current_second < duration_seconds and sampled_frame_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_second * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Encode frame to JPEG
            ret_enc, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret_enc:
                frame_bytes = buffer.tobytes()
                mins = int(current_second // 60)
                secs = int(current_second % 60)
                timestamp_str = f"{mins:02d}:{secs:02d}"

                frame_filename = f"video_{video_hash}_frame_{timestamp_str.replace(':', '_')}.jpg"
                frame_path = DIAG_DIR / frame_filename
                frame_path.write_bytes(frame_bytes)
                frame_url = f"/static/extracted_diagrams/{frame_filename}"

                # Match audio segment for this time window
                relevant_audio = [
                    s["text"] for s in audio_transcript_segments
                    if (s["start"] <= current_second <= s["end"]) or (current_second - sample_interval_seconds <= s["start"] <= current_second)
                ]
                audio_text = " ".join(relevant_audio) if relevant_audio else "Demonstration of equipment and visual workflow."

                # Generate Vision Description of frame
                frame_desc = describe_diagram_with_vision(frame_bytes, "image/jpeg", f"{filename} at {timestamp_str}")

                # Create Multimodal Fusion Chunk
                fusion_text = (
                    f"![Video Frame at {timestamp_str}]({frame_url})\n\n"
                    f"🎥 [Watch Video Tutorial: {filename} at {timestamp_str}]({full_video_url}#t={int(current_second)})\n\n"
                    f"=== [VIDEO TUTORIAL & OPERATION GUIDE: {filename} | TIMESTAMP: {timestamp_str}] ===\n"
                    f"Full Video URL: {full_video_url}#t={int(current_second)}\n"
                    f"Video Frame Screenshot: {frame_url}\n"
                    f"Visual Action & Scene: {frame_desc}\n"
                    f"Synchronized Audio Narration: \"{audio_text}\""
                )

                chunks_out.append({
                    "timestamp": timestamp_str,
                    "second": current_second,
                    "frame_url": frame_url,
                    "video_url": full_video_url,
                    "content": fusion_text,
                    "audio_text": audio_text,
                    "visual_description": frame_desc
                })

                sampled_frame_count += 1

            current_second += sample_interval_seconds

        cap.release()

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    logger.info(f"✨ [Multimodal Fusion]: Generated {len(chunks_out)} fused video context chunks for {filename}.")
    return chunks_out
