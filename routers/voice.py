import io
import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
import speech_recognition as sr

router = APIRouter(prefix="/api/voice", tags=["voice"])

recognizer = sr.Recognizer()

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes audio recorded in browser (WAV, WEBM, OGG).
    Works in all browsers including Firefox and mobile.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes or len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Empty or invalid audio data.")

        text = ""
        # Direct WAV reading
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
        except Exception as wav_err:
            # Conversion fallback if audio format is WebM / OGG
            try:
                from pydub import AudioSegment
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
                    tmp_in.write(audio_bytes)
                    tmp_in_path = tmp_in.name

                tmp_wav_path = tmp_in_path + ".wav"
                segment = AudioSegment.from_file(tmp_in_path)
                segment = segment.set_frame_rate(16000).set_channels(1)
                segment.export(tmp_wav_path, format="wav")

                with sr.AudioFile(tmp_wav_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)

                for p in [tmp_in_path, tmp_wav_path]:
                    if os.path.exists(p):
                        os.remove(p)
            except Exception as conv_err:
                print(f"[Audio Transcribe Error]: Direct WAV error ({wav_err}), Conversion error ({conv_err})")
                raise HTTPException(status_code=400, detail="Could not transcribe audio. Please ensure microphone audio is clear.")

        if not text:
            raise HTTPException(status_code=400, detail="No speech detected in audio.")

        return {"text": text.strip()}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Voice API Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Voice processing error: {str(e)}")

