import asyncio
import base64
import json
import datetime
import pyaudio
import threading
import websockets
import time
import os
from config import Config
from insight_generator import get_insight

# ────────── global state ──────────
display_transcript = []
display_insights   = []
full_transcript    = []
is_recording       = False
recording_thread   = None

def get_state():
    """Return a copy of the transcript and insights for the front-end."""
    return display_transcript.copy(), display_insights.copy()

def start_recording():
    """Kick off background recording + websocket thread."""
    global is_recording, recording_thread, display_transcript, display_insights, full_transcript
    display_transcript = []
    display_insights   = []
    full_transcript    = []
    is_recording       = True

    recording_thread = threading.Thread(target=recording_worker)
    recording_thread.start()

def stop_recording():
    """Signal the thread to stop and add a final insight."""
    global is_recording
    is_recording = False
    if recording_thread:
        recording_thread.join(timeout=5)

    # Final insight for the full meeting
    if full_transcript:
        full_text = " ".join(full_transcript)
        if full_text.strip():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            insight = loop.run_until_complete(get_insight(full_text))
            display_insights.append(f"FINAL INSIGHT: {insight}")
            loop.close()

# ────────── helper: non-blocking insight task ──────────
async def _generate_insight(text: str):
    insight = await get_insight(text)
    display_insights.append(f"INSIGHT: {insight}")

# ────────── background worker ──────────
def recording_worker():
    global display_transcript, display_insights, full_transcript, is_recording

    # Silence ALSA warnings
    os.environ["PYTHONWARNINGS"] = "ignore"

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=Config.RATE,
        input=True,
        frames_per_buffer=Config.CHUNK_SIZE,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_websocket():
        nonlocal stream
        uri = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={Config.RATE}"
        async with websockets.connect(
            uri,
            extra_headers={"Authorization": Config.API_KEY},
            ping_interval=5,
            ping_timeout=20,
        ) as ws:
            print("Connected to AssemblyAI")

            transcript_buffer = []

            while is_recording:
                # ── send audio ──
                data = stream.read(Config.CHUNK_SIZE, exception_on_overflow=False)
                await ws.send(
                    json.dumps({"audio_data": base64.b64encode(data).decode("utf-8")})
                )

                # ── receive text ──
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    msg = json.loads(response)

                    # partial updates
                    if msg.get("message_type") == "PartialTranscript":
                        if display_transcript and display_transcript[-1].startswith("PARTIAL:"):
                            display_transcript.pop()
                        display_transcript.append(f"PARTIAL: {msg.get('text','')}")
                        continue

                    # finalised segment
                    if msg.get("message_type") == "FinalTranscript":
                        final_text = msg.get("text", "").strip()
                        if not final_text:
                            continue

                        if display_transcript and display_transcript[-1].startswith("PARTIAL:"):
                            display_transcript.pop()

                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        display_transcript.append(f"[{ts}] {final_text}")
                        full_transcript.append(final_text)
                        transcript_buffer.append(final_text)

                        # ── trigger insight generation asynchronously ──
                        if len(" ".join(transcript_buffer)) > Config.INSIGHT_THRESHOLD:
                            text_for_insight = " ".join(transcript_buffer)
                            transcript_buffer = []            # clear immediately
                            asyncio.create_task(_generate_insight(text_for_insight))

                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass

                await asyncio.sleep(0.01) 

    try:
        loop.run_until_complete(run_websocket())
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        loop.close()
        print("Recording stopped")
