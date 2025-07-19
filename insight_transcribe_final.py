import asyncio
import base64
import datetime
import json
import os
from typing import Dict, List

import pyaudio
import torch
import websockets
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from pdf_report import generate_pdf_report  # local module

# ───────────────────────── USER CONFIG ─────────────────────────
API_KEY            = "925ecc0fb4d749938155be1c3e75c8d3"
BASE_MODEL_ID      = "Qwen/Qwen2.5-0.5B"
ADAPTER_PATH       = "./llm_feature/qwen_lora_finetuned"

RATE               = 16_000
CHUNK              = 1_024
INSIGHT_THRESHOLD  = 200

MAX_TOKENS   = 256
TEMPERATURE  = 0.3
TOP_P        = 0.7
REPETITION_PENALTY = 1.15

# ─────────────────── LOAD TOKENIZER + MODEL ───────────────────
print("⏳ Loading Qwen-2.5-0.5B base model …")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
    trust_remote_code=True,
)

print("⏳ Applying LoRA adapter from", ADAPTER_PATH)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    device_map={"": "cpu"},
    offload_folder=None
)
model.eval()
print("✅ Local Qwen-LoRA ready\n")

# ───────────────────── MICROPHONE STREAM ──────────────────────
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

# ──────────────────────── UTILITIES ────────────────────────────
insight_queue: asyncio.Queue[str] = asyncio.Queue()
buffer_chunk: List[str] = []
full_transcript: List[Dict[str, str]] = []


def run_llm(text_chunk: str) -> str:
    """Blocking helper, executed in a thread."""
    prompt = f"""
You are a helpful assistant with strong computer science knowledge. From the input text below:

1. Identify and list all *unique* computer science–related terms (e.g., algorithm, quantum computing, etc.).
2. For each term, provide a short one-sentence explanation.
3. Then write a concise overall takeaway or insight.
4. Finally, add a 1–2 sentence summary if appropriate.

Keep the format readable in plain text. No need for JSON or markdown formatting.

Input text:
\"\"\"{text_chunk}\"\"\"
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_cfg = GenerationConfig(
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_cfg)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full[len(prompt):].strip()

# ───────────────────────── TASKS ───────────────────────────────
async def send_audio(ws):
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        encoded = base64.b64encode(data).decode()
        await ws.send(json.dumps({"audio_data": encoded}))
        await asyncio.sleep(0.01)

async def recv_transcripts(ws):
    global buffer_chunk, full_transcript
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("message_type") == "PartialTranscript":
            text = msg.get("text", "")
            print(f"🟡 {text.ljust(80)}", end="\r", flush=True)
        elif msg.get("message_type") == "FinalTranscript":
            text = msg.get("text", "").strip()
            if text:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"\n📝 [{ts}] {text}")
                buffer_chunk.append(text + " ")
                full_transcript.append({"timestamp": ts, "text": text})
                if sum(len(x) for x in buffer_chunk) >= INSIGHT_THRESHOLD:
                    await insight_queue.put("".join(buffer_chunk).strip())
                    buffer_chunk.clear()

async def insight_worker():
    while True:
        chunk = await insight_queue.get()
        if chunk is None:
            break
        print("\n🤖 Generating insight…\n")
        insight = await asyncio.to_thread(run_llm, chunk)
        print("💡 INSIGHT\n" + insight + "\n")
        insight_queue.task_done()

async def flush_and_report():
    if buffer_chunk:
        await insight_queue.put("".join(buffer_chunk).strip())
        buffer_chunk.clear()
    await insight_queue.join()
    await insight_queue.put(None)

    if full_transcript:
        print("🧾 Final transcript received, generating summary + PDF...")
        raw_text = " ".join([e["text"] for e in full_transcript])[:2000]
        final_insight = await asyncio.to_thread(run_llm, raw_text)
        print("✅ Final insight generated.")
        filename = generate_pdf_report(full_transcript, final_insight)
        print(f"📄 PDF report written to {filename}")
    else:
        print("❌ No transcript available — PDF not generated.")

# ─────────────────────────  MAIN  ──────────────────────────────
async def main():
    uri = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}"
    async with websockets.connect(
        uri,
        additional_headers={"Authorization": API_KEY},
        ping_interval=5,
        ping_timeout=20,
    ) as ws:
        print("🎙️  Speak – press Ctrl+C to stop.\n")
        worker_task = asyncio.create_task(insight_worker())
        try:
            await asyncio.gather(
                send_audio(ws),
                recv_transcripts(ws),
            )
        except asyncio.CancelledError:
            raise
        finally:
            await flush_and_report()
            await worker_task

# ───────────────────────── ENTRY ───────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down…")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
