import base64
import requests

API = "http://localhost:7860"
# make sure the model is loaded first…
resp = requests.post(
    f"{API}/load",
    data={
        "model_id": "models/Qwen3-TTS-12Hz-1.7B-Base",
        # "models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        # "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    },
)

print(resp.json())

# 1. voice_clone (default)
data = {
    "text": "Hello from qian1 wen4 3-TTS! This is a voice clone demo.",
    "mode": "voice_clone",
    # either a file or a preset name:
    # "ref_preset": "ref_audio",
}
files = {
    "ref_audio": open(
        r"D:\CosyVoice\参考音频\古话说：“天下大势，分久必合，合久必分。”.wav", "rb"
    )
}
r = requests.post(f"{API}/generate", data=data, files=files)

# # 2. custom
#   { id: 'Vivian',   desc: 'Chinese — Bright young female' },
#   { id: 'Serena',   desc: 'Chinese — Warm gentle female' },
#   { id: 'Uncle_Fu', desc: 'Chinese — Seasoned low male' },
#   { id: 'Dylan',    desc: 'Chinese — Youthful Beijing male' },
#   { id: 'Eric',     desc: 'Chinese (Sichuan) — Lively male' },
#   { id: 'Ryan',     desc: 'English — Dynamic male' },
#   { id: 'Aiden',    desc: 'English — Sunny American male' },
#   { id: 'Ono_Anna', desc: 'Japanese — Playful female' },
#   { id: 'Sohee',    desc: 'Korean — Warm female' },
# data = {
#     "text": "This is a custom speaker.",
#     "mode": "custom",
#     "speaker": "Uncle_Fu",  # must be one of the IDs returned by /statusconst defaultSpeakers = [
#     "instruct": "Speak like a robot.",
# }
# r = requests.post(f"{API}/generate", data=data)

# # 3. voice_design
# data = {
#     "text": "Let your voice be warm and slow.",
#     "mode": "voice_design",
#     "instruct": "Add a gentle, storyteller tone.",
# }
# r = requests.post(f"{API}/generate", data=data)


r.raise_for_status()
result = r.json()

# the demo server returns the base64-encoded waveform under the
# `audio_b64` key (formerly the web UI used an `audio` list).
# adjust accordingly to avoid KeyError.
wav_b64 = result.get("audio_b64")
if wav_b64 is None:
    raise RuntimeError(f"unexpected response: {result}")
wav_bytes = base64.b64decode(wav_b64)
with open("out.wav", "wb") as f:
    f.write(wav_bytes)
print("saved out.wav")

# for streaming you can hit /generate/stream and process SSE events
