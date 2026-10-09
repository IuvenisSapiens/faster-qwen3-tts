# qwentts.cpp GGML Backend

This repo keeps `faster-qwen3-tts` as the user-facing package and uses
`qwentts-cpp-python` as its native GGML runtime.

## Package Layout

```text
faster-qwen3-tts
  Existing Torch/CUDA-graph backend
  Optional GGML adapter in faster_qwen3_tts.ggml_backend

qwentts-cpp-python
  Pure Python ctypes wrapper over qwentts.cpp's C ABI
  Platform wheel packaging for libqwen + libggml

qwentts.cpp
  Pascal's C++/GGML implementation, built separately with CMake
```

On native Apple Silicon Python, the default install includes the Metal runtime:

```bash
pip install faster-qwen3-tts
```

On other platforms, GGML is opt-in through the `ggml` extra. Both paths require
`qwentts-cpp-python>=0.5.0`. PyPI version 0.5.0 provides a Metal wheel for
macOS 14+ with native Apple Silicon Python and CUDA 12.8 wheels for supported
Linux hosts:

```bash
pip install "faster-qwen3-tts[ggml]"
```

On Apple Silicon, the wheel includes `libqwen` and its Metal dependencies; no
Homebrew libraries, local native build, or `qwentts_library_path` are needed.
The Torch backend requires NVIDIA CUDA or AMD ROCm. Intel Macs and macOS
older than 14 are not covered by the Metal wheel.

Version 0.5.0 uses qwentts.cpp ABI v5. If you pass an explicit
`qwentts_library_path`, rebuild that native library using the wrapper
release's pinned revision; older ABI v2 libraries are incompatible. The
wrapper checks the native revision before passing C parameter structures.

The GGML adapter passes `log_level="warning"` to the native wrapper before
model initialization, so Python applications and the CLI hide routine startup
and synthesis diagnostics while keeping warnings and errors visible. Download
progress remains visible. Put `--verbose` before the CLI subcommand or pass
`qwentts_log_level="debug"` to `FasterQwen3TTS.from_pretrained(...)` to show
all native diagnostics. Direct `GGMLQwen3TTS` constructors accept `log_level`;
`"info"` and `"error"` are also supported. Native logging is process-wide: the
most recent setting applies to all GGML contexts. Configure it when no other
thread is running native inference.

To verify Metal execution with local GGUF weights, set `GGML_BACKEND=MTL0`.
This selects the Metal device explicitly, so generation fails if only a CPU
backend is available:

```bash
GGML_BACKEND=MTL0 faster-qwen3-tts \
  --backend ggml \
  --gguf-model /path/to/qwen-talker-1.7b-base-Q8_0.gguf \
  --gguf-codec /path/to/qwen-tokenizer-12hz-Q8_0.gguf \
  clone --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio reference.wav --xvec-only --streaming \
  --text "Hello from Metal." --output metal-smoke.wav
```

For CUDA 13 / DGX Spark, CUDA 12.4, CPU-only Linux, or older Linux hosts
whose glibc cannot use the PyPI wheel, install a backend-specific wrapper
wheel with version 0.5.0 or newer before installing the extra:

```bash
# Ubuntu 22.04 / older Linux with CUDA 12.8
pip install "qwentts-cpp-python==0.5.0+cu128" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu128

# CUDA 13 / DGX Spark
pip install "qwentts-cpp-python==0.5.0+cu130" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu130

pip install "faster-qwen3-tts[ggml]"
```

The CUDA 12.4 and CPU flavors are `+cu124` and `+cpu`; they also require
wrapper version 0.5.0 or newer.

For a local build, clone the wrapper repo beside this checkout and use a
version 0.5.0 or newer:

```bash
git clone https://github.com/andimarafioti/qwentts-cpp-python ../qwentts-cpp-python
cd ../qwentts-cpp-python
git checkout v0.5.0
```

Build with the CUDA toolkit installed on the target Linux machine. Use
`--backend cpu` instead for a CPU-only build:

```bash
python scripts/build_native.py --backend cuda --clean
python -m pip install .
cd ../faster-qwen3-tts
python -m pip install ".[ggml]"
```

### AMD GPUs (HIP)

The GGML adapter was validated on AMD Instinct MI300X VF (`gfx942`) with
ROCm 7.2.4, `qwentts-cpp-python==0.5.0`, and the pinned qwentts.cpp revision
`6fae92914045cd83364d2845ceaa0f7969727319` (ABI v5). BF16 and Q4_K_M worked
for full and streaming generation with 0.6B/1.7B Base, 1.7B CustomVoice, and
1.7B VoiceDesign. Other AMD GPUs have not been tested.

The published Linux wheels are CUDA builds. AMD requires a source-built HIP
library and an explicit `--qwentts-lib` path. Install `faster-qwen3-tts` first
as described in the [README](../README.md#amd-gpus-rocm), in a ROCm development
environment with HIP, hipBLAS, and rocBLAS available. The ROCm PyTorch image
listed there was tested. Then build the native runtime:

```bash
pip install cmake ninja
git clone --branch v0.5.0 https://github.com/andimarafioti/qwentts-cpp-python
cd qwentts-cpp-python
git clone --no-checkout https://github.com/ServeurpersoCom/qwentts.cpp third_party/qwentts.cpp
git -C third_party/qwentts.cpp checkout 6fae92914045cd83364d2845ceaa0f7969727319
git -C third_party/qwentts.cpp submodule update --init --recursive

cmake -S third_party/qwentts.cpp -B build/hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DQWEN_SHARED=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF
cmake --build build/hip --target qwen -j 8
pip install --no-deps -e .

export LD_LIBRARY_PATH="$PWD/build/hip:${LD_LIBRARY_PATH:-}"
GGML_BACKEND=ROCm0 faster-qwen3-tts --backend ggml --quant BF16 \
  --qwentts-lib "$PWD/build/hip/libqwen.so" design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "A warm, calm narrator." \
  --text "Hello from AMD." --language English --output amd.wav
```

Keep the native build directory: the adapter loads `libqwen.so` and
its sibling GGML libraries from it. `GGML_BACKEND=ROCm0` forces the AMD device
and fails if it is unavailable, avoiding a CPU-only result. For Python, pass
`qwentts_library_path="/path/to/build/hip/libqwen.so"` with `backend="ggml"`.

## Python Usage

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    backend="ggml",
    quant="BF16",
)

audio_list, sr = model.generate_voice_design(
    text="Welcome to the show.",
    instruct="Warm, confident narrator with slight British accent",
    language="English",
)
```

Local GGUF paths are supported:

```python
model = FasterQwen3TTS.from_pretrained(
    "unused",
    backend="ggml",
    gguf_talker_path="qwen-talker-1.7b-voicedesign-BF16.gguf",
    gguf_codec_path="qwen-tokenizer-12hz-BF16.gguf",
)
```

CLI usage:

```bash
faster-qwen3-tts --backend ggml --quant BF16 design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "Warm, confident narrator" \
  --text "Welcome to the show." \
  --language English \
  --output out.wav
```

Raw reference audio is cached automatically after the first base voice-clone
request. The adapter stores qwentts-compatible `.spk` and `.rvq` latents under
`~/.cache/faster-qwen3-tts/qwentts_refs` by default, or under
`FQWEN3TTS_QWENTTS_REF_CACHE_DIR` / `--qwentts-ref-cache-dir` when set.

Precomputed qwentts.cpp references are also supported for base voice cloning:

```python
audio_list, sr = model.generate_voice_clone(
    text="Cached speaker and RVQ latents avoid reference audio encoding.",
    language="English",
    ref_spk="freeman.spk",
    ref_rvq="freeman.rvq",
    ref_text="The transcript for the cached reference audio.",
)
```

Use `ref_spk` by itself for speaker-only conditioning. Use `ref_spk` +
`ref_rvq` + `ref_text` for cached ICL conditioning. Raw `ref_audio` and
explicit cached references are mutually exclusive.

## Current ABI Gaps

The qwentts.cpp C ABI is already enough for buffered and streaming
synthesis, voice cloning, CustomVoice, and VoiceDesign. These gaps remain
before treating the backend as full parity:

- no `non_streaming_mode` switch; requesting
  `non_streaming_mode=False` emits a warning because qwentts.cpp ignores
  that step-by-step text-feed mode and uses its native prompt layout
- base-model `instruct` is rejected by qwentts.cpp
- KV-cache length is fixed in qwentts.cpp

The public GGUF model repo used by the resolver is
`Serveurperso/Qwen3-TTS-GGUF`.

## TTFA Profiling

The GGML adapter attaches a `ggml_profile` snapshot to the first streamed
chunk's `timing` dict. It includes Python-visible boundaries such as ctypes
parameter packing, lock wait, native `qt_synthesize()` entry, first native
audio callback, callback copy/queue cost, and first Python yield.

Run the focused diagnostic with:

```bash
python benchmarks/profile_ggml_ttfa.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --mode custom \
  --speaker aiden \
  --quant BF16 \
  --cache-dir .cache/qwentts \
  --local-files-only \
  --chunk-sizes 2,4,8,16
```

With a `libqwen` that includes native profiling markers, the same command also
prints native phase splits for prompt build, first talker prefill, first code
predictor step, first emit entry, and first codec decode. Use `--warmup-runs 0`
to inspect true first-request latency; use the default warmup to inspect
steady-state server latency after CUDA/GGML graphs and prefix caches are hot.

For a direct Torch-vs-GGML comparison, use the benchmark entry point:

```bash
MODEL_SIZE=1.7B MODE=custom QUANT=BF16 ./benchmark.sh backends
MODEL_SIZE=0.6B QUANT=BF16 ./benchmark.sh backend-base
```

The legacy CUDA-graph-only benchmarks still run with `./benchmark.sh`.

## Wheel Distribution

`qwentts-cpp-python==0.5.0` is published on PyPI with Linux CUDA 12.8 and
macOS 14+ Apple Silicon Metal wheels. Pip selects the matching platform wheel
for `pip install "faster-qwen3-tts[ggml]"`.

The public Linux PyPI wheels use `manylinux_2_39`. Additional local-version
wheels are hosted on Hugging Face Hub:

```bash
pip install "qwentts-cpp-python==0.5.0+cpu" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cpu

pip install "qwentts-cpp-python==0.5.0+cu124" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu124

pip install "qwentts-cpp-python==0.5.0+cu128" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu128

pip install "qwentts-cpp-python==0.5.0+cu130" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu130
```

Hugging Face file hosting is used as a `--find-links` wheelhouse. For CUDA 13 /
DGX Spark, install the `+cu130` wheel first. For Ubuntu 22.04 / older Linux
hosts, install `+cu128` from the wheelhouse so pip can select the
`manylinux_2_35` CUDA 12.8 build. If no wheel matches your platform, build
version 0.5.0 or newer from source as shown above.

For publishing new wrapper wheels, use the manual GitHub Actions workflow:

```text
andimarafioti/qwentts-cpp-python:.github/workflows/publish-hf-wheels.yml
```

The workflow builds Linux x86_64 and Linux aarch64 wheels for CPU, CUDA 12.4,
CUDA 12.8, and CUDA 13.0, then uploads static wheel index pages to the HF
dataset.

Local development builds still use the wrapper build script:

```bash
cd ../qwentts-cpp-python
python scripts/build_native.py \
  --source third_party/qwentts.cpp \
  --backend cuda \
  --clean \
  --cmake-arg=-G \
  --cmake-arg=Ninja \
  --cmake-arg="-DCMAKE_CUDA_ARCHITECTURES=75-virtual;80-real;86-real;90-real;121-real"
python -m build --wheel
```

CUDA-linked `libqwen` depends on CUDA runtime libraries and an NVIDIA driver at
runtime. Choose the wrapper wheel that matches the runtime and GPU target first;
the `faster-qwen3-tts` package only selects the Python adapter.
