# Ollama - NPU / ONNX Runtime Inference

Experimental support for running ONNX models on **NPU**, **GPU**, or **CPU** via
[ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) and
[DirectML](https://github.com/microsoft/DirectML).

This enables hardware-accelerated inference on Windows ARM64 devices with NPUs
(Qualcomm Snapdragon X Elite/Plus, Intel Core Ultra, AMD Ryzen AI, etc.)
without requiring vendor-specific SDKs.

## NPU Benchmark Results

Tested on **Snapdragon X Elite** (X1E78100) with **Phi-3-mini 3.8B int4**.
CPU baseline uses the DirectML ONNX model with `OLLAMA_ONNX_PROVIDER=cpu`.
NPU uses a QNN-native model (`llmware/phi-3-mini-4k-instruct-onnx-qnn`) with
pre-compiled HTP context binaries on Hexagon NPU. All results via `cmd/bench`
with `NumCtx=512, PromptTokens=512, MaxTokens=128`.

### Warm (steady-state, 8 epochs) — all p=0.000

| Metric | CPU ORT | NPU QNN | Change |
|---|---|---|---|
| **Decode** | 4.34 tok/s (±3%) | 11.23 tok/s (±9%) | **+159% (2.6x)** |
| **TTFT** | 32.86s (±2%) | 1.83s (±2%) | **-94% (18x faster)** |
| **Total** | 62.30s (±1%) | 10.02s (±7%) | **-84% (6.2x faster)** |
| **Load** | 29ms | 36ms | ~ (same) |

### Cold (first request, single sample)

| Metric | CPU ORT | NPU QNN |
|---|---|---|
| **Decode** | 4.35 tok/s | 3.48 tok/s* |
| **TTFT** | 38.0s | 2.5s |
| **Load** | 5.24s | 33ms |
| **Total** | 67.4s | 29.2s |

\*Cold NPU decode is slower on first request due to QNN graph compilation
overhead; warm steady-state is 2.6x faster.

**Summary**: The Snapdragon X Elite NPU with a QNN-native model delivers:
- **2.6x faster decode** (11.2 vs 4.3 tok/s)
- **18x faster time-to-first-token** (1.8s vs 32.9s)
- **6.2x faster end-to-end** (10s vs 62s)
- **160x faster cold load** (33ms vs 5.2s)

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Windows 11 (ARM64 or x64) |
| **NPU Driver** | Device-specific — must expose a D3D12 adapter via DXCore |
| **Go** | 1.24+ with CGo enabled |
| **C Compiler** | MSVC (VS 2022) or LLVM/Clang targeting `*-windows-msvc` |
| **ONNX Model** | A GenAI-compatible ONNX model directory (see [Models](#models)) |

### NPU Driver Compatibility

NPU inference requires a driver that registers the NPU as a D3D12 adapter.
Verify your hardware is supported:

| Vendor | Chipset | NPU Driver | EP |
|---|---|---|---|
| Qualcomm | Snapdragon X Elite / Plus | Adreno GPU + Hexagon NPU (Windows Update) | `dml` or `qnn` |
| Intel | Core Ultra (Meteor Lake+) | Intel NPU Driver (Windows Update) | `dml` |
| AMD | Ryzen AI 300+ | AMD IPU Driver | `dml` |

To check if your NPU is accessible, open PowerShell:

```powershell
Get-CimInstance Win32_VideoController | Format-Table Name, Status -AutoSize
```

You should see your NPU listed with Status `OK`. If it only appears as a
display adapter with no D3D12 user-mode driver, NPU targeting will silently
fall back to GPU.

## Setup

### 1. Download DLLs

Place these three DLLs in `lib/ollama/ortgenai/` (or any directory you point
`OLLAMA_ORT_PATH` to):

| DLL | Source | Version Tested |
|---|---|---|
| `onnxruntime-genai.dll` | [onnxruntime-genai releases](https://github.com/microsoft/onnxruntime-genai/releases) | v0.12.1 ARM64 |
| `onnxruntime.dll` | NuGet: `Microsoft.ML.OnnxRuntime.DirectML` | v1.21.1 ARM64 |
| `DirectML.dll` | NuGet: `Microsoft.AI.DirectML` | v1.15.4 ARM64 |

**Important:** Match the architecture (ARM64 vs x64) to your OS. The
`onnxruntime.dll` **must** come from the DirectML NuGet package — the system
copy (often v1.17 from Edge/Office) is too old and will cause API version
errors.

### 2. Download an ONNX Model

You need a model in ORT GenAI format (a directory containing `genai_config.json`,
`model.onnx`, and tokenizer files).

#### DirectML Models (CPU/GPU)

```bash
# Example: Phi-3-mini 4K Instruct (DirectML, int4 quantized, ~2 GB)
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
cd Phi-3-mini-4k-instruct-onnx
git lfs pull --include="directml/directml-int4-awq-block-128/*"
```

#### QNN-Native Models (Snapdragon NPU)

For Qualcomm Snapdragon devices, QNN-native models contain pre-compiled HTP
context binaries that run directly on the Hexagon NPU. These use a split
pipeline architecture (`embeddings.onnx`, `context_ctx.onnx`, `iterator_ctx.onnx`,
`lm_head.onnx`) with QNN provider options baked into `genai_config.json`.

```bash
# Example: Phi-3-mini QNN (int4, pre-compiled for Snapdragon X Elite, ~2 GB)
git lfs install
git clone https://huggingface.co/llmware/phi-3-mini-4k-instruct-onnx-qnn C:\models\phi3-mini-onnx-qnn
```

Other QNN-native models from the [llmware NPU-QNN collection](https://huggingface.co/collections/llmware/npu-qnn):
- `llmware/phi-3.5-onnx-qnn`
- `llmware/llama-3.2-3b-onnx-qnn`
- `llmware/qwen2.5-1.5b-instruct-onnx-qnn`
- `llmware/qwen2.5-7b-instruct-onnx-qnn`

> **Important:** DirectML and QNN models are **not interchangeable**. A DirectML
> model forced to run with `OLLAMA_ONNX_PROVIDER=qnn` will fall back to CPU.
> A QNN model with baked-in provider options will use NPU regardless of env vars.

Other models that work with ORT GenAI + DirectML:
- `microsoft/Phi-4-mini-instruct-onnx`
- `microsoft/Phi-3.5-mini-instruct-onnx`
- Any [Hugging Face model with `onnx` + `genai` tags](https://huggingface.co/models?library=onnxruntime&sort=trending)

### 3. Build

On Windows ARM64 without MinGW GCC, use the `cgo-clang` wrapper scripts
(included in this repo) that filter GCC-specific flags for MSVC-targeted
Clang:

```bash
CC=c:/path/to/cgo-clang.exe \
CXX=c:/path/to/cgo-clang++.exe \
CGO_ENABLED=1 \
CGO_LDFLAGS="-ladvapi32" \
go build -o ollama.exe .
```

With MSVC or standard MinGW:

```bash
CGO_ENABLED=1 CGO_LDFLAGS="-ladvapi32" go build -o ollama.exe .
```

## Usage

### Quick Start (env var override)

The fastest way to test is using `OLLAMA_ONNX_MODEL` to route any model
request through the ORT GenAI runner:

```bash
OLLAMA_ORT_PATH=lib/ollama/ortgenai \
OLLAMA_ONNX_MODEL="c:/path/to/your/onnx-model-dir" \
./ollama.exe serve
```

Then in another terminal:

```bash
ollama run phi3
# Or any model name — it routes to the ONNX model regardless
```

### Targeting the NPU

Set `OLLAMA_ORT_DEVICE_TYPE=npu` to direct inference to the NPU via DirectML:

```bash
OLLAMA_ORT_PATH=lib/ollama/ortgenai \
OLLAMA_ONNX_MODEL="c:/path/to/your/onnx-model-dir" \
OLLAMA_ORT_DEVICE_TYPE=npu \
./ollama.exe serve
```

If the NPU is not available (no D3D12 driver), DirectML silently falls back to
GPU.

### Targeting a Specific Device

Use `OLLAMA_ORT_DEVICE_ID` to select a device by DXGI enumeration index:

```bash
OLLAMA_ORT_DEVICE_ID=1 ./ollama.exe serve
```

### Using QNN (Qualcomm Snapdragon only)

For Qualcomm Snapdragon devices, the QNN execution provider targets the
Hexagon HTP (NPU) directly. This provides the best performance — in benchmarks
on Snapdragon X Elite, QNN-native models achieve **~2.6x faster decode** and
**~18x faster TTFT** compared to CPU ORT inference.

**Recommended: QNN-native model (pre-compiled HTP context binaries)**

```powershell
$env:OLLAMA_ONNX_MODEL = "C:\models\phi3-mini-onnx-qnn"
$env:OLLAMA_ONNX_PROVIDER = "qnn"
$env:OLLAMA_ORT_QNN_BACKEND_TYPE = "htp"
$env:OLLAMA_ORT_QNN_BACKEND_PATH = "QnnHtp.dll"
$env:OLLAMA_ORT_PATH = "lib\ollama\ortgenai"
$env:OLLAMA_CONTEXT_LENGTH = "512"
.\ollama.exe serve
```

Then in another terminal:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
.\ollama.exe run phi3:mini
```

> **Note:** QNN-native models have QNN provider options baked into their
> `genai_config.json` per pipeline stage. The `OLLAMA_ONNX_PROVIDER` env var
> controls the top-level session, but per-stage overrides take precedence —
> a QNN model will always route transformer stages to the NPU.

This requires a QNN-specific ONNX model (not the same as DirectML models).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_ORT_PATH` | `lib/ollama/ortgenai` | Directory containing ORT GenAI DLLs |
| `OLLAMA_ONNX_MODEL` | *(none)* | Path to ONNX model directory; overrides normal model routing |
| `OLLAMA_ONNX_PROVIDER` | *(auto-detect)* | Execution provider: `dml`, `qnn`, `cpu`, or any ORT EP name. Auto-detects QNN if `onnxruntime_providers_qnn.dll` is found. |
| `OLLAMA_ACCEL_MODE` | *(none)* | NPU acceleration mode: `npu` (full NPU via ORT GenAI) |
| `OLLAMA_CONTEXT_LENGTH` | *(model default)* | Context length (num_ctx). Strongly recommended for ORT GenAI models (e.g., `512`). |
| `OLLAMA_ORT_DEVICE_TYPE` | *(none)* | Device filter: `npu` or `gpu` (DML only) |
| `OLLAMA_ORT_DEVICE_ID` | *(none)* | Explicit device index (overrides device type filter) |
| `OLLAMA_ONNX_NPU` | `0` | Set to `1` to switch to QNN provider |
| `OLLAMA_ORT_QNN_BACKEND_TYPE` | `htp` | QNN backend type: `htp` (NPU/Hexagon) or `cpu`. Mutually exclusive with `BACKEND_PATH`. |
| `OLLAMA_ORT_QNN_BACKEND_PATH` | *(none)* | Path to QNN backend DLL (e.g., `QnnHtp.dll`). Mutually exclusive with `BACKEND_TYPE`. |
| `OLLAMA_ORT_QNN_HTP_ARCH` | *(auto)* | QNN HTP architecture version (e.g., `73` for v73). Empty = auto-detect. |
| `OLLAMA_ORT_QNN_SOC_MODEL` | *(auto)* | QNN SoC model number (e.g., `60` for Snapdragon X Elite). Empty = auto-detect. |

## Architecture

```
ollama serve
  |
  |-- server/sched.go  (checks OLLAMA_ONNX_MODEL or model.IsONNX())
  |     |
  |     +-- ortrunner.NewClient(modelDir)
  |           |
  |           +-- spawns subprocess:
  |                 ollama runner --ortgenai-engine --model <dir> --port <port>
  |
  +-- x/ortrunner/server.go    HTTP server (health, completion, tokenize)
  +-- x/ortrunner/runner.go    Model loading + EP configuration
  +-- x/ortrunner/pipeline.go  Token-by-token generation loop
  +-- x/ortrunner/client.go    Subprocess client (implements llm.LlamaServer)
  +-- x/ortrunner/oga/         CGo bindings: dynamic loading of onnxruntime-genai.dll
```

The runner loads `onnxruntime-genai.dll` at runtime via `LoadLibraryA` +
`GetProcAddress` (no link-time dependency). A `SetDllDirectoryA` call ensures
the correct `onnxruntime.dll` is loaded from the same directory, avoiding
conflicts with system copies.

## Troubleshooting

### "The requested API version [23] is not available"

The wrong `onnxruntime.dll` is being loaded (system copy from Edge/Office).
Make sure `OLLAMA_ORT_PATH` points to the directory with **your** copy of
`onnxruntime.dll` (v1.21+).

### Model loads but output is garbage

Ensure the model variant matches the execution provider. DirectML models need
the DML provider; QNN models need the QNN provider. Using a DML-quantized
model with `OLLAMA_ONNX_PROVIDER=cpu` may produce incorrect results.

### "failed to load ORT GenAI dynamic library"

Check that all three DLLs are present and match your architecture:
```bash
ls lib/ollama/ortgenai/
# Should contain: onnxruntime-genai.dll  onnxruntime.dll  DirectML.dll
```

### NPU targeting has no effect

Run the `npu_access_probe` tool in `_tools/` to verify your NPU has a working
D3D12 driver. If `D3D12CreateDevice` fails for the NPU adapter, DirectML
cannot use it and will fall back to GPU.

For Snapdragon NPUs (Generic ML-only, no D3D12), use the QNN provider path
instead: `set OLLAMA_ONNX_PROVIDER=qnn`. Run `ollama debug ortgenai` to verify.

### "DML provider requested, but ... not built with DML support"

Your `onnxruntime-genai.dll` is a QNN build that does not include DML. Either:
1. Install a DML-enabled GenAI build if you need DML, or
2. Set `OLLAMA_ONNX_PROVIDER=qnn` for Snapdragon NPU (recommended for NPU path)

### "DspTransport.openSession" / "0x80000406" / QNN HTP transport errors

QNN HTP failed to open a DSP session. Check with `ollama debug ortgenai`:
1. **Missing DLLs**: Ensure `QnnHtp.dll`, `QnnSystem.dll`, `QnnHtpPrepare.dll`,
   `QnnHtpV*Stub.dll` are all in `OLLAMA_ORT_PATH`
2. **Transport DLLs**: Some systems require `cdsprpc.dll` — check if it's
   shipped with your QNN SDK and copy it to `OLLAMA_ORT_PATH`
3. **Architecture mismatch**: Try setting `OLLAMA_ORT_QNN_HTP_ARCH` and
   `OLLAMA_ORT_QNN_SOC_MODEL` explicitly
4. **Driver/runtime**: Ensure the Qualcomm NPU driver is installed and up to date

### Two NPU paths

There are two distinct NPU acceleration paths with different requirements:

**Path A — D3D12/DirectML** (ggml-directml style):
- Requires `D3D12CreateDevice()` to succeed on the NPU adapter
- Requires DirectML device creation
- `ollama debug npu` should show `D3D12=yes` and `DML=yes`

**Path B — ORT GenAI + QNN EP** (Snapdragon NPU, recommended):
- Does NOT require D3D12
- Requires ORT GenAI + QNN provider DLLs + QNN backend (HTP)
- Requires a QNN-compatible model (not a DirectML model)
- `ollama debug ortgenai` shows provider support and missing DLLs
- **Best performance**: ~2.6x decode speedup, ~18x TTFT improvement vs CPU

## Benchmarking

### Quick benchmark

Use the `cmd/bench` tool to benchmark model performance:

```powershell
go build -o .\ollama-bench.exe .\cmd\bench

# Benchmark with context length (required for ORT GenAI stability)
.\ollama-bench.exe -model phi3:mini -epochs 8 -warmup 1 -num-ctx 512 -prompt-tokens 512 -max-tokens 128
```

The `-num-ctx` flag is critical for ORT GenAI models — without it, you may
hit KV allocation errors or `max_length (-1)` issues.

### CPU vs NPU comparison

Use `scripts/bench-npu.ps1` to run automated cold + warm benchmarks across
CPU and NPU profiles:

```powershell
go build -o .\ollama.exe .
go build -o .\ollama-bench.exe .\cmd\bench
.\scripts\bench-npu.ps1 -Model phi3:mini -NumCtx 512 -PromptTokens 512 -MaxTokens 128 -WarmEpochs 8
```

The script:
1. Starts isolated `ollama serve` instances on dynamic ports per profile
2. Runs a **cold** benchmark (epochs=1, warmup=0) to capture first-request latency
3. Runs a **warm** benchmark (warmup=1, epochs=N) for steady-state throughput
4. Outputs benchstat-format files for statistical comparison

Compare results with `benchstat`:

```powershell
go install golang.org/x/perf/cmd/benchstat@latest
benchstat (Get-ChildItem .\results-npu-bench\bench-*-warm.txt | ForEach-Object { $_.FullName })
```

Edit the `$profiles` block in the script to configure model paths and env vars
for each profile. The default profiles compare:
- **cpu_ort**: DirectML model on CPU via ORT GenAI
- **npu_qnn**: QNN-native model on NPU via ORT GenAI + Hexagon HTP

### Reference benchmark results (Snapdragon X Elite)

Phi-3-mini 3.8B, int4 quantized, NumCtx=512, PromptTokens=512, MaxTokens=128:

| Metric | CPU ORT | NPU QNN | Change |
|---|---|---|---|
| **Decode** | 4.3 tok/s | 11.2 tok/s | **+159% (2.6x)** |
| **TTFT** | 32.9s | 1.8s | **-94% (18x faster)** |
| **Total** | 62.3s | 10.0s | **-84% (6.2x faster)** |
| **Load (warm)** | 29ms | 36ms | ~ (same) |
