# NPU Acceleration (Windows ARM64)

This fork adds support for NPU (Neural Processing Unit) acceleration on Windows ARM64 devices such as Snapdragon X Elite/Plus laptops. NPU support enables three modes of operation:

- **NPU-only inference** for ONNX/GenAI models routed directly to the NPU
- **NPU-assisted speculative decoding** where a small draft model on the NPU accelerates GGUF inference on the main GPU/CPU
- **NPU diagnostics** to verify hardware availability

## Prerequisites

- Windows 11 ARM64 with a DXCore-compatible NPU (e.g., Qualcomm Adreno NPU on Snapdragon X)
- DirectML-compatible drivers installed
- For ONNX models: an ONNX model directory compatible with ORT GenAI
- For speculative decoding: a small ONNX draft model with the same tokenizer as the main GGUF model

## Diagnostics

Verify that your NPU is detected and usable:

```shell
ollama debug npu
```

Example output:

```
DXCore: available
Adapters:
  #0  Qualcomm Adreno (NPU)  SharedMem=16384MB  D3D12=yes  DML=yes  FL=1_0_GENERIC
  #1  Qualcomm Adreno (GPU)  SharedMem=16384MB  D3D12=yes  DML=yes  FL=11_0
Summary: 1 NPU, 1 GPU
```

## Environment Variables

| Variable | Values | Description |
|---|---|---|
| `OLLAMA_ACCEL_MODE` | `off`, `npu`, `npu_assist` | Controls NPU acceleration mode |
| `OLLAMA_NPU_DEVICE_ID` | adapter index (e.g., `0`) | Target a specific DXCore adapter |
| `OLLAMA_NPU_STRICT` | `0`, `1` | If `1`, fail with error when NPU is unavailable instead of falling back |
| `OLLAMA_DRAFT_MODEL` | path to ONNX model dir | Path to the draft model for speculative decoding |
| `OLLAMA_DRAFT_K` | integer (default `5`) | Number of draft tokens proposed per iteration |
| `OLLAMA_NPU_STATS` | `0`, `1` | Include NPU speculative decoding stats in completion responses |

## Mode 1: NPU-Only Inference (ONNX Models)

Route an ONNX model entirely to the NPU using DirectML:

```shell
OLLAMA_ACCEL_MODE=npu ollama serve
```

Then run an ONNX-format model as normal. The scheduler will route it to the NPU via the ORT GenAI runner with DirectML EP targeting NPU.

To fail loudly if the NPU is not available (instead of falling back to GPU):

```shell
OLLAMA_ACCEL_MODE=npu OLLAMA_NPU_STRICT=1 ollama serve
```

### GenAI-QNN Models

Models with the `genai-qnn` format are automatically routed to the ORT GenAI runner with the QNN execution provider, regardless of `OLLAMA_ACCEL_MODE`. This is for models specifically compiled for Qualcomm's QNN runtime.

## Mode 2: NPU-Assisted Speculative Decoding (GGUF Models)

This is the most impactful mode for everyday use. A small ONNX draft model runs on the NPU and proposes tokens speculatively. The main GGUF model on GPU/CPU then verifies them in a single forward pass, accepting correct predictions. This can significantly improve tokens-per-second for greedy decoding workloads.

### How It Works

1. The main model generates a token normally
2. The draft model on NPU proposes K tokens greedily
3. All K draft tokens are evaluated by the main model in one batch
4. Matching tokens are accepted; on mismatch, the main model's token is used
5. The draft model is notified of accepted/rejected tokens and corrects its state

### Setup

```shell
OLLAMA_ACCEL_MODE=npu_assist \
OLLAMA_DRAFT_MODEL=/path/to/small-onnx-model \
ollama serve
```

Then use `ollama run` as normal. The speculative decoding is transparent to the user.

### Requirements and Limitations

- **Greedy decoding only (V1):** Speculative decoding only activates when `temperature=0`. For non-zero temperature, inference falls back to normal single-token generation automatically.
- **Tokenizer compatibility:** The draft and main models must share the same tokenizer. A compatibility check runs at startup and disables speculative decoding if a mismatch is detected.
- **Single sequence:** Designed for single-sequence inference. Works normally with `parallel=1` (the default).
- **Graceful fallback:** If the draft model fails to start, crashes, or produces errors, speculative decoding is disabled silently and inference continues normally.

### Tuning

Adjust the number of draft tokens per iteration:

```shell
OLLAMA_DRAFT_K=8 ollama serve
```

A higher K means more tokens are proposed and potentially accepted per iteration, but increases the cost of rejected tokens. The default of 5 is a good starting point. Optimal values depend on how well the draft model matches the main model.

### Monitoring Performance

Enable NPU stats in the completion response:

```shell
OLLAMA_NPU_STATS=1 ollama serve
```

The final response for each completion will include:

```json
{
  "done": true,
  "npu_draft_proposed": 512,
  "npu_draft_accepted": 401,
  "npu_accept_rate": 0.783
}
```

- `npu_draft_proposed`: total draft tokens proposed during the request
- `npu_draft_accepted`: total draft tokens verified as correct
- `npu_accept_rate`: acceptance ratio (higher is better; >0.7 is good)

## Mode 3: GGUF on DirectML

When `OLLAMA_ACCEL_MODE=npu` or `npu_assist` is set with a GGUF model (not ONNX), Ollama implicitly enables the DirectML backend (`OLLAMA_DIRECTML=1`). The ggml-directml backend will route operations to NPU or GPU based on the adapter priority established by DXCore.

## Troubleshooting

**`ollama debug npu` shows no NPU adapters:**
- Ensure you're on Windows 11 ARM64 with NPU drivers installed
- Check Device Manager for the NPU device
- Update to the latest Qualcomm or Intel NPU drivers

**Speculative decoding disabled at startup with "tokenizer mismatch":**
- The draft model uses a different tokenizer than the main model. Use a draft model from the same model family or one that shares the same tokenizer vocabulary.

**Low accept rate (<0.5):**
- The draft model is a poor match for the main model. Try a draft model from the same family (e.g., a smaller quantization of the same architecture).
- Increase `OLLAMA_DRAFT_K` slightly — a larger window gives the draft model more chances, though rejection cost also increases.

**Draft model fails to start:**
- Check that `OLLAMA_DRAFT_MODEL` points to a valid ONNX model directory
- Ensure ORT GenAI dependencies are available
- Check logs for specific errors: `ollama serve 2>&1 | grep draft`
