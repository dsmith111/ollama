# NPU Acceleration (Windows ARM64)

This fork adds support for NPU (Neural Processing Unit) acceleration on Windows ARM64 devices such as Snapdragon X Elite/Plus laptops. NPU support enables:

- **NPU-only inference** for ONNX/GenAI models routed directly to the NPU
- **NPU diagnostics** to verify hardware availability

## Prerequisites

- Windows 11 ARM64 with a DXCore-compatible NPU (e.g., Qualcomm Adreno NPU on Snapdragon X)
- DirectML-compatible drivers installed
- For ONNX models: an ONNX model directory compatible with ORT GenAI
- For QNN-native models: a QNN-specific ONNX model with pre-compiled HTP context binaries

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
| `OLLAMA_ACCEL_MODE` | `off`, `npu` | Controls NPU acceleration mode |
| `OLLAMA_NPU_DEVICE_ID` | adapter index (e.g., `0`) | Target a specific DXCore adapter |
| `OLLAMA_NPU_STRICT` | `0`, `1` | If `1`, fail with error when NPU is unavailable instead of falling back |

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

### QNN-Native Models (Snapdragon, recommended)

For Qualcomm Snapdragon devices, QNN-native models provide the best performance.
These contain pre-compiled HTP context binaries that run directly on the Hexagon NPU.

```powershell
$env:OLLAMA_ONNX_MODEL = "C:\models\phi3-mini-onnx-qnn"
$env:OLLAMA_ONNX_PROVIDER = "qnn"
$env:OLLAMA_ORT_QNN_BACKEND_TYPE = "htp"
$env:OLLAMA_ORT_QNN_BACKEND_PATH = "QnnHtp.dll"
$env:OLLAMA_ORT_PATH = "lib\ollama\ortgenai"
$env:OLLAMA_CONTEXT_LENGTH = "512"
.\ollama.exe serve
```

See `x/ortrunner/README.md` for full setup instructions, available QNN models, and benchmark results.

### GenAI-QNN Models

Models with the `genai-qnn` format are automatically routed to the ORT GenAI runner with the QNN execution provider, regardless of `OLLAMA_ACCEL_MODE`. This is for models specifically compiled for Qualcomm's QNN runtime.

## Mode 2: GGUF on DirectML

When `OLLAMA_ACCEL_MODE=npu` is set with a GGUF model (not ONNX), Ollama implicitly enables the DirectML backend (`OLLAMA_DIRECTML=1`). The ggml-directml backend will route operations to NPU or GPU based on the adapter priority established by DXCore.

## Troubleshooting

**`ollama debug npu` shows no NPU adapters:**
- Ensure you're on Windows 11 ARM64 with NPU drivers installed
- Check Device Manager for the NPU device
- Update to the latest Qualcomm or Intel NPU drivers

**NPU targeting has no effect:**
- Run the `npu_access_probe` tool in `_tools/` to verify your NPU has a working D3D12 driver
- For Snapdragon NPUs (Generic ML-only, no D3D12), use the QNN provider: `set OLLAMA_ONNX_PROVIDER=qnn`
- Run `ollama debug ortgenai` to verify provider support
