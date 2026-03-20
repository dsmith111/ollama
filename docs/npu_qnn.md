# ORT GenAI / QNN Model Support

Ollama supports packaging and running ORT GenAI models natively. This enables
Snapdragon NPU inference via the QNN execution provider without manual DLL
copying or model folder management.

## Model Format

ORT GenAI models use `model_format: "ortgenai"` in their Ollama config. The
model directory typically contains:

```
model-dir/
├── genai_config.json          # ORT GenAI configuration (required)
├── embeddings.onnx            # ONNX graph files (at least one required)
├── lm_head.onnx
├── context_ctx.onnx           # QNN context binary caches (optional)
├── iterator_ctx.onnx
├── tokenizer.json             # Tokenizer (at least one required)
├── tokenizer.model            # SentencePiece tokenizer (alternative)
├── tokenizer_config.json      # Tokenizer config
├── special_tokens_map.json    # Special tokens
└── chat_template.jinja        # Chat template (optional)
```

### Required Files

- At least one `*.onnx` file (model graphs)
- `genai_config.json` (ORT GenAI runtime configuration)
- `tokenizer.json` or `tokenizer.model` (tokenizer)

## Creating a Model

Given an ORT GenAI model directory, create a `Modelfile`:

```
FROM ./path/to/onnx-model-dir
```

Then run:

```sh
ollama create myuser/phi3:mini-qnn -f Modelfile
```

Ollama automatically detects the ORT GenAI format when it finds `.onnx` files
alongside `genai_config.json`. All files are stored as blobs in the standard
Ollama model store.

## Pushing and Pulling

Once created, the model can be pushed and pulled like any other Ollama model:

```sh
ollama push myuser/phi3:mini-qnn
ollama pull myuser/phi3:mini-qnn
```

When pulled on another machine, Ollama automatically reconstructs the model
directory from the stored blob layers. No manual file management is needed.

## Running

```sh
ollama run myuser/phi3:mini-qnn
```

The scheduler automatically routes `ortgenai` models to the ORT GenAI runner.
Ensure the ORT GenAI runtime is installed:

```sh
ollama npu setup
```

This downloads and installs the required ONNX Runtime and GenAI DLLs for
Snapdragon QNN/HTP inference.

## Runtime Dependencies

ORT GenAI NPU inference requires:

- **Windows arm64** (Snapdragon)
- ORT GenAI runtime DLLs (installed via `ollama npu setup`)
- QNN SDK transport libraries (typically pre-installed on Snapdragon devices)

### Diagnostics

```sh
ollama npu doctor    # Full health check
ollama debug npu     # NPU adapter information
ollama debug ortgenai # ORT GenAI DLL diagnostics
```

## Benchmarking

Compare CPU vs NPU performance:

```sh
ollama benchmark npu --model myuser/phi3:mini-qnn --ctx 512 --predict 128
```

## Technical Details

### Blob Storage

Each file in the ORT GenAI directory is stored as a separate blob in the Ollama
model store with media type `application/vnd.ollama.image.ortgenai`. The
original filename is preserved in the layer's `name` field.

### Directory Materialization

When loading an ortgenai model, Ollama reconstructs the model directory by
symlinking (or copying) blobs into a cache directory at:

```
$OLLAMA_MODELS/ortgenai/<digest>/
```

This directory is passed to the ORT GenAI runner as the model path.

### Model Config

The `ConfigV2` for ortgenai models sets:

- `model_format: "ortgenai"`
- `file_type: "onnx"`
- `capabilities: ["completion"]`
