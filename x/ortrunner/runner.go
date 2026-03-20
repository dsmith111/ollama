package ortrunner

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/ortrunner/oga"
)

// Runner manages ORT GenAI model loading and inference.
type Runner struct {
	model     *oga.Model
	tokenizer *oga.Tokenizer
	config    *oga.Config
	modelDir  string

	Requests chan Request
}

// Request represents a completion request.
type Request struct {
	TextCompletionsRequest
	Responses chan CompletionResponse
	Ctx       context.Context
}

// TextCompletionsRequest is the JSON body for /v1/completions.
type TextCompletionsRequest struct {
	Prompt  string `json:"prompt"`
	Options struct {
		Temperature float32 `json:"temperature"`
		TopP        float32 `json:"top_p"`
		TopK        int     `json:"top_k"`
		MaxTokens   int     `json:"max_tokens"`
		NumPredict  int     `json:"num_predict"`
		NumCtx      int     `json:"num_ctx"`
	} `json:"options"`
}

// CompletionResponse is a single JSONL line streamed back to the client.
type CompletionResponse struct {
	Content    string           `json:"content,omitempty"`
	Done       bool             `json:"done"`
	DoneReason int              `json:"done_reason,omitempty"`
	Error      *api.StatusError `json:"error,omitempty"`

	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

// Load loads the ORT GenAI model from a directory.
func (r *Runner) Load(modelDir string) error {
	r.modelDir = modelDir

	cfg, err := oga.NewConfig(modelDir)
	if err != nil {
		return err
	}
	r.config = cfg

	// Configure execution provider based on environment
	provider := os.Getenv("OLLAMA_ONNX_PROVIDER")
	if provider == "" {
		if os.Getenv("OLLAMA_ONNX_NPU") == "1" {
			provider = "qnn"
		} else {
			// Auto-detect: prefer QNN if QNN provider DLL exists, else DML, else CPU
			provider = detectBestProvider()
		}
	}

	// QNN EP options from environment
	qnnBackendType := os.Getenv("OLLAMA_ORT_QNN_BACKEND_TYPE")
	if qnnBackendType == "" {
		qnnBackendType = "htp" // default: Hexagon Tensor Processor (NPU)
	}
	qnnBackendPath := os.Getenv("OLLAMA_ORT_QNN_BACKEND_PATH") // e.g. "QnnHtp.dll" or absolute path
	qnnHTPArch := os.Getenv("OLLAMA_ORT_QNN_HTP_ARCH")         // e.g. "73" for v73; empty = auto
	qnnSocModel := os.Getenv("OLLAMA_ORT_QNN_SOC_MODEL")       // e.g. "60" for Snapdragon X Elite; empty = auto

	// Device targeting: OLLAMA_ORT_DEVICE_TYPE=npu|gpu, OLLAMA_ORT_DEVICE_ID=<index>
	deviceType := os.Getenv("OLLAMA_ORT_DEVICE_TYPE")
	deviceID := os.Getenv("OLLAMA_ORT_DEVICE_ID")

	slog.Info("configuring ORT GenAI execution provider", "provider", provider, "device_type", deviceType, "device_id", deviceID,
		"qnn_backend_type", qnnBackendType, "qnn_backend_path", qnnBackendPath,
		"qnn_htp_arch", qnnHTPArch, "qnn_soc_model", qnnSocModel)

	if err := cfg.ClearProviders(); err != nil {
		slog.Warn("failed to clear providers, using defaults", "error", err)
	} else {
		switch provider {
		case "qnn":
			if err := cfg.AppendProvider("QNN"); err != nil {
				return err
			}
			// QNN EP requires exactly one of backend_type or backend_path, not both.
			if qnnBackendPath != "" {
				if err := cfg.SetProviderOption("QNN", "backend_path", qnnBackendPath); err != nil {
					slog.Warn("failed to set QNN backend_path", "error", err)
				}
			} else {
				if err := cfg.SetProviderOption("QNN", "backend_type", qnnBackendType); err != nil {
					slog.Warn("failed to set QNN backend_type", "error", err)
				}
			}
			if qnnHTPArch != "" {
				if err := cfg.SetProviderOption("QNN", "htp_arch", qnnHTPArch); err != nil {
					slog.Warn("failed to set QNN htp_arch", "error", err)
				}
			}
			if qnnSocModel != "" {
				if err := cfg.SetProviderOption("QNN", "soc_model", qnnSocModel); err != nil {
					slog.Warn("failed to set QNN soc_model", "error", err)
				}
			}
		case "dml":
			if err := cfg.AppendProvider("dml"); err != nil {
				return err
			}
			// Apply device targeting options for DML
			if deviceID != "" {
				// Explicit device index takes priority (DXGI enumeration order)
				if err := cfg.SetProviderOption("dml", "device_id", deviceID); err != nil {
					slog.Warn("failed to set DML device_id", "error", err)
				}
			} else if deviceType != "" {
				// Use DXCore device_filter for type-based targeting
				switch deviceType {
				case "npu", "NPU":
					if err := cfg.SetProviderOption("dml", "device_filter", "npu"); err != nil {
						slog.Warn("failed to set DML device_filter=npu, trying performance_preference", "error", err)
						// Fallback: prefer minimum power (NPU > GPU in sort order)
						if err := cfg.SetProviderOption("dml", "performance_preference", "minimum_power"); err != nil {
							slog.Warn("failed to set DML performance_preference", "error", err)
						}
					}
				case "gpu", "GPU":
					if err := cfg.SetProviderOption("dml", "device_filter", "gpu"); err != nil {
						slog.Warn("failed to set DML device_filter=gpu", "error", err)
					}
				default:
					slog.Warn("unknown OLLAMA_ORT_DEVICE_TYPE, ignoring", "device_type", deviceType)
				}
			}
		case "cpu":
			// No provider needed, CPU is the default fallback
		default:
			if err := cfg.AppendProvider(provider); err != nil {
				return err
			}
		}
	}

	slog.Info("loading ORT GenAI model", "dir", modelDir)
	model, err := oga.NewModel(cfg)
	if err != nil {
		return classifyLoadError(err, provider)
	}
	r.model = model

	tok, err := oga.NewTokenizer(model)
	if err != nil {
		return err
	}
	r.tokenizer = tok

	slog.Info("ORT GenAI model loaded successfully")
	return nil
}

// detectBestProvider checks for provider DLL availability and returns the best
// provider string. For Snapdragon/QNN NPU: "qnn"; for DirectML GPU: "dml"; fallback: "cpu".
func detectBestProvider() string {
	// Build list of directories to search for provider DLLs
	var searchDirs []string
	if ortPath, ok := os.LookupEnv("OLLAMA_ORT_PATH"); ok {
		searchDirs = append(searchDirs, filepath.SplitList(ortPath)...)
	}
	// Include the default runtime install directory
	if installDir := DefaultRuntimeInstallDir(); installDir != "" {
		searchDirs = append(searchDirs, installDir)
	}
	if exe, err := os.Executable(); err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		searchDirs = append(searchDirs, filepath.Dir(exe))
	}

	hasQNN := false
	hasDML := false
	for _, dir := range searchDirs {
		if !hasQNN {
			if _, err := os.Stat(filepath.Join(dir, "onnxruntime_providers_qnn.dll")); err == nil {
				hasQNN = true
			}
		}
		if !hasDML {
			// DML support is built into onnxruntime.dll for DML builds; check for DirectML.dll
			if _, err := os.Stat(filepath.Join(dir, "DirectML.dll")); err == nil {
				hasDML = true
			}
		}
	}

	if hasQNN {
		slog.Info("auto-detected QNN provider DLLs, selecting provider=qnn")
		return "qnn"
	}
	if hasDML {
		return "dml"
	}
	// No provider DLLs found — use CPU fallback rather than assuming DML is built in.
	// This avoids NPU_PROVIDER_MISMATCH errors with QNN-only builds.
	slog.Info("no QNN or DML provider DLLs found, defaulting to provider=cpu")
	return "cpu"
}

// classifyLoadError examines an ORT GenAI model load error and returns a more
// actionable error message keyed to known failure signatures.
func classifyLoadError(err error, provider string) error {
	msg := err.Error()

	switch {
	case strings.Contains(msg, "DML provider requested") && strings.Contains(msg, "not been built with DML support"):
		return fmt.Errorf("NPU_PROVIDER_MISMATCH: provider=%s but your onnxruntime-genai.dll was not built with DML support. "+
			"Either install a DML-enabled GenAI build, or set OLLAMA_ONNX_PROVIDER=qnn for Snapdragon NPU: %w", provider, err)

	case strings.Contains(msg, "openSession") || strings.Contains(msg, "0x80000406") || strings.Contains(msg, "DspTransport"):
		missing := checkMissingQNNDLLs()
		hint := "QNN HTP transport failed to open a DSP session. "
		if missing != "" {
			hint += "Missing DLLs in OLLAMA_ORT_PATH: " + missing + ". "
		}
		hint += "Check: (1) all QNN runtime DLLs are present, (2) DSP RPC transport DLLs are available, " +
			"(3) try setting OLLAMA_ORT_QNN_HTP_ARCH and OLLAMA_ORT_QNN_SOC_MODEL explicitly"
		return fmt.Errorf("NPU_QNN_TRANSPORT_OPENSESSION_FAILED: %s: %w", hint, err)

	case strings.Contains(msg, "Unable to load lib"):
		missing := checkMissingQNNDLLs()
		hint := "QNN backend library load failed. "
		if missing != "" {
			hint += "Missing DLLs: " + missing + ". "
		}
		hint += "Ensure all QNN/HTP DLLs are in OLLAMA_ORT_PATH and that directory is first in PATH"
		return fmt.Errorf("NPU_QNN_LIBRARY_LOAD_FAILED: %s: %w", hint, err)

	case strings.Contains(msg, "requested API version") && strings.Contains(msg, "not available"):
		return fmt.Errorf("ORT_API_VERSION_MISMATCH: onnxruntime-genai.dll and onnxruntime.dll versions are incompatible. "+
			"Ensure both DLLs come from the same NuGet package and OLLAMA_ORT_PATH is first in PATH: %w", err)
	}

	return err
}

// checkMissingQNNDLLs scans the ORT directory for expected QNN runtime DLLs
// and returns a comma-separated list of missing ones.
func checkMissingQNNDLLs() string {
	searchDirs := ortSearchDirs()

	// Expected DLLs for a complete QNN HTP runtime
	expected := []string{
		"onnxruntime.dll",
		"onnxruntime-genai.dll",
		"onnxruntime_providers_shared.dll",
		"onnxruntime_providers_qnn.dll",
		"QnnHtp.dll",
		"QnnSystem.dll",
		"QnnHtpPrepare.dll",
	}

	var missing []string
	for _, dll := range expected {
		found := false
		for _, dir := range searchDirs {
			if _, err := os.Stat(filepath.Join(dir, dll)); err == nil {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, dll)
		}
	}

	return strings.Join(missing, ", ")
}

// ortSearchDirs returns the list of directories to scan for ORT/QNN DLLs.
func ortSearchDirs() []string {
	var dirs []string
	if ortPath, ok := os.LookupEnv("OLLAMA_ORT_PATH"); ok {
		dirs = append(dirs, filepath.SplitList(ortPath)...)
	}
	// Include the default runtime install directory
	if installDir := DefaultRuntimeInstallDir(); installDir != "" {
		dirs = append(dirs, installDir)
	}
	if exe, err := os.Executable(); err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		dirs = append(dirs, filepath.Dir(exe))
	}
	return dirs
}

// DefaultRuntimeInstallDir returns the standard per-user install directory
// for the ORT GenAI runtime bundle on Windows.
func DefaultRuntimeInstallDir() string {
	if runtime.GOOS != "windows" {
		return ""
	}
	localAppData := os.Getenv("LOCALAPPDATA")
	if localAppData == "" {
		return ""
	}
	return filepath.Join(localAppData, "Ollama", "runtimes", "ortgenai", "win-arm64-qnn", "0.12.2_1.24.1")
}

// ORTDirReport describes the contents and health of an ORT runtime directory.
type ORTDirReport struct {
	SearchDirs    []string
	FoundDLLs     []string
	MissingDLLs   []string
	HasQNN        bool
	HasDML        bool
	HasTransport  bool // cdsprpc / DSP transport DLL found
	Provider      string
	BackendType   string
	BackendPath   string
	HTPArch       string
	SocModel      string
}

// ValidateORTDir inspects the ORT runtime directory and returns a diagnostic report.
func ValidateORTDir() ORTDirReport {
	dirs := ortSearchDirs()

	report := ORTDirReport{
		SearchDirs:  dirs,
		Provider:    os.Getenv("OLLAMA_ONNX_PROVIDER"),
		BackendType: os.Getenv("OLLAMA_ORT_QNN_BACKEND_TYPE"),
		BackendPath: os.Getenv("OLLAMA_ORT_QNN_BACKEND_PATH"),
		HTPArch:     os.Getenv("OLLAMA_ORT_QNN_HTP_ARCH"),
		SocModel:    os.Getenv("OLLAMA_ORT_QNN_SOC_MODEL"),
	}

	// All DLLs we check for
	allDLLs := []string{
		"onnxruntime.dll",
		"onnxruntime-genai.dll",
		"onnxruntime_providers_shared.dll",
		"onnxruntime_providers_qnn.dll",
		"QnnHtp.dll",
		"QnnSystem.dll",
		"QnnHtpPrepare.dll",
		"DirectML.dll",
	}

	for _, dll := range allDLLs {
		found := false
		for _, dir := range dirs {
			if _, err := os.Stat(filepath.Join(dir, dll)); err == nil {
				found = true
				report.FoundDLLs = append(report.FoundDLLs, dll)
				break
			}
		}
		if !found {
			report.MissingDLLs = append(report.MissingDLLs, dll)
		}
	}

	// Check for transport/RPC DLLs (glob for cdsprpc patterns)
	for _, dir := range dirs {
		matches, _ := filepath.Glob(filepath.Join(dir, "*cdsprpc*"))
		if len(matches) > 0 {
			report.HasTransport = true
			for _, m := range matches {
				report.FoundDLLs = append(report.FoundDLLs, filepath.Base(m))
			}
		}
		// Also check for QnnHtpV*Stub.dll
		stubs, _ := filepath.Glob(filepath.Join(dir, "QnnHtpV*Stub.dll"))
		for _, s := range stubs {
			report.FoundDLLs = append(report.FoundDLLs, filepath.Base(s))
		}
	}

	for _, dll := range report.FoundDLLs {
		switch {
		case dll == "onnxruntime_providers_qnn.dll":
			report.HasQNN = true
		case dll == "DirectML.dll":
			report.HasDML = true
		}
	}

	return report
}

// Close frees all ORT GenAI resources.
func (r *Runner) Close() {
	if r.tokenizer != nil {
		r.tokenizer.Close()
	}
	if r.model != nil {
		r.model.Close()
	}
	if r.config != nil {
		r.config.Close()
	}
}
