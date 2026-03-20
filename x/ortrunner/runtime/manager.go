package runtime

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
)

// Provider constants for execution provider selection.
const (
	ProviderAuto = "auto"
	ProviderQNN  = "qnn"
	ProviderDML  = "dml"
	ProviderCPU  = "cpu"
)

// InstallConfig controls runtime installation behavior.
type InstallConfig struct {
	Provider string // auto, qnn, dml, cpu
	Force    bool   // reinstall even if present
}

// InstallResult reports the outcome of a runtime installation.
type InstallResult struct {
	Dir       string // final installation directory
	Provider  string // resolved provider
	Installed bool   // true if new files were installed (false if already present)
	Validated bool   // true if validation passed
	Message   string // human-readable status
}

// DefaultInstallDir returns the standard per-user install directory for ORT GenAI runtimes.
// On Windows: %LOCALAPPDATA%\Ollama\runtimes\ortgenai\win-arm64-qnn\<genaiVer>_<ortVer>
func DefaultInstallDir() string {
	if runtime.GOOS != "windows" {
		return ""
	}

	localAppData := os.Getenv("LOCALAPPDATA")
	if localAppData == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		localAppData = filepath.Join(home, "AppData", "Local")
	}

	return filepath.Join(
		localAppData,
		"Ollama", "runtimes", "ortgenai",
		fmt.Sprintf("win-arm64-qnn"),
		fmt.Sprintf("%s_%s", PinnedGenAIVersion, PinnedORTVersion),
	)
}

// IsInstalled checks if a valid runtime installation exists at the default directory.
func IsInstalled() bool {
	dir := DefaultInstallDir()
	if dir == "" {
		return false
	}

	// Check for the key DLL
	if _, err := os.Stat(filepath.Join(dir, "onnxruntime-genai.dll")); err != nil {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, "onnxruntime.dll")); err != nil {
		return false
	}
	return true
}

// Install downloads and installs the ORT GenAI runtime bundle.
func Install(cfg InstallConfig) (*InstallResult, error) {
	if runtime.GOOS != "windows" || runtime.GOARCH != "arm64" {
		return nil, fmt.Errorf("ORT GenAI NPU runtime is only supported on Windows arm64")
	}

	provider := cfg.Provider
	if provider == "" || provider == ProviderAuto {
		provider = ProviderQNN
	}

	dir := DefaultInstallDir()
	if dir == "" {
		return nil, fmt.Errorf("unable to determine install directory (LOCALAPPDATA not set)")
	}

	// Check existing installation
	if !cfg.Force && IsInstalled() {
		slog.Info("ORT GenAI runtime already installed", "dir", dir)
		return &InstallResult{
			Dir:       dir,
			Provider:  provider,
			Installed: false,
			Validated: true,
			Message:   "Runtime already installed",
		}, nil
	}

	slog.Info("installing ORT GenAI runtime", "dir", dir, "provider", provider)

	// Atomic install: download to temp dir, validate, then rename to final
	tempDir, err := os.MkdirTemp(filepath.Dir(dir), ".ortgenai-install-*")
	if err != nil {
		// Parent dir may not exist yet — create it
		if mkErr := os.MkdirAll(filepath.Dir(dir), 0o755); mkErr != nil {
			return nil, fmt.Errorf("failed to create parent directory: %w", mkErr)
		}
		tempDir, err = os.MkdirTemp(filepath.Dir(dir), ".ortgenai-install-*")
		if err != nil {
			return nil, fmt.Errorf("failed to create temp directory: %w", err)
		}
	}
	defer os.RemoveAll(tempDir) // cleanup on failure

	// Download and extract NuGet packages
	for _, pkg := range PinnedPackages {
		slog.Info("downloading NuGet package", "id", pkg.ID, "version", pkg.Version)
		if err := DownloadAndExtractNuGet(pkg, tempDir); err != nil {
			return nil, fmt.Errorf("failed to install %s: %w", pkg.ID, err)
		}
	}

	// Validate the installation
	if err := validateInstallDir(tempDir); err != nil {
		return nil, fmt.Errorf("validation failed after extraction: %w", err)
	}

	// Atomic rename: remove existing dir and move temp into place
	if cfg.Force {
		os.RemoveAll(dir)
	}
	if err := os.Rename(tempDir, dir); err != nil {
		// On Windows, rename can fail if the target exists. Try remove+rename.
		os.RemoveAll(dir)
		if err := os.Rename(tempDir, dir); err != nil {
			return nil, fmt.Errorf("failed to finalize install directory: %w", err)
		}
	}

	slog.Info("ORT GenAI runtime installed successfully", "dir", dir)
	return &InstallResult{
		Dir:       dir,
		Provider:  provider,
		Installed: true,
		Validated: true,
		Message:   "Runtime installed successfully",
	}, nil
}

// validateInstallDir checks that the essential DLLs are present in a directory.
func validateInstallDir(dir string) error {
	required := []string{
		"onnxruntime.dll",
		"onnxruntime-genai.dll",
	}
	for _, dll := range required {
		if _, err := os.Stat(filepath.Join(dir, dll)); err != nil {
			return fmt.Errorf("required DLL missing: %s", dll)
		}
	}
	return nil
}
