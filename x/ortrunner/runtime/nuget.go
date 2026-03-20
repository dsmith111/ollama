package runtime

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// NuGetPackage describes a pinned NuGet package to download.
type NuGetPackage struct {
	ID      string // NuGet package ID
	Version string // Exact version
	SHA256  string // Expected SHA-256 hash of the .nupkg (empty to skip verification)
}

// Pinned known-good versions. Bump these for updates.
const (
	PinnedGenAIVersion = "0.12.2"
	PinnedORTVersion   = "1.24.1"
)

// PinnedPackages lists the NuGet packages needed for QNN NPU inference.
var PinnedPackages = []NuGetPackage{
	{
		ID:      "Microsoft.ML.OnnxRuntimeGenAI.QNN",
		Version: PinnedGenAIVersion,
	},
	{
		ID:      "Microsoft.ML.OnnxRuntime.QNN",
		Version: PinnedORTVersion,
	},
}

// nugetDownloadURL constructs the NuGet v3 flat-container download URL.
func nugetDownloadURL(pkg NuGetPackage) string {
	lower := strings.ToLower(pkg.ID)
	return fmt.Sprintf("https://api.nuget.org/v3-flatcontainer/%s/%s/%s.%s.nupkg",
		lower, pkg.Version, lower, pkg.Version)
}

// DownloadAndExtractNuGet downloads a NuGet package and extracts the
// runtimes/win-arm64/native/** tree into destDir.
// It skips .lib and .pdb files, and verifies the SHA256 hash if provided.
func DownloadAndExtractNuGet(pkg NuGetPackage, destDir string) error {
	url := nugetDownloadURL(pkg)
	slog.Info("downloading NuGet package", "url", url)

	// Download to a temp file
	tmpFile, err := os.CreateTemp("", "ollama-nupkg-*.zip")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	resp, err := http.Get(url)
	if err != nil {
		tmpFile.Close()
		return fmt.Errorf("download %s: %w", pkg.ID, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		tmpFile.Close()
		return fmt.Errorf("download %s: HTTP %d", pkg.ID, resp.StatusCode)
	}

	// Hash while downloading
	hasher := sha256.New()
	writer := io.MultiWriter(tmpFile, hasher)
	if _, err := io.Copy(writer, resp.Body); err != nil {
		tmpFile.Close()
		return fmt.Errorf("download %s: %w", pkg.ID, err)
	}
	tmpFile.Close()

	// Verify hash if provided
	if pkg.SHA256 != "" {
		actual := hex.EncodeToString(hasher.Sum(nil))
		if !strings.EqualFold(actual, pkg.SHA256) {
			return fmt.Errorf("SHA256 mismatch for %s: expected %s, got %s", pkg.ID, pkg.SHA256, actual)
		}
		slog.Info("SHA256 verified", "package", pkg.ID)
	}

	// Extract runtimes/win-arm64/native/** from the nupkg (which is a ZIP)
	return extractNativeFiles(tmpPath, destDir)
}

// extractNativeFiles opens a .nupkg (ZIP) and extracts files from
// runtimes/win-arm64/native/ into destDir, skipping .lib and .pdb files.
func extractNativeFiles(zipPath, destDir string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return fmt.Errorf("open nupkg: %w", err)
	}
	defer r.Close()

	const prefix = "runtimes/win-arm64/native/"
	extracted := 0

	for _, f := range r.File {
		// Normalize to forward slashes for comparison
		name := strings.ReplaceAll(f.Name, "\\", "/")

		if !strings.HasPrefix(name, prefix) {
			continue
		}

		// Get the relative path within the native directory
		relPath := strings.TrimPrefix(name, prefix)
		if relPath == "" || strings.HasSuffix(relPath, "/") {
			continue // skip directory entries
		}

		// Skip .lib and .pdb files
		ext := strings.ToLower(filepath.Ext(relPath))
		if ext == ".lib" || ext == ".pdb" {
			slog.Debug("skipping non-runtime file", "file", relPath)
			continue
		}

		// Security: prevent path traversal
		destPath := filepath.Join(destDir, filepath.FromSlash(relPath))
		if !strings.HasPrefix(filepath.Clean(destPath), filepath.Clean(destDir)+string(filepath.Separator)) {
			slog.Warn("skipping file with suspicious path", "file", relPath)
			continue
		}

		// Create parent directories
		if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
			return fmt.Errorf("mkdir for %s: %w", relPath, err)
		}

		if err := extractZipFile(f, destPath); err != nil {
			return fmt.Errorf("extract %s: %w", relPath, err)
		}

		slog.Debug("extracted", "file", relPath)
		extracted++
	}

	if extracted == 0 {
		return fmt.Errorf("no native files found in package (expected files under %s)", prefix)
	}

	slog.Info("extracted native files", "count", extracted)
	return nil
}

// extractZipFile writes a single ZIP file entry to disk.
func extractZipFile(f *zip.File, destPath string) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()

	out, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer out.Close()

	// Limit extraction size to 500MB per file as a safety measure
	const maxFileSize = 500 << 20
	if _, err := io.Copy(out, io.LimitReader(rc, maxFileSize)); err != nil {
		return err
	}

	return nil
}
