package server

import (
	"crypto/sha256"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
)

// ortgenaiModelDirMu protects concurrent materialization of the same model directory.
var ortgenaiModelDirMu sync.Map // keyed by manifest digest

// materializeORTGenAIDir reconstructs an ORT GenAI model directory from manifest layers.
// It creates symlinks (or copies) from the blob store into a cache directory so that
// the ORT GenAI runner can load the model as a flat directory.
//
// Returns the path to the materialized directory.
func materializeORTGenAIDir(mf *manifest.Manifest) (string, error) {
	// Use the manifest digest as a stable directory key
	digestKey := mf.Digest()
	if digestKey == "" {
		// Fallback: hash the layer digests
		h := sha256.New()
		for _, layer := range mf.Layers {
			h.Write([]byte(layer.Digest))
		}
		digestKey = fmt.Sprintf("sha256:%x", h.Sum(nil))
	}

	// Per-model lock to prevent concurrent materialization races
	mu, _ := ortgenaiModelDirMu.LoadOrStore(digestKey, &sync.Mutex{})
	mu.(*sync.Mutex).Lock()
	defer mu.(*sync.Mutex).Unlock()

	// Cache dir: $OLLAMA_MODELS/ortgenai/<digest-short>/
	shortDigest := digestKey
	if len(shortDigest) > 20 {
		shortDigest = shortDigest[len(shortDigest)-16:]
	}
	cacheDir := filepath.Join(envconfig.Models(), "ortgenai", shortDigest)

	// Check if already materialized (idempotent)
	if isORTGenAIDirComplete(cacheDir, mf) {
		slog.Debug("ORT GenAI model directory already materialized", "dir", cacheDir)
		return cacheDir, nil
	}

	slog.Info("materializing ORT GenAI model directory", "dir", cacheDir)

	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", fmt.Errorf("create ORT GenAI cache dir: %w", err)
	}

	for _, layer := range mf.Layers {
		if layer.MediaType != MediaTypeORTGenAI {
			continue
		}

		filename := layer.Name
		if filename == "" {
			slog.Warn("ORT GenAI layer missing filename", "digest", layer.Digest)
			continue
		}

		blobPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return "", fmt.Errorf("resolve blob for %s: %w", filename, err)
		}

		destPath := filepath.Join(cacheDir, filename)
		if err := createLink(blobPath, destPath); err != nil {
			return "", fmt.Errorf("link %s: %w", filename, err)
		}
	}

	slog.Info("ORT GenAI model directory materialized", "dir", cacheDir)
	return cacheDir, nil
}

// isORTGenAIDirComplete checks if all expected ORT GenAI files are present.
func isORTGenAIDirComplete(dir string, mf *manifest.Manifest) bool {
	for _, layer := range mf.Layers {
		if layer.MediaType != MediaTypeORTGenAI {
			continue
		}
		if layer.Name == "" {
			continue
		}
		fi, err := os.Stat(filepath.Join(dir, layer.Name))
		if err != nil || fi.Size() != layer.Size {
			return false
		}
	}
	return true
}
