package ortrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"time"
)

// DraftClient manages a draft model ORT GenAI runner subprocess for speculative decoding.
// It communicates with the draft runner via HTTP on localhost using the
// /internal/draft/* endpoints.
type DraftClient struct {
	port   int
	client *http.Client
	inner  *Client // reuses Client for subprocess lifecycle
	mu     sync.Mutex
}

// NewDraftClient spawns a new ORT GenAI runner subprocess configured for NPU
// draft model inference. The model at modelDir should be a small ONNX model
// suitable for speculative draft generation.
func NewDraftClient(modelDir string) (*DraftClient, error) {
	// Determine device targeting — draft models run on NPU
	deviceType := "npu"
	deviceID := os.Getenv("OLLAMA_NPU_DEVICE_ID")
	provider := os.Getenv("OLLAMA_ONNX_PROVIDER")
	if provider == "" {
		// Auto-detect best provider for NPU draft models
		provider = detectBestProvider()
	}

	inner, err := NewClientWithOpts(ClientOptions{
		ModelDir:   modelDir,
		DeviceType: deviceType,
		DeviceID:   deviceID,
		Provider:   provider,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to start draft model runner: %w", err)
	}

	slog.Info("draft model runner started",
		"model", modelDir, "port", inner.port,
		"device_type", deviceType, "provider", provider)

	return &DraftClient{
		port:   inner.port,
		client: &http.Client{Timeout: 30 * time.Second},
		inner:  inner,
	}, nil
}

// Init initializes the draft session with the given prompt text.
func (d *DraftClient) Init(promptText string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	body, _ := json.Marshal(struct {
		Prompt string `json:"prompt"`
	}{Prompt: promptText})

	return d.post("/internal/draft/init", body, nil)
}

// Propose asks the draft model to generate up to k tokens greedily.
// Returns the proposed token IDs.
func (d *DraftClient) Propose(k int) ([]int32, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	body, _ := json.Marshal(struct {
		K int `json:"k"`
	}{K: k})

	var resp struct {
		Tokens []int32 `json:"tokens"`
		Count  int     `json:"count"`
	}
	if err := d.post("/internal/draft/propose", body, &resp); err != nil {
		return nil, err
	}
	return resp.Tokens, nil
}

// Accept notifies the draft model how many of the proposed tokens were accepted,
// and optionally provides a correction token (the main model's choice at the
// mismatch point).
func (d *DraftClient) Accept(acceptedCount, lastProposedCount int, correctionToken int32) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	body, _ := json.Marshal(struct {
		AcceptedCount     int   `json:"accepted_count"`
		LastProposedCount int   `json:"last_proposed_count"`
		CorrectionToken   int32 `json:"correction_token,omitempty"`
	}{
		AcceptedCount:     acceptedCount,
		LastProposedCount: lastProposedCount,
		CorrectionToken:   correctionToken,
	})

	return d.post("/internal/draft/accept", body, nil)
}

// Tokenize tokenizes text using the draft model's tokenizer.
func (d *DraftClient) Tokenize(ctx context.Context, text string) ([]int, error) {
	return d.inner.Tokenize(ctx, text)
}

// Close terminates the draft model subprocess.
func (d *DraftClient) Close() error {
	return d.inner.Close()
}

// HasExited returns true if the draft model subprocess has exited.
func (d *DraftClient) HasExited() bool {
	return d.inner.HasExited()
}

// post sends a JSON POST to the draft runner and optionally decodes the response.
func (d *DraftClient) post(path string, body []byte, result any) error {
	url := fmt.Sprintf("http://127.0.0.1:%d%s", d.port, path)
	req, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(req)
	if err != nil {
		return fmt.Errorf("draft runner request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("draft runner error (%d): %s", resp.StatusCode, string(respBody))
	}

	if result != nil {
		return json.NewDecoder(resp.Body).Decode(result)
	}
	return nil
}
