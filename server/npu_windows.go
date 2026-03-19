//go:build windows

package server

import (
	"fmt"
	"log/slog"
	"strconv"

	"github.com/ollama/ollama/discover/dxcore"
)

// NPUStatus describes the availability of an NPU on this system.
type NPUStatus struct {
	Available   bool
	AdapterIdx  int
	Description string
	HasDML      bool
	HasD3D12    bool
	GenericML   bool // True if Generic ML-only (no D3D12 device creation)
	// QNNCapable indicates the NPU may be usable via the QNN execution provider
	// (e.g., Snapdragon Hexagon NPU). This is true when an NPU is found even
	// without D3D12, because QNN uses a separate runtime path.
	QNNCapable bool
}

// ValidateNPU checks that an NPU is present and functional.
// If deviceID is non-empty, it looks for that specific adapter index.
func ValidateNPU(deviceID string) (NPUStatus, error) {
	result := dxcore.Probe()

	if !result.DXCoreAvailable {
		return NPUStatus{}, fmt.Errorf("DXCore not available: %s", result.Error)
	}

	if result.NPUCount == 0 {
		return NPUStatus{}, fmt.Errorf("no NPU adapters found on this system")
	}

	// If a specific device ID was requested, find that adapter
	if deviceID != "" {
		idx, err := strconv.Atoi(deviceID)
		if err != nil {
			return NPUStatus{}, fmt.Errorf("invalid NPU device ID %q: %w", deviceID, err)
		}
		for _, a := range result.Adapters {
			if a.Index == idx && a.IsNPU {
				isGenericML := a.HasD3D12GenericML && !a.HasD3D12
				status := NPUStatus{
					Available:   a.HasD3D12,
					AdapterIdx:  a.Index,
					Description: a.Description,
					HasDML:      a.HasDML,
					HasD3D12:    a.HasD3D12,
					GenericML:   isGenericML,
					QNNCapable:  true, // Any NPU adapter is potentially QNN-capable
				}
				if !a.HasD3D12 {
					return status, fmt.Errorf("NPU #%d (%s) found but D3D12 device creation failed (may still work via QNN provider)", idx, a.Description)
				}
				return status, nil
			}
		}
		return NPUStatus{}, fmt.Errorf("NPU adapter #%d not found", idx)
	}

	// Find the first usable NPU
	for _, a := range result.Adapters {
		if a.IsNPU {
			isGenericML := a.HasD3D12GenericML && !a.HasD3D12
			status := NPUStatus{
				Available:   a.HasD3D12,
				AdapterIdx:  a.Index,
				Description: a.Description,
				HasDML:      a.HasDML,
				HasD3D12:    a.HasD3D12,
				GenericML:   isGenericML,
				QNNCapable:  true, // Any NPU adapter is potentially QNN-capable
			}
			if a.HasD3D12 {
				slog.Info("NPU validated", "adapter", a.Index, "description", a.Description,
					"d3d12", a.HasD3D12, "dml", a.HasDML, "feature_level", a.FeatureLevel)
				return status, nil
			}
			// NPU found but not usable via D3D12 — may still be usable via QNN
			if a.HasD3D12GenericML {
				slog.Warn("NPU is Generic ML-only — not usable via ggml-directml, but may work via QNN EP or ONNX Runtime DML EP",
					"adapter", a.Index, "description", a.Description)
				return status, fmt.Errorf("NPU #%d (%s) is Generic ML-only — D3D12 device creation not supported; use QNN provider (set OLLAMA_ONNX_PROVIDER=qnn) for Snapdragon NPU",
					a.Index, a.Description)
			}
			return status, fmt.Errorf("NPU #%d (%s) found but D3D12 device creation failed (try QNN provider for Snapdragon devices)", a.Index, a.Description)
		}
	}

	return NPUStatus{}, fmt.Errorf("no NPU adapters found (unexpected)")
}

// ValidateNPUForD3D12 checks that an NPU is present and usable via the D3D12/DirectML path.
// This requires D3D12 device creation to succeed — it is NOT valid for QNN-only NPUs.
func ValidateNPUForD3D12(deviceID string) (NPUStatus, error) {
	status, err := ValidateNPU(deviceID)
	if err != nil {
		return status, err
	}
	if !status.HasD3D12 {
		return status, fmt.Errorf("NPU #%d (%s) found but D3D12 device creation failed — not usable for DirectML/ggml-directml path",
			status.AdapterIdx, status.Description)
	}
	return status, nil
}

// ValidateNPUForOrtQNN checks that an NPU is present and potentially usable via the
// ORT GenAI + QNN execution provider path. This does NOT require D3D12 — it only requires
// that an NPU adapter exists (QNN uses its own runtime, not D3D12).
func ValidateNPUForOrtQNN(deviceID string) (NPUStatus, error) {
	status, err := ValidateNPU(deviceID)
	// For QNN path, we only need the NPU to exist — D3D12 failure is expected and OK
	if err != nil && !status.QNNCapable {
		return status, err
	}
	if status.QNNCapable {
		return status, nil
	}
	return status, fmt.Errorf("no QNN-capable NPU found")
}
