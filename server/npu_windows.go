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
				status := NPUStatus{
					Available:   a.HasD3D12,
					AdapterIdx:  a.Index,
					Description: a.Description,
					HasDML:      a.HasDML,
					HasD3D12:    a.HasD3D12,
					GenericML:   a.HasD3D12GenericML && !a.HasD3D12,
				}
				if !a.HasD3D12 {
					return status, fmt.Errorf("NPU #%d (%s) found but D3D12 device creation failed", idx, a.Description)
				}
				return status, nil
			}
		}
		return NPUStatus{}, fmt.Errorf("NPU adapter #%d not found", idx)
	}

	// Find the first usable NPU
	for _, a := range result.Adapters {
		if a.IsNPU {
			status := NPUStatus{
				Available:   a.HasD3D12,
				AdapterIdx:  a.Index,
				Description: a.Description,
				HasDML:      a.HasDML,
				HasD3D12:    a.HasD3D12,
				GenericML:   a.HasD3D12GenericML && !a.HasD3D12,
			}
			if a.HasD3D12 {
				slog.Info("NPU validated", "adapter", a.Index, "description", a.Description,
					"d3d12", a.HasD3D12, "dml", a.HasDML, "feature_level", a.FeatureLevel)
				return status, nil
			}
			// NPU found but not usable via D3D12
			if a.HasD3D12GenericML {
				slog.Warn("NPU is Generic ML-only, may work via ONNX Runtime DML EP but not ggml-directml",
					"adapter", a.Index, "description", a.Description)
				return status, fmt.Errorf("NPU #%d (%s) is Generic ML-only — D3D12 device creation not supported; may work via ONNX Runtime DML EP",
					a.Index, a.Description)
			}
			return status, fmt.Errorf("NPU #%d (%s) found but D3D12 device creation failed", a.Index, a.Description)
		}
	}

	return NPUStatus{}, fmt.Errorf("no NPU adapters found (unexpected)")
}
