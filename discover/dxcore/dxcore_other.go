//go:build !windows

// Package dxcore provides NPU detection via DXCore adapter enumeration.
// On non-Windows platforms, all functions return empty/false results.
package dxcore

// AdapterInfo describes a single DXCore adapter (NPU, GPU, etc).
type AdapterInfo struct {
	Index             int
	Description       string
	IsNPU             bool
	IsGPU             bool
	IsIntegrated      bool
	IsHardware        bool
	HasD3D12          bool
	HasDML            bool
	HasD3D12Graphics  bool
	HasD3D12CoreComp  bool
	HasD3D12GenericML bool
	FeatureLevel      string
	DedicatedMemBytes uint64
	SharedMemBytes    uint64
	DriverVersion     uint64
	LUID              [8]byte
}

// DedicatedMemMB returns dedicated memory in megabytes.
func (a AdapterInfo) DedicatedMemMB() float64 { return 0 }

// SharedMemMB returns shared system memory in megabytes.
func (a AdapterInfo) SharedMemMB() float64 { return 0 }

// TypeString returns "NPU", "GPU", "iGPU", or "unknown".
func (a AdapterInfo) TypeString() string { return "unknown" }

// ProbeResult contains the full probe output.
type ProbeResult struct {
	DXCoreAvailable bool
	Adapters        []AdapterInfo
	NPUCount        int
	GPUCount        int
	Error           string
}

// String returns a human-readable summary.
func (r ProbeResult) String() string {
	return "DXCore: unavailable (not Windows)\n"
}

// Probe returns an empty result on non-Windows platforms.
func Probe() ProbeResult {
	return ProbeResult{Error: "DXCore only available on Windows"}
}

// HasUsableNPU always returns false on non-Windows.
func HasUsableNPU() bool { return false }

// HasAnyNPU always returns false on non-Windows.
func HasAnyNPU() bool { return false }

// NPUAdapters always returns nil on non-Windows.
func NPUAdapters() []AdapterInfo { return nil }
