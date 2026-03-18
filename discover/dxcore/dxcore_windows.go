//go:build windows

// Package dxcore provides NPU detection via DXCore adapter enumeration on Windows.
// It dynamically loads dxcore.dll, d3d12.dll, and DirectML.dll to probe for
// NPU and GPU adapters, testing D3D12 and DirectML device creation.
package dxcore

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ld3d12 -ldxgi -lole32
// #include "dxcore_probe.h"
import "C"

import (
	"fmt"
	"strings"
	"unsafe"
)

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
func (a AdapterInfo) DedicatedMemMB() float64 {
	return float64(a.DedicatedMemBytes) / (1024 * 1024)
}

// SharedMemMB returns shared system memory in megabytes.
func (a AdapterInfo) SharedMemMB() float64 {
	return float64(a.SharedMemBytes) / (1024 * 1024)
}

// TypeString returns "NPU", "GPU", "iGPU", or "unknown".
func (a AdapterInfo) TypeString() string {
	if a.IsNPU {
		return "NPU"
	}
	if a.IsGPU {
		if a.IsIntegrated {
			return "iGPU"
		}
		return "GPU"
	}
	return "unknown"
}

// ProbeResult contains the full probe output.
type ProbeResult struct {
	DXCoreAvailable bool
	Adapters        []AdapterInfo
	NPUCount        int
	GPUCount        int
	Error           string
}

// String returns a human-readable summary of the probe result.
func (r ProbeResult) String() string {
	var b strings.Builder
	if r.Error != "" {
		fmt.Fprintf(&b, "DXCore: unavailable (%s)\n", r.Error)
		return b.String()
	}
	fmt.Fprintf(&b, "DXCore: available\n")
	fmt.Fprintf(&b, "Adapters:\n")
	for _, a := range r.Adapters {
		d3d12Str := "no"
		if a.HasD3D12 {
			d3d12Str = "yes"
		}
		dmlStr := "no"
		if a.HasDML {
			dmlStr = "yes"
		}
		mem := a.SharedMemMB()
		memLabel := "SharedMem"
		if a.DedicatedMemBytes > 0 {
			mem = a.DedicatedMemMB()
			memLabel = "DedicatedMem"
		}
		fmt.Fprintf(&b, "  #%d  %s (%s)  %s=%.0fMB  D3D12=%s  DML=%s  FL=%s\n",
			a.Index, a.Description, a.TypeString(),
			memLabel, mem, d3d12Str, dmlStr, a.FeatureLevel)
	}
	fmt.Fprintf(&b, "Summary: %d NPU(s), %d GPU(s)\n", r.NPUCount, r.GPUCount)
	return b.String()
}

// Probe enumerates DXCore adapters and tests D3D12/DML device creation.
func Probe() ProbeResult {
	var cResult C.DxcoreProbeResult
	C.dxcore_probe(&cResult)

	result := ProbeResult{
		DXCoreAvailable: cResult.dxcore_available != 0,
		NPUCount:        int(cResult.npu_count),
		GPUCount:        int(cResult.gpu_count),
		Error:           C.GoString(&cResult.error[0]),
	}

	for i := 0; i < int(cResult.adapter_count); i++ {
		ca := cResult.adapters[i]
		a := AdapterInfo{
			Index:             int(ca.index),
			Description:       C.GoString(&ca.description[0]),
			IsNPU:             ca.is_npu != 0,
			IsGPU:             ca.is_gpu != 0,
			IsIntegrated:      ca.is_integrated != 0,
			IsHardware:        ca.is_hardware != 0,
			HasD3D12:          ca.has_d3d12 != 0,
			HasDML:            ca.has_dml != 0,
			HasD3D12Graphics:  ca.has_d3d12_graphics != 0,
			HasD3D12CoreComp:  ca.has_d3d12_core_compute != 0,
			HasD3D12GenericML: ca.has_d3d12_generic_ml != 0,
			FeatureLevel:      C.GoString(&ca.feature_level[0]),
			DedicatedMemBytes: uint64(ca.dedicated_mem_bytes),
			SharedMemBytes:    uint64(ca.shared_mem_bytes),
			DriverVersion:     uint64(ca.driver_version),
		}
		for j := 0; j < 8; j++ {
			a.LUID[j] = byte(*(*C.uint8_t)(unsafe.Pointer(uintptr(unsafe.Pointer(&ca.luid[0])) + uintptr(j))))
		}
		result.Adapters = append(result.Adapters, a)
	}

	return result
}

// HasUsableNPU returns true if at least one NPU with D3D12 support is present.
func HasUsableNPU() bool {
	r := Probe()
	for _, a := range r.Adapters {
		if a.IsNPU && a.HasD3D12 {
			return true
		}
	}
	return false
}

// HasAnyNPU returns true if at least one NPU adapter is enumerated (even without D3D12).
func HasAnyNPU() bool {
	r := Probe()
	return r.NPUCount > 0
}

// NPUAdapters returns only the NPU adapters from the probe.
func NPUAdapters() []AdapterInfo {
	r := Probe()
	var npus []AdapterInfo
	for _, a := range r.Adapters {
		if a.IsNPU {
			npus = append(npus, a)
		}
	}
	return npus
}
