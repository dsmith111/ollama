//go:build !windows

package server

import "fmt"

// NPUStatus describes the availability of an NPU on this system.
type NPUStatus struct {
	Available   bool
	AdapterIdx  int
	Description string
	HasDML      bool
	HasD3D12    bool
	GenericML   bool
}

// ValidateNPU is not available on non-Windows platforms.
func ValidateNPU(deviceID string) (NPUStatus, error) {
	return NPUStatus{}, fmt.Errorf("NPU support is only available on Windows")
}
