//go:build windows

package cmd

import (
	"fmt"

	"github.com/ollama/ollama/discover/dxcore"
	"github.com/spf13/cobra"
)

// DebugNPUHandler runs the DXCore NPU probe and prints results.
func DebugNPUHandler(cmd *cobra.Command, args []string) error {
	result := dxcore.Probe()
	fmt.Print(result.String())

	if result.Error != "" {
		return nil
	}

	// Summary of usability
	for _, a := range result.Adapters {
		if a.IsNPU && !a.HasD3D12 {
			if a.HasD3D12GenericML {
				fmt.Printf("\nNote: NPU #%d (%s) supports Generic ML but D3D12 device creation failed.\n",
					a.Index, a.Description)
				fmt.Println("  This NPU may be usable via ONNX Runtime DML EP but not via ggml-directml.")
			} else {
				fmt.Printf("\nWarning: NPU #%d (%s) was enumerated but D3D12 device creation failed.\n",
					a.Index, a.Description)
				fmt.Println("  This NPU is not usable for DirectML-based inference.")
			}
		}
		if a.IsNPU && a.HasD3D12 && !a.HasDML {
			fmt.Printf("\nWarning: NPU #%d (%s) has D3D12 but DirectML device creation failed.\n",
				a.Index, a.Description)
			fmt.Println("  Ensure DirectML.dll is available.")
		}
		if a.IsNPU && a.HasD3D12 && a.HasDML {
			fmt.Printf("\nNPU #%d (%s) is fully usable for DirectML inference.\n",
				a.Index, a.Description)
		}
	}

	if result.NPUCount == 0 {
		fmt.Println("\nNo NPU adapters found. NPU acceleration is not available on this system.")
	}

	return nil
}
