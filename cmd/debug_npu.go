//go:build windows

package cmd

import (
	"fmt"
	"os"
	"path/filepath"

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
				fmt.Println("  For Snapdragon NPUs, use the QNN execution provider path instead:")
				fmt.Println("    set OLLAMA_ONNX_PROVIDER=qnn")
				fmt.Println("    set OLLAMA_ORT_QNN_BACKEND_TYPE=htp")
			} else {
				fmt.Printf("\nWarning: NPU #%d (%s) was enumerated but D3D12 device creation failed.\n",
					a.Index, a.Description)
				fmt.Println("  This NPU is not usable for DirectML-based inference.")
				fmt.Println("  If this is a Snapdragon device, try the QNN execution provider:")
				fmt.Println("    set OLLAMA_ONNX_PROVIDER=qnn")
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

	// QNN provider DLL check
	fmt.Println("\n--- QNN Provider Status ---")
	qnnFound := false
	var searchDirs []string
	if ortPath, ok := os.LookupEnv("OLLAMA_ORT_PATH"); ok {
		searchDirs = append(searchDirs, filepath.SplitList(ortPath)...)
	}
	if exe, err := os.Executable(); err == nil {
		if eval, err := filepath.EvalSymlinks(exe); err == nil {
			exe = eval
		}
		searchDirs = append(searchDirs, filepath.Dir(exe))
	}

	for _, dir := range searchDirs {
		if _, err := os.Stat(filepath.Join(dir, "onnxruntime_providers_qnn.dll")); err == nil {
			fmt.Printf("  QNN provider DLL found: %s\\onnxruntime_providers_qnn.dll\n", dir)
			qnnFound = true
		}
		if _, err := os.Stat(filepath.Join(dir, "QnnHtp.dll")); err == nil {
			fmt.Printf("  QNN HTP backend DLL found: %s\\QnnHtp.dll\n", dir)
		}
	}
	if !qnnFound {
		fmt.Println("  QNN provider DLL not found in search paths.")
		fmt.Println("  To use QNN, install ORT GenAI QNN package and set OLLAMA_ORT_PATH.")
	}

	return nil
}
