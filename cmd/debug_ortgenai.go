//go:build windows

package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/x/ortrunner"
	"github.com/spf13/cobra"
)

// DebugORTGenAIHandler prints ORT GenAI environment diagnostics.
func DebugORTGenAIHandler(cmd *cobra.Command, args []string) error {
	report := ortrunner.ValidateORTDir()

	fmt.Println("=== ORT GenAI Environment Report ===")
	fmt.Println()

	// Search directories
	fmt.Println("Search directories:")
	for _, d := range report.SearchDirs {
		fmt.Printf("  %s\n", d)
	}
	fmt.Println()

	// Environment
	fmt.Println("Environment:")
	fmt.Printf("  OLLAMA_ORT_PATH          = %s\n", os.Getenv("OLLAMA_ORT_PATH"))
	fmt.Printf("  OLLAMA_ONNX_PROVIDER     = %s\n", valOrDefault(report.Provider, "(auto-detect)"))
	fmt.Printf("  OLLAMA_ORT_QNN_BACKEND_TYPE = %s\n", valOrDefault(report.BackendType, "htp (default)"))
	fmt.Printf("  OLLAMA_ORT_QNN_BACKEND_PATH = %s\n", valOrDefault(report.BackendPath, "(not set)"))
	fmt.Printf("  OLLAMA_ORT_QNN_HTP_ARCH  = %s\n", valOrDefault(report.HTPArch, "(auto)"))
	fmt.Printf("  OLLAMA_ORT_QNN_SOC_MODEL = %s\n", valOrDefault(report.SocModel, "(auto)"))
	fmt.Println()

	// Found DLLs
	fmt.Println("Found DLLs:")
	if len(report.FoundDLLs) == 0 {
		fmt.Println("  (none)")
	} else {
		// Deduplicate for display
		seen := make(map[string]bool)
		for _, dll := range report.FoundDLLs {
			if !seen[dll] {
				seen[dll] = true
				fmt.Printf("  %s\n", dll)
			}
		}
	}
	fmt.Println()

	// Missing DLLs
	if len(report.MissingDLLs) > 0 {
		fmt.Println("Missing DLLs:")
		for _, dll := range report.MissingDLLs {
			fmt.Printf("  %s\n", dll)
		}
		fmt.Println()
	}

	// Provider support
	fmt.Println("Provider support:")
	fmt.Printf("  QNN (Qualcomm):  %s\n", boolYesNo(report.HasQNN))
	fmt.Printf("  DML (DirectML):  %s\n", boolYesNo(report.HasDML))
	fmt.Printf("  DSP transport:   %s\n", boolYesNo(report.HasTransport))
	fmt.Println()

	// Actionable warnings
	if report.Provider == "dml" && !report.HasDML {
		fmt.Println("WARNING: OLLAMA_ONNX_PROVIDER=dml but DirectML.dll was not found.")
		fmt.Println("  Your GenAI build may not include DML support.")
		fmt.Println("  For Snapdragon NPU, use: set OLLAMA_ONNX_PROVIDER=qnn")
		fmt.Println()
	}

	if report.Provider == "qnn" || report.HasQNN {
		if !report.HasQNN {
			fmt.Println("WARNING: provider=qnn requested but onnxruntime_providers_qnn.dll not found.")
			fmt.Println("  Install the QNN ORT GenAI package and set OLLAMA_ORT_PATH.")
			fmt.Println()
		}

		// Check specific QNN dependencies
		qnnRequired := []string{"QnnHtp.dll", "QnnSystem.dll"}
		var qnnMissing []string
		for _, dll := range qnnRequired {
			found := false
			for _, f := range report.FoundDLLs {
				if strings.EqualFold(f, dll) {
					found = true
					break
				}
			}
			if !found {
				qnnMissing = append(qnnMissing, dll)
			}
		}
		if len(qnnMissing) > 0 {
			fmt.Printf("WARNING: QNN HTP backend DLLs missing: %s\n", strings.Join(qnnMissing, ", "))
			fmt.Println("  These are required for QNN HTP (NPU) inference.")
			fmt.Println()
		}

		if !report.HasTransport {
			fmt.Println("NOTE: No DSP transport/RPC DLLs (cdsprpc) found in search directories.")
			fmt.Println("  If QNN HTP fails with 'DspTransport.openSession' errors, you may need")
			fmt.Println("  the DSP RPC transport DLL in OLLAMA_ORT_PATH or the system QNN SDK path.")
			fmt.Println()
		}
	}

	// Check for stale onnxruntime.dll in other PATH locations
	if pathEnv := os.Getenv("PATH"); pathEnv != "" {
		ortInPath := []string{}
		for _, dir := range filepath.SplitList(pathEnv) {
			if _, err := os.Stat(filepath.Join(dir, "onnxruntime.dll")); err == nil {
				// Skip our own search dirs
				isOurs := false
				for _, sd := range report.SearchDirs {
					if strings.EqualFold(dir, sd) {
						isOurs = true
						break
					}
				}
				if !isOurs {
					ortInPath = append(ortInPath, dir)
				}
			}
		}
		if len(ortInPath) > 0 {
			fmt.Println("WARNING: Other onnxruntime.dll found in PATH (may cause version conflicts):")
			for _, p := range ortInPath {
				fmt.Printf("  %s\\onnxruntime.dll\n", p)
			}
			fmt.Println("  Ensure OLLAMA_ORT_PATH directory is first in PATH to avoid API mismatches.")
			fmt.Println()
		}
	}

	return nil
}

func valOrDefault(val, def string) string {
	if val == "" {
		return def
	}
	return val
}

func boolYesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}
