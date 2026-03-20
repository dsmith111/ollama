//go:build windows

package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/discover/dxcore"
	"github.com/ollama/ollama/x/ortrunner"
	"github.com/ollama/ollama/x/ortrunner/oga"
	"github.com/ollama/ollama/x/ortrunner/runtime"
	"github.com/spf13/cobra"
)

// NPUSetupHandler installs the ORT GenAI runtime for NPU inference.
func NPUSetupHandler(cmd *cobra.Command, args []string) error {
	provider, _ := cmd.Flags().GetString("provider")
	force, _ := cmd.Flags().GetBool("force")
	printPath, _ := cmd.Flags().GetBool("print-path")
	validateOnly, _ := cmd.Flags().GetBool("validate-only")

	if printPath {
		dir := runtime.DefaultInstallDir()
		if dir == "" {
			return fmt.Errorf("unable to determine install directory")
		}
		fmt.Println(dir)
		return nil
	}

	if validateOnly {
		report := ortrunner.ValidateORTDir()
		printValidationReport(report)
		return nil
	}

	fmt.Println("=== Ollama NPU Setup ===")
	fmt.Println()

	// Platform check
	fmt.Print("Checking platform... ")
	result := dxcore.Probe()
	if result.NPUCount > 0 {
		fmt.Println("OK (NPU detected)")
		for _, a := range result.Adapters {
			if a.IsNPU {
				fmt.Printf("  NPU: %s\n", a.Description)
			}
		}
	} else {
		fmt.Println("WARNING: No NPU detected")
		fmt.Println("  Setup will continue, but NPU inference may not work on this device.")
	}
	fmt.Println()

	// Install runtime
	fmt.Println("Installing ORT GenAI runtime...")
	installResult, err := runtime.Install(runtime.InstallConfig{
		Provider: provider,
		Force:    force,
	})
	if err != nil {
		return fmt.Errorf("installation failed: %w", err)
	}

	fmt.Printf("  Status:   %s\n", installResult.Message)
	fmt.Printf("  Dir:      %s\n", installResult.Dir)
	fmt.Printf("  Provider: %s\n", installResult.Provider)
	fmt.Println()

	// Validate
	fmt.Print("Validating installation... ")
	if installResult.Validated {
		fmt.Println("READY")
	} else {
		fmt.Println("NOT READY")
	}
	fmt.Println()

	fmt.Println("Next steps:")
	fmt.Println("  ollama run <model>          # Run a model with NPU acceleration")
	fmt.Println("  ollama npu doctor           # Check full NPU health")
	fmt.Println("  ollama benchmark npu        # Run NPU benchmark")

	return nil
}

// NPUDoctorHandler diagnoses NPU health and ORT GenAI readiness.
func NPUDoctorHandler(cmd *cobra.Command, args []string) error {
	fmt.Println("=== Ollama NPU Doctor ===")
	fmt.Println()
	allPassed := true

	// 1. NPU Hardware Check
	fmt.Println("--- NPU Hardware ---")
	result := dxcore.Probe()
	if result.Error != "" {
		fmt.Printf("  FAIL: DXCore probe error: %s\n", result.Error)
		allPassed = false
	} else if result.NPUCount == 0 {
		fmt.Println("  FAIL: No NPU adapters found")
		fmt.Println("  Action: This device does not have an NPU. NPU inference is not available.")
		allPassed = false
	} else {
		for _, a := range result.Adapters {
			if a.IsNPU {
				fmt.Printf("  PASS: NPU #%d: %s\n", a.Index, a.Description)
				if a.HasD3D12GenericML && !a.HasD3D12 {
					fmt.Println("    Note: Generic ML support detected (Snapdragon). QNN provider recommended.")
				}
			}
		}
	}
	fmt.Println()

	// 2. ORT Runtime Validation
	fmt.Println("--- ORT GenAI Runtime ---")
	report := ortrunner.ValidateORTDir()

	if len(report.FoundDLLs) == 0 {
		fmt.Println("  FAIL: No ORT GenAI DLLs found")
		fmt.Println("  Action: Run 'ollama npu setup' to install the runtime")
		allPassed = false
	} else {
		// Check core DLLs
		coreDLLs := []string{"onnxruntime.dll", "onnxruntime-genai.dll"}
		for _, dll := range coreDLLs {
			found := false
			for _, f := range report.FoundDLLs {
				if strings.EqualFold(f, dll) {
					found = true
					break
				}
			}
			if found {
				fmt.Printf("  PASS: %s found\n", dll)
			} else {
				fmt.Printf("  FAIL: %s missing\n", dll)
				allPassed = false
			}
		}

		// Provider-specific checks
		if report.HasQNN {
			fmt.Println("  PASS: QNN provider available")
		} else {
			fmt.Println("  WARN: QNN provider not found (onnxruntime_providers_qnn.dll)")
			fmt.Println("    Action: Run 'ollama npu setup --force' to reinstall")
		}

		if !report.HasTransport {
			fmt.Println("  WARN: No DSP transport DLLs found (cdsprpc)")
			fmt.Println("    Note: QNN HTP may fail without DSP transport. Check QNN SDK installation.")
		}
	}
	fmt.Println()

	// 3. Library Load Selftest
	fmt.Println("--- Library Load Selftest ---")
	if err := oga.CheckInit(); err != nil {
		fmt.Printf("  FAIL: ORT GenAI library load failed: %v\n", err)
		fmt.Println("  Action: Run 'ollama npu setup' to install compatible DLLs")
		allPassed = false
	} else {
		fmt.Println("  PASS: ORT GenAI library loaded successfully")
	}
	fmt.Println()

	// Summary
	fmt.Println("--- Summary ---")
	if allPassed {
		fmt.Println("  Status: READY")
		fmt.Println("  All checks passed. NPU inference should be available.")
	} else {
		fmt.Println("  Status: NOT READY")
		fmt.Println("  Some checks failed. Review the actions above.")
		fmt.Println("  Common fix: ollama npu setup")
	}

	return nil
}

func printValidationReport(report ortrunner.ORTDirReport) {
	fmt.Println("=== ORT GenAI Validation ===")
	fmt.Println()

	fmt.Println("Search directories:")
	for _, d := range report.SearchDirs {
		fmt.Printf("  %s\n", d)
	}
	fmt.Println()

	fmt.Println("Found DLLs:")
	if len(report.FoundDLLs) == 0 {
		fmt.Println("  (none)")
	} else {
		seen := make(map[string]bool)
		for _, dll := range report.FoundDLLs {
			if !seen[dll] {
				seen[dll] = true
				fmt.Printf("  %s\n", dll)
			}
		}
	}
	if len(report.MissingDLLs) > 0 {
		fmt.Println()
		fmt.Println("Missing DLLs:")
		for _, dll := range report.MissingDLLs {
			fmt.Printf("  %s\n", dll)
		}
	}
	fmt.Println()

	fmt.Printf("QNN:       %s\n", boolYesNo(report.HasQNN))
	fmt.Printf("DML:       %s\n", boolYesNo(report.HasDML))
	fmt.Printf("Transport: %s\n", boolYesNo(report.HasTransport))

	if os.Getenv("OLLAMA_ORT_PATH") != "" {
		fmt.Printf("\nOLLAMA_ORT_PATH = %s\n", os.Getenv("OLLAMA_ORT_PATH"))
	}
}
