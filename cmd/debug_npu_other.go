//go:build !windows

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// DebugNPUHandler is not available on non-Windows platforms.
func DebugNPUHandler(cmd *cobra.Command, args []string) error {
	fmt.Println("NPU diagnostics are only available on Windows.")
	return nil
}
