//go:build !windows

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// BenchmarkNPUHandler is not available on non-Windows platforms.
func BenchmarkNPUHandler(cmd *cobra.Command, args []string) error {
	fmt.Println("NPU benchmark is only available on Windows.")
	return nil
}
