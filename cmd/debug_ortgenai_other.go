//go:build !windows

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// DebugORTGenAIHandler is not available on non-Windows platforms.
func DebugORTGenAIHandler(cmd *cobra.Command, args []string) error {
	fmt.Println("ORT GenAI diagnostics are only available on Windows.")
	return nil
}
