//go:build !windows

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// NPUSetupHandler is not available on non-Windows platforms.
func NPUSetupHandler(cmd *cobra.Command, args []string) error {
	fmt.Println("NPU setup is only available on Windows arm64.")
	return nil
}

// NPUDoctorHandler is not available on non-Windows platforms.
func NPUDoctorHandler(cmd *cobra.Command, args []string) error {
	fmt.Println("NPU doctor is only available on Windows.")
	return nil
}
