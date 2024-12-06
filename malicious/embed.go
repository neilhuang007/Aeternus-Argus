package testcases

import (
	"embed"
)

//go:embed all:*/*/*.white all:*/*/*.black
var EmbedTestCasesFS embed.FS
