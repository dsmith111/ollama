param(
  [string]$Root = (Resolve-Path "."),
  [string]$Model = "phi3:mini",
  [int]$NumCtx = 512,
  [int]$PromptTokens = 512,
  [int]$MaxTokens = 128,
  [int]$WarmEpochs = 8,
  [string]$ResultsDir = ".\results-npu-bench"
)

$ErrorActionPreference = "Continue"

function Get-FreePort {
  $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
  $l.Start()
  $p = $l.LocalEndpoint.Port
  $l.Stop()
  return $p
}

function Wait-Ready([string]$HostUrl, [int]$TimeoutSec = 60) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    try {
      Invoke-RestMethod -Method Get -Uri "$HostUrl/api/version" | Out-Null
      return
    } catch {
      Start-Sleep -Milliseconds 200
    }
  }
  throw "Server did not become ready at $HostUrl within $TimeoutSec seconds."
}

function Start-Ollama([hashtable]$EnvMap, [string]$HostUrl) {
  # Apply env vars (empty string => remove)
  foreach ($k in $EnvMap.Keys) {
    $v = $EnvMap[$k]
    if ($null -eq $v -or $v -eq "") {
      Remove-Item "Env:$k" -ErrorAction SilentlyContinue
    } else {
      Set-Item "Env:$k" -Value $v
    }
  }

  # Ensure host is set for both serve + bench client
  $env:OLLAMA_HOST = $HostUrl

  $stdout = Join-Path $ResultsDir ("serve-" + ($HostUrl -replace '[:/\.]','_') + ".out.log")
  $stderr = Join-Path $ResultsDir ("serve-" + ($HostUrl -replace '[:/\.]','_') + ".err.log")

  $p = Start-Process `
    -FilePath (Join-Path $Root "ollama.exe") `
    -ArgumentList "serve" `
    -WorkingDirectory $Root `
    -PassThru `
    -NoNewWindow `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError  $stderr

  Wait-Ready -HostUrl $HostUrl
  return $p
}

function Stop-Ollama($proc) {
  if (-not $proc) { return }
  $procId = $proc.Id
  # Kill the serve process and its child tree
  try { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue } catch {}
  # Wait briefly for runner subprocesses to exit
  Start-Sleep -Seconds 2
  # Kill any leftover child runner processes (but not unrelated ollama instances)
  Get-CimInstance Win32_Process -Filter "ParentProcessId = $procId" -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

New-Item -ItemType Directory -Force $ResultsDir | Out-Null

# ---- Profiles ----
# CPU uses the DirectML ONNX model (single model.onnx, no baked-in QNN provider).
# NPU uses the QNN-native model (split pipeline with pre-compiled HTP context binaries).
$commonBase = @{
  "OLLAMA_CONTEXT_LENGTH" = "$NumCtx"
  "OLLAMA_ORT_PATH"       = (Join-Path $Root "lib\ollama\ortgenai")
  "OLLAMA_DEBUG"           = "1"
}

$profiles = @(
  @{
    Name = "cpu_ort"
    Env  = $commonBase + @{
      "OLLAMA_ONNX_MODEL"           = "C:\models\phi3-mini-onnx\directml\directml-int4-awq-block-128"
      "OLLAMA_ACCEL_MODE"           = ""
      "OLLAMA_ONNX_PROVIDER"        = "cpu"
      "OLLAMA_ORT_QNN_BACKEND_TYPE" = ""
      "OLLAMA_ORT_QNN_BACKEND_PATH" = ""
    }
  },
  @{
    Name = "npu_qnn"
    Env  = $commonBase + @{
      "OLLAMA_ONNX_MODEL"           = "C:\models\phi3-mini-onnx-qnn"
      "OLLAMA_ACCEL_MODE"           = "npu"
      "OLLAMA_ONNX_PROVIDER"        = "qnn"
      "OLLAMA_ORT_QNN_BACKEND_TYPE" = "htp"
      "OLLAMA_ORT_QNN_BACKEND_PATH" = "QnnHtp.dll"
    }
  }
)

foreach ($prof in $profiles) {
  $port = Get-FreePort
  $hostUrl = "http://127.0.0.1:$port"

  Write-Host ""
  Write-Host "=== Profile: $($prof.Name) @ $hostUrl ==="

  $p = $null
  try {
    $p = Start-Ollama -EnvMap $prof.Env -HostUrl $hostUrl

    # COLD: 1 request, no warmup (captures "first request burst")
    $coldOut = Join-Path $ResultsDir ("bench-" + $prof.Name + "-cold.txt")
    & (Join-Path $Root "ollama-bench.exe") `
      -model $Model `
      -epochs 1 `
      -warmup 0 `
      -prompt-tokens $PromptTokens `
      -max-tokens $MaxTokens `
      -num-ctx $NumCtx `
      -temperature 0 `
      -seed 1 `
      -format benchstat `
      -output $coldOut

    # WARM: warmup + multiple epochs
    $warmOut = Join-Path $ResultsDir ("bench-" + $prof.Name + "-warm.txt")
    & (Join-Path $Root "ollama-bench.exe") `
      -model $Model `
      -epochs $WarmEpochs `
      -warmup 1 `
      -prompt-tokens $PromptTokens `
      -max-tokens $MaxTokens `
      -num-ctx $NumCtx `
      -temperature 0 `
      -seed 1 `
      -format benchstat `
      -output $warmOut
  }
  finally {
    if ($p) { Stop-Ollama $p }
  }
}

Write-Host ""
Write-Host "Done. Results in: $ResultsDir"
Write-Host "Tip: go install golang.org/x/perf/cmd/benchstat@latest"
Write-Host "Then: benchstat $ResultsDir\bench-*-warm.txt"
