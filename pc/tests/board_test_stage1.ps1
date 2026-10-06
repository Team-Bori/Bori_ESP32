<#
Stage 1 board verification (tflm_runtime + mlp on the new partition table).

Run in a PowerShell where ESP-IDF is activated (export.ps1), from the repository root:
  powershell -ExecutionPolicy Bypass -File pc\tests\board_test_stage1.ps1 -Port COM6

Flashes the board several times. Everything is logged to
models\generated\board_test_stage1.log (git-ignored) for review.

Steps
  A  tflm_runtime: erased model partition (no_package, max_arena_bytes), sine_regression package
     (verify/info/infer/bench/eval/run-test/metrics), fault packages, old v1 client
  B  mlp firmware + MLP1 model on the new layout, old v1 client
#>
param(
    [string]$Port = "COM6",
    # Python with pyserial + numpy for esp_monitor.py (the ESP-IDF python has no numpy).
    [string]$HostPython = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    [switch]$SkipMlp,
    [switch]$SkipFaults
)

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Root
$LogDir = Join-Path $Root "models\generated"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Log = Join-Path $LogDir "board_test_stage1.log"
"# board_test_stage1 $(Get-Date -Format s) port=$Port" | Out-File $Log -Encoding utf8

function Step([string]$title) {
    $line = "`n===== $title ====="
    Write-Host $line -ForegroundColor Cyan
    $line | Out-File $Log -Append -Encoding utf8
}

function Run([string]$exe, [string[]]$argv) {
    "> $exe $($argv -join ' ')" | Out-File $Log -Append -Encoding utf8
    $out = & $exe @argv 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $out | Out-File $Log -Append -Encoding utf8
    $out | ForEach-Object { Write-Host $_ }
    "exit=$code" | Out-File $Log -Append -Encoding utf8
    return $code
}

function Esptool([string[]]$argv) {
    Run "python" (@("-m", "esptool", "--chip", "esp32", "-p", $Port, "-b", "921600") + $argv) | Out-Null
}

function Mon([string[]]$argv) {
    return Run $HostPython (@("rpi\esp_monitor.py", $Port) + $argv)
}

function WriteModel([string]$file) {
    Esptool @("write_flash", "0x200000", $file)
}

# The protocol v1 client exactly as it was before this change, for the compatibility check.
$V1Client = Join-Path $LogDir "esp_monitor_v1.py"
git show 66a9038:rpi/esp_monitor.py | Out-File $V1Client -Encoding utf8

# ---------------------------------------------------------------- A
Step "A1 flash tflm_runtime, erase model partition"
Run $HostPython @("pc\validate_firmware.py", "firmware\tflm_runtime.bin") | Out-Null
Esptool @("write_flash", "0x0", "firmware\tflm_runtime.bin")
Esptool @("erase_region", "0x200000", "0x200000")

Step "A2 empty partition: expect no_package, board stays up; info gives max_arena_bytes"
Mon @("verify", "--allow-no-model", "--expect-firmware", "tflm_runtime") | Out-Null
Mon @("info") | Out-Null

Step "A3 sine_regression package"
WriteModel "models\sine_regression\default_model.bin"
Mon @("verify", "--expect-checksum", "0xfc8a647d", "--expect-firmware", "tflm_runtime") | Out-Null
Mon @("info") | Out-Null
Mon @("infer") | Out-Null
Mon @("bench") | Out-Null
Mon @("eval") | Out-Null

Step "A4 streamed test file at 921600 and at 115200"
Mon @("run-test", "models\sine_regression\test_data.npz") | Out-Null
Mon @("run-test", "models\sine_regression\test_data.npz", "--baud", "0", "--emit-samples", "--limit", "3") | Out-Null
Mon @("run-test", "models\sine_regression\test_data.npz", "--no-labels") | Out-Null
Mon @("info") | Out-Null

Step "A5 periodic metrics for 16 s (expect seq without gaps every 5 s)"
Mon @("listen", "--duration", "16") | Out-Null

Step "A6 old v1 client against tflm_runtime"
Run $HostPython @($V1Client, $Port, "info") | Out-Null
Run $HostPython @($V1Client, $Port, "bench") | Out-Null

if (-not $SkipFaults) {
    Run $HostPython @("pc\tests\make_fault_packages.py") | Out-Null
    $faults = [ordered]@{
        "bad_checksum"    = "checksum_mismatch"
        "unsupported_op"  = "unsupported_op"
        "arena_too_small" = "arena_too_small"
        "arena_too_big"   = "arena_alloc_failed"
        "tensor_mismatch" = "package_invalid"
        "mlp_package"     = "package_invalid"
    }
    foreach ($name in $faults.Keys) {
        Step "A7 fault package $name (expect error $($faults[$name]), no reboot)"
        WriteModel "models\generated\faults\$name.bin"
        Mon @("verify", "--allow-no-model", "--expect-firmware", "tflm_runtime", "--settle", "6") | Out-Null
        Mon @("info") | Out-Null
    }
    Step "A8 restore sine_regression"
    WriteModel "models\sine_regression\default_model.bin"
    Mon @("verify", "--expect-checksum", "0xfc8a647d", "--expect-firmware", "tflm_runtime") | Out-Null
}

# ---------------------------------------------------------------- B
if (-not $SkipMlp) {
    Step "B1 mlp firmware + MLP1 model on the new partition table"
    Run $HostPython @("pc\validate_firmware.py", "firmware\mlp.bin") | Out-Null
    Esptool @("write_flash", "0x0", "firmware\mlp.bin")
    Esptool @("erase_region", "0x200000", "0x200000")
    WriteModel "models\mlp\default_model.bin"
    Mon @("verify", "--expect-checksum", "0x12550a83", "--expect-firmware", "mlp") | Out-Null
    Mon @("info") | Out-Null
    Mon @("infer") | Out-Null
    Mon @("bench") | Out-Null

    Step "B2 old v1 client against mlp"
    Run $HostPython @($V1Client, $Port, "verify", "--expect-checksum", "0x12550a83") | Out-Null
    Run $HostPython @($V1Client, $Port, "info") | Out-Null

    Step "B3 v2 tools refuse to pair: eval/run-test on mlp must fail cleanly"
    Mon @("run-test", "models\sine_regression\test_data.npz") | Out-Null
}

Step "done"
Write-Host "`nLog: $Log" -ForegroundColor Green
