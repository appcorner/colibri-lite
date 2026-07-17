[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TestExecutable,

    [Parameter(Mandatory = $true)]
    [string]$ArtifactRoot,

    [Parameter(Mandatory = $true)]
    [string]$DiagnosticRoot,

    [Parameter(Mandatory = $true)]
    [string]$FullLogitsRoot,

    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,

    [Parameter(Mandatory = $true)]
    [string]$RunLabel,

    [Parameter(Mandatory = $true)]
    [string]$FilesystemCacheAssumption
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-AvailablePhysicalBytes {
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem
        return [uint64]$os.FreePhysicalMemory * 1024
    }
    catch {
        return $null
    }
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Value
    )

    [System.IO.File]::WriteAllText(
        $Path,
        $Value,
        [System.Text.UTF8Encoding]::new($false)
    )
}

$testPath = [System.IO.Path]::GetFullPath($TestExecutable)
$artifactPath = [System.IO.Path]::GetFullPath($ArtifactRoot)
$diagnosticPath = [System.IO.Path]::GetFullPath($DiagnosticRoot)
$logitsPath = [System.IO.Path]::GetFullPath($FullLogitsRoot)
$runPath = [System.IO.Path]::GetFullPath($RunDirectory)

if (-not [System.IO.File]::Exists($testPath)) {
    throw "Test executable does not exist: $testPath"
}
if ([System.IO.Directory]::Exists($runPath)) {
    throw "Run directory already exists: $runPath"
}
[System.IO.Directory]::CreateDirectory($runPath) | Out-Null

$metricsPath = [System.IO.Path]::Combine($runPath, "rust-metrics.tsv")
$stdoutPath = [System.IO.Path]::Combine($runPath, "stdout.log")
$stderrPath = [System.IO.Path]::Combine($runPath, "stderr.log")
$jsonPath = [System.IO.Path]::Combine($runPath, "run-metrics.json")
$incompleteJsonPath = "$jsonPath.incomplete"

$availableBefore = Get-AvailablePhysicalBytes
$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($runPath))
$freeBefore = [uint64]$drive.AvailableFreeSpace

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $testPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.Arguments = "full_model_validation_tests::short_cached_generation_matches_transformers --exact --nocapture"
$startInfo.EnvironmentVariables["COLIBRI_ARTIFACT_ROOT"] = $artifactPath
$startInfo.EnvironmentVariables["COLIBRI_RMS_DIAGNOSTIC_ROOT"] = $diagnosticPath
$startInfo.EnvironmentVariables["COLIBRI_FULL_LOGITS_ROOT"] = $logitsPath
$startInfo.EnvironmentVariables["COLIBRI_METRICS_OUTPUT"] = $metricsPath
$startInfo.EnvironmentVariables["COLIBRI_FS_CACHE_ASSUMPTION"] = $FilesystemCacheAssumption

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$process = [System.Diagnostics.Process]::Start($startInfo)
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
$peakSampledWorkingSet = [uint64]0
$peakSampledPrivateBytes = [uint64]0
$lastWorkingSet = [uint64]0
$lastPrivateBytes = [uint64]0
$samples = 0

while (-not $process.HasExited) {
    try {
        $process.Refresh()
        $lastWorkingSet = [uint64]$process.WorkingSet64
        $lastPrivateBytes = [uint64]$process.PrivateMemorySize64
        $peakSampledWorkingSet = [Math]::Max($peakSampledWorkingSet, $lastWorkingSet)
        $peakSampledPrivateBytes = [Math]::Max($peakSampledPrivateBytes, $lastPrivateBytes)
        $samples += 1
    }
    catch {
        # The process may exit between HasExited and Refresh.
    }
    Start-Sleep -Milliseconds 100
}

$process.WaitForExit()
$stopwatch.Stop()
$stdout = $stdoutTask.GetAwaiter().GetResult()
$stderr = $stderrTask.GetAwaiter().GetResult()
Write-Utf8NoBom -Path $stdoutPath -Value $stdout
Write-Utf8NoBom -Path $stderrPath -Value $stderr

try {
    $process.Refresh()
    $processPeakWorkingSet = [uint64]$process.PeakWorkingSet64
    $processorSeconds = $process.TotalProcessorTime.TotalSeconds
}
catch {
    $processPeakWorkingSet = $peakSampledWorkingSet
    $processorSeconds = $null
}

$availableAfter = Get-AvailablePhysicalBytes
$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($runPath))
$freeAfter = [uint64]$drive.AvailableFreeSpace

if ($process.ExitCode -ne 0) {
    throw "M4.2-05 measured process failed with exit code $($process.ExitCode); see $stderrPath"
}
if (-not [System.IO.File]::Exists($metricsPath)) {
    throw "Rust metrics output was not created: $metricsPath"
}

$rustMetrics = @(Import-Csv -Delimiter "`t" -LiteralPath $metricsPath)
function Get-RustMetricValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Record,
        [Parameter(Mandatory = $true)]
        [string]$Phase,
        [Parameter(Mandatory = $true)]
        [string]$Metric
    )

    $matching = @($rustMetrics | Where-Object {
        $_.record -eq $Record -and $_.phase -eq $Phase -and $_.metric -eq $Metric
    })
    if ($matching.Count -ne 1) {
        throw "Expected one Rust metric $Record/$Phase/$Metric, found $($matching.Count)"
    }
    return $matching[0].value
}

if ((Get-RustMetricValue -Record "correctness" -Phase "total" -Metric "generated_token_ids") -ne "1096,374") {
    throw "Measured run changed generated token IDs"
}
if ((Get-RustMetricValue -Record "correctness" -Phase "total" -Metric "f32_classifications") -ne "exact_match_safe,exact_match_safe") {
    throw "Measured run changed F32 classifications"
}
if ((Get-RustMetricValue -Record "kv_cache" -Phase "total" -Metric "previous_position_overwrite") -ne "false") {
    throw "Measured run reported a KV-cache overwrite"
}

$result = [ordered]@{
    schema_version = 1
    task = "M4.2-05"
    run_label = $RunLabel
    filesystem_cache_assumption = $FilesystemCacheAssumption
    timing_scope = [ordered]@{
        rust_phase_clock = "monotonic high-resolution System.Diagnostics.Instant equivalent"
        process_wall_includes = "test harness and validation comparisons"
        rust_inference_wall_excludes = "reference preparation and validation comparisons"
    }
    process = [ordered]@{
        exit_code = $process.ExitCode
        external_wall_seconds = $stopwatch.Elapsed.TotalSeconds
        cpu_seconds = $processorSeconds
        peak_working_set_bytes = [Math]::Max($processPeakWorkingSet, $peakSampledWorkingSet)
        peak_sampled_private_bytes = $peakSampledPrivateBytes
        final_sampled_working_set_bytes = $lastWorkingSet
        final_sampled_private_bytes = $lastPrivateBytes
        memory_sample_count = $samples
        sample_interval_milliseconds = 100
    }
    system = [ordered]@{
        available_memory_before_bytes = $availableBefore
        available_memory_after_bytes = $availableAfter
        run_volume_free_before_bytes = $freeBefore
        run_volume_free_after_bytes = $freeAfter
    }
    correctness = [ordered]@{
        input_token_ids = @(9707, 11, 1879, 0)
        generated_token_ids = @(1096, 374)
        f32_classifications = @("exact_match_safe", "exact_match_safe")
        kv_cache_valid = $true
    }
    rust_metrics = $rustMetrics
}

$json = $result | ConvertTo-Json -Depth 8
Write-Utf8NoBom -Path $incompleteJsonPath -Value "$json`n"
[System.IO.File]::Move($incompleteJsonPath, $jsonPath)
Write-Output $jsonPath
