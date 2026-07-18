[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$TestExecutable,
    [Parameter(Mandatory = $true)] [string]$ArtifactRoot,
    [Parameter(Mandatory = $true)] [string]$TraceOutput,
    [Parameter(Mandatory = $true)] [string]$FullLogitsRoot,
    [Parameter(Mandatory = $true)] [string]$DiagnosticRoot,
    [Parameter(Mandatory = $true)] [string]$InstrumentationCommit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$testPath = [System.IO.Path]::GetFullPath($TestExecutable)
$artifactPath = [System.IO.Path]::GetFullPath($ArtifactRoot)
$tracePath = [System.IO.Path]::GetFullPath($TraceOutput)
$logitsPath = [System.IO.Path]::GetFullPath($FullLogitsRoot)
$diagnosticPath = [System.IO.Path]::GetFullPath($DiagnosticRoot)
if (-not [System.IO.File]::Exists($testPath)) { throw "Test executable does not exist: $testPath" }
if (-not [System.IO.Directory]::Exists($artifactPath)) { throw "Artifact root does not exist: $artifactPath" }
if (-not [System.IO.Directory]::Exists($logitsPath)) { throw "Full-logit root does not exist: $logitsPath" }
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($tracePath)) | Out-Null
[System.IO.Directory]::CreateDirectory($diagnosticPath) | Out-Null

$metricsPath = [System.IO.Path]::Combine($diagnosticPath, "m5.1-00-rust-metrics.tsv")
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $testPath
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.Arguments = "full_model_validation_tests::short_cached_generation_matches_transformers --exact --nocapture"
$env:COLIBRI_ARTIFACT_ROOT = $artifactPath
$env:COLIBRI_RMS_DIAGNOSTIC_ROOT = $diagnosticPath
$env:COLIBRI_FULL_LOGITS_ROOT = $logitsPath
$env:COLIBRI_METRICS_OUTPUT = $metricsPath
$env:COLIBRI_FS_CACHE_ASSUMPTION = "M5.1-00 replay; timing not baseline"
$env:COLIBRI_EXPERT_TRACE_OUTPUT = $tracePath
$env:COLIBRI_TRACE_INSTRUMENTATION_COMMIT = $InstrumentationCommit
$env:COLIBRI_TRACE_ONLY = "1"

$process = [System.Diagnostics.Process]::Start($startInfo)
$stdout = $process.StandardOutput.ReadToEndAsync()
$stderr = $process.StandardError.ReadToEndAsync()
$process.WaitForExit()
$stdoutText = $stdout.GetAwaiter().GetResult()
$stderrText = $stderr.GetAwaiter().GetResult()
if ($stdoutText) { [Console]::Out.WriteLine($stdoutText) }
if ($stderrText) { [Console]::Error.WriteLine($stderrText) }
if ($process.ExitCode -ne 0) { throw "Authoritative replay failed with exit code $($process.ExitCode)" }
if (-not [System.IO.File]::Exists($tracePath)) { throw "Trace output was not created: $tracePath" }
Write-Output "trace=$tracePath"
Write-Output "replay_command=$testPath full_model_validation_tests::short_cached_generation_matches_transformers --exact --nocapture"
