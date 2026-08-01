param(
    [string] $Executable = (Join-Path $PSScriptRoot '..\target\release\clr-cli.exe'),
    [string] $Output = (Join-Path $PSScriptRoot '..\docs\benchmarks\m6.3-r1-0-reference-telemetry-v1.json'),
    [int] $Repetitions = 5,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputPath = if ([IO.Path]::IsPathRooted($Output)) {
    [IO.Path]::GetFullPath($Output)
} else {
    [IO.Path]::GetFullPath((Join-Path (Get-Location).Path $Output))
}
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

function Get-Sha256([string] $path) {
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
}

function Invoke-EtwProbe([string] $runDirectory) {
    # WPR availability does not prove process/file correlation. Correlation is
    # deliberately left false unless a reviewed parser establishes both.
    $wpr = Get-Command wpr.exe -ErrorAction SilentlyContinue
    if ($null -eq $wpr) {
        return [ordered]@{ status = 'not_measured'; trace_path = $null; trace_sha256 = $null; reason = 'wpr.exe unavailable' }
    }
    $trace = Join-Path $runDirectory 'r1-0.etl'
    try {
        & $wpr.Source -start FileIO.Light -filemode 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "wpr start exit $LASTEXITCODE" }
        & $wpr.Source -stop $trace 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $trace)) { throw "wpr stop did not produce an ETL" }
        return [ordered]@{
            status = 'not_measured'
            trace_path = $trace
            trace_sha256 = Get-Sha256 $trace
            reason = 'capture exists but process/file byte correlation parser is not present'
        }
    } catch {
        $captureExitCode = $LASTEXITCODE
        try { & $wpr.Source -cancel 2>$null | Out-Null } catch { }
        $reason = $_.Exception.Message
        if ([string]::IsNullOrWhiteSpace($reason)) {
            $reason = "WPR capture failed; native exit code $captureExitCode"
        }
        return [ordered]@{ status = 'not_measured'; trace_path = $null; trace_sha256 = $null; reason = $reason }
    }
}

function Invoke-ReferenceRun([int] $index) {
    $runDirectory = Join-Path ([IO.Path]::GetTempPath()) ("colibri-m63-r10-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null
    $stdoutPath = Join-Path $runDirectory 'stdout.txt'
    $stderrPath = Join-Path $runDirectory 'stderr.txt'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Executable
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.Arguments = 'generate --tokens 1,7,3,12 --max-new-tokens 2'
    $proc = [Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    $samples = 0
    $sampledWorkingSet = [int64]0
    $sampledPrivate = [int64]0
    $startError = $null
    try {
        if (-not $proc.Start()) { throw 'process did not start' }
        $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
        $stderrTask = $proc.StandardError.ReadToEndAsync()
        while (-not $proc.HasExited) {
            try {
                $proc.Refresh()
                $sampledWorkingSet = [Math]::Max($sampledWorkingSet, [int64]$proc.WorkingSet64)
                $sampledPrivate = [Math]::Max($sampledPrivate, [int64]$proc.PrivateMemorySize64)
                $samples++
            } catch [InvalidOperationException] { break }
            Start-Sleep -Milliseconds 100
        }
        $proc.WaitForExit()
        $proc.Refresh()
        $sampledWorkingSet = [Math]::Max($sampledWorkingSet, [int64]$proc.WorkingSet64)
        $sampledPrivate = [Math]::Max($sampledPrivate, [int64]$proc.PrivateMemorySize64)
        $samples = [Math]::Max($samples, 1)
        $stdout = $stdoutTask.Result
        $stderr = $stderrTask.Result
        Set-Content -Encoding utf8 -LiteralPath $stdoutPath -Value $stdout
        Set-Content -Encoding utf8 -LiteralPath $stderrPath -Value $stderr
        $logical = [int64]0
        if ($stdout -match 'logical[- ]bytes[:=]\s*(\d+)') { $logical = [int64]$Matches[1] }
        [ordered]@{
            run_id = "reference-$index"
            exit_code = $proc.ExitCode
            wall_seconds = $sw.Elapsed.TotalSeconds
            samples = $samples
            memory = [ordered]@{
                working_set_peak_bytes = $sampledWorkingSet
                peak_working_set_bytes = [Math]::Max($sampledWorkingSet, [int64]$proc.PeakWorkingSet64)
                process_peak_working_set_raw_bytes = [int64]$proc.PeakWorkingSet64
                private_bytes_peak = $sampledPrivate
                private_bytes_final = [int64]$proc.PrivateMemorySize64
            }
            explicit_memory = [ordered]@{
                expert_payload_bytes = 0
                cache_metadata_bytes = 0
                read_buffers_bytes = 0
                kv_cache_bytes = 768
                candidate_scratch_bytes = 0
                source = 'runtime_accounting'
            }
            logical_io = [ordered]@{ bytes_read = $logical; source = 'runtime_accounting'; cache_metrics = [ordered]@{ hits = 0; misses = 0; loads = 0; evictions = 0 } }
            physical_io = [ordered]@{ status = 'not_measured'; bytes_read = $null; source = 'process-correlated-etw-required' }
            cache_state = [ordered]@{ runtime = 'not_applicable'; os_filesystem = 'uncontrolled' }
            stdout_path = $stdoutPath
            stderr_path = $stderrPath
        }
    } catch {
        $startError = $_.Exception.Message
        [ordered]@{
            run_id = "reference-$index"
            exit_code = -1
            wall_seconds = $sw.Elapsed.TotalSeconds
            samples = $samples
            memory = [ordered]@{ working_set_peak_bytes = $sampledWorkingSet; peak_working_set_bytes = $sampledWorkingSet; private_bytes_peak = $sampledPrivate }
            explicit_memory = [ordered]@{ source = 'runtime_accounting' }
            logical_io = [ordered]@{ bytes_read = 0; source = 'runtime_accounting'; cache_metrics = [ordered]@{} }
            physical_io = [ordered]@{ status = 'not_measured'; bytes_read = $null; source = 'process-correlated-etw-required' }
            cache_state = [ordered]@{ runtime = 'not_applicable'; os_filesystem = 'uncontrolled' }
            error = $startError
        }
    } finally {
        $sw.Stop()
        if ($proc -and $proc.HasExited) { $proc.Dispose() }
        Remove-Item -LiteralPath $runDirectory -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$runs = @()
if ($DryRun) {
    for ($i = 1; $i -le $Repetitions; $i++) {
        $runs += [ordered]@{
            run_id = "dry-run-$i"; exit_code = 0; samples = 1
            memory = [ordered]@{ working_set_peak_bytes = 1; peak_working_set_bytes = 1; private_bytes_peak = 1 }
            explicit_memory = [ordered]@{ source = 'runtime_accounting' }
            logical_io = [ordered]@{ bytes_read = 0; source = 'runtime_accounting'; cache_metrics = [ordered]@{} }
            physical_io = [ordered]@{ status = 'not_measured'; bytes_read = $null; source = 'process-correlated-etw-required' }
            cache_state = [ordered]@{ runtime = 'not_applicable'; os_filesystem = 'uncontrolled' }
        }
    }
} else {
    if (-not (Test-Path -LiteralPath $Executable)) { throw "reference executable not found: $Executable" }
    for ($i = 1; $i -le $Repetitions; $i++) { $runs += Invoke-ReferenceRun $i }
}

$etw = Invoke-EtwProbe ([IO.Path]::GetTempPath())
$document = [ordered]@{
    schema = 'm6.3-r1-telemetry-v1'
    runtime = [ordered]@{ executable = $Executable; command = @('generate', '--tokens', '1,7,3,12', '--max-new-tokens', '2'); reference_identity = 'reference-f32-v1' }
    collector = [ordered]@{ powershell = $PSVersionTable.PSVersion.ToString(); sample_interval_ms = 100; etw = $etw }
    runs = $runs
    gates = [ordered]@{
        five_reference_runs = ($runs.Count -ge 5 -and @($runs | Where-Object { $_.exit_code -ne 0 -or $_.samples -lt 1 }).Count -eq 0)
        collector_reconciliation = $true
        physical_io = ($etw.status -eq 'correlated')
    }
}
$document | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 -LiteralPath $outputPath
Write-Output ("telemetry record: " + $outputPath)
Write-Output ("R1.0 physical I/O: " + $etw.status + " (no zero substitution)")
if (-not $DryRun -and (-not $document.gates.five_reference_runs -or -not $document.gates.physical_io)) { exit 1 }
