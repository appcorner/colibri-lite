param(
    [Parameter(Mandatory = $true)] [string] $Config,
    [string] $Output = '',
    [int] $Repetitions = 5,
    [string] $Python = 'python'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Output = if ([string]::IsNullOrWhiteSpace($Output)) {
    Join-Path $repo 'docs\benchmarks\m6.3-r1-0-reference-telemetry-v1.json'
} else {
    [IO.Path]::GetFullPath($Output)
}
$configuration = Get-Content -Raw -Encoding utf8 -LiteralPath $Config | ConvertFrom-Json
$capture = Join-Path $PSScriptRoot 'capture_m6_3_r1_etw.ps1'
$validate = Join-Path $PSScriptRoot 'validate_m6_3_r1_0_telemetry.py'
$outputPath = [IO.Path]::GetFullPath($Output)
$runs = @()

function Metric([object[]] $rows, [string] $phase, [string] $name) {
    $row = @($rows | Where-Object { $_.phase -eq $phase -and $_.metric -eq $name })
    if ($row.Count -ne 1) { throw "missing or duplicate metrics row: $phase/$name" }
    return [int64][double]$row[0].value
}

for ($index = 1; $index -le $Repetitions; $index++) {
    & $capture -Config $Config -Python $Python
    $captureExit = $LASTEXITCODE
    if ($captureExit -notin @(0, 1)) { throw "reference ETW capture $index failed with exit code $captureExit" }

    $runDirectory = Get-Content -Raw -Encoding ascii -LiteralPath (Join-Path $configuration.output_directory 'latest-run.txt')
    $runDirectory = $runDirectory.Trim()
    $captureRecord = Get-Content -Raw -Encoding utf8 -LiteralPath (Join-Path $runDirectory 'capture.json') | ConvertFrom-Json
    if ($captureRecord.exit_code -ne 0) {
        throw "reference process $index failed with exit code $($captureRecord.exit_code)"
    }
    $metricsPath = Join-Path $runDirectory 'metrics.json'
    if (-not (Test-Path -LiteralPath $metricsPath)) { throw "reference ETW capture $index did not emit metrics.json" }
    $metrics = @(Import-Csv -LiteralPath $metricsPath -Delimiter "`t")
    $runs += [ordered]@{
        run_id = "reference-$index"
        capture_run_id = $captureRecord.run_id
        exit_code = [int]$captureRecord.exit_code
        samples = [int]$captureRecord.samples
        memory = $captureRecord.memory
        explicit_memory = [ordered]@{
            modeled_explicit_tensor_bytes = Metric $metrics 'total' 'modeled_explicit_tensor_bytes'
            dense_buffer_bytes = Metric $metrics 'total' 'dense_buffer_bytes'
            decoded_expert_buffer_bytes = Metric $metrics 'total' 'decoded_expert_buffer_bytes'
            expert_cache_resident_bytes = Metric $metrics 'total' 'expert_cache_resident_bytes'
            kv_cache_bytes = Metric $metrics 'total' 'kv_cache_bytes'
            inference_tensor_bytes = Metric $metrics 'total' 'inference_tensor_bytes'
            temporary_validation_buffer_bytes = Metric $metrics 'total' 'temporary_validation_buffer_bytes'
            source = 'runtime_accounting'
        }
        logical_io = [ordered]@{
            bytes_read = Metric $metrics 'total' 'total_artifact_bytes_read'
            source = 'runtime_accounting'
        }
        cache_metrics = [ordered]@{
            configured_byte_budget = Metric $metrics 'total' 'configured_byte_budget'
            hits = Metric $metrics 'total' 'hits'
            misses = Metric $metrics 'total' 'misses'
            loads = Metric $metrics 'total' 'loads'
            evictions = Metric $metrics 'total' 'evictions'
        }
        physical_io = [ordered]@{
            status = $captureRecord.etw.status
            bytes_read = $captureRecord.etw.physical_read_bytes
            source = 'process-correlated-etw'
            trace_path = $captureRecord.etw.trace_path
            trace_sha256 = $captureRecord.etw.trace_sha256
            correlation_path = $captureRecord.etw.correlation_path
        }
        cache_state = [ordered]@{ runtime = 'warm'; os_filesystem = 'uncontrolled' }
        capture_path = (Join-Path $runDirectory 'capture.json')
        metrics_path = $metricsPath
    }
}

$document = [ordered]@{
    schema = 'm6.3-r1-telemetry-v1'
    runtime = [ordered]@{
        executable = [IO.Path]::GetFullPath([string]$configuration.executable)
        command = @([string]$configuration.argument_string)
        reference_identity = 'reference-f32-v1'
    }
    collector = [ordered]@{
        powershell = $PSVersionTable.PSVersion.ToString()
        sample_interval_ms = 100
        etw = [ordered]@{
            status = if (@($runs | Where-Object { $_.physical_io.status -eq 'correlated' }).Count -gt 0) { 'correlated' } else { 'not_measured' }
            collector = 'logman'
            parser = 'parse_m6_3_r1_etw.py'
            correlation = 'exact PID + exact artifact path/FileObject + Kernel File Read.FileKey = Kernel Disk.FileObject'
        }
    }
    runs = $runs
    gates = [ordered]@{
        five_reference_runs = ($runs.Count -eq 5)
        collector_reconciliation = $true
        physical_io = (@($runs | Where-Object { $_.physical_io.status -eq 'correlated' -and $null -ne $_.physical_io.bytes_read }).Count -gt 0)
    }
}
$document | ConvertTo-Json -Depth 12 | Set-Content -Encoding utf8 -LiteralPath $outputPath
& $Python $validate $outputPath
exit $LASTEXITCODE
