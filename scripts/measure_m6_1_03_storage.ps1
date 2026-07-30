[CmdletBinding()]
param(
    [string]$OutputPath = "docs/benchmarks/m6.1-03-storage-benchmark-v1.json",
    [string]$TempRoot = "D:\tmp\colibri-lite-runs"
)

$ErrorActionPreference = "Stop"

$TaskId = "m6.1-03"
$ExpertPayloadBytes = 18874368
$ExpertPayloadCount = 56
$PayloadBytes = [int64]$ExpertPayloadBytes * $ExpertPayloadCount
$SequentialBufferBytes = 4MB
$RandomBufferBytes = 1MB
$WarmSequentialSamples = 9
$CreatedAt = (Get-Date).ToUniversalTime().ToString("o")
$Commit = (git rev-parse HEAD).Trim()
$RunId = [guid]::NewGuid().ToString("N")

function Assert-RunDirectory([string]$Candidate, [string]$Root, [string]$ExpectedLeaf) {
    $resolvedRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $resolvedCandidate = [System.IO.Path]::GetFullPath($Candidate)
    if (-not $resolvedCandidate.StartsWith("$resolvedRoot\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "run directory must remain under temp root: $resolvedCandidate"
    }
    if ([System.IO.Path]::GetFileName($resolvedCandidate) -ne $ExpectedLeaf) {
        throw "run directory leaf does not match the generated run ID"
    }
}

function New-Distribution([double[]]$Values) {
    if ($Values.Count -lt 3) { throw "a distribution requires at least three samples" }
    $samples = @($Values | Sort-Object)
    $percentile = {
        param([double]$p)
        $index = [int][math]::Round(($samples.Count - 1) * $p, 0, [System.MidpointRounding]::AwayFromZero)
        return $samples[$index]
    }
    return [ordered]@{
        samples = $samples
        minimum = $samples[0]
        p10 = & $percentile 0.10
        median = & $percentile 0.50
        p90 = & $percentile 0.90
        maximum = $samples[$samples.Count - 1]
    }
}

function Read-Sequential([string]$Path, [int]$BufferBytes) {
    $buffer = New-Object byte[] $BufferBytes
    $checksum = [int64]0
    $bytesRead = [int64]0
    $stream = [System.IO.FileStream]::new(
        $Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read, $BufferBytes, [System.IO.FileOptions]::SequentialScan)
    try {
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $bytesRead += $read
            $checksum += $buffer[0] + $buffer[$read - 1]
        }
        $timer.Stop()
        return [ordered]@{ seconds = $timer.Elapsed.TotalSeconds; bytes = $bytesRead; checksum = $checksum }
    }
    finally { $stream.Dispose() }
}

function Read-ExpertPayload([string]$Path, [int64]$Offset, [int]$PayloadBytes, [int]$BufferBytes) {
    $buffer = New-Object byte[] $BufferBytes
    $remaining = [int64]$PayloadBytes
    $checksum = [int64]0
    $stream = [System.IO.FileStream]::new(
        $Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read, $BufferBytes, [System.IO.FileOptions]::RandomAccess)
    try {
        $stream.Seek($Offset, [System.IO.SeekOrigin]::Begin) | Out-Null
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        while ($remaining -gt 0) {
            $requested = [int][math]::Min($buffer.Length, $remaining)
            $read = $stream.Read($buffer, 0, $requested)
            if ($read -eq 0) { throw "unexpected EOF at offset $Offset" }
            $remaining -= $read
            $checksum += $buffer[0] + $buffer[$read - 1]
        }
        $timer.Stop()
        return [ordered]@{ seconds = $timer.Elapsed.TotalSeconds; bytes = $PayloadBytes; checksum = $checksum }
    }
    finally { $stream.Dispose() }
}

function Get-RandomOrder([int]$Count, [int]$Seed) {
    $random = [System.Random]::new($Seed)
    $order = [int[]](0..($Count - 1))
    for ($index = $order.Length - 1; $index -gt 0; $index--) {
        $other = $random.Next($index + 1)
        $temporary = $order[$index]
        $order[$index] = $order[$other]
        $order[$other] = $temporary
    }
    return $order
}

function Write-Payload([string]$Path, [int64]$Bytes, [int]$BufferBytes) {
    $buffer = New-Object byte[] $BufferBytes
    $written = [int64]0
    $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $stream = [System.IO.FileStream]::new(
        $Path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None, $BufferBytes, [System.IO.FileOptions]::WriteThrough)
    try {
        while ($written -lt $Bytes) {
            $count = [int][math]::Min($buffer.Length, $Bytes - $written)
            $random.GetBytes($buffer)
            $stream.Write($buffer, 0, $count)
            $written += $count
        }
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
        $random.Dispose()
    }
    return $written
}

New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
$runLeaf = "$TaskId-$RunId"
$runDirectory = Join-Path $TempRoot $runLeaf
Assert-RunDirectory $runDirectory $TempRoot $runLeaf
$drive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($TempRoot)).TrimEnd(':', '\'))
$freeBytesBefore = $drive.Free
$safetyReserveBytes = [int64][math]::Max(1GB, [math]::Ceiling($PayloadBytes * 0.05))
$requiredFreeBytes = $PayloadBytes + $safetyReserveBytes
if ($freeBytesBefore -lt $requiredFreeBytes) {
    throw "disk preflight failed: free=$freeBytesBefore, required=$requiredFreeBytes"
}

$payloadPath = Join-Path $runDirectory "storage-payload.bin"
$success = $false
try {
    New-Item -ItemType Directory -Path $runDirectory | Out-Null
    $writtenBytes = Write-Payload $payloadPath $PayloadBytes $SequentialBufferBytes
    $freeBytesAfterPayloadWrite = (Get-PSDrive -Name $drive.Name).Free

    $firstSequential = Read-Sequential $payloadPath $SequentialBufferBytes
    $warmSequentialThroughputs = [System.Collections.Generic.List[double]]::new()
    $sequentialChecksum = $firstSequential.checksum
    for ($index = 0; $index -lt $WarmSequentialSamples; $index++) {
        $sample = Read-Sequential $payloadPath $SequentialBufferBytes
        $warmSequentialThroughputs.Add(($sample.bytes / $sample.seconds) / 1GB)
        $sequentialChecksum += $sample.checksum
    }

    $firstTouchLatencies = [System.Collections.Generic.List[double]]::new()
    $firstTouchThroughputs = [System.Collections.Generic.List[double]]::new()
    $warmRandomLatencies = [System.Collections.Generic.List[double]]::new()
    $warmRandomThroughputs = [System.Collections.Generic.List[double]]::new()
    $randomChecksum = [int64]0
    foreach ($slot in (Get-RandomOrder $ExpertPayloadCount 6102)) {
        $sample = Read-ExpertPayload $payloadPath ([int64]$slot * $ExpertPayloadBytes) $ExpertPayloadBytes $RandomBufferBytes
        $firstTouchLatencies.Add($sample.seconds * 1000.0)
        $firstTouchThroughputs.Add(($sample.bytes / $sample.seconds) / 1MB)
        $randomChecksum += $sample.checksum
    }
    foreach ($slot in (Get-RandomOrder $ExpertPayloadCount 6103)) {
        $sample = Read-ExpertPayload $payloadPath ([int64]$slot * $ExpertPayloadBytes) $ExpertPayloadBytes $RandomBufferBytes
        $warmRandomLatencies.Add($sample.seconds * 1000.0)
        $warmRandomThroughputs.Add(($sample.bytes / $sample.seconds) / 1MB)
        $randomChecksum += $sample.checksum
    }

    Assert-RunDirectory $runDirectory $TempRoot $runLeaf
    Remove-Item -LiteralPath $runDirectory -Recurse -Force
    $runDirectoryRemoved = -not (Test-Path -LiteralPath $runDirectory)
    $freeBytesAfterCleanup = (Get-PSDrive -Name $drive.Name).Free
    $document = [ordered]@{
        schema = "colibri-lite-m6.1-03-storage-benchmark-v1"
        schema_version = 1
        profile_id = "windows-x64-$($env:COMPUTERNAME.ToLowerInvariant())-m6.1-03"
        created_at = $CreatedAt
        runtime = [ordered]@{ name = "colibri-lite-rs"; commit = $Commit; target_arch = "x86_64"; build_profile = "release" }
        storage_target = [ordered]@{ path = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($TempRoot)); free_bytes_before = $freeBytesBefore; free_bytes_after_payload_write = $freeBytesAfterPayloadWrite; free_bytes_after_cleanup = $freeBytesAfterCleanup }
        preflight = [ordered]@{ expected_new_output_bytes = $PayloadBytes; expected_peak_temporary_bytes = 0; safety_reserve_bytes = $safetyReserveBytes; required_free_bytes = $requiredFreeBytes; passed = $true }
        test_payload = [ordered]@{ logical_bytes = $PayloadBytes; expert_payload_bytes = $ExpertPayloadBytes; expert_payload_count = $ExpertPayloadCount; write_mode = "FileOptions.WriteThrough followed by Flush(true)"; created_in_unique_flat_run_directory = $true }
        cache_semantics = [ordered]@{
            first_touch = "first read of each range after test-file write and Flush(true); not claimed as a cold-device measurement because Windows physical filesystem-cache eviction was not requested or assumed"
            warm = "a prior complete read of the same file/ranges occurred in this process; measured as likely warm but not guaranteed resident"
        }
        results = [ordered]@{
            sequential_read = [ordered]@{
                unit = "GiB/s"
                first_touch_throughput = ($firstSequential.bytes / $firstSequential.seconds) / 1GB
                warm_throughput = New-Distribution $warmSequentialThroughputs.ToArray()
                payload_bytes = $PayloadBytes
                buffer_bytes = $SequentialBufferBytes
                warmup_passes = 1
                repetitions = $WarmSequentialSamples
            }
            expert_sized_random_read = [ordered]@{
                latency_unit = "ms"
                throughput_unit = "MiB/s"
                first_touch_latency = New-Distribution $firstTouchLatencies.ToArray()
                first_touch_throughput = New-Distribution $firstTouchThroughputs.ToArray()
                warm_latency = New-Distribution $warmRandomLatencies.ToArray()
                warm_throughput = New-Distribution $warmRandomThroughputs.ToArray()
                payload_bytes = $ExpertPayloadBytes
                buffer_bytes = $RandomBufferBytes
                repetitions_per_cache_state = $ExpertPayloadCount
                random_seeds = @(6102, 6103)
            }
        }
        checksums = [ordered]@{ sequential = $sequentialChecksum; random = $randomChecksum }
        cleanup = [ordered]@{ run_directory = $runDirectory; payload_bytes_removed = $PayloadBytes; run_directory_removed = $runDirectoryRemoved; retained_debug = $false }
        limitations = @(
            "This is a storage microbenchmark, not end-to-end model inference throughput.",
            "Windows physical filesystem-cache eviction was not requested, so first-touch is not claimed as a cold-device measurement.",
            "The test payload has the frozen F32 per-expert byte size but does not contain model weights.",
            "The host was not isolated or clock-pinned; background I/O can affect dispersion."
        )
    }
    $outputDirectory = Split-Path -Parent $OutputPath
    if ($outputDirectory) { New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null }
    [System.IO.File]::WriteAllText(
        [System.IO.Path]::GetFullPath($OutputPath),
        ($document | ConvertTo-Json -Depth 12),
        [System.Text.UTF8Encoding]::new($false))
    $success = $true
}
finally {
    if (Test-Path -LiteralPath $runDirectory) {
        Assert-RunDirectory $runDirectory $TempRoot $runLeaf
        Remove-Item -LiteralPath $runDirectory -Recurse -Force
    }
}

if (-not $success) { throw "storage benchmark did not complete" }
Write-Output "wrote $OutputPath"
