param(
    [Parameter(Mandatory = $true)] [string] $Config,
    [switch] $ValidateConfigOnly
)

$ErrorActionPreference = 'Stop'
$failurePath = $null
trap {
    if ($failurePath) { ($_ | Out-String) | Set-Content -Encoding utf8 -LiteralPath $failurePath }
    else { Write-Error $_ }
    exit 2
}

$configuration = Get-Content -Raw -Encoding utf8 -LiteralPath $Config | ConvertFrom-Json
$Executable = [IO.Path]::GetFullPath([string]$configuration.executable)
$WorkingDirectory = [IO.Path]::GetFullPath([string]$configuration.working_directory)
$ArgumentString = [string]$configuration.argument_string
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw 'executable not found' }
if (-not (Test-Path -LiteralPath $WorkingDirectory -PathType Container)) { throw 'working directory not found' }

$OutputBase = [IO.Path]::GetFullPath([string]$configuration.output_directory)
if (-not (Test-Path -LiteralPath $OutputBase -PathType Container)) { throw 'output directory not found' }
$Environment = @{}
if ($configuration.environment) {
    $configuration.environment.psobject.Properties | ForEach-Object { $Environment[$_.Name] = [string]$_.Value }
}
foreach ($name in @($configuration.required_environment | ForEach-Object { [string]$_ })) {
    if (-not $Environment.ContainsKey($name) -or [string]::IsNullOrWhiteSpace([string]$Environment[$name])) {
        throw "required environment binding is missing: $name"
    }
}
foreach ($identity in @($configuration.artifact_identities)) {
    $path = [IO.Path]::GetFullPath([string]$identity.path)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "artifact not found: $path" }
    if ((Get-Item -LiteralPath $path).Length -ne [int64]$identity.bytes) { throw "artifact byte length mismatch: $path" }
}
if ($ValidateConfigOnly) {
    Write-Output '{"status":"passed","collector":"d5c-memory-v1"}'
    exit 0
}

$runId = 'run-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N')
$outputRoot = Join-Path $OutputBase $runId
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
Set-Content -Encoding ascii -LiteralPath (Join-Path $OutputBase 'latest-run.txt') -Value $outputRoot
$failurePath = Join-Path $outputRoot 'capture.failure.txt'
Set-Content -Encoding utf8 -LiteralPath $failurePath -Value 'capture started'
$stdoutPath = Join-Path $outputRoot 'process.stdout.txt'
$stderrPath = Join-Path $outputRoot 'process.stderr.txt'
$resultPath = Join-Path $outputRoot 'capture.json'

$ArgumentString = $ArgumentString.Replace('{run_dir}', $outputRoot)
foreach ($name in @($Environment.Keys)) { $Environment[$name] = ([string]$Environment[$name]).Replace('{run_dir}', $outputRoot) }
$readyMarker = ([string]$configuration.ready_marker).Replace('{run_dir}', $outputRoot)
$goMarker = ([string]$configuration.go_marker).Replace('{run_dir}', $outputRoot)
if ([string]::IsNullOrWhiteSpace($readyMarker) -or [string]::IsNullOrWhiteSpace($goMarker)) { throw 'READY/GO markers are required' }
$readyMarker = [IO.Path]::GetFullPath($readyMarker)
$goMarker = [IO.Path]::GetFullPath($goMarker)

$process = $null
try {
    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Executable
    $psi.Arguments = $ArgumentString
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($name in $Environment.Keys) { $psi.EnvironmentVariables[$name] = $Environment[$name] }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $psi
    $processWall = [Diagnostics.Stopwatch]::StartNew()
    if (-not $process.Start()) { throw 'benchmark process did not start' }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $sampledWorkingSet = [int64]0
    $sampledPrivate = [int64]0
    $samples = 0
    $readyDeadline = [DateTime]::UtcNow.AddMinutes(10)
    while (-not (Test-Path -LiteralPath $readyMarker)) {
        if ($process.HasExited) { throw 'benchmark process exited before READY marker' }
        $process.Refresh()
        $sampledWorkingSet = [Math]::Max($sampledWorkingSet, [int64]$process.WorkingSet64)
        $sampledPrivate = [Math]::Max($sampledPrivate, [int64]$process.PrivateMemorySize64)
        $samples++
        if ([DateTime]::UtcNow -ge $readyDeadline) { throw 'READY marker timeout' }
        Start-Sleep -Milliseconds 100
    }
    Set-Content -Encoding ascii -LiteralPath $goMarker -Value 'go'
    while (-not $process.HasExited) {
        $process.Refresh()
        $sampledWorkingSet = [Math]::Max($sampledWorkingSet, [int64]$process.WorkingSet64)
        $sampledPrivate = [Math]::Max($sampledPrivate, [int64]$process.PrivateMemorySize64)
        $samples++
        Start-Sleep -Milliseconds 100
    }
    $process.WaitForExit()
    $processWall.Stop()
    $process.Refresh()
    $sampledWorkingSet = [Math]::Max($sampledWorkingSet, [int64]$process.WorkingSet64)
    $sampledPrivate = [Math]::Max($sampledPrivate, [int64]$process.PrivateMemorySize64)
    Set-Content -Encoding utf8 -LiteralPath $stdoutPath -Value $stdoutTask.Result
    Set-Content -Encoding utf8 -LiteralPath $stderrPath -Value $stderrTask.Result
    $record = [ordered]@{
        schema = 'm6.3-r2.2-d5c-memory-capture-v1'
        run_id = $runId
        pid = $process.Id
        executable = $Executable
        arguments = $ArgumentString
        working_directory = $WorkingDirectory
        exit_code = $process.ExitCode
        process_wall_seconds = $processWall.Elapsed.TotalSeconds
        samples = [Math]::Max(1, $samples)
        handshake = [ordered]@{ ready_marker = $readyMarker; go_marker = $goMarker; ready_before_go = $true; memory_sampler_active = $true }
        memory = [ordered]@{
            working_set_peak_bytes = $sampledWorkingSet
            peak_working_set_bytes = [Math]::Max($sampledWorkingSet, [int64]$process.PeakWorkingSet64)
            private_bytes_peak = $sampledPrivate
        }
        artifacts = @($configuration.artifact_identities)
        collector = 'd5c-memory-v1'
    }
    $record | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 -LiteralPath $resultPath
    if ($process.ExitCode -ne 0) { exit 1 }
} finally {
    if ($process) { $process.Dispose() }
}
