param(
    [Parameter(Mandatory = $true)] [string] $Config,
    [string] $Python = 'python',
    [switch] $ValidateConfigOnly
)

$ErrorActionPreference = 'Stop'
$failurePath = $null
trap {
    if ($failurePath) {
        ($_ | Out-String) | Set-Content -Encoding utf8 -LiteralPath $failurePath
    } else {
        Write-Error $_
    }
    exit 2
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$configuration = Get-Content -Raw -Encoding utf8 -LiteralPath $Config | ConvertFrom-Json
$Executable = [string]$configuration.executable
$ArgumentString = [string]$configuration.argument_string
$workingDirectoryValue = [string]$configuration.working_directory
if ([string]::IsNullOrWhiteSpace($workingDirectoryValue)) {
    throw 'working_directory is required'
}
$WorkingDirectory = (Resolve-Path -LiteralPath $workingDirectoryValue -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $WorkingDirectory -PathType Container)) {
    throw 'working_directory must resolve to a directory'
}
$repoPrefix = $repo.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if ($WorkingDirectory -ne $repo -and -not $WorkingDirectory.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'working_directory must be the repository root or one of its descendants'
}
$Artifact = @($configuration.artifacts | ForEach-Object { [string]$_ })
if ($configuration.artifact_directories) {
    foreach ($directory in @($configuration.artifact_directories | ForEach-Object { [string]$_ })) {
        if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
            throw "artifact directory not found: $directory"
        }
        $Artifact += @(Get-ChildItem -LiteralPath $directory -File -Filter '*.bin' |
            Sort-Object -Property FullName |
            ForEach-Object { $_.FullName })
    }
}
$Artifact = @($Artifact | Sort-Object -Unique)
if ($Artifact.Count -eq 0) { throw 'at least one artifact file is required' }
$OutputDirectory = [string]$configuration.output_directory
$outputBase = [IO.Path]::GetFullPath($OutputDirectory)
$useOutputDirectoryAsRun = [bool]$configuration.use_output_directory_as_run
if ($useOutputDirectoryAsRun) {
    if (-not (Test-Path -LiteralPath $outputBase -PathType Container)) {
        throw 'fixed flat run directory must exist before ETW capture'
    }
    if ((Split-Path -Leaf (Split-Path -Parent $outputBase)) -ne 'colibri-lite-runs') {
        throw 'fixed flat run directory must be directly under colibri-lite-runs'
    }
    $outputRoot = $outputBase
    $runId = Split-Path -Leaf $outputRoot
    foreach ($artifactPath in $Artifact) {
        if ((Split-Path -Parent ([IO.Path]::GetFullPath($artifactPath))) -ne $outputRoot) {
            throw 'candidate artifact must be a direct child of the fixed flat run directory'
        }
    }
} else {
    if ($ValidateConfigOnly) {
        $runId = 'validation-only'
        $outputRoot = $outputBase
    } else {
        $runId = 'run-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [guid]::NewGuid().ToString('N')
        $outputRoot = Join-Path $outputBase $runId
        New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
        Set-Content -Encoding ascii -LiteralPath (Join-Path $outputBase 'latest-run.txt') -Value $outputRoot
    }
}
$ArgumentString = $ArgumentString.Replace('{run_dir}', $outputRoot)
$Environment = @{}
if ($configuration.environment) {
    $configuration.environment.psobject.Properties | ForEach-Object {
        $Environment[$_.Name] = ([string]$_.Value).Replace('{run_dir}', $outputRoot)
    }
}
$requiredEnvironment = @($configuration.required_environment | ForEach-Object { [string]$_ })
foreach ($name in $requiredEnvironment) {
    if ([string]::IsNullOrWhiteSpace($name) -or -not $Environment.ContainsKey($name) -or
        [string]::IsNullOrWhiteSpace([string]$Environment[$name])) {
        throw "required environment binding is missing: $name"
    }
}
$isCandidateRun = $Environment.ContainsKey('COLIBRI_R1_1A_CANDIDATE_ID') -or
    $Environment.ContainsKey('COLIBRI_R1_1A_CANDIDATE_PATH')
$coldCacheAuthorization = $null
if ($isCandidateRun) {
    if ($Artifact.Count -ne 1) {
        throw 'candidate capture requires exactly one artifact file'
    }
    foreach ($name in @(
        'COLIBRI_ARTIFACT_ROOT',
        'COLIBRI_R1_1A_CANDIDATE_ID',
        'COLIBRI_R1_1A_CANDIDATE_PATH',
        'COLIBRI_R1_1A_COLD_CACHE_AUTHORIZATION'
    )) {
        if (-not $requiredEnvironment.Contains($name) -or -not $Environment.ContainsKey($name) -or
            [string]::IsNullOrWhiteSpace([string]$Environment[$name])) {
            throw "candidate configuration requires binding: $name"
        }
    }
    $coldCacheAuthorization = [IO.Path]::GetFullPath([string]$Environment['COLIBRI_R1_1A_COLD_CACHE_AUTHORIZATION'])
    if (-not (Test-Path -LiteralPath $coldCacheAuthorization -PathType Leaf)) {
        throw 'cold-cache authorization file is missing'
    }
    if ([IO.Path]::GetFullPath([string]$Environment['COLIBRI_R1_1A_CANDIDATE_PATH']) -ne
        [IO.Path]::GetFullPath($Artifact[0])) {
        throw 'candidate path must equal the captured artifact path'
    }
}
if ($ValidateConfigOnly) {
    [ordered]@{
        status = 'passed'
        executable = [IO.Path]::GetFullPath($Executable)
        working_directory = $WorkingDirectory
        artifact_count = $Artifact.Count
        required_environment = $requiredEnvironment
    } | ConvertTo-Json -Depth 4
    exit 0
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'capture_m6_3_r1_etw.ps1 requires an elevated Administrator PowerShell'
}
$coldCacheConsumption = $null
if ($isCandidateRun) {
    $coldCacheTool = Join-Path $PSScriptRoot 'm6_3_r1_1a_cold_cache.py'
    $coldCacheContract = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r1-1a-cold-cache-contract-v1.json'
    $coldCacheOutput = & $Python $coldCacheTool verify-launch `
        --contract $coldCacheContract `
        --authorization $coldCacheAuthorization `
        --candidate-id ([string]$Environment['COLIBRI_R1_1A_CANDIDATE_ID']) `
        --artifact ([string]$Environment['COLIBRI_R1_1A_CANDIDATE_PATH']) `
        --consume
    if ($LASTEXITCODE -ne 0) {
        throw "cold-cache authorization validation failed: $($coldCacheOutput | Out-String)"
    }
    $coldCacheConsumption = $coldCacheOutput | ConvertFrom-Json
}
$parser = Join-Path $PSScriptRoot 'parse_m6_3_r1_etw.py'
$providerFile = Join-Path $PSScriptRoot 'm6_3_r1_etw-providers.txt'
$failurePath = Join-Path $outputRoot 'capture.failure.txt'
$etl = Join-Path $outputRoot 'kernel-file-disk.etl'
$csv = Join-Path $outputRoot 'kernel-file-disk.csv'
$summary = Join-Path $outputRoot 'kernel-file-disk.summary.txt'
$correlation = Join-Path $outputRoot 'correlation.json'
$stdout = Join-Path $outputRoot 'process.stdout.txt'
$stderr = Join-Path $outputRoot 'process.stderr.txt'
$resultPath = Join-Path $outputRoot 'capture.json'
$traceStarted = $false
$process = $null
$sessionName = 'ColibriM63-' + [guid]::NewGuid().ToString('N')
Set-Content -Encoding utf8 -LiteralPath $failurePath -Value 'capture started'

function Stop-TraceSession {
    param([Parameter(Mandatory = $true)] [string] $Name)

    $stopClient = Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\logman.exe') `
        -ArgumentList @('stop', $Name, '-ets') -PassThru -WindowStyle Hidden
    $clientTimedOut = -not $stopClient.WaitForExit(30000)
    if ($clientTimedOut) {
        # The control client can remain blocked after ETW has already stopped the session.
        $stopClient.Kill()
        $stopClient.WaitForExit()
    }
    $stopClient.Dispose()

    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        & logman.exe query $Name -ets 2>$null | Out-Null
        $active = $LASTEXITCODE -eq 0
        if (-not $active -and (Test-Path -LiteralPath $etl)) {
            return [ordered]@{ stopped = $true; client_timed_out = $clientTimedOut }
        }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "ETW session '$Name' did not stop within 30 seconds"
}

try {
    # Reserve enough kernel logger buffers for the full reference trace.  The
    # default logger buffer pool dropped events under the 49-artifact workload.
    & logman.exe create trace $sessionName -pf $providerFile -o $etl -ow -bs 1024 -nb 512 512 -ets
    if ($LASTEXITCODE -ne 0) { throw "logman start failed with exit code $LASTEXITCODE" }
    $traceStarted = $true

    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = [IO.Path]::GetFullPath($Executable)
    $psi.Arguments = $ArgumentString
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    foreach ($name in $Environment.Keys) { $psi.EnvironmentVariables[$name] = $Environment[$name] }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $psi
    if (-not $process.Start()) { throw 'benchmark process did not start' }
    $pidValue = $process.Id
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $sampledWorkingSet = [int64]0
    $sampledPrivate = [int64]0
    $samples = 0
    while (-not $process.HasExited) {
        $process.Refresh()
        $sampledWorkingSet = [Math]::Max($sampledWorkingSet, [int64]$process.WorkingSet64)
        $sampledPrivate = [Math]::Max($sampledPrivate, [int64]$process.PrivateMemorySize64)
        $samples++
        Start-Sleep -Milliseconds 100
    }
    $process.WaitForExit()
    $process.Refresh()
    $sampledWorkingSet = [Math]::Max($sampledWorkingSet, [int64]$process.WorkingSet64)
    $sampledPrivate = [Math]::Max($sampledPrivate, [int64]$process.PrivateMemorySize64)
    Set-Content -Encoding utf8 -LiteralPath $stdout -Value $stdoutTask.Result
    Set-Content -Encoding utf8 -LiteralPath $stderr -Value $stderrTask.Result

    $stopResult = Stop-TraceSession -Name $sessionName
    $traceStarted = $false

    & tracerpt.exe $etl -of CSV -o $csv -summary $summary -y
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $csv) -or -not (Test-Path -LiteralPath $summary)) {
        throw "tracerpt failed with exit code $LASTEXITCODE"
    }

    $parserArguments = @('--csv', $csv, '--summary', $summary, '--pid', "$pidValue")
    foreach ($path in $Artifact) { $parserArguments += @('--artifact', [IO.Path]::GetFullPath($path)) }
    $parserArguments += @('--output', $correlation)
    & $Python $parser @parserArguments
    $parserExit = $LASTEXITCODE
    $parsed = Get-Content -Raw -Encoding utf8 -LiteralPath $correlation | ConvertFrom-Json
    $record = [ordered]@{
        schema = 'm6.3-r1-etw-capture-v1'
        run_id = $runId
        pid = $pidValue
        executable = [IO.Path]::GetFullPath($Executable)
        arguments = $ArgumentString
        working_directory = $WorkingDirectory
        environment = $Environment
        cold_cache = if ($isCandidateRun) {
            [ordered]@{
                authorization_path = $coldCacheAuthorization
                consumption_path = $coldCacheConsumption.consumed_path
                status = 'consumed_before_etw_launch'
            }
        } else { $null }
        exit_code = $process.ExitCode
        samples = [Math]::Max(1, $samples)
        memory = [ordered]@{
            working_set_peak_bytes = $sampledWorkingSet
            peak_working_set_bytes = [Math]::Max($sampledWorkingSet, [int64]$process.PeakWorkingSet64)
            private_bytes_peak = $sampledPrivate
        }
        etw = [ordered]@{
            status = $parsed.status
            trace_path = $etl
            trace_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $etl).Hash.ToLowerInvariant()
            trace_bytes = (Get-Item -LiteralPath $etl).Length
            csv_path = $csv
            correlation_path = $correlation
            physical_read_bytes = $parsed.physical_read_bytes
            parser_exit_code = $parserExit
            collector = 'logman'
            session_name = $sessionName
            stop_client_timed_out = $stopResult.client_timed_out
            provider_file = $providerFile
            provider_file_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $providerFile).Hash.ToLowerInvariant()
            tracerpt_version = (Get-Item "$env:SystemRoot\System32\tracerpt.exe").VersionInfo.FileVersion
        }
        artifacts = @($Artifact | ForEach-Object {
            [ordered]@{
                path = [IO.Path]::GetFullPath($_)
                bytes = (Get-Item -LiteralPath $_).Length
                sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_).Hash.ToLowerInvariant()
            }
        })
    }
    $record | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 -LiteralPath $resultPath
    if ($process.ExitCode -ne 0 -or $parsed.status -ne 'correlated') { exit 1 }
} finally {
    if ($traceStarted) { Stop-TraceSession -Name $sessionName | Out-Null }
    if ($process) { $process.Dispose() }
}
