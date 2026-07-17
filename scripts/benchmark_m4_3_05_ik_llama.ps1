param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [Parameter(Mandatory = $true)] [string] $Executable,
    [Parameter(Mandatory = $true)] [string] $OutputRoot,
    [int] $Threads = 8,
    [string] $RunPrefix = 'run'
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$common = @(
    '-m', $Model,
    '-p', '"Hello, world!"',
    '-t', "$Threads",
    '-tb', "$Threads",
    '-c', '128',
    '-n', '2',
    '--no-kv-offload',
    '--no-warmup',
    '--no-display-prompt',
    '--simple-io',
    '--temp', '0',
    '--top-k', '0',
    '--top-p', '1',
    '--min-p', '0',
    '--repeat-penalty', '1',
    '--seed', '0',
    '--no-context-shift'
)

function Invoke-IkRun([string] $Label, [int] $Predict, [string] $CacheAssumption) {
    $stdoutPath = Join-Path $OutputRoot "$RunPrefix-$Label.stdout.txt"
    $stderrPath = Join-Path $OutputRoot "$RunPrefix-$Label.stderr.txt"
    $args = [System.Collections.Generic.List[string]]::new()
    foreach ($arg in $common) { [void] $args.Add($arg) }
    $args[$args.IndexOf('-n') + 1] = "$Predict"

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Executable
    $psi.Arguments = ($args -join ' ')
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    [void] $proc.Start()
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $peakWorkingSet = [int64] 0
    $peakPrivate = [int64] 0
    $peakCpuSeconds = [double] 0
    $samples = 0
    while (-not $proc.HasExited) {
        try {
            $proc.Refresh()
            $peakWorkingSet = [Math]::Max($peakWorkingSet, [int64] $proc.WorkingSet64)
            $peakPrivate = [Math]::Max($peakPrivate, [int64] $proc.PrivateMemorySize64)
            $peakCpuSeconds = [Math]::Max($peakCpuSeconds, [double] $proc.TotalProcessorTime.TotalSeconds)
            $samples++
        } catch [System.InvalidOperationException] {
            break
        }
        Start-Sleep -Milliseconds 100
    }
    $proc.WaitForExit()
    $sw.Stop()
    try {
        $proc.Refresh()
        $peakWorkingSet = [Math]::Max($peakWorkingSet, [int64] $proc.WorkingSet64)
        $peakPrivate = [Math]::Max($peakPrivate, [int64] $proc.PrivateMemorySize64)
    } catch [System.InvalidOperationException] { }

    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    Set-Content -Encoding utf8 -LiteralPath $stdoutPath -Value $stdout
    Set-Content -Encoding utf8 -LiteralPath $stderrPath -Value $stderr
    $load = $null
    $prompt = $null
    $eval = $null
    foreach ($line in ($stderr -split "`r?`n")) {
        if ($line -match 'load time\s*=\s*([0-9.+-]+) ms') { $load = [double]$Matches[1] }
        if ($line -match 'prompt eval time\s*=\s*([0-9.+-]+) ms') { $prompt = [double]$Matches[1] }
        if ($line -match 'eval time\s*=\s*([0-9.+-]+) ms') { $eval = [double]$Matches[1] }
    }
    [ordered]@{
        label = "$RunPrefix-$Label"
        predict_tokens = $Predict
        cache_assumption = $CacheAssumption
        exit_status = $proc.ExitCode
        wall_seconds = $sw.Elapsed.TotalSeconds
        cpu_seconds = $peakCpuSeconds
        model_load_ms = $load
        prompt_eval_ms = $prompt
        decode_eval_ms = $eval
        prompt_tokens = 4
        generated_tokens_requested = $Predict
        prompt_tokens_per_second = if ($prompt -is [double] -and $prompt -gt 0) { 4000.0 / $prompt } else { $null }
        generated_tokens_per_second = if ($eval -is [double] -and $eval -gt 0) { 1000.0 * $Predict / $eval } else { $null }
        peak_working_set_bytes = $peakWorkingSet
        peak_private_bytes = $peakPrivate
        memory_sample_count = $samples
        generated_text = $stdout.Trim()
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
        command = ($Executable + ' ' + ($args -join ' '))
    }
}

$results = @()
$results += Invoke-IkRun 'fresh-a' 2 'fresh process; filesystem cache uncontrolled'
$results += Invoke-IkRun 'fresh-b' 2 'fresh process after A; filesystem cache uncontrolled'
$results += Invoke-IkRun 'fresh-c' 2 'fresh process after B; potentially filesystem-cache-warm'
$results += Invoke-IkRun 'warm-repeat' 2 'new process; explicitly potentially filesystem-cache-warm'
$results += Invoke-IkRun 'long-decode' 32 'new process; filesystem cache uncontrolled; long decode timing'

$metadata = [ordered]@{
    schema = 'm4.3-05-ik-llama-run-v1'
    model = [ordered]@{ path = $Model; bytes = (Get-Item -LiteralPath $Model).Length }
    executable = [ordered]@{ path = $Executable; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Executable).Hash }
    cpu_only = $true
    gpu_layers = 0
    threads = $Threads
    prompt = 'Hello, world!'
    prompt_token_ids = @(9707, 11, 1879, 0)
    context_size = 128
    kv_cache = 'f16; runtime-managed; exact allocation not exposed by CLI'
    mmap = 'enabled by default'
    runs = $results
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 -LiteralPath (Join-Path $OutputRoot 'ik-runs.json')
