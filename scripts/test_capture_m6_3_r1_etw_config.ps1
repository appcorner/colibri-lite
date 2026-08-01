$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$capture = Join-Path $PSScriptRoot 'capture_m6_3_r1_etw.ps1'
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('colibri-etw-config-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$candidate = Join-Path $testRoot 'candidate.bin'
$authorization = Join-Path $testRoot 'cold-cache-authorization.json'
Set-Content -Encoding ascii -LiteralPath $candidate -Value 'candidate'
Set-Content -Encoding utf8 -LiteralPath $authorization -Value '{}'

function Write-Configuration([string] $name, [object] $document) {
    $path = Join-Path $testRoot $name
    $document | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 -LiteralPath $path
    return $path
}

function Invoke-Validation([string] $path) {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = & $powershell -NoProfile -ExecutionPolicy Bypass -File $capture `
        -Config $path -ValidateConfigOnly 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    return [ordered]@{ exit_code = $exitCode; output = ($output | Out-String) }
}

try {
    $base = [ordered]@{
        executable = $powershell
        argument_string = '-NoProfile -Command exit 0'
        working_directory = (Join-Path $repo 'crates\clr-qwen3-moe')
        artifacts = @($candidate)
        output_directory = $testRoot
        use_output_directory_as_run = $false
        environment = [ordered]@{
            COLIBRI_ARTIFACT_ROOT = 'D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1'
            COLIBRI_R1_1A_CANDIDATE_ID = 'test-candidate'
            COLIBRI_R1_1A_CANDIDATE_PATH = $candidate
            COLIBRI_R1_1A_COLD_CACHE_AUTHORIZATION = $authorization
        }
        required_environment = @(
            'COLIBRI_ARTIFACT_ROOT',
            'COLIBRI_R1_1A_CANDIDATE_ID',
            'COLIBRI_R1_1A_CANDIDATE_PATH',
            'COLIBRI_R1_1A_COLD_CACHE_AUTHORIZATION'
        )
    }
    $valid = Invoke-Validation (Write-Configuration 'valid.json' $base)
    if ($valid.exit_code -ne 0 -or $valid.output -notmatch '"status"\s*:\s*"passed"') {
        throw 'valid ETW configuration was rejected'
    }

    $missing = [ordered]@{}
    foreach ($key in $base.Keys) { if ($key -ne 'working_directory') { $missing[$key] = $base[$key] } }
    $missingResult = Invoke-Validation (Write-Configuration 'missing.json' $missing)
    if ($missingResult.exit_code -eq 0 -or $missingResult.output -notmatch 'working_directory is required') {
        throw 'missing working_directory was not rejected'
    }

    $outside = [ordered]@{}
    foreach ($key in $base.Keys) { $outside[$key] = $base[$key] }
    $outside.working_directory = $testRoot
    $outsideResult = Invoke-Validation (Write-Configuration 'outside.json' $outside)
    if ($outsideResult.exit_code -eq 0 -or $outsideResult.output -notmatch 'repository\s+root\s+or\s+one\s+of\s+its') {
        throw 'out-of-repository working_directory was not rejected'
    }

    $missingEnvironment = [ordered]@{}
    foreach ($key in $base.Keys) { $missingEnvironment[$key] = $base[$key] }
    $missingEnvironment.environment = [ordered]@{
        COLIBRI_R1_1A_CANDIDATE_ID = 'test-candidate'
        COLIBRI_R1_1A_CANDIDATE_PATH = $candidate
        COLIBRI_R1_1A_COLD_CACHE_AUTHORIZATION = $authorization
    }
    $environmentResult = Invoke-Validation (Write-Configuration 'missing-environment.json' $missingEnvironment)
    if ($environmentResult.exit_code -eq 0 -or $environmentResult.output -notmatch 'COLIBRI_ARTIFACT_ROOT') {
        throw 'missing required environment binding was not rejected'
    }

    $missingColdCache = [ordered]@{}
    foreach ($key in $base.Keys) { $missingColdCache[$key] = $base[$key] }
    $missingColdCache.environment = [ordered]@{
        COLIBRI_ARTIFACT_ROOT = 'D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1'
        COLIBRI_R1_1A_CANDIDATE_ID = 'test-candidate'
        COLIBRI_R1_1A_CANDIDATE_PATH = $candidate
    }
    $missingColdCache.required_environment = @(
        'COLIBRI_ARTIFACT_ROOT',
        'COLIBRI_R1_1A_CANDIDATE_ID',
        'COLIBRI_R1_1A_CANDIDATE_PATH'
    )
    $coldCacheResult = Invoke-Validation (Write-Configuration 'missing-cold-cache.json' $missingColdCache)
    if ($coldCacheResult.exit_code -eq 0 -or $coldCacheResult.output -notmatch 'COLD_CACHE_AUTHORIZATION') {
        throw 'missing cold-cache authorization binding was not rejected'
    }

    $multipleArtifacts = [ordered]@{}
    foreach ($key in $base.Keys) { $multipleArtifacts[$key] = $base[$key] }
    $multipleArtifacts.artifacts = @($candidate, (Join-Path $repo 'Cargo.toml'))
    $multipleResult = Invoke-Validation (Write-Configuration 'multiple-artifacts.json' $multipleArtifacts)
    if ($multipleResult.exit_code -eq 0 -or $multipleResult.output -notmatch 'exactly\s+one\s+artifact') {
        throw 'multiple candidate artifacts were not rejected'
    }

    Write-Output 'ETW config validation tests: 6 passed'
} finally {
    $resolvedTestRoot = [IO.Path]::GetFullPath($testRoot)
    $resolvedTempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ($resolvedTestRoot.StartsWith($resolvedTempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
