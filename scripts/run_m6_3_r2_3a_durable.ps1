$ErrorActionPreference = 'Stop'

$repo = 'D:\sandboxs\colibri-lite'
$binary = Join-Path $repo 'target\release\deps\clr_qwen3_moe-98ff24c36e070e31.exe'
$harness = Join-Path $repo 'crates\clr-qwen3-moe\src\r2_3a_reference_tests.rs'
$contract = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-3-implementation-contract-v1.json'
$planResult = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-3-all-layer-plan-review-result-v1.json'
$reference = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r1-2-f32-multitoken-reference-v1.tsv'
$modelManifest = Join-Path $repo 'models\qwen3-30b-a3b\model-manifest-v1.json'
$artifactRoot = 'D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1'
$runRoot = 'D:\tmp\colibri-lite-r2-3\r2-3a-durable-v1'
$output = Join-Path $runRoot 'm6.3-r2-3a-f32-four-token-reference-v1.tsv'

$expected = @{
    binary = '3419b81c80f343cca7fe91c003a656a43155c86fbe3be45118198ca494c75ec9'
    harness = '0710e566ef9550cc46601617f14ee952aa4830163afcfacd87a8a73b79b7dfe9'
    contract = '97dd33b5cdb576d1ebcf9b3b7661f5462c912be40c7485c7e6bd7f8773d7771b'
    plan_result = 'c525cfc8ed536fb7d0033104f41e5e2f64a0d9a2842c4fcddba5b32dfac7e6e2'
    reference = 'fa8f1a4f6b30624b78cf8c004f379fdb0633a08a65c102791acc57d0205d54ff'
    model_manifest = 'f133d733612840ad691d637732d4ef2de1e0242c4bb1d92521b49dfcfb1b8cd2'
}

function Hash-Lower([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (Test-Path -LiteralPath $runRoot) { throw 'R2.3a durable run root must be new' }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

try {
    if ($env:COMPUTERNAME -ne 'ANUSORN-NB') { throw 'host identity mismatch' }
    if ((Get-Item -LiteralPath $binary).Length -ne 50015744) { throw 'release binary byte length mismatch' }
    if ((Hash-Lower $binary) -ne $expected.binary) { throw 'release binary SHA mismatch' }
    if ((Hash-Lower $harness) -ne $expected.harness) { throw 'R2.3a harness SHA mismatch' }
    if ((Hash-Lower $contract) -ne $expected.contract) { throw 'R2.3 implementation contract SHA mismatch' }
    if ((Hash-Lower $planResult) -ne $expected.plan_result) { throw 'R2.3 plan-review result SHA mismatch' }
    if ((Hash-Lower $reference) -ne $expected.reference) { throw 'old reference SHA mismatch' }
    if ((Hash-Lower $modelManifest) -ne $expected.model_manifest) { throw 'model manifest SHA mismatch' }
    if (-not (Test-Path -LiteralPath $artifactRoot -PathType Container)) { throw 'artifact root missing' }
    if (Test-Path -LiteralPath $output) { throw 'R2.3a output must be new' }

    $preflight = [ordered]@{
        schema = 'm6.3-r2.3a-durable-preflight-v1'
        host = $env:COMPUTERNAME
        binary_sha256 = $expected.binary
        harness_sha256 = $expected.harness
        implementation_contract_sha256 = $expected.contract
        plan_review_result_sha256 = $expected.plan_result
        old_reference_sha256 = $expected.reference
        model_manifest_sha256 = $expected.model_manifest
        generated_token_count = 4
        fixtures = @('short_english', 'short_thai')
    }
    $preflight | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $runRoot 'preflight.json')

    $env:COLIBRI_ARTIFACT_ROOT = $artifactRoot
    $env:COLIBRI_EXPERT_CACHE_BUDGET_BYTES = '18874368'
    $env:COLIBRI_R1_2_REFERENCE_PATH = $reference
    $env:COLIBRI_R2_3A_OUTPUT = $output

    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $args = @(
        'full_model_validation_tests::r2_3a_reference_tests::m6_3_r2_3a_freeze_four_token_f32_reference',
        '--exact',
        '--nocapture'
    )
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $native = Start-Process -FilePath $binary -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    $watch.Stop()
    $code = $native.ExitCode
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') $code

    $completion = [ordered]@{
        schema = 'm6.3-r2.3a-durable-completion-v1'
        exit_code = $code
        runtime_seconds = $watch.Elapsed.TotalSeconds
        output_exists = (Test-Path -LiteralPath $output)
        output_bytes = if (Test-Path -LiteralPath $output) { (Get-Item -LiteralPath $output).Length } else { 0 }
        stderr_bytes = if (Test-Path -LiteralPath $stderr) { (Get-Item -LiteralPath $stderr).Length } else { 0 }
    }
    $completion | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runRoot 'completion.json')
    exit $code
}
catch {
    $_ | Out-String | Set-Content -Encoding utf8 (Join-Path $runRoot 'worker-error.txt')
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') 97
    exit 97
}
