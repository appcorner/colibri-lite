$ErrorActionPreference = 'Stop'

$repo = 'D:\sandboxs\colibri-lite'
$binary = Join-Path $repo 'target\release\deps\clr_qwen3_moe-98ff24c36e070e31.exe'
$harness = Join-Path $repo 'crates\clr-qwen3-moe\src\r2_2_d4_tests.rs'
$contract = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d4-hybrid-quality-contract-v1.json'
$reference = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r1-2-f32-multitoken-reference-v1.tsv'
$output = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d4-held-out-quality-v1.tsv'
$group32Root = 'D:\tmp\colibri-lite-r2-2\artifacts'
$runRoot = 'D:\tmp\colibri-lite-r2-2\d4-durable-v1'

$expected = @{
    binary = '90878330b6cf8341010a7bc979302984ccb98024c80f9d1104290c4b5dd8ba20'
    harness = '12c7a0507611e0f5276cf8fed2a513243aeaada3205b13d2d174831fc630f2c2'
    contract = '5f2b27b623a0456ad15e14b542816e4d5ad17bf32dacd91099184adfc86fb4fe'
    reference = 'fa8f1a4f6b30624b78cf8c004f379fdb0633a08a65c102791acc57d0205d54ff'
}

function Hash-Lower([string]$Path) {
    return (Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
if (Test-Path $runRoot) { throw 'D4 durable run root must be new' }
New-Item -ItemType Directory -Path $runRoot | Out-Null

try {
    if ($env:COMPUTERNAME -ne 'ANUSORN-NB') { throw 'host identity mismatch' }
    if (Test-Path $output) { throw 'D4 evidence output must be new' }
    if ((Get-Item $binary).Length -ne 47668224) { throw 'release binary byte length mismatch' }
    if ((Hash-Lower $binary) -ne $expected.binary) { throw 'release binary SHA mismatch' }
    if ((Hash-Lower $harness) -ne $expected.harness) { throw 'D4 harness SHA mismatch' }
    if ((Hash-Lower $contract) -ne $expected.contract) { throw 'D4 contract SHA mismatch' }
    if ((Hash-Lower $reference) -ne $expected.reference) { throw 'reference SHA mismatch' }

    $artifacts = @(
        @('layer00-group32-repro.bin', 679477248, '777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2'),
        @('layer24-group32.bin', 679477248, '890936e78829cf013a141b225c7819cb2c10134fbd2c787e4131f7f41844d9b0')
    )
    foreach ($entry in $artifacts) {
        $path = Join-Path $group32Root $entry[0]
        if ((Get-Item $path).Length -ne $entry[1]) { throw "artifact byte length mismatch: $path" }
        if ((Hash-Lower $path) -ne $entry[2]) { throw "artifact SHA mismatch: $path" }
    }

    $preflight = [ordered]@{
        schema = 'm6.3-r2.2-d4-durable-preflight-v1'
        host = $env:COMPUTERNAME
        binary_sha256 = $expected.binary
        harness_sha256 = $expected.harness
        contract_sha256 = $expected.contract
        reference_sha256 = $expected.reference
        verified_artifact_count = $artifacts.Count
    }
    $preflight | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $runRoot 'preflight.json')

    $env:COLIBRI_ARTIFACT_ROOT = 'D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1'
    $env:COLIBRI_R2_2_ARTIFACT_ROOT = $group32Root
    $env:COLIBRI_R1_2_REFERENCE_PATH = $reference
    $env:COLIBRI_R2_2_D4_OUTPUT = $output
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $args = @(
        'full_model_validation_tests::r2_2_d4_tests::m6_3_r2_2_d4_hybrid_pass_held_out_quality',
        '--exact',
        '--nocapture'
    )
    $native = Start-Process -FilePath $binary -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    $code = $native.ExitCode
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') $code

    $completion = [ordered]@{
        schema = 'm6.3-r2.2-d4-durable-completion-v1'
        exit_code = $code
        d4_output_exists = (Test-Path $output)
        d4_output_bytes = if (Test-Path $output) { (Get-Item $output).Length } else { 0 }
    }
    $completion | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runRoot 'completion.json')
    exit $code
}
catch {
    $_ | Out-String | Set-Content -Encoding utf8 (Join-Path $runRoot 'worker-error.txt')
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') 97
    exit 97
}

