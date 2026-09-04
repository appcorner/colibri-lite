$ErrorActionPreference = 'Stop'

$repo = 'D:\sandboxs\colibri-lite'
$binary = Join-Path $repo 'target\release\deps\clr_qwen3_moe-98ff24c36e070e31.exe'
$harness = Join-Path $repo 'crates\clr-qwen3-moe\src\r2_2_quality_tests.rs'
$reference = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r1-2-f32-multitoken-reference-v1.tsv'
$artifactManifest = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-sentinel-artifacts-v1.json'
$qualityOutput = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-held-out-quality-v1.tsv'
$sentinelRoot = 'D:\tmp\colibri-lite-r2-2\artifacts'
$runRoot = 'D:\tmp\colibri-lite-r2-2\quality-durable-v2'

$expected = @{
    binary = '00fac75370364ec6592f5255af9ca3ea4dd8757ee429e08732715debbd0f3dbb'
    harness = '6d30857a02e7fe5e1d0941b5de0a5237198c588b8ef619e4dd2fd84d4c9a1d35'
    reference = 'fa8f1a4f6b30624b78cf8c004f379fdb0633a08a65c102791acc57d0205d54ff'
    manifest = '7abcb5f393472e3e63caf3bb7f06320aebcf292743a0709896d9c0de58a4d8f1'
    layer0 = '777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2'
    layer24 = '890936e78829cf013a141b225c7819cb2c10134fbd2c787e4131f7f41844d9b0'
    layer47 = 'a2a45a1f407d5cd2303286843001684fc7800953cd2b67995a2c5e62a9e66a33'
}

function Hash-Lower([string]$Path) {
    return (Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
if (Test-Path $runRoot) { throw 'durable quality run root must be new' }
New-Item -ItemType Directory -Path $runRoot | Out-Null

try {
    if ($env:COMPUTERNAME -ne 'ANUSORN-NB') { throw 'host identity mismatch' }
    if (Test-Path $qualityOutput) { throw 'quality output must be new' }
    if ((Get-Item $binary).Length -ne 44539904) { throw 'release binary byte length mismatch' }
    if ((Hash-Lower $binary) -ne $expected.binary) { throw 'release binary SHA mismatch' }
    if ((Hash-Lower $harness) -ne $expected.harness) { throw 'quality harness SHA mismatch' }
    if ((Hash-Lower $reference) -ne $expected.reference) { throw 'reference SHA mismatch' }
    if ((Hash-Lower $artifactManifest) -ne $expected.manifest) { throw 'artifact manifest SHA mismatch' }

    $sentinels = @(
        @('layer00-group32-repro.bin', $expected.layer0),
        @('layer24-group32.bin', $expected.layer24),
        @('layer47-group32.bin', $expected.layer47)
    )
    foreach ($entry in $sentinels) {
        $path = Join-Path $sentinelRoot $entry[0]
        if ((Get-Item $path).Length -ne 679477248) { throw "sentinel byte length mismatch: $path" }
        if ((Hash-Lower $path) -ne $entry[1]) { throw "sentinel SHA mismatch: $path" }
    }

    $preflight = [ordered]@{
        schema = 'm6.3-r2.2-quality-durable-preflight-v1'
        host = $env:COMPUTERNAME
        binary_sha256 = $expected.binary
        harness_sha256 = $expected.harness
        reference_sha256 = $expected.reference
        artifact_manifest_sha256 = $expected.manifest
    }
    $preflight | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $runRoot 'preflight.json')

    $env:COLIBRI_ARTIFACT_ROOT = 'D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1'
    $env:COLIBRI_R2_2_ARTIFACT_ROOT = $sentinelRoot
    $env:COLIBRI_R1_2_REFERENCE_PATH = $reference
    $env:COLIBRI_R2_2_QUALITY_OUTPUT = $qualityOutput

    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $args = @(
        'full_model_validation_tests::r2_2_quality_tests::m6_3_r2_2_three_sentinel_layers_pass_held_out_quality',
        '--exact',
        '--nocapture'
    )
    $native = Start-Process -FilePath $binary -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    $code = $native.ExitCode
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') $code

    $completion = [ordered]@{
        schema = 'm6.3-r2.2-quality-durable-completion-v1'
        exit_code = $code
        quality_output_exists = (Test-Path $qualityOutput)
        quality_output_bytes = if (Test-Path $qualityOutput) { (Get-Item $qualityOutput).Length } else { 0 }
    }
    $completion | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runRoot 'completion.json')
    exit $code
}
catch {
    $_ | Out-String | Set-Content -Encoding utf8 (Join-Path $runRoot 'worker-error.txt')
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') 97
    exit 97
}
