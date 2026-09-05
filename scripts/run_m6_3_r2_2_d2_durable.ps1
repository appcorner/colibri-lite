$ErrorActionPreference = 'Stop'

$repo = 'D:\sandboxs\colibri-lite'
$binary = Join-Path $repo 'target\release\deps\clr_qwen3_moe-98ff24c36e070e31.exe'
$harness = Join-Path $repo 'crates\clr-qwen3-moe\src\r2_2_d2_tests.rs'
$reference = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r1-2-f32-multitoken-reference-v1.tsv'
$artifactManifest = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d2-artifacts-v1.json'
$output = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d2-evidence-v1.tsv'
$group32Root = 'D:\tmp\colibri-lite-r2-2\artifacts'
$d2Root = 'D:\tmp\colibri-lite-r2-2\d2-artifacts'
$runRoot = 'D:\tmp\colibri-lite-r2-2\d2-durable-v1'

$expected = @{
    binary = 'caa27c71e0d75e7f45ceb2c24b9e48d82db6199a1c07b32b8bfc6fcc32d3cc33'
    harness = '457a3ba73312e93fbd5b7b86c1cdda6b62ec5c1491b6236491a316e18b80c1a8'
    reference = 'fa8f1a4f6b30624b78cf8c004f379fdb0633a08a65c102791acc57d0205d54ff'
    manifest = 'bd54c87ae20826086a9f8e6b1e06011bb4f1c65fa973494804c3fcb6b79e0a46'
}

function Hash-Lower([string]$Path) {
    return (Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
if (Test-Path $runRoot) { throw 'D2 durable run root must be new' }
New-Item -ItemType Directory -Path $runRoot | Out-Null

try {
    if ($env:COMPUTERNAME -ne 'ANUSORN-NB') { throw 'host identity mismatch' }
    if (Test-Path $output) { throw 'D2 evidence output must be new' }
    if ((Get-Item $binary).Length -ne 46561280) { throw 'release binary byte length mismatch' }
    if ((Hash-Lower $binary) -ne $expected.binary) { throw 'release binary SHA mismatch' }
    if ((Hash-Lower $harness) -ne $expected.harness) { throw 'D2 harness SHA mismatch' }
    if ((Hash-Lower $reference) -ne $expected.reference) { throw 'reference SHA mismatch' }
    if ((Hash-Lower $artifactManifest) -ne $expected.manifest) { throw 'D2 artifact manifest SHA mismatch' }

    $artifacts = @(
        @($group32Root, 'layer00-group32-repro.bin', 679477248, '777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2'),
        @($group32Root, 'layer24-group32.bin', 679477248, '890936e78829cf013a141b225c7819cb2c10134fbd2c787e4131f7f41844d9b0'),
        @($group32Root, 'layer47-group32.bin', 679477248, 'a2a45a1f407d5cd2303286843001684fc7800953cd2b67995a2c5e62a9e66a33'),
        @($d2Root, 'layer24-group16.bin', 754974720, '569b221f448b521488764da8b0e99058f8daded9f41ba10ef7f1f2706ac36416'),
        @($d2Root, 'layer24-group8.bin', 905969664, '4d51c17a2ba040229a8c20c895c5be8f030ab273db3151ccb739df7289965435'),
        @($d2Root, 'layer47-group16.bin', 754974720, 'ab11ed88973d394e01e3430b9607f0b42c2752cd1631ed8eff2c0c716a639c19'),
        @($d2Root, 'layer47-group8.bin', 905969664, '7a57ce8bcf1d05b0d82491c88e644c994d2adafedf0285596a62da81c29c200d')
    )
    foreach ($entry in $artifacts) {
        $path = Join-Path $entry[0] $entry[1]
        if ((Get-Item $path).Length -ne $entry[2]) { throw "artifact byte length mismatch: $path" }
        if ((Hash-Lower $path) -ne $entry[3]) { throw "artifact SHA mismatch: $path" }
    }

    $preflight = [ordered]@{
        schema = 'm6.3-r2.2-d2-durable-preflight-v1'
        host = $env:COMPUTERNAME
        binary_sha256 = $expected.binary
        harness_sha256 = $expected.harness
        reference_sha256 = $expected.reference
        artifact_manifest_sha256 = $expected.manifest
        verified_artifact_count = $artifacts.Count
    }
    $preflight | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $runRoot 'preflight.json')

    $env:COLIBRI_ARTIFACT_ROOT = 'D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1'
    $env:COLIBRI_R2_2_ARTIFACT_ROOT = $group32Root
    $env:COLIBRI_R2_2_D2_ARTIFACT_ROOT = $d2Root
    $env:COLIBRI_R1_2_REFERENCE_PATH = $reference
    $env:COLIBRI_R2_2_D2_OUTPUT = $output

    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $args = @(
        'full_model_validation_tests::r2_2_d2_tests::m6_3_r2_2_d2_characterize_depth_sensitive_precision',
        '--exact',
        '--nocapture'
    )
    $native = Start-Process -FilePath $binary -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    $code = $native.ExitCode
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') $code

    $completion = [ordered]@{
        schema = 'm6.3-r2.2-d2-durable-completion-v1'
        exit_code = $code
        d2_output_exists = (Test-Path $output)
        d2_output_bytes = if (Test-Path $output) { (Get-Item $output).Length } else { 0 }
    }
    $completion | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runRoot 'completion.json')
    exit $code
}
catch {
    $_ | Out-String | Set-Content -Encoding utf8 (Join-Path $runRoot 'worker-error.txt')
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') 97
    exit 97
}
