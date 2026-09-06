$ErrorActionPreference = 'Stop'

$repo = 'D:\sandboxs\colibri-lite'
$binary = Join-Path $repo 'target\release\deps\clr_qwen3_moe-98ff24c36e070e31.exe'
$harness = Join-Path $repo 'crates\clr-qwen3-moe\src\r2_2_d4d3_tests.rs'
$contract = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d4d3-sequence-aware-down-precision-contract-v1.json'
$reference = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r1-2-f32-multitoken-reference-v1.tsv'
$output = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d4d3-evidence-v1.tsv'
$group32Root = 'D:\tmp\colibri-lite-r2-2\artifacts'
$d2Root = 'D:\tmp\colibri-lite-r2-2\d2-artifacts'
$runRoot = 'D:\tmp\colibri-lite-r2-2\d4d3-durable-v1'

$expected = @{
    binary = '7aa878e205323879c85605a8829c69c530b2c8e33fd00de5d75496f8a359a2ef'
    harness = '5191b300a7ac7d4f2b74c6d7507759da7fb38efe53128443413dd33210da052b'
    contract = 'dddc8b4ec320835686e5c5286b420e5b39bb4f3bbb3ef95e0f6b82bab473470a'
    reference = 'fa8f1a4f6b30624b78cf8c004f379fdb0633a08a65c102791acc57d0205d54ff'
}

function Hash-Lower([string]$Path) {
    return (Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
if (Test-Path $runRoot) { throw 'D4D3 durable run root must be new' }
New-Item -ItemType Directory -Path $runRoot | Out-Null

try {
    if ($env:COMPUTERNAME -ne 'ANUSORN-NB') { throw 'host identity mismatch' }
    if (Test-Path $output) { throw 'D4D3 evidence output must be new' }
    if ((Get-Item $binary).Length -ne 48793088) { throw 'release binary byte length mismatch' }
    if ((Hash-Lower $binary) -ne $expected.binary) { throw 'release binary SHA mismatch' }
    if ((Hash-Lower $harness) -ne $expected.harness) { throw 'D4D3 harness SHA mismatch' }
    if ((Hash-Lower $contract) -ne $expected.contract) { throw 'D4D3 contract SHA mismatch' }
    if ((Hash-Lower $reference) -ne $expected.reference) { throw 'reference SHA mismatch' }

    $artifacts = @(
        @($group32Root, 'layer00-group32-repro.bin', 679477248, '777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2'),
        @($group32Root, 'layer24-group32.bin', 679477248, '890936e78829cf013a141b225c7819cb2c10134fbd2c787e4131f7f41844d9b0'),
        @($d2Root, 'layer24-group16.bin', 754974720, '569b221f448b521488764da8b0e99058f8daded9f41ba10ef7f1f2706ac36416'),
        @($d2Root, 'layer24-group8.bin', 905969664, '4d51c17a2ba040229a8c20c895c5be8f030ab273db3151ccb739df7289965435')
    )
    foreach ($entry in $artifacts) {
        $path = Join-Path $entry[0] $entry[1]
        if ((Get-Item $path).Length -ne $entry[2]) { throw "artifact byte length mismatch: $path" }
        if ((Hash-Lower $path) -ne $entry[3]) { throw "artifact SHA mismatch: $path" }
    }

    $preflight = [ordered]@{
        schema = 'm6.3-r2.2-d4d3-durable-preflight-v1'
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
    $env:COLIBRI_R2_2_D2_ARTIFACT_ROOT = $d2Root
    $env:COLIBRI_R1_2_REFERENCE_PATH = $reference
    $env:COLIBRI_R2_2_D4D3_OUTPUT = $output
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $args = @(
        'full_model_validation_tests::r2_2_d4d3_tests::m6_3_r2_2_d4d3_characterize_sequence_aware_layer24_down_precision',
        '--exact',
        '--nocapture'
    )
    $native = Start-Process -FilePath $binary -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    $code = $native.ExitCode
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') $code

    $completion = [ordered]@{
        schema = 'm6.3-r2.2-d4d3-durable-completion-v1'
        exit_code = $code
        d4d3_output_exists = (Test-Path $output)
        d4d3_output_bytes = if (Test-Path $output) { (Get-Item $output).Length } else { 0 }
    }
    $completion | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runRoot 'completion.json')
    exit $code
}
catch {
    $_ | Out-String | Set-Content -Encoding utf8 (Join-Path $runRoot 'worker-error.txt')
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') 97
    exit 97
}

