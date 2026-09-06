$ErrorActionPreference = 'Stop'

$repo = 'D:\sandboxs\colibri-lite'
$binary = Join-Path $repo 'target\release\deps\clr_qwen3_moe-98ff24c36e070e31.exe'
$harness = Join-Path $repo 'crates\clr-qwen3-moe\src\r2_2_d5b_tests.rs'
$contract = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d5-contract-v1.json'
$d5aResult = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d5a-result-v1.json'
$reference = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r1-2-f32-multitoken-reference-v1.tsv'
$output = Join-Path $repo 'models\qwen3-30b-a3b\m6.3-r2-2-d5b-held-out-quality-v1.tsv'
$group32Root = 'D:\tmp\colibri-lite-r2-2\artifacts'
$d5Root = 'D:\tmp\colibri-lite-r2-2\d5-artifacts'
$runRoot = 'D:\tmp\colibri-lite-r2-2\d5b-durable-v1'

$expected = @{
    binary = '606496675c81b4efd16708ea13f13206908d2ca363d156935f727e7a3469c781'
    harness = '33b65e2b4b442902c6133f93962008492318c93d068e4da27a2a6f19a0c60e07'
    contract = '34b5a5a032bf6251b7d724c115bad9db44fc11f08092d235155e32c304d8acd3'
    d5a_result = '8ca431533ed94c8a07fe3e88512cf3300ef41d5a78e09027a920f84c8e269836'
    reference = 'fa8f1a4f6b30624b78cf8c004f379fdb0633a08a65c102791acc57d0205d54ff'
}

function Hash-Lower([string]$Path) {
    return (Get-FileHash $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
if (Test-Path $runRoot) { throw 'D5B durable run root must be new' }
New-Item -ItemType Directory -Path $runRoot | Out-Null

try {
    if ($env:COMPUTERNAME -ne 'ANUSORN-NB') { throw 'host identity mismatch' }
    if (Test-Path $output) { throw 'D5B evidence output must be new' }
    if ((Get-Item $binary).Length -ne 49923584) { throw 'release binary byte length mismatch' }
    if ((Hash-Lower $binary) -ne $expected.binary) { throw 'release binary SHA mismatch' }
    if ((Hash-Lower $harness) -ne $expected.harness) { throw 'D5B harness SHA mismatch' }
    if ((Hash-Lower $contract) -ne $expected.contract) { throw 'D5B contract SHA mismatch' }
    if ((Hash-Lower $d5aResult) -ne $expected.d5a_result) { throw 'D5a result SHA mismatch' }
    if ((Hash-Lower $reference) -ne $expected.reference) { throw 'reference SHA mismatch' }

    $artifacts = @(
        @($group32Root, 'layer00-group32-repro.bin', 679477248, '777820df3bfd7fa918035757245611c033f80eda9c92bbfca577f61ef504b1a2'),
        @($d5Root, 'layer24-hybrid-f32-gate-up-group8-down.bin', 1912602624, 'd94d12cbea648e2f2911573c893f564254cbde526d60132c600ed24e88317ac2')
    )
    foreach ($entry in $artifacts) {
        $path = Join-Path $entry[0] $entry[1]
        if ((Get-Item $path).Length -ne $entry[2]) { throw "artifact byte length mismatch: $path" }
        if ((Hash-Lower $path) -ne $entry[3]) { throw "artifact SHA mismatch: $path" }
    }

    $preflight = [ordered]@{
        schema = 'm6.3-r2.2-d5b-durable-preflight-v1'
        host = $env:COMPUTERNAME
        binary_sha256 = $expected.binary
        harness_sha256 = $expected.harness
        contract_sha256 = $expected.contract
        d5a_result_sha256 = $expected.d5a_result
        reference_sha256 = $expected.reference
        verified_artifact_count = $artifacts.Count
    }
    $preflight | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $runRoot 'preflight.json')

    $env:COLIBRI_ARTIFACT_ROOT = 'D:\models\colibri-lite\qwen3-30b-a3b\artifact-v1'
    $env:COLIBRI_R2_2_ARTIFACT_ROOT = $group32Root
    $env:COLIBRI_R2_2_D5_ARTIFACT_ROOT = $d5Root
    $env:COLIBRI_R1_2_REFERENCE_PATH = $reference
    $env:COLIBRI_R2_2_D5B_OUTPUT = $output
    $stdout = Join-Path $runRoot 'stdout.log'
    $stderr = Join-Path $runRoot 'stderr.log'
    $args = @(
        'full_model_validation_tests::r2_2_d5b_tests::m6_3_r2_2_d5b_group8_hybrid_pass_held_out_quality',
        '--exact',
        '--nocapture'
    )
    $native = Start-Process -FilePath $binary -ArgumentList $args -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
    $code = $native.ExitCode
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') $code

    $completion = [ordered]@{
        schema = 'm6.3-r2.2-d5b-durable-completion-v1'
        exit_code = $code
        d5b_output_exists = (Test-Path $output)
        d5b_output_bytes = if (Test-Path $output) { (Get-Item $output).Length } else { 0 }
    }
    $completion | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $runRoot 'completion.json')
    exit $code
}
catch {
    $_ | Out-String | Set-Content -Encoding utf8 (Join-Path $runRoot 'worker-error.txt')
    Set-Content -Encoding ascii (Join-Path $runRoot 'exit.code') 97
    exit 97
}


