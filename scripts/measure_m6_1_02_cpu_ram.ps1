[CmdletBinding()]
param(
    [string]$OutputPath = "docs/benchmarks/m6.1-02-cpu-ram-benchmark-v1.json"
)

$ErrorActionPreference = "Stop"

$processor = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1
$computer = Get-CimInstance -ClassName Win32_ComputerSystem
$operatingSystem = Get-CimInstance -ClassName Win32_OperatingSystem
$commit = (git rev-parse HEAD).Trim()
$createdAt = (Get-Date).ToUniversalTime().ToString("o")
$profileId = "windows-x64-$($env:COMPUTERNAME.ToLowerInvariant())-m6.1-02"

& cargo run --release -p clr-core --example m6_1_cpu_ram_bench -- `
    --output $OutputPath `
    --profile-id $profileId `
    --created-at $createdAt `
    --runtime-commit $commit `
    --os-name $operatingSystem.Caption `
    --os-version $operatingSystem.Version `
    --cpu-model $processor.Name `
    --logical-core-count $processor.NumberOfLogicalProcessors `
    --ram-total-bytes $computer.TotalPhysicalMemory
