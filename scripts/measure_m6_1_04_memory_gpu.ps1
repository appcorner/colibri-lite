[CmdletBinding()]
param(
    [string]$OutputPath = "docs/benchmarks/m6.1-04-memory-gpu-profile-v1.json"
)

$ErrorActionPreference = "Stop"

function New-NotRunBenchmark([string]$Reason) {
    return [ordered]@{
        status = "not_run"
        unit = "GiB/s"
        cache_state = "not_applicable"
        payload_bytes = 0
        repetitions = 0
        reason = $Reason
    }
}

function Test-CommandAvailable([string]$Name) {
    return $null -ne (Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

$createdAt = (Get-Date).ToUniversalTime().ToString("o")
$commit = (git rev-parse HEAD).Trim()
$operatingSystem = Get-CimInstance -ClassName Win32_OperatingSystem
$computer = Get-CimInstance -ClassName Win32_ComputerSystem
$videoControllers = @(Get-CimInstance -ClassName Win32_VideoController)
$totalRamBytes = [int64]$computer.TotalPhysicalMemory
$availableRamBytes = [int64]$operatingSystem.FreePhysicalMemory * 1KB
$safetyReserveBytes = [int64][math]::Max(4GB, [math]::Ceiling($totalRamBytes * 0.15))
$safeRamBudgetBytes = [int64][math]::Max(0, $availableRamBytes - $safetyReserveBytes)

$adapters = @(
    foreach ($controller in $videoControllers) {
        [ordered]@{
            name = $controller.Name
            driver_version = $controller.DriverVersion
            pnp_device_id = $controller.PNPDeviceID
            reported_adapter_ram_bytes = if ($null -eq $controller.AdapterRAM) { 0 } else { [int64]$controller.AdapterRAM }
            classification = "detection_only; reported adapter RAM is not admitted as usable VRAM without a measured colibri backend"
        }
    }
)

$cudaDetected = Test-CommandAvailable "nvidia-smi"
$vulkanDetected = Test-CommandAvailable "vulkaninfo"
$directMlLibrary = Join-Path $env:WINDIR "System32\DirectML.dll"

$backends = @(
    [ordered]@{
        backend_id = "cuda"
        availability = "unavailable"
        detection = [ordered]@{
            probe = "Get-Command nvidia-smi"
            command_present = $cudaDetected
            result = if ($cudaDetected) { "CUDA runtime tooling may be installed, but colibri has no CUDA backend in M6.1" } else { "nvidia-smi was not found" }
        }
        usable_vram_bytes = 0
        host_to_device = New-NotRunBenchmark "No usable colibri CUDA backend was detected; transfer measurement is not meaningful."
        device_to_host = New-NotRunBenchmark "No usable colibri CUDA backend was detected; transfer measurement is not meaningful."
    },
    [ordered]@{
        backend_id = "directml"
        availability = "unavailable"
        detection = [ordered]@{
            probe = "DirectML.dll presence plus video-controller inventory"
            directml_library_present = (Test-Path -LiteralPath $directMlLibrary)
            detected_adapter_count = $adapters.Count
            result = "No colibri DirectML backend exists in M6.1; detected adapters are inventory only."
        }
        usable_vram_bytes = 0
        host_to_device = New-NotRunBenchmark "No usable colibri DirectML backend was detected; transfer measurement is not meaningful."
        device_to_host = New-NotRunBenchmark "No usable colibri DirectML backend was detected; transfer measurement is not meaningful."
    },
    [ordered]@{
        backend_id = "vulkan"
        availability = "unavailable"
        detection = [ordered]@{
            probe = "Get-Command vulkaninfo"
            command_present = $vulkanDetected
            result = if ($vulkanDetected) { "Vulkan tooling may be installed, but colibri has no Vulkan backend in M6.1" } else { "vulkaninfo was not found" }
        }
        usable_vram_bytes = 0
        host_to_device = New-NotRunBenchmark "No usable colibri Vulkan backend was detected; transfer measurement is not meaningful."
        device_to_host = New-NotRunBenchmark "No usable colibri Vulkan backend was detected; transfer measurement is not meaningful."
    }
)

$document = [ordered]@{
    schema = "colibri-lite-m6.1-04-memory-gpu-profile-v1"
    schema_version = 1
    profile_id = "windows-x64-$($env:COMPUTERNAME.ToLowerInvariant())-m6.1-04"
    created_at = $createdAt
    runtime = [ordered]@{
        name = "colibri-lite-rs"
        commit = $commit
        target_arch = "x86_64"
        build_profile = "release"
    }
    ram = [ordered]@{
        total_physical_bytes = $totalRamBytes
        available_physical_bytes = $availableRamBytes
        safety_reserve_bytes = $safetyReserveBytes
        usable_budget_bytes = $safeRamBudgetBytes
        measurement_semantics = "Win32_OperatingSystem.FreePhysicalMemory snapshot minus max(4 GiB, 15% of installed RAM); advisory only"
    }
    adapters = $adapters
    backends = $backends
    recommendations = [ordered]@{
        ram_budget_bytes = $safeRamBudgetBytes
        vram_budget_bytes = 0
        confidence = "partial"
    }
    limitations = @(
        "Adapter inventory does not make a backend usable.",
        "No GPU backend is implemented in colibri during M6.1, so usable VRAM is zero and host/device transfer is not run.",
        "RAM availability is a point-in-time OS snapshot; runtime admission must still enforce the user's explicit budget.",
        "No GPU compute or transfer benchmark was attempted before M6.3 backend review."
    )
}

$outputDirectory = Split-Path -Parent $OutputPath
if ($outputDirectory) { New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null }
[System.IO.File]::WriteAllText(
    [System.IO.Path]::GetFullPath($OutputPath),
    ($document | ConvertTo-Json -Depth 10),
    [System.Text.UTF8Encoding]::new($false))

Write-Output "wrote $OutputPath"
