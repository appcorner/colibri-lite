param(
    [Parameter(Mandatory = $true)] [string] $Path,
    [long] $Bytes = 67108864
)

$buffer = New-Object byte[] 1048576
$stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
$total = 0
try {
    while ($total -lt $Bytes) {
        $requested = [int][Math]::Min([long]$buffer.Length, $Bytes - $total)
        $count = $stream.Read($buffer, 0, $requested)
        if ($count -eq 0) { break }
        $total += $count
    }
} finally {
    $stream.Dispose()
}
Write-Output "logical_bytes_read=$total"
