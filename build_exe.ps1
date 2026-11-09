Set-Location $PSScriptRoot

$assets = (Resolve-Path zulf_nmr\gui\assets).Path

$versionFile = (Resolve-Path zulf_nmr\version.py).Path
$versionText = [IO.File]::ReadAllText($versionFile)
$version = [regex]::Match($versionText, '(?m)^VERSION = "([^"]+)"').Groups[1].Value
$build = [int][regex]::Match($versionText, '(?m)^BUILD = (\d+)').Groups[1].Value
$builtFor = [regex]::Match($versionText, '(?m)^BUILT_FOR = "([^"]*)"').Groups[1].Value
if ($builtFor -ne $version) { $build = 1 } else { $build += 1 }
$newText = [regex]::Replace($versionText, '(?m)^BUILD = \d+', "BUILD = $build")
$newText = [regex]::Replace($newText, '(?m)^BUILT_FOR = "[^"]*"', "BUILT_FOR = `"$version`"")
[IO.File]::WriteAllText($versionFile, $newText)
Write-Host "ZULF-SCPI $version.$build"

& .venv\Scripts\python -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name ZULF-SCPI --paths . `
    --icon "$assets\icon.ico" --add-data "$assets;zulf_nmr\gui\assets" `
    --distpath exe --workpath build --specpath build `
    zulf_nmr\gui\__main__.py 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) {
    [IO.File]::WriteAllText($versionFile, $versionText)
    Write-Error "PyInstaller failed ($LASTEXITCODE) - build number not changed"
    exit $LASTEXITCODE
}
$ErrorActionPreference = "Stop"

foreach ($folder in "config", "data") {
    $source = Join-Path "zulf_nmr" $folder
    Get-ChildItem $source -Recurse -File |
        Where-Object { $_.FullName -notlike "*\data\results\*" } |
        ForEach-Object {
            $relative = $_.FullName.Substring((Resolve-Path $source).Path.Length + 1)
            $target = Join-Path (Join-Path "exe" $folder) $relative
            if (-not (Test-Path $target)) {
                New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
                Copy-Item $_.FullName $target
                Write-Host "copied $target"
            }
        }
}
Write-Host "Done: exe\ZULF-SCPI.exe - version $version.$build"
