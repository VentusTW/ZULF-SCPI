# Builds exe\ZULF-SCPI.exe (a single file, no console window) and puts config\ and
# data\ next to it - the executable reads and writes them there.
# Existing files in exe\config and exe\data are kept, so edited settings survive a rebuild.
# Usage: .\build_exe.ps1   (requires: .venv\Scripts\python -m pip install pyinstaller)

Set-Location $PSScriptRoot

# The logo: the .exe file's icon, and the assets folder inside it for the window icons.
# Absolute paths - PyInstaller resolves relative ones against the spec folder (build\).
$assets = (Resolve-Path zulf_nmr\gui\assets).Path

# PyInstaller logs to stderr; with ErrorActionPreference=Stop Windows PowerShell
# would treat that as a failure, so only the exit code is checked.
& .venv\Scripts\python -m PyInstaller --noconfirm --clean --onefile --windowed `
    --name ZULF-SCPI --paths . `
    --icon "$assets\icon.ico" --add-data "$assets;zulf_nmr\gui\assets" `
    --distpath exe --workpath build --specpath build `
    zulf_nmr\gui\__main__.py 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) { Write-Error "PyInstaller failed ($LASTEXITCODE)"; exit $LASTEXITCODE }
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
Write-Host "Done: exe\ZULF-SCPI.exe"
