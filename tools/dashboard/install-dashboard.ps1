# Installs the Dense-Armor dashboard for the current user and puts an icon on the desktop.
# It copies the program to %LOCALAPPDATA%\Dense-Armor, makes a private Python environment
# there, installs what the dashboard needs and creates the shortcut "Dense-Armor Dashboard".
# Run it by double-clicking install-dashboard.bat. Running it again updates the installation.
param([string]$Target = (Join-Path $env:LOCALAPPDATA 'Dense-Armor'),
      [string]$ShortcutDir = [Environment]::GetFolderPath('Desktop'),
      [switch]$ShortcutOnly)

$ErrorActionPreference = 'Stop'
$src = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

function Find-Python {
    foreach ($c in @(@('py', '-3.13'), @('py', '-3.12'), @('py', '-3.11'), @('python'))) {
        $exe = Get-Command $c[0] -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        $rest = @($c | Select-Object -Skip 1)
        try {
            $v = & $c[0] @rest -c "import sys; print(sys.version_info >= (3, 11))" 2>$null
            if ($v -eq 'True') { return ,$c }
        } catch { }
    }
    return $null
}

if (-not $ShortcutOnly) {
    $py = Find-Python
    if (-not $py) {
        Write-Host 'Python 3.11 or later was not found: installing Python 3.12 with winget...'
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
        $py = Find-Python
        if (-not $py) { throw 'Python could not be installed. Install Python 3.12 from python.org, then run this again.' }
    }

    Write-Host "Copying the program to $Target ..."
    New-Item -ItemType Directory -Force $Target | Out-Null
    robocopy $src $Target /MIR /XD .git .venv run __pycache__ docs benchmarks /XF *.pyc /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }

    $venv = Join-Path $Target '.venv'
    if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
        Write-Host 'Making the private Python environment ...'
        $rest = @($py | Select-Object -Skip 1)
        & $py[0] @rest -m venv $venv
    }
    $vpy = Join-Path $venv 'Scripts\python.exe'
    Write-Host 'Installing the packages (the first time it takes a few minutes) ...'
    & $vpy -m pip install --quiet --upgrade pip
    & $vpy -m pip install --quiet -e $Target flask sqlalchemy
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
}

$bat = Join-Path $Target 'tools\dashboard\start-dashboard.bat'
$ico = Join-Path $Target 'tools\dashboard\dense-armor.ico'
if (-not (Test-Path $bat)) { throw "$bat not found: run the installation first." }
$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut((Join-Path $ShortcutDir 'Dense-Armor Dashboard.lnk'))
$lnk.TargetPath = $bat
$lnk.WorkingDirectory = Join-Path $Target 'tools\dashboard'
$lnk.IconLocation = $ico
$lnk.Description = 'Dense-Armor dashboard'
$lnk.WindowStyle = 7
$lnk.Save()
Write-Host ''
Write-Host 'Done. The icon "Dense-Armor Dashboard" is on the desktop: double-click it to start.'
