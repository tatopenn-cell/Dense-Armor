# Run by the setup wizard after the files are copied: finds Python 3.11+ (installs 3.12 if the computer has none),
# makes the private environment in <App>\.venv and installs the packages. The log is <App>\install.log.
param([Parameter(Mandatory = $true)][string]$App)

$ErrorActionPreference = 'Stop'
$log = Join-Path $App 'install.log'
function Say([string]$t) { Add-Content $log $t; Write-Host $t }
function Run([string]$exe, [string[]]$a) {
    $o = Join-Path $App 'out.tmp'; $e = Join-Path $App 'err.tmp'
    $p = Start-Process -FilePath $exe -ArgumentList $a -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    Get-Content $o, $e -ErrorAction SilentlyContinue | Add-Content $log
    Remove-Item $o, $e -ErrorAction SilentlyContinue
    return $p.ExitCode
}
function Find-Python {
    foreach ($c in @(@('py', '-3.13'), @('py', '-3.12'), @('py', '-3.11'), @('python'))) {
        if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
        $rest = @($c | Select-Object -Skip 1)
        try {
            $path = & $c[0] @rest -c "import sys; print(sys.executable if sys.version_info >= (3, 11) else '')" 2>$null
            if ($path -and (Test-Path $path)) { return $path }
        } catch { }
    }
    return $null
}

try {
    '' | Set-Content $log
    $py = Find-Python
    if (-not $py) {
        Say 'Python 3.11 or later not found: installing Python 3.12 for this user.'
        $inst = Join-Path $env:TEMP 'python-3.12-installer.exe'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe' -OutFile $inst
        $rc = Run $inst @('/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_test=0', 'Include_launcher=0')
        if ($rc -ne 0) { throw "The Python installer failed (code $rc)." }
        $py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
        if (-not (Test-Path $py)) { throw 'Python was installed but python.exe was not found.' }
    }
    Say "Python: $py"
    $venv = Join-Path $App '.venv'
    $vpy = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Path $vpy)) {
        if ((Run $py @('-m', 'venv', $venv)) -ne 0) { throw 'The Python environment could not be created.' }
    }
    $null = Run $vpy @('-m', 'pip', 'install', '--upgrade', 'pip')
    if ((Run $vpy @('-m', 'pip', 'install', '-e', $App, 'flask', 'sqlalchemy')) -ne 0) { throw 'The packages could not be installed.' }
    Say 'Done.'
    exit 0
} catch {
    Say ('ERROR: ' + $_.Exception.Message)
    exit 1
}
