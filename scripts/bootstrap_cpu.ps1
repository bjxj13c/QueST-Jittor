param(
    [string]$BasePython = 'D:\anaconda\envs\research_practice\python.exe',
    [switch]$SkipInstall
)
$ErrorActionPreference = 'Stop'
$questRoot = Split-Path -Parent $PSScriptRoot
$questEnvironment = Join-Path $questRoot '.venv-torch'
$questPython = Join-Path $questEnvironment 'Scripts\python.exe'
$env:PIP_CACHE_DIR = Join-Path $questRoot 'cache\pip'
$env:TORCH_HOME = Join-Path $questRoot 'cache\torch'
$env:HF_HOME = Join-Path $questRoot 'cache\huggingface'
$env:MPLCONFIGDIR = Join-Path $questRoot 'cache\matplotlib'
$env:TEMP = Join-Path $questRoot 'cache\tmp'
$env:TMP = $env:TEMP
foreach ($questCache in @($env:PIP_CACHE_DIR, $env:TORCH_HOME, $env:HF_HOME, $env:MPLCONFIGDIR, $env:TEMP)) {
    New-Item -ItemType Directory -Path $questCache -Force | Out-Null
}
if (!(Test-Path -LiteralPath $questPython)) {
    & $BasePython -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10 for the pinned CPU baseline"'
    if ($LASTEXITCODE -ne 0) { throw 'Python version check failed' }
    & $BasePython -m venv $questEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
if (!$SkipInstall) {
    & $questPython -m pip install pip==25.3 setuptools==75.8.2 --index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
    # Install Torch's ordinary dependencies first; never let an unconstrained dependency pull a different Torch build.
    & $questPython -m pip install wheel==0.45.1 numpy==1.26.4 pillow==10.4.0 filelock==3.16.1 typing-extensions==4.12.2 sympy==1.13.1 networkx==3.4.2 jinja2==3.1.6 fsspec==2024.6.1 --index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw 'Core dependency installation failed' }
    & $questPython -m pip install --no-deps torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
    & $questPython -m pip install -r (Join-Path $questRoot 'requirements\cpu.txt') --index-url https://pypi.org/simple
    if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed' }
    & $questPython -m pip install --no-deps --no-build-isolation -e $questRoot
    if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
}
& $questPython (Join-Path $PSScriptRoot 'doctor.py') --root $questRoot --output (Join-Path $questRoot 'reports\doctor_cpu.json')
if ($LASTEXITCODE -ne 0) { throw 'Doctor failed' }
