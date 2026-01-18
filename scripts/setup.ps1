# Run this from the project root in PowerShell
#   .\scripts\setup.ps1

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

if (!(Test-Path -Path .\venv)) {
  python -m venv venv
}

. .\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

jupyter notebook
