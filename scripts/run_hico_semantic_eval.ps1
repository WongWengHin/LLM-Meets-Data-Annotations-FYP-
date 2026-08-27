<# *** Use this PowerShell script to launch HICO semantic evaluation. *** #>

$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDirectory
Set-Location $projectRoot

python scripts/run_hico_semantic_eval.py @args
