<# *** 用于启动离线评估命令行入口的 PowerShell 脚本。 *** #>

$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDirectory
Set-Location $projectRoot

python -m app.evaluation.run_eval @args
