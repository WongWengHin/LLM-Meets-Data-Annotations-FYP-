<# *** 用于启动本地 FastAPI 标注服务的脚本入口。 *** #>

$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDirectory
Set-Location $projectRoot

if (-not (Test-Path ".env")) {
    Write-Host "Missing .env file. Copy .env_example to .env and fill in your API key first." -ForegroundColor Yellow
    exit 1
}

function Clear-InvalidLoopbackProxies {
    $proxyKeys = @(
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy"
    )

    foreach ($key in $proxyKeys) {
        $value = [System.Environment]::GetEnvironmentVariable($key, "Process")
        if ([string]::IsNullOrWhiteSpace($value)) {
            continue
        }

        try {
            $uri = [System.Uri]$value
        } catch {
            continue
        }

        if (($uri.Host -in @("127.0.0.1", "localhost")) -and $uri.Port -eq 9) {
            [System.Environment]::SetEnvironmentVariable($key, $null, "Process")
            Write-Host "Cleared invalid proxy: $key=$value" -ForegroundColor DarkYellow
        }
    }
}

Clear-InvalidLoopbackProxies

Write-Host "Starting FastAPI annotation server from $projectRoot" -ForegroundColor Cyan
python main.py
