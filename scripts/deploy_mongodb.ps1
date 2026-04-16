# Deploy MongoDB via Docker Desktop (optional; rank_jobs_app uses JSON files by default).
#
# Example URI: mongodb://127.0.0.1:27017
#
# Examples:
#   .\scripts\deploy_mongodb.ps1 -Action Up
#   .\scripts\deploy_mongodb.ps1 -Action Down
#   .\scripts\deploy_mongodb.ps1 -Action Status
#   .\scripts\deploy_mongodb.ps1 -Action Logs

param(
    [ValidateSet('Up', 'Down', 'Status', 'Logs')]
    [string] $Action = 'Up',
    [string] $ContainerName = 'speechjudge-mongo',
    [string] $Image = 'mongo:7',
    [int] $Port = 27017,
    [string] $DataVolume = 'speechjudge_mongo_data'
)

$ErrorActionPreference = 'Stop'

function Assert-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'docker not found; install Docker Desktop first.'
    }
}

Assert-Docker

switch ($Action) {
    'Up' {
        $exists = docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $ContainerName }
        $running = docker ps --format '{{.Names}}' | Where-Object { $_ -eq $ContainerName }
        if ($exists) {
            if ($running) {
                Write-Host "MongoDB container '$ContainerName' is already running."
            }
            else {
                Write-Host "Starting existing container '$ContainerName'..."
                docker start $ContainerName
            }
        }
        else {
            Write-Host "Creating MongoDB $Image as '$ContainerName' on host port $Port..."
            docker run -d `
                --name $ContainerName `
                --restart unless-stopped `
                -p "${Port}:27017" `
                -v "${DataVolume}:/data/db" `
                $Image `
                --bind_ip_all
        }
        Write-Host ''
        Write-Host 'Ready. Example environment:'
        Write-Host "  `$env:SPEECHJUDGE_MONGO_URI = 'mongodb://127.0.0.1:$Port'"
        Write-Host '  $env:SPEECHJUDGE_MONGO_DB = ''speechjudge'''
        Write-Host '  $env:SPEECHJUDGE_MONGO_COLLECTION = ''rank_jobs'''
    }
    'Down' {
        docker rm -f $ContainerName 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Container '$ContainerName' not found (nothing to remove)."
        }
    }
    'Status' {
        docker ps -a --filter "name=^/$ContainerName$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    }
    'Logs' {
        docker logs -f $ContainerName
    }
}
