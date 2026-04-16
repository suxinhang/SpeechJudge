# Run scripts/start.sh with Git Bash (avoids WSL relay: "execvpe(/bin/bash) failed").
# start.sh itself targets a Linux-style layout (/root/SpeechJudge, conda); use WSL/Linux for real deploy.

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot

$candidates = @(
    "${env:ProgramFiles}\Git\bin\bash.exe"
    "${env:ProgramFiles(x86)}\Git\bin\bash.exe"
    "${env:LOCALAPPDATA}\Programs\Git\bin\bash.exe"
)

$bashExe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $bashExe) {
    Write-Host @"
Git for Windows bash.exe not found. Options:
  1) Install Git for Windows, then re-run:  .\scripts\start.ps1
  2) From Git Bash:  cd /d/work/tts/SpeechJudge/scripts && bash ./start.sh
  3) Fix WSL so /bin/bash exists, or use:  wsl -d <DistroName> -- bash -lc '...'
"@ -ForegroundColor Yellow
    exit 1
}

# Git Bash: D:\a\b -> /d/a/b
function ConvertTo-GitBashPath([string]$WinPath) {
    $p = $WinPath -replace '\\', '/'
    if ($p -match '^([A-Za-z]):(/.*)?$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = if ($Matches[2]) { $Matches[2] } else { '' }
        return "/$drive$rest"
    }
    return $p
}
$unixHere = ConvertTo-GitBashPath $here

Write-Host "Using: $bashExe" -ForegroundColor Cyan
& $bashExe -lc "set -e; cd '$unixHere'; bash ./start.sh"
