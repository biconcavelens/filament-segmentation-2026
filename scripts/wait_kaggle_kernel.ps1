# Detached watcher: wait for a Kaggle kernel to finish, then download matching outputs (with retries).
# Runs outside the agent's background-task time limit.
#   powershell -File scripts\wait_kaggle_kernel.ps1 -Kernel owner/name -OutDir dir -Pattern 'regex' -Log file
param([string]$Kernel, [string]$OutDir, [string]$Pattern, [string]$Log)
Set-Location (Split-Path $PSScriptRoot -Parent)
function Note($m) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $Log }
Note "watching $Kernel"
while ($true) {
    $s = (kaggle kernels status $Kernel 2>&1 | Out-String)
    if ($s -match 'KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL\w*)') { Note "status: $($Matches[0])"; break }
    Start-Sleep -Seconds 300
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
for ($i = 1; $i -le 6; $i++) {
    kaggle kernels output $Kernel -p $OutDir --file-pattern $Pattern -o -q 2>&1 | Out-Null
    $files = Get-ChildItem $OutDir -File | Where-Object { $_.Length -gt 0 }
    Note "download attempt ${i}: $($files.Name -join ', ')"
    if ($files.Count -ge 2) { break }
    Start-Sleep -Seconds 60
}
Note "DONE"
