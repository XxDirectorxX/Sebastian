# 1. Backup current state
$backupPath = "R:\sebastian\backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item -Path "R:\sebastian\Core" -Destination $backupPath -Recurse

# 2. Create new structure with error handling
$directories = @(
    "Base",
    "Constants",
    "Quantum\Processors",
    "Quantum\Operations",
    "Reality\Core",
    "Reality\Manipulation",
    "Visualization\Rendering",
    "Visualization\Effects",
    "Services",
    "Neural\Configuration",
    "Neural\Learning",
    "Neural\Models",
    "Utils"
)

foreach ($dir in $directories) {
    $path = "R:\sebastian\Core\$dir"
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force
        Write-Host "Created directory: $path"
    }
}
