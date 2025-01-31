$ROOT_DIR = "R:\sebastian"
$UNIFIED_DIR = "$ROOT_DIR\Unified"

# Create output directory
New-Item -ItemType Directory -Path $UNIFIED_DIR -Force

# Define consolidation mapping
$consolidation = @{
    "UnifiedQuantum.cs" = Get-ChildItem -Path $ROOT_DIR -Recurse -File | Where-Object { $_.Name -like "*Quantum*" }
    "UnifiedVoice.cs" = Get-ChildItem -Path $ROOT_DIR -Recurse -File | Where-Object { $_.Name -like "*Voice*" }
    "UnifiedNeural.cs" = Get-ChildItem -Path $ROOT_DIR -Recurse -File | Where-Object { $_.Name -like "*Neural*" }
    "UnifiedGui.xaml" = Get-ChildItem -Path $ROOT_DIR -Recurse -File | Where-Object { $_.Extension -eq ".xaml" }
}

# Execute consolidation
foreach ($target in $consolidation.Keys) {
    $files = $consolidation[$target]
    $outputPath = Join-Path $UNIFIED_DIR $target
    
    # Create new file
    New-Item -ItemType File -Path $outputPath -Force
    
    # Merge content
    foreach ($file in $files) {
        Add-Content -Path $outputPath -Value "// Source: $($file.FullName)"
        Get-Content $file.FullName | Add-Content -Path $outputPath
        Add-Content -Path $outputPath -Value "`n"
    }
}

Write-Host "Consolidation complete"