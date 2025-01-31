# Define source and mapping
$fileMapping = @{
    "Quantum\Processors" = "*Quantum*Processor*.cs"
    "Quantum\Operations" = "*Field*Operations*.cs"
    "Reality\Core" = "*Reality*Core*.cs"
    "Reality\Manipulation" = "*Manipulation*.cs"
    "Visualization\Rendering" = "*Renderer*.cs"
    "Visualization\Effects" = "*Effects*.cs"
    "Services" = "*Service*.cs"
    "Neural\Configuration" = "*.nn"
    "Utils" = "*Helper*.cs"
}

# Execute moves with logging
foreach ($destination in $fileMapping.Keys) {
    $pattern = $fileMapping[$destination]
    Get-ChildItem -Path "R:\sebastian\Core" -Filter $pattern | 
    ForEach-Object {
        $destPath = "R:\sebastian\Core\$destination"
        Move-Item $_.FullName -Destination $destPath -Force
        Write-Host "Moved $($_.Name) to $destPath"
    }
}
