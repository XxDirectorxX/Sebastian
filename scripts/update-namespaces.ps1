# Update all .cs files with new namespaces
Get-ChildItem -Path "R:\sebastian\Core" -Filter "*.cs" -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName
    $directory = Split-Path (Split-Path $_.FullName -Parent) -Leaf
    $newNamespace = "Sebastian.Core.$directory"
    
    $updatedContent = $content -replace 'namespace Sebastian\.Core', "namespace $newNamespace"
    Set-Content $_.FullName $updatedContent
}
