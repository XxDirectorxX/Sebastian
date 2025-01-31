# Update namespace references
Get-ChildItem -Path "R:\sebastian\Core" -Filter "*.cs" -Recurse | ForEach-Object {
    (Get-Content $_.FullName) | ForEach-Object {
        $_ -replace 'Sebastian\.Core', 'Sebastian.Core.NewNamespace'
    } | Set-Content $_.FullName
}
