# Create new directory structure
New-Item -ItemType Directory -Path "R:\sebastian\Core\{Base,Constants,Quantum,Reality,Visualization,Services,Neural,Utils}"

# Move and reorganize files
Move-Item -Path "R:\sebastian\Core\*Processor.cs" -Destination "R:\sebastian\Core\Base\"
Move-Item -Path "R:\sebastian\Core\*Constants.cs" -Destination "R:\sebastian\Core\Constants\"
