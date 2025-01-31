import os
import shutil
from pathlib import Path

def organize_quantum_files():
    root = Path('R:/sebastian/Core')
    print(f"Scanning all directories recursively...")
    
    # Recursive search for all files
    files = list(root.rglob('*.cs')) + list(root.rglob('*.nn'))
    print(f"Found {len(files)} files to organize")
    
    for file in files:
        if 'Quantum' in file.name:
            dest = root / 'Quantum/Processors' / file.name
        elif 'Field' in file.name:
            dest = root / 'Quantum/Operations' / file.name
        elif 'Reality' in file.name:
            dest = root / 'Reality/Core' / file.name
        elif 'Visual' in file.name:
            dest = root / 'Visualization/Rendering' / file.name
        elif '.nn' in file.name:
            dest = root / 'Neural/Configuration' / file.name
        else:
            dest = root / 'Utils' / file.name
            
        shutil.move(str(file), str(dest))
        print(f"Moved: {file.name}")

if __name__ == '__main__':
    organize_quantum_files()
