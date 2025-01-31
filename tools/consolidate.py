import os
import shutil

ROOT_DIR = r"R:\sebastian\Unified"
FINAL_DIR = r"R:\sebastian\Unified\Final"

TARGETS = {
    'UnifiedQuantum.cs': ['Quantum', '.cs'],
    'UnifiedVoice.cs': ['Voice', '.cs'],
    'UnifiedNeural.cs': ['Neural', '.cs'],
    'UnifiedGui.xaml': ['', '.xaml']
}

def consolidate():
    os.makedirs(FINAL_DIR, exist_ok=True)
    
    for output_file, (pattern, ext) in TARGETS.items():
        with open(os.path.join(FINAL_DIR, output_file), 'wb') as outfile:
            for root, _, files in os.walk(ROOT_DIR):
                for file in files:
                    if pattern in file and file.endswith(ext):
                        file_path = os.path.join(root, file)
                        print(f"Consolidating: {file_path}")
                        with open(file_path, 'rb') as infile:
                            outfile.write(f"// Source: {file_path}\n".encode('utf-8'))
                            outfile.write(infile.read())
                            outfile.write(b"\n\n")

    # Clean up original files
    for root, dirs, files in os.walk(ROOT_DIR, topdown=False):
        for name in files:
            if root != FINAL_DIR:
                os.remove(os.path.join(root, name))
        for name in dirs:
            if root != FINAL_DIR:
                os.rmdir(os.path.join(root, name))

if __name__ == "__main__":
    consolidate()
