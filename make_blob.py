# make_blob.py
import os

def collect_python_files(root_dir="."):
    py_files = []
    exclude_dirs = {'.venv', '__pycache__', '.git', 'build', 'dist'}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # modify dirnames in-place to skip excluded dirs
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for filename in filenames:
            if filename.endswith(".py"):
                py_files.append(os.path.join(dirpath, filename))
    return py_files

def make_blob(py_files, output_file="code_blob_all.txt"):
    with open(output_file, "w", encoding="utf-8") as out:
        for file_path in py_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                    out.write(f"\n\n# --- FILE: {file_path} ---\n")
                    out.write(code)
                    out.write("\n# --- END OF FILE ---\n")
            except Exception as e:
                print(f"⚠️ Skipped {file_path}: {e}")
    print(f"✅ Blob created: {output_file}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    files = collect_python_files(project_root)
    print(f"📦 Found {len(files)} Python files (excluding .venv and cache).")
    make_blob(files)
