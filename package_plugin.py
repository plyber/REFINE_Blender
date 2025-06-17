import zipfile
import os

PLUGIN_NAME = "refine_plugin"
PLUGIN_VERSION = "1.2.2"
OUTPUT_ZIP = f"{PLUGIN_NAME}-v{PLUGIN_VERSION}.zip"

#exclude
EXCLUDE_DIRS = {"__pycache__", "deps", ".git"}
EXCLUDE_EXTS = {".pyc", ".pyo"}

def should_exclude(path):
    for part in path.split(os.sep):
        if part in EXCLUDE_DIRS:
            return True
    if os.path.splitext(path)[1] in EXCLUDE_EXTS:
        return True
    return False

def zip_plugin():
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PLUGIN_NAME):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PLUGIN_NAME)
                arcname = os.path.join(PLUGIN_NAME, rel_path)
                if not should_exclude(rel_path):
                    zipf.write(full_path, arcname)
    print(f"[✔] Plugin packaged as {OUTPUT_ZIP}")

if __name__ == "__main__":
    zip_plugin()
