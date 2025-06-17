import shutil
from pathlib import Path

# virtualenv site-packages
SITE_PACKAGES = Path("refine_plugin/deps/Lib/site-packages")
DEST_LIBS = Path("refine_plugin/libs")

def copy_all_packages():
    print(f"[i] Copying everything from: {SITE_PACKAGES}")
    DEST_LIBS.mkdir(parents=True, exist_ok=True)
    for item in SITE_PACKAGES.iterdir():
        dst = DEST_LIBS / item.name
        if dst.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
        print(f"[+] Copied: {item.name}")

if __name__ == "__main__":
    copy_all_packages()
    print("All dependencies copied.")
