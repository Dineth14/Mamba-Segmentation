
import os
import subprocess
import sys

WEIGHTS_DIR = "/storage2/ChangeDetection/NSST-mamba/Mamba-Segmentation/VisionMamba/weights"

# Mapping: Variant -> (Repo, Filename)
WEIGHTS_MAP = {
    "tiny": (
        "hustvl/Vim-tiny-midclstok",
        "vim_t_midclstok_76p1acc.pth"
    ),
    "small": (
        "hustvl/Vim-small-midclstok",
        "vim_s_midclstok_80p5acc.pth"
    ),
    "base": (
        "hustvl/Vim-base-midclstok",
        "vim_b_midclstok_81p9acc.pth"
    ),
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
        # Use wget for reliability
        subprocess.check_call(["wget", "-q", "--show-progress", "-O", dest_path, url])
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}: {e}")
        # Clean up partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)

def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    for variant, (repo, filename) in WEIGHTS_MAP.items():
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        dest_path = os.path.join(WEIGHTS_DIR, filename)
        
        print(f"processing {variant}...")
        download_file(url, dest_path)

if __name__ == "__main__":
    main()
