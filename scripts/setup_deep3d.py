"""
Deep3D Face Reconstruction — Model Setup Script
=================================================
Downloads and prepares all required models for the Deep3D integration.

Usage:
    python scripts/setup_deep3d.py

What this script does:
  1. Downloads epoch_20.pth (289 MB) pretrained ResNet50 from Google Drive
  2. Downloads Exp_Pca.bin (51 MB) expression basis from Google Drive
  3. Checks/processes BFM_model_front.mat from BFM09 raw model
  4. Verifies all required files

You MUST manually download 01_MorphableModel.mat:
  → https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads
  (Free registration, ~50 MB download)
  Place at: models/deep3d/BFM/01_MorphableModel.mat
"""

import os
import sys
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DEEP3D_DIR = os.path.join(PROJECT_ROOT, "models", "deep3d")
BFM_DIR = os.path.join(DEEP3D_DIR, "BFM")
CKPT_DIR = os.path.join(DEEP3D_DIR, "checkpoints")
VENDOR_DIR = os.path.join(PROJECT_ROOT, "vendor", "Deep3DFaceRecon_pytorch")

CHECKPOINT_URL = "https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRMHVoSx"
EXP_PCA_URL = "https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view"

# BFM support files bundled in vendor repo
BFM_SUPPORT_FILES = [
    "BFM_exp_idx.mat", "BFM_front_idx.mat", "facemodel_info.mat",
    "similarity_Lm3D_all.mat", "std_exp.txt", "select_vertex_id.mat",
]


def ensure_dirs():
    os.makedirs(BFM_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"[OK] Directories ready: {DEEP3D_DIR}")


def download_checkpoint():
    ckpt_path = os.path.join(CKPT_DIR, "epoch_20.pth")
    if os.path.isfile(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"[OK] Checkpoint already exists: epoch_20.pth ({size_mb:.0f} MB)")
        return True

    print("[...] Downloading epoch_20.pth (289 MB) from Google Drive...")
    try:
        import gdown
        gdown.download_folder(CHECKPOINT_URL, output=CKPT_DIR, quiet=False)
        if os.path.isfile(ckpt_path):
            print(f"[OK] Checkpoint downloaded: {ckpt_path}")
            return True
        else:
            print("[FAIL] Download completed but epoch_20.pth not found.")
            return False
    except ImportError:
        print("[FAIL] gdown not installed. Run: pip install gdown")
        return False
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
        print(f"  Manual download: https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRMHVoSx")
        print(f"  Place epoch_20.pth at: {ckpt_path}")
        return False


def download_exp_pca():
    exp_path = os.path.join(BFM_DIR, "Exp_Pca.bin")
    if os.path.isfile(exp_path):
        size_mb = os.path.getsize(exp_path) / 1e6
        print(f"[OK] Expression basis already exists: Exp_Pca.bin ({size_mb:.0f} MB)")
        return True

    print("[...] Downloading Exp_Pca.bin (51 MB) from Google Drive...")
    try:
        import gdown
        gdown.download(
            "https://drive.google.com/uc?id=1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6",
            exp_path, quiet=False
        )
        if os.path.isfile(exp_path):
            print(f"[OK] Expression basis downloaded: {exp_path}")
            return True
        else:
            print("[FAIL] Download completed but Exp_Pca.bin not found.")
            return False
    except ImportError:
        print("[FAIL] gdown not installed. Run: pip install gdown")
        return False
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
        return False


def copy_bfm_support_files():
    vendor_bfm = os.path.join(VENDOR_DIR, "BFM")
    copied = 0
    for fname in BFM_SUPPORT_FILES:
        dst = os.path.join(BFM_DIR, fname)
        if os.path.isfile(dst):
            continue

        src = os.path.join(vendor_bfm, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"[WARN] Support file not found: {fname}")

    if copied > 0:
        print(f"[OK] Copied {copied} BFM support files from vendor repo")
    else:
        print("[OK] All BFM support files present")

    return True


def check_bfm09():
    """Check for BFM09 model and process if available."""
    front_path = os.path.join(BFM_DIR, "BFM_model_front.mat")
    if os.path.isfile(front_path):
        size_mb = os.path.getsize(front_path) / 1e6
        print(f"[OK] BFM_model_front.mat already exists ({size_mb:.1f} MB)")
        return True

    raw_path = os.path.join(BFM_DIR, "01_MorphableModel.mat")
    if os.path.isfile(raw_path):
        print("[...] Processing 01_MorphableModel.mat → BFM_model_front.mat ...")
        try:
            from src.reconstruction.deep3d_recon import transfer_bfm09
            transfer_bfm09(BFM_DIR)
            print(f"[OK] BFM_model_front.mat created at: {front_path}")
            return True
        except Exception as e:
            print(f"[FAIL] BFM processing failed: {e}")
            return False
    else:
        print("")
        print("=" * 70)
        print("  MANUAL DOWNLOAD REQUIRED: Basel Face Model 2009")
        print("=" * 70)
        print("")
        print("  The Deep3D reconstruction requires the BFM09 face model.")
        print("  This model is free but requires registration.")
        print("")
        print("  Step 1: Go to https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads")
        print("  Step 2: Register a free account (academic email not required)")
        print("  Step 3: Download '01_MorphableModel.mat'")
        print(f"  Step 4: Place it here → {raw_path}")
        print("  Step 5: Run this script again")
        print("")
        print("=" * 70)
        return False


def verify_installation():
    """Verify all required files are present."""
    print("\n" + "=" * 50)
    print("  VERIFICATION  ")
    print("=" * 50)

    required = {
        "Checkpoint (epoch_20.pth)": os.path.join(CKPT_DIR, "epoch_20.pth"),
        "BFM Model (BFM_model_front.mat)": os.path.join(BFM_DIR, "BFM_model_front.mat"),
        "Expression Basis (Exp_Pca.bin)": os.path.join(BFM_DIR, "Exp_Pca.bin"),
        "BFM Landmarks (similarity_Lm3D_all.mat)": os.path.join(BFM_DIR, "similarity_Lm3D_all.mat"),
        "BFM Info (facemodel_info.mat)": os.path.join(BFM_DIR, "facemodel_info.mat"),
        "BFM Front Index (BFM_front_idx.mat)": os.path.join(BFM_DIR, "BFM_front_idx.mat"),
        "BFM Exp Index (BFM_exp_idx.mat)": os.path.join(BFM_DIR, "BFM_exp_idx.mat"),
        "Std Expressions (std_exp.txt)": os.path.join(BFM_DIR, "std_exp.txt"),
    }

    all_ok = True
    for name, path in required.items():
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        size = f" ({os.path.getsize(path)/1e6:.1f} MB)" if exists else ""
        print(f"  [{status}] {name}{size}")
        if not exists:
            all_ok = False

    print()
    if all_ok:
        print("  ALL MODELS READY — Deep3D Face Reconstruction is operational!")
    else:
        print("  SOME FILES MISSING — Please resolve the issues above.")
    print("=" * 50)
    return all_ok


def main():
    print("\n" + "=" * 60)
    print("  Deep3D Face Reconstruction — Model Setup")
    print("  CA_MONK v5.1")
    print("=" * 60 + "\n")

    ensure_dirs()
    download_checkpoint()
    download_exp_pca()
    copy_bfm_support_files()
    check_bfm09()
    return verify_installation()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
