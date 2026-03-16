import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download, snapshot_download

# Configuration
MODELS = [
    {
        "name": "CodeFormer ONNX",
        "repo": "bluefoxcreation/Codeformer-ONNX",
        "filename": "codeformer.onnx",
        "target": "models/codeformer.onnx",
    },
    {
        "name": "AdaFace ONNX (Fallback)",
        "repo": "adaface-neurips/adaface-models",
        "filename": "models/insightface/models/antelopev2/arcface.onnx",
        "target": "models/adaface_ir101_webface12m.onnx",
    },
]


def ensure_local_ollama_model(model_name: str) -> None:
    print(f"\nChecking local Ollama model '{model_name}'...")
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        print("FAIL: Ollama CLI not found in PATH.")
        return

    show = subprocess.run([ollama_bin, "show", model_name], capture_output=True, text=True)
    if show.returncode == 0:
        print(f"PASS: Ollama model '{model_name}' is installed locally.")
        return

    print(f"MISSING: Ollama model '{model_name}'. Attempting pull...")
    pull = subprocess.run([ollama_bin, "pull", model_name], capture_output=False, text=True)
    if pull.returncode == 0:
        print(f"SUCCESS: Pulled Ollama model '{model_name}'.")
    else:
        print(f"FAIL: Could not pull Ollama model '{model_name}'.")


def install_deps():
    os.makedirs("models", exist_ok=True)

    # 1. Pipeline model files
    print("Checking specific model files...")
    for model in MODELS:
        if os.path.exists(model["target"]) and os.path.getsize(model["target"]) > 0:
            print(f"PASS: {model['name']} already exists at {model['target']}")
            continue

        print(f"MISSING: {model['name']}. Downloading from {model['repo']}...")
        try:
            hf_hub_download(
                repo_id=model["repo"],
                filename=model["filename"],
                local_dir="models",
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            downloaded_path = os.path.join("models", model["filename"])
            if os.path.abspath(downloaded_path) != os.path.abspath(model["target"]):
                os.makedirs(os.path.dirname(model["target"]), exist_ok=True)
                print(f"Moving {downloaded_path} to {model['target']}...")
                shutil.copy2(downloaded_path, model["target"])

            print(f"SUCCESS: Installed {model['name']}")
        except Exception as e:
            print(f"FAIL: Could not download {model['name']}. Error: {e}")

    # 2. SD 1.5 Realistic Vision — REMOVED in v5.0
    # Reconstruction now uses Deep3D Forensic Pipeline (MediaPipe + OpenCV + CodeFormer).
    # No diffusers models needed. The models/sd15_rv6 and models/vae_ft_mse directories
    # can be safely deleted to free ~2.5GB of disk space.
    print("\n[v5.0] SD 1.5 Realistic Vision REMOVED — using Deep3D Forensic Pipeline.")
    print("  Tip: Delete models/sd15_rv6/ and models/vae_ft_mse/ to free ~2.5GB disk space.")

    # 3. Local Ollama model
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    ensure_local_ollama_model(ollama_model)


if __name__ == "__main__":
    install_deps()
