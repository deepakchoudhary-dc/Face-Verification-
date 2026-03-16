from src.reconstruction.generative import OpenVINOForensicReconstructor


class SDXLTurboReconstructor(OpenVINOForensicReconstructor):
    """
    Backward-compatible alias for older imports.
    Now backed by Deep3D Forensic Pipeline (MediaPipe + OpenCV + CodeFormer).
    SD 1.5 and SDXL-Turbo have been fully removed in v5.0.
    """

