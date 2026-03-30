from src.reconstruction.generative import OpenVINOForensicReconstructor
from src.reconstruction.sdxl_reconstructor import SDXLTurboReconstructor  # legacy alias
from src.reconstruction.deep3d_recon import Deep3DFaceReconstructor
from src.reconstruction.expression_transfer import Deep3DExpressionTransferService
from src.reconstruction.expression_suite import Deep3DExpressionSuiteService

__all__ = [
    "OpenVINOForensicReconstructor",
    "SDXLTurboReconstructor",
    "Deep3DFaceReconstructor",
    "Deep3DExpressionTransferService",
    "Deep3DExpressionSuiteService",
]

# v5.1 — Deep3DFaceReconstruction Integration (Microsoft's actual repo):
#   https://github.com/microsoft/Deep3DFaceReconstruction
#   PyTorch successor: https://github.com/sicxu/Deep3DFaceRecon_pytorch
#
# OpenVINOForensicReconstructor now runs:
#   ResNet50 → 257 BFM coefficients → 35K-vertex 3D mesh → CPU-rendered face
#   → Upscale → Occlusion Removal → Lighting Normalization
#   → Forensic Reconstruction → Super Resolution → CodeFormer ONNX
#
# Deep3DFaceReconstructor available as standalone:
#   from src.reconstruction import Deep3DFaceReconstructor
#   recon = Deep3DFaceReconstructor()
#   result = recon.reconstruct(image_bgr)
