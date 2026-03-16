"""Test new renderer quality — save all output renders."""
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reconstruction.deep3d_recon import Deep3DFaceReconstructor

d = Deep3DFaceReconstructor()
img = cv2.imread("test_data/applicant/primary/image.png")
print(f"Input: {img.shape}")

r = d.reconstruct(img)

os.makedirs("evidence_cards", exist_ok=True)
cv2.imwrite("evidence_cards/v6_rendered_512.jpg", r["rendered"])
cv2.imwrite("evidence_cards/v6_geometry_512.jpg", r["geometry"])
cv2.imwrite("evidence_cards/v6_normals_512.jpg", r["normal_map"])
cv2.imwrite("evidence_cards/v6_depth_512.jpg", r["depth_colored"])
cv2.imwrite("evidence_cards/v6_overlay_512.jpg", r["overlay"])
cv2.imwrite("evidence_cards/v6_sideview_224.jpg", r["side_view"])
cv2.imwrite("evidence_cards/v6_sidegeom_224.jpg", r["side_geometry"])

mask = r["face_mask"]
coverage = (mask > 0).sum() / (mask.shape[0] * mask.shape[1]) * 100
print(f"Rendered: {r['rendered'].shape}")
print(f"Geometry: {r['geometry'].shape}")
print(f"Normal map: {r['normal_map'].shape}")
print(f"Depth: {r['depth_colored'].shape}")
print(f"Overlay: {r['overlay'].shape}")
print(f"Side view: {r['side_view'].shape}")
print(f"Face mask coverage: {coverage:.1f}%")
print(f"Elapsed: {r['elapsed']:.2f}s")
print("All renders saved to evidence_cards/v6_*")
