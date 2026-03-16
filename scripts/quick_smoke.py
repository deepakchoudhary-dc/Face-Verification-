"""Quick smoke test for Deep3D integration."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reconstruction.deep3d_recon import Deep3DFaceReconstructor
import cv2

d = Deep3DFaceReconstructor()
img = cv2.imread("test_data/applicant/primary/image.png")
print(f"Image: {img.shape}")

t0 = time.time()
r = d.reconstruct(img)
t1 = time.time()
print(f"First call: {t1-t0:.2f}s")
print(f"Rendered: {r['rendered'].shape}")
print(f"Vertices: {r['mesh_vertices'].shape}")
print(f"Faces: {r['mesh_faces'].shape}")
print(f"Landmarks: {r['landmarks_68'].shape}")

t2 = time.time()
r2 = d.reconstruct(img)
t3 = time.time()
print(f"Cached call: {t3-t2:.2f}s")
print("ALL OK")
