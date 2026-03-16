"""Test Deep3D on a real face image."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print('=== Deep3D — Real Face Image Test ===\n')

from src.reconstruction.deep3d_recon import Deep3DFaceReconstructor
import cv2

# Test image
img_path = r'test_data\applicant\primary\image.png'
print(f'Input: {img_path}')
image = cv2.imread(img_path)
if image is None:
    print(f'Cannot read image: {img_path}')
    sys.exit(1)
print(f'Image size: {image.shape}')

# Full reconstruction
print('\nInitializing Deep3D (first call loads model)...')
t0 = time.time()
recon = Deep3DFaceReconstructor()
result = recon.reconstruct(image)
t1 = time.time()
print(f'First call (includes model load): {t1-t0:.2f}s')

if result is None:
    print('FAIL: No face detected')
    sys.exit(1)

print(f'  Vertices: {result["mesh_vertices"].shape[0]}')
print(f'  Landmark count: {result["landmarks_68"].shape[0]}')
print(f'  Coefficients: {result["coefficients"].shape}')
print(f'  Elapsed (reconstruction only): {result["elapsed"]*1000:.0f}ms')

# Save outputs
os.makedirs('evidence_cards', exist_ok=True)
cv2.imwrite('evidence_cards/deep3d_real_render.png', result['rendered'])
cv2.imwrite('evidence_cards/deep3d_real_depth.png', result['depth_colored'])
cv2.imwrite('evidence_cards/deep3d_real_overlay.png', result['overlay'])
cv2.imwrite('evidence_cards/deep3d_real_aligned.png', result['aligned_input'])
recon.save_obj(result, 'evidence_cards/deep3d_real_mesh.obj')

# Draw landmarks on aligned image
aligned_lm = result['aligned_input'].copy()
for lm in result['landmarks_68']:
    x, y = int(lm[0]), int(lm[1])
    cv2.circle(aligned_lm, (x, y), 1, (0, 255, 0), -1)
cv2.imwrite('evidence_cards/deep3d_real_landmarks.png', aligned_lm)

print('\nSaved outputs:')
for f in ['render', 'depth', 'overlay', 'aligned', 'landmarks', 'mesh']:
    ext = '.obj' if f == 'mesh' else '.png'
    path = f'evidence_cards/deep3d_real_{f}{ext}'
    if os.path.exists(path):
        sz = os.path.getsize(path)
        print(f'  {path} ({sz/1024:.1f} KB)')

# Second call (cached)
print('\nSecond call (cached model)...')
t0 = time.time()
result2 = recon.reconstruct(image)
t1 = time.time()
print(f'Second call: {t1-t0:.2f}s ({(t1-t0)*1000:.0f}ms)')

print('\n=== SUCCESS ===')
