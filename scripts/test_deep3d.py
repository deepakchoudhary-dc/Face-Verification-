"""Quick test: Deep3D pipeline components."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['INSIGHTFACE_LOG_LEVEL'] = '0'

print('=== Deep3D Face Reconstruction — Full Pipeline Test ===\n')

# 1. Load BFM
print('[1/5] Loading Basel Face Model...')
from src.reconstruction.deep3d_recon import ParametricFaceModel
bfm = ParametricFaceModel()
n_verts = bfm.mean_shape.shape[1] // 3
n_faces = bfm.face_buf.shape[0]
print(f'  BFM loaded: {n_verts} vertices, {n_faces} triangles, 68 keypoints')

# 2. Load ResNet50
print('[2/5] Loading ResNet50...')
import torch
from src.reconstruction.deep3d_recon import ReconNetWrapper
net = ReconNetWrapper(use_last_fc=False)
sd = torch.load('models/deep3d/checkpoints/epoch_20.pth', map_location='cpu')
net.load_state_dict(sd['net_recon'], strict=True)
net.eval()
params = sum(p.numel() for p in net.parameters())
print(f'  ResNet50 loaded: {params:,} params')

# 3. Test forward pass
print('[3/5] Running forward pass on dummy input...')
t0 = time.time()
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224)
    coeffs = net(dummy)
t1 = time.time()
print(f'  ResNet50 inference: {(t1-t0)*1000:.0f}ms, output shape: {list(coeffs.shape)}')

# 4. BFM reconstruction
print('[4/5] Running BFM reconstruction...')
t0 = time.time()
result = bfm.reconstruct(coeffs)
t1 = time.time()
vshape = list(result['face_vertex'].shape)
cshape = list(result['face_color'].shape)
lshape = list(result['landmarks_2d'].shape)
print(f'  BFM reconstruction: {(t1-t0)*1000:.0f}ms')
print(f'  face_vertex: {vshape}')
print(f'  face_color: {cshape}')
print(f'  landmarks_2d: {lshape}')

# 5. CPU Rendering
print('[5/5] Running CPU renderer...')
from src.reconstruction.deep3d_recon import CPUMeshRenderer
import cv2
renderer = CPUMeshRenderer()
t0 = time.time()
verts = result['face_vertex'][0].numpy()
colors = result['face_color'][0].numpy()
faces = result['face_buf']
if isinstance(faces, torch.Tensor):
    faces = faces.numpy()
render_out = renderer.render(verts, faces, colors)
t1 = time.time()
mask_px = int((render_out['mask'] > 0).sum())
print(f'  CPU render: {(t1-t0)*1000:.0f}ms')
print(f'  rendered image: {list(render_out["rendered"].shape)}')
print(f'  face mask pixels: {mask_px}')

os.makedirs('evidence_cards', exist_ok=True)
cv2.imwrite('evidence_cards/deep3d_test_render.png', render_out['rendered'])
depth_col = cv2.applyColorMap(render_out['depth'], cv2.COLORMAP_INFERNO)
cv2.imwrite('evidence_cards/deep3d_test_depth.png', depth_col)
print('\n=== SUCCESS: All components operational ===')
print('Saved: evidence_cards/deep3d_test_render.png')
print('Saved: evidence_cards/deep3d_test_depth.png')
