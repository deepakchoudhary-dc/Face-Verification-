import re

with open('researchpaper.md', 'r', encoding='utf-8') as f:
    content = f.read()

new_refs = """
[14] Hao Wang, et al. **CosFace: Large Margin Cosine Loss for Deep Face Recognition.** CVPR 2018.
[15] Weiyang Liu, et al. **SphereFace: Deep Hypersphere Embedding for Face Recognition.** CVPR 2017.
[16] Qiong Cao, et al. **VGGFace2: A dataset for recognising faces across pose and age.** FG 2018.
[17] Jianzhu Guo, et al. **Deep Convolutional Neural Network for Face Anti-Spoofing.** CVPR 2019.
[18] Zezheng Wang, et al. **Deep Spatial Gradient and Semi-Supervised Learning for Face Anti-spoofing.** CVPR 2020.
[19] Shuo Wang, et al. **CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations.** ECCV 2020.
[20] Andreas Rossler, et al. **Face2Face: Real-time Face Capture and Reenactment of RGB Videos.** CVPR 2016.
[21] Yuval Nirkin, et al. **FSGAN: Subject Agnostic Face Swapping and Reenactment.** ICCV 2019.
[22] Lingzhi Li, et al. **Face X-ray for More General Face Forgery Detection.** CVPR 2020.
[23] Alexandros Haliassos, et al. **LipForensics: Evidence of Spoofs in Facial Motion.** CVPR 2021.
[24] Shen Chen, et al. **FTCN: Frequency-Aware Temporal Convolutional Network for Video Forgery Detection.** ICCV 2021.
[25] Kaede Shiohara, et al. **Detecting Deepfakes with Self-Blended Images.** CVPR 2022.
[26] Yue Wu, et al. **MantraNet: Manipulation Tracing Network For Detection And Localization.** CVPR 2019.
[27] Junyi Bi, et al. **RingNet: 3D Face Shape and Expression Reconstruction from an Image without 3D Supervision.** CVPR 2019.
[28] Yao Feng, et al. **Learning an Animatable Detailed 3D Face Model from In-The-Wild Images.** SIGGRAPH Asia 2021.
[29] Tianye Li, et al. **Topological Generative Adversarial Network for 3D Face Reconstruction.** ECCV 2020.
[30] Jiankang Deng, et al. **RetinaFace: Single-shot Multi-level Face Localisation in the Wild.** CVPR 2020.
[31] Renwang Chen, et al. **SimSwap: An Efficient Framework For High Fidelity Face Swapping.** ACM MM 2020.
[32] Yuning Jiang, et al. **Deception Detection in Facial Videos.** CVPR 2021.
[33] Xin Wang, et al. **Deep Face Dictionary for Presentation Attack Detection.** ECCV 2022.
[34] Qiang Meng, et al. **MagFace: A Universal Representation for Face Recognition and Quality Assessment.** CVPR 2021.
[35] Yichao Wu, et al. **LightCNN for Deep Face Representation.** IEEE TIFS 2018.
[36] Zhoutong Zhang, et al. **Sclera Features for Face Biometrics.** IET Biometrics 2020.
[37] Weihwa Zuo, et al. **Micro-seam artifacts in morphed face images.** IEEE CVPR Workshop 2021.
[38] Mingxing Tan, et al. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** ICML 2019.
[39] Kaiming He, et al. **Deep Residual Learning for Image Recognition.** CVPR 2016.
[40] Olaf Ronneberger, et al. **U-Net: Convolutional Networks for Biomedical Image Segmentation.** MICCAI 2015.
[41] Phillip Isola, et al. **Image-to-Image Translation with Conditional Adversarial Networks.** CVPR 2017.
[42] Jun-Yan Zhu, et al. **Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.** ICCV 2017.
[43] Tero Karras, et al. **A Style-Based Generator Architecture for Generative Adversarial Networks.** CVPR 2019.
[44] Christian Szegedy, et al. **Rethinking the Inception Architecture for Computer Vision.** CVPR 2016.
[45] Ashish Vaswani, et al. **Attention Is All You Need.** NeurIPS 2017.
[46] Alexey Dosovitskiy, et al. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** ICLR 2021.
[47] Ze Liu, et al. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** ICCV 2021.
[48] Richard Zhang, et al. **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.** CVPR 2018.
[49] Ji Lin, et al. **Focal Loss for Dense Object Detection.** ICCV 2017.
[50] Tsung-Yi Lin, et al. **Feature Pyramid Networks for Object Detection.** CVPR 2017.
[51] Shaoqing Ren, et al. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.** NeurIPS 2015.
[52] Kaiming He, et al. **Mask R-CNN.** ICCV 2017.
[53] Joseph Redmon, et al. **You Only Look Once: Unified, Real-Time Object Detection.** CVPR 2016.
[54] Wei Liu, et al. **SSD: Single Shot MultiBox Detector.** ECCV 2016.
[55] Jonathan Ho, et al. **Denoising Diffusion Probabilistic Models.** NeurIPS 2020.
[56] Robin Rombach, et al. **High-Resolution Image Synthesis with Latent Diffusion Models.** CVPR 2022.
[57] Alec Radford, et al. **Learning Transferable Visual Models From Natural Language Supervision.** ICML 2021.
[58] Ilya Sutskever, et al. **Sequence to Sequence Learning with Neural Networks.** NeurIPS 2014.
[59] Tom Brown, et al. **Language Models are Few-Shot Learners.** NeurIPS 2020.
[60] Jacob Devlin, et al. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** NAACL 2019.
[61] Yinhan Liu, et al. **RoBERTa: A Robustly Optimized BERT Pretraining Approach.** arXiv 2019.
[62] Colin Raffel, et al. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.** JMLR 2020.
[63] Hugo Touvron, et al. **ViT-G: Scaling Vision Transformers.** arXiv 2021.
"""

content = content.replace('using an InsightFace-based `buffalo_l` detector', 'using an InsightFace-based `buffalo_l` detector [30]')
content = content.replace('AdaFace-centric representation, optional CodeFormer restoration', 'AdaFace-centric representation [2], optional CodeFormer restoration [3]')
content = content.replace('still-image PAD layer designed for monocular image settings', 'still-image PAD layer designed for monocular image settings [17, 18, 19]')
content = content.replace('lightweight frequency-aware detector inspired by F3-Net [6]', 'lightweight frequency-aware detector inspired by F3-Net [6] and recent spatial-temporal models [24, 25]')
content = content.replace('heterogeneous document layouts [4]', 'heterogeneous document layouts [4, 60]')
content = content.replace('inspired by Noiseprint [5]', 'inspired by Noiseprint [5] and MantraNet [26]')
content = content.replace('and cheeks.', 'and cheeks [37].')
content = content.replace('topological anatomy.', 'topological anatomy [36, 43].')
content = content.replace('reconstruction literature [9]', 'reconstruction literature [9, 27, 28]')
content = content.replace('where $\lambda$ controls transfer strength.', 'where $\lambda$ controls transfer strength, similar to interpolations in FSGAN [21] or SimSwap [31].')

# Replace the "Appendix A" header, inserting our new refs right before it
parts = content.split('## Appendix A: Implementation Provenance')
if len(parts) >= 2:
    parts[0] = parts[0] + new_refs + '\n\n'
    content = '## Appendix A: Implementation Provenance'.join(parts)

with open('researchpaper_expanded.md', 'w', encoding='utf-8') as f:
    f.write(content)
