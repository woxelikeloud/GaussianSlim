# GaussianSlim

Official code repository for the paper:

> **GaussianSlim: Geometry-Aware Compact Gaussian Reconstruction for Mobile Web Rendering**

GaussianSlim is a compact 3D Gaussian Splatting system designed for efficient mobile Web deployment. The method combines geometry-aware Gaussian reconstruction with a compressed attribute loading and rendering pipeline.

## Repository structure

### `recon/`

The reconstruction algorithm described in the paper. This directory contains the geometry-aware densification and sparsification code, including curvature-guided asymmetric cloning and splitting, multi-view photometric gating, and capacity-constrained pruning. It produces compact Gaussian scenes while preserving the standard Gaussian attributes required by the renderer.

### `renderer/`

The mobile Web renderer. This directory contains the Three.js/WebGL frontend, Web Workers, WebAssembly decoder, compressed attribute recovery, GPU texture paths, visibility processing, asynchronous depth sorting, and browser demo. See [`renderer/README.md`](renderer/README.md) for build and usage instructions.

## System workflow

```text
Input images and camera poses
            ↓
recon/  Geometry-aware compact Gaussian reconstruction
            ↓
Compact Gaussian attributes and compressed asset
            ↓
renderer/  WebAssembly decoding, GPU upload, and mobile Web rendering
```

The two directories are modular: `recon/` controls the number and structure of retained Gaussians, while `renderer/` decodes and displays the resulting asset in a browser. The renderer consumes the standard ordered Gaussian attributes exported by the reconstruction stage.

<!-- ## Citation

```bibtex
@article{gaussianslim2026,
  title   = {GaussianSlim: Geometry-Aware Compact Gaussian Reconstruction for Mobile Web Rendering},
  author  = {Zhong, Houqiang and Li, Bate and Zhu, Tianchi and Cheng, Zhengxue and Hu, Qiang and Song, Li},
  journal = {arXiv preprint},
  year    = {2026}
}
``` -->

## License

See the license files in the relevant subdirectories. Renderer dependencies and third-party components retain their respective licenses.
