# GaussianSlim Compact Gaussian Web Renderer

This repository extends [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) into a WebGL reference renderer for GaussianSlim Compact Gaussian (3D Gaussian Splatting). The product code combines a Three.js/ES module frontend with a C++/Emscripten WASM decoder and provides GaussianSlim format parsing, reconstruction, compressed SH texture paths, and a browser demo.

## Device Requirements

The backend requires a Linux server for NPM compilation, service deployment, and runtime execution. Cloudflared is used to establish a public access tunnel and expose the local service to the Internet.

The frontend requires a network-enabled smartphone, which accesses the public URL provided by Cloudflared through a web browser or WeChat to load and run the frontend service.

## File Structure

The main file structure of the current project is as follows:

```
gaussianslim-renderer/

│   │   ├── SplatMaterial.js
│   │   ├── SplatMaterial2D.js
│   │   ├── SplatMaterial3D.js
│   │   └── SplatScene.js
│   │
│   ├── splattree/
│   │   └── SplatTree.js         # Visible-region management and spatial organization
│   │
│   ├── worker/
│   │   ├── SortWorker.js        # Transparent Gaussian sorting Worker
│   │   ├── sorter.cpp
│   │   ├── sorter.wasm
│   │   └── sorter_*             # SIMD/non-SIMD and shared/non-shared variants
│   │
│   ├── raycaster/               # Ray casting
│   ├── ui/
│   │   ├── InfoPanel.js         # Parameter panel opened with the I key and render resolution display
│   │   ├── LoadingProgressBar.js
│   │   └── LoadingSpinner.js
│   ├── webxr/                   # AR/VR support
│   └── three-shim/              # Three.js WebGL capability compatibility layer
│
├── 3DGS_Render_glb_unpack/      # C++ decoder and Emscripten build project
│   ├── CMakeLists.txt
│   ├── WasmBridge.cpp           # Coordinator WASM interface
│   ├── ReconstructionBridge.cpp # Reconstruction WASM interface
│   ├── build_wasm.sh
│   ├── build_wasm_new.sh
│   ├── build_wasm_windows.ps1
│   │
│   ├── compactgs_gsdecoder/
│   │   ├── gaussian_model/
│   │   │   ├── gs_decoder.cpp   # GSBS model-level decoding and preparation
│   │   │   ├── gs_data.cpp
│   │   │   └── reconstruction_kernel.cpp
│   │   ├── processor/
│   │   │   ├── stream.cpp       # GSBS bitstream parsing
│   │   │   ├── codec_decoder.cpp
│   │   │   ├── prediction.cpp
│   │   │   ├── quantizer.cpp
│   │   │   ├── transform.cpp
│   │   │   ├── unpacker.cpp
│   │   │   ├── texture_transcoder.cpp
│   │   │   └── platform/
│   │   │       ├── ffmpeg_video_decoder.cpp
│   │   │       ├── android_video_decoder.cpp
│   │   │       ├── apple_video_decoder.cpp
│   │   │       └── ohos_video_decoder.cpp
│   │   ├── test_codec0.cpp
│   │   └── test_decoder.cpp
│   │
│   └── thirdparty/
│       ├── astc-encoder
│       ├── bc7enc_rdo
│       ├── ffmpeg
│       ├── glm
│       ├── nlohmann
│       ├── zlib
│       └── zstd
│
├── util/
│   ├── server.js                # Local static server
│   ├── create-ksplat.js
│   └── import-base-64.js
│
├── emsdk/                       # Emscripten SDK directory/gitlink
├── build/                       # npm build outputs; not part of the main source code
└── node_modules/                # npm dependencies
```

The overall architecture can be divided into four main layers:

1. `demo/`: Handles local file selection, DPR input, and Viewer initialization.
2. `src/loaders/compactLoader/`: Handles GaussianSlim requests, capability-based strategy selection, Worker management, and WASM scheduling.
3. `3DGS_Render_glb_unpack/`: Handles C++ model decoding, video processing, texture decoding, and chunk-based reconstruction.
4. `src/splatmesh/` and `src/worker/`: Handle GPU resource construction, shading, and transparent Gaussian sorting.

## Supported Inputs

- The upstream `.ply`, `.splat`, `.ksplat`, and `.spz` loaders remain available.
- The GaussianSlim path accepts GaussianSlim-specific `.glb` containers with GSBS payloads; it does not support arbitrary glTF/GLB files.
- The demo accepts local GaussianSlim `.glb` files. The URL `.glb` branch in `Viewer.addSplatScene()` is currently blocked by an outer format guard and is therefore unreachable. This is a known limitation, not a supported API.

## Runtime Overview

- The coordinator decodes the complete input once and dispatches block-aligned shards to a fixed pool of `2` reconstruction Web Workers.
- The coordinator selects pthread WASM only when the page is cross-origin isolated and `SharedArrayBuffer` is available; otherwise it uses the single-thread fallback.
- The compressed SH texture policy uses ASTC after its GPU upload canary succeeds, then BC3 when both the browser and WASM build support it, and finally CPU spherical harmonics (SH). BC7 has an internal implementation, but the current capability mask does not publish it.
- Codec ID `0` raw YUV444P uses a passthrough path. Eligible HEVC Main 8-bit 4:2:0 can use WebCodecs after I420/NV12 frame-layout validation. Incompatible, failed, timed-out, and H.264 inputs use the WASM FFmpeg fallback. The presence of these paths is not evidence of target-device hardware-decoding acceptance.

## Quick Start

When the frontend dependencies and complete WASM deployment artifacts are available, run these commands from the repository root:

```bash
npm install
npm run build
npm run demo
```

Then open `http://localhost:8080/index.html`. `npm run demo` serves `build/demo`, so the build must run first. See the detailed guides for preparing WASM from a clean clone, exact Windows/Linux commands, mobile LAN debugging, cache invalidation, and validation checklists.

## Validation Status

The repository includes JS unit tests, ESLint, and frontend builds, but it has no automated acceptance suite for the complete browser rendering path. It does not currently claim accepted first-frame time, FPS, VRAM savings, or a completed device/browser matrix. Those conclusions require reproducible measurements with fixed samples on target devices.

## Documentation

- [Upstream repository and documentation](https://github.com/mkkellogg/GaussianSplats3D): original GaussianSplats3D API and background. The upstream npm package does not contain this fork's GaussianSlim extensions.

## Complete startup procedure

Install Node.js 18+, npm, Emscripten, and the native dependencies referenced by `3DGS_Render_glb_unpack/build_wasm_new.sh`. Initialize submodules when they are present:

```bash
git submodule update --init --recursive
npm install
```

Build the decoder and copy its six generated artifacts into `src/loaders/compactLoader/`:

```bash
source /path/to/emsdk/emsdk_env.sh
cd 3DGS_Render_glb_unpack
bash build_wasm_new.sh
cp output/gaussianslim_wasm.{js,wasm} ../src/loaders/compactLoader/
cp output/gaussianslim_wasm_pthread.{js,wasm} ../src/loaders/compactLoader/
cp output/gaussianslim_reconstruction_wasm.{js,wasm} ../src/loaders/compactLoader/
cd ..
npm run build
npm run demo
```

The server listens on `http://localhost:8080/`. For a phone or tablet on the same network, open `http://<host-ip>:8080/index.html`. The server sends COOP/COEP headers; verify `crossOriginIsolated` in `browser.html` before expecting pthread decoding. Without isolation or `SharedArrayBuffer`, the loader uses its single-thread fallback. `browser.html` probes WebGL2, ASTC/BC texture upload, SIMD, Workers, and transferable buffers.

## Runtime configuration and troubleshooting

The coordinator dispatches block-aligned work to two reconstruction Workers. Runtime texture selection follows ASTC, BC3, and CPU SH fallback according to capability probes. Eligible HEVC Main 8-bit 4:2:0 streams may use WebCodecs after planar-layout validation; unsupported or failed paths use the WASM FFmpeg fallback. The compressed loader accepts GaussianSlim GLB/GSBS assets and does not accept arbitrary glTF files.

For stale resources, clear site data, disable cache in DevTools, use a fresh profile, or deploy versioned Worker/JavaScript/WASM URLs. Android Chrome can be inspected through `chrome://inspect/#devices`; always check the current page URL, resource responses, and `crossOriginIsolated` state. Timing JSON records file, Worker, WASM, video, reconstruction, merge, texture upload, and first-frame stages; overlapping stages must not be added as independent wall-clock durations.

Run the available checks from the repository root:

```bash
node --test src/loaders/compactLoader/__tests__/*.test.js
npm run lint
npm run build
```

If Emscripten is missing, reactivate `emsdk_env.sh`. If FFmpeg or a codec library is missing, compare the error with the exact include and static-library paths required by the build script. A desktop build or unit test does not replace browser and target-device checks; record the sample, browser, device, build mode, cache state, and raw timing data for reproducible measurements.
