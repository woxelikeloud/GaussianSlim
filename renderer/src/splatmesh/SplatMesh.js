import * as THREE from 'three';
import { SplatMaterial3D } from './SplatMaterial3D.js';
import { SplatMaterial2D } from './SplatMaterial2D.js';
import { getCompressedTextureSampleAddress } from './SplatMaterial.js';
import { SplatGeometry } from './SplatGeometry.js';
import { SplatScene } from './SplatScene.js';
import { SplatTree } from '../splattree/SplatTree.js';
import { WebGLExtensions } from '../three-shim/WebGLExtensions.js';
import { WebGLCapabilities } from '../three-shim/WebGLCapabilities.js';
import { uintEncodedFloat, rgbaArrayToInteger } from '../Util.js';
import { Constants } from '../Constants.js';
import { SceneRevealMode } from '../SceneRevealMode.js';
import { SplatRenderMode } from '../SplatRenderMode.js';
import { LogLevel } from '../LogLevel.js';
import { clamp, getSphericalHarmonicsComponentCountForDegree } from '../Util.js';

const dummyGeometry = new THREE.BufferGeometry();
const dummyMaterial = new THREE.MeshBasicMaterial();

const COVARIANCES_ELEMENTS_PER_SPLAT = 6;
const CENTER_COLORS_ELEMENTS_PER_SPLAT = 4;

const COVARIANCES_ELEMENTS_PER_TEXEL_STORED = 4;
const COVARIANCES_ELEMENTS_PER_TEXEL_ALLOCATED = 4;
const COVARIANCES_ELEMENTS_PER_TEXEL_COMPRESSED_STORED = 6;
const COVARIANCES_ELEMENTS_PER_TEXEL_COMPRESSED_ALLOCATED = 8;
const SCALES_ROTATIONS_ELEMENTS_PER_TEXEL = 4;
const CENTER_COLORS_ELEMENTS_PER_TEXEL = 4;
const SCENE_INDEXES_ELEMENTS_PER_TEXEL = 1;

const SCENE_FADEIN_RATE_FAST = 0.012;
const SCENE_FADEIN_RATE_GRADUAL = 0.003;

const VISIBLE_REGION_EXPANSION_DELTA = 1;
const EMPTY_UINT32_ARRAY = new Uint32Array(0);

const ASTC_FORMATS = {
    '4x4': THREE.RGBA_ASTC_4x4_Format,
    '5x5': THREE.RGBA_ASTC_5x5_Format,
    '6x6': THREE.RGBA_ASTC_6x6_Format,
    '8x8': THREE.RGBA_ASTC_8x8_Format,
    '10x10': THREE.RGBA_ASTC_10x10_Format,
    '12x12': THREE.RGBA_ASTC_12x12_Format
};

function compressedTextureThreeFormat(format, blockWidth, blockHeight) {
    if (format === 'astc') return ASTC_FORMATS[`${blockWidth}x${blockHeight}`];
    if (blockWidth !== 4 || blockHeight !== 4) return undefined;
    if (format === 'bc7') return THREE.RGBA_BPTC_Format;
    if (format === 'bc3') return THREE.RGBA_S3TC_DXT5_Format;
    return undefined;
}

function validateCompressedTexturePayload(payload) {
    if (!payload || typeof payload !== 'object') throw new Error('Compressed texture payload is missing.');
    const { format, raw, metas } = payload;
    const width = Number(payload.width ?? metas?.[1]);
    const height = Number(payload.height ?? metas?.[2]);
    const layers = Number(payload.layers ?? (metas ? metas.length / 6 : 0));
    const blockWidth = Number(payload.blockWidth ?? metas?.[0]);
    const blockHeight = Number(payload.blockHeight ?? metas?.[0]);
    const singleWidth = Number(payload.singleWidth ?? metas?.[3]);
    const regionPixelCount = Number(payload.regionPixelCount ?? metas?.[4]);
    const singleHeight = Number(payload.singleHeight ?? regionPixelCount / singleWidth);
    const threeFormat = compressedTextureThreeFormat(format, blockWidth, blockHeight);

    if (!(raw instanceof Uint8Array) || raw.byteLength === 0) {
        throw new Error(`Compressed texture payload ${format || 'unknown'} has no byte data.`);
    }
    if (![width, height, layers, blockWidth, blockHeight, singleWidth, singleHeight,
        regionPixelCount].every(Number.isInteger) ||
        width <= 0 || height <= 0 || layers <= 0 || blockWidth <= 0 || blockHeight <= 0 || singleWidth <= 0 ||
        singleHeight <= 0 || regionPixelCount !== singleWidth * singleHeight ||
        singleWidth > width || singleHeight > height || width > 16384 || height > 16384 || layers !== 1 || !threeFormat ||
        !Number.isFinite(payload.shnMin) || !Number.isFinite(payload.shnMax) || payload.shnMax < payload.shnMin) {
        throw new Error(`Compressed texture payload ${format || 'unknown'} has invalid format or dimensions.`);
    }
    if (metas && (metas.length !== layers * 6 ||
        payload.textureNum !== undefined && payload.textureNum !== layers)) {
        throw new Error(`Compressed texture payload ${format} has inconsistent layer metadata.`);
    }
    if (metas) {
        let metadataBytes = 0;
        for (let layer = 0; layer < layers; layer++) {
            const offset = layer * 6;
            if (metas[offset] !== blockWidth || metas[offset + 1] !== width ||
                metas[offset + 2] !== height || metas[offset + 3] !== singleWidth ||
                metas[offset + 4] !== regionPixelCount) {
                throw new Error(`Compressed texture payload ${format} has incompatible array-layer metadata.`);
            }
            metadataBytes += metas[offset + 5];
        }
        if (metadataBytes !== raw.byteLength) {
            throw new Error(`Compressed texture payload ${format} stream sizes do not match its byte data.`);
        }
    }

    const expectedRawSize = Math.ceil(width / blockWidth) * Math.ceil(height / blockHeight) * 16 * layers;
    if (raw.byteLength !== expectedRawSize) {
        throw new Error(
            `Compressed texture payload ${format} has ${raw.byteLength} bytes; expected ${expectedRawSize}.`
        );
    }
    if (payload.uvs instanceof Uint32Array && payload.uvs.length >= 2) {
        const shDegree = Math.max(1, Math.min(2, Number(payload.shDegree) || 1));
        const coefficientCount = getSphericalHarmonicsComponentCountForDegree(shDegree) / 3;
        const lastAddress = getCompressedTextureSampleAddress(
            payload.uvs[0], payload.uvs[1], coefficientCount - 1,
            singleWidth, singleHeight, width
        );
        if (lastAddress.x >= width || lastAddress.y >= height || lastAddress.layer !== 0) {
            throw new Error(`Compressed texture payload ${format} cannot address its SH regions.`);
        }
    }
    return {
        format, raw, width, height, layers, blockWidth, blockHeight,
        singleWidth, singleHeight, threeFormat
    };
}

const recordProcessingDuration = (processingProfile, key, startTime) => {
    if (!processingProfile) return;
    const elapsedMs = performance.now() - startTime;
    processingProfile[key] = (processingProfile[key] || 0) + elapsedMs;
};

// Based on my own observations across multiple devices, OSes and browsers, using textures that have one dimension
// greater than 4096 while the other is greater than or equal to 4096 causes issues (Essentially any texture larger
// than 4096 x 4096 (16777216) texels). Specifically it seems all texture data beyond the 4096 x 4096 texel boundary
// is corrupted, while data below that boundary is usable. In these cases the texture has been valid in the eyes of
// both Three.js and WebGL, and the texel format (RG, RGBA, etc.) has not mattered. More investigation will be needed,
// but for now the work-around is to split the spherical harmonics into three textures (one for each color channel).
const MAX_TEXTURE_TEXELS = 16777216;

/**
 * SplatMesh: Container for one or more splat scenes, abstracting them into a single unified container for
 * splat data. Additionally contains data structures and code to make the splat data renderable as a Three.js mesh.
 */
export class SplatMesh extends THREE.Mesh {

    constructor(splatRenderMode = SplatRenderMode.ThreeD, dynamicMode = false, enableOptionalEffects = false,
                halfPrecisionCovariancesOnGPU = false, devicePixelRatio = 1, enableDistancesComputationOnGPU = true,
                integerBasedDistancesComputation = false, antialiased = false, maxScreenSpaceSplatSize = 324, logLevel = LogLevel.None,
                sphericalHarmonicsDegree = 0, sceneFadeInRateMultiplier = 1.0, kernel2DSize = 0.18) {
        super(dummyGeometry, dummyMaterial);

        // Reference to a Three.js renderer
        this.renderer = undefined;

        // Determine how the splats are rendered
        this.splatRenderMode = splatRenderMode;

        // When 'dynamicMode' is true, scenes are assumed to be non-static. Dynamic scenes are handled differently
        // and certain optimizations cannot be made for them. Additionally, by default, all splat data retrieved from
        // this splat mesh will not have their scene transform applied to them if the splat mesh is dynamic. That
        // can be overriden via parameters to the individual functions that are used to retrieve splat data.
        this.dynamicMode = dynamicMode;

        // When true, allows for usage of extra properties and attributes during rendering for effects such as opacity adjustment.
        // Default is false for performance reasons. These properties are separate from transform properties (scale, rotation, position)
        // that are enabled by the 'dynamicScene' parameter.
        this.enableOptionalEffects = enableOptionalEffects;

        // Use 16-bit floating point values when storing splat covariance data in textures, instead of 32-bit
        this.halfPrecisionCovariancesOnGPU = halfPrecisionCovariancesOnGPU;

        // Ratio of the resolution in physical pixels to the resolution in CSS pixels for the current display device
        this.devicePixelRatio = devicePixelRatio;

        // Use a transform feedback to calculate splat distances from the camera
        this.enableDistancesComputationOnGPU = enableDistancesComputationOnGPU;

        // Use a faster integer-based approach for calculating splat distances from the camera
        this.integerBasedDistancesComputation = integerBasedDistancesComputation;

        // When true, will perform additional steps during rendering to address artifacts caused by the rendering of gaussians at a
        // substantially different resolution than that at which they were rendered during training. This will only work correctly
        // for models that were trained using a process that utilizes this compensation calculation. For more details:
        // https://github.com/nerfstudio-project/gsplat/pull/117
        // https://github.com/graphdeco-inria/gaussian-splatting/issues/294#issuecomment-1772688093
        this.antialiased = antialiased;

        // The size of the 2D kernel used for splat rendering
        // This will adjust the 2D kernel size after the projection
        this.kernel2DSize = kernel2DSize;

        // Specify the maximum clip space splat size, can help deal with large splats that get too unwieldy
        this.maxScreenSpaceSplatSize = maxScreenSpaceSplatSize;

        // The verbosity of console logging
        this.logLevel = logLevel;

        // Degree 0 means no spherical harmonics
        this.sphericalHarmonicsDegree = sphericalHarmonicsDegree;
        this.minSphericalHarmonicsDegree = 0;

        this.sceneFadeInRateMultiplier = sceneFadeInRateMultiplier;

        // The individual splat scenes stored in this splat mesh, each containing their own transform
        this.scenes = [];

        // Special octree tailored to SplatMesh instances
        this.splatTree = null;
        this.baseSplatTree = null;

        // Cache textures and the intermediate data used to populate them
        this.splatDataTextures = {};

        this.distancesTransformFeedback = {
            'id': null,
            'vertexShader': null,
            'fragmentShader': null,
            'program': null,
            'centersBuffer': null,
            'sceneIndexesBuffer': null,
            'outDistancesBuffer': null,
            'centersLoc': -1,
            'modelViewProjLoc': -1,
            'sceneIndexesLoc': -1,
            'transformsLocs': []
        };

        this.globalSplatIndexToLocalSplatIndexMap = [];
        this.globalSplatIndexToSceneIndexMap = [];

        this.lastBuildSplatCount = 0;
        this.lastBuildScenes = [];
        this.lastBuildMaxSplatCount = 0;
        this.lastBuildSceneCount = 0;
        this.firstRenderTime = -1;
        this.finalBuild = false;

        this.webGLUtils = null;

        this.boundingBox = new THREE.Box3();
        this.calculatedSceneCenter = new THREE.Vector3();
        this.maxSplatDistanceFromSceneCenter = 0;
        this.visibleRegionBufferRadius = 0;
        this.visibleRegionRadius = 0;
        this.visibleRegionFadeStartRadius = 0;
        this.visibleRegionChanging = false;

        this.splatScale = 1.0;
        this.pointCloudModeEnabled = false;

        this.useCompressedTexture = false;
        // Compatibility alias retained for integrations that inspect this flag.
        this.useASTC = false;
        this.compactGaussianMode = false;
        this.compactGaussianDistanceCenters = null;

        this.disposed = false;
        this.lastRenderer = null;
        this.visible = false;
    }

    /**
     * Build a container for each scene managed by this splat mesh based on an instance of SplatBuffer, along with optional
     * transform data (position, scale, rotation) passed to the splat mesh during the build process.
     * @param {Array<THREE.Matrix4>} splatBuffers SplatBuffer instances containing splats for each scene
     * @param {Array<object>} sceneOptions Array of options objects: {
     *
     *         position (Array<number>):   Position of the scene, acts as an offset from its default position, defaults to [0, 0, 0]
     *
     *         rotation (Array<number>):   Rotation of the scene represented as a quaternion, defaults to [0, 0, 0, 1]
     *
     *         scale (Array<number>):      Scene's scale, defaults to [1, 1, 1]
     * }
     * @return {Array<THREE.Matrix4>}
     */
    static buildScenes(parentObject, splatBuffers, sceneOptions) {
        const scenes = [];
        scenes.length = splatBuffers.length;
        for (let i = 0; i < splatBuffers.length; i++) {
            const splatBuffer = splatBuffers[i];
            const options = sceneOptions[i] || {};
            let positionArray = options['position'] || [0, 0, 0];
            let rotationArray = options['rotation'] || [0, 0, 0, 1];
            let scaleArray = options['scale'] || [1, 1, 1];
            const position = new THREE.Vector3().fromArray(positionArray);
            const rotation = new THREE.Quaternion().fromArray(rotationArray);
            const scale = new THREE.Vector3().fromArray(scaleArray);
            const scene = SplatMesh.createScene(splatBuffer, position, rotation, scale,
                                                options.splatAlphaRemovalThreshold || 1, options.opacity, options.visible);
            parentObject.add(scene);
            scenes[i] = scene;
        }
        return scenes;
    }

    static createScene(splatBuffer, position, rotation, scale, minimumAlpha, opacity = 1.0, visible = true) {
        return new SplatScene(splatBuffer, position, rotation, scale, minimumAlpha, opacity, visible);
    }

    /**
     * Build data structures that map global splat indexes (based on a unified index across all splat buffers) to
     * local data within a single scene.
     * @param {Array<SplatBuffer>} splatBuffers Instances of SplatBuffer off which to build the maps
     * @return {object}
     */
    static buildSplatIndexMaps(splatBuffers) {
        const localSplatIndexMap = [];
        const sceneIndexMap = [];
        let totalSplatCount = 0;
        for (let s = 0; s < splatBuffers.length; s++) {
            const splatBuffer = splatBuffers[s];
            const maxSplatCount = splatBuffer.getMaxSplatCount();
            for (let i = 0; i < maxSplatCount; i++) {
                localSplatIndexMap[totalSplatCount] = i;
                sceneIndexMap[totalSplatCount] = s;
                totalSplatCount++;
            }
        }
        return {
            localSplatIndexMap,
            sceneIndexMap
        };
    }

    /**
     * Build an instance of SplatTree (a specialized octree) for the given splat mesh.
     * @param {Array<number>} minAlphas Array of minimum splat slphas for each scene
     * @param {function} onSplatTreeIndexesUpload Function to be called when the upload of splat centers to the splat tree
     *                                            builder worker starts and finishes.
     * @param {function} onSplatTreeConstruction Function to be called when the conversion of the local splat tree from
     *                                           the format produced by the splat tree builder worker starts and ends.
     * @return {SplatTree}
     */
     buildSplatTree = function(minAlphas = [], onSplatTreeIndexesUpload, onSplatTreeConstruction) {
        return new Promise((resolve) => {
            this.disposeSplatTree();
            // TODO: expose SplatTree constructor parameters (maximumDepth and maxCentersPerNode) so that they can
            // be configured on a per-scene basis
            this.baseSplatTree = new SplatTree(8, 1000);
            const buildStartTime = performance.now();
            const splatColor = new THREE.Vector4();
            this.baseSplatTree.processSplatMesh(this, (splatIndex) => {
                this.getSplatColor(splatIndex, splatColor);
                const sceneIndex = this.getSceneIndexForSplat(splatIndex);
                const minAlpha = minAlphas[sceneIndex] || 1;
                return splatColor.w >= minAlpha;
            }, onSplatTreeIndexesUpload, onSplatTreeConstruction)
            .then(() => {
                const buildTime = performance.now() - buildStartTime;
                if (this.logLevel >= LogLevel.Info) console.log('SplatTree build: ' + buildTime + ' ms');
                if (this.disposed) {
                    resolve();
                } else {

                    this.splatTree = this.baseSplatTree;
                    this.baseSplatTree = null;

                    let leavesWithVertices = 0;
                    let avgSplatCount = 0;
                    let maxSplatCount = 0;
                    let nodeCount = 0;

                    this.splatTree.visitLeaves((node) => {
                        const nodeSplatCount = node.data.indexes.length;
                        if (nodeSplatCount > 0) {
                            avgSplatCount += nodeSplatCount;
                            maxSplatCount = Math.max(maxSplatCount, nodeSplatCount);
                            nodeCount++;
                            leavesWithVertices++;
                        }
                    });
                    if (this.logLevel >= LogLevel.Info) {
                        console.log(`SplatTree leaves: ${this.splatTree.countLeaves()}`);
                        console.log(`SplatTree leaves with splats:${leavesWithVertices}`);
                        avgSplatCount = avgSplatCount / nodeCount;
                        console.log(`Avg splat count per node: ${avgSplatCount}`);
                        console.log(`Total splat count: ${this.getSplatCount()}`);
                    }
                    resolve();
                }
            });
        });
    };

    /**
     * Construct this instance of SplatMesh.
     * @param {Array<SplatBuffer>} splatBuffers The base splat data, instances of SplatBuffer
     * @param {Array<object>} sceneOptions Dynamic options for each scene {
     *
     *         splatAlphaRemovalThreshold: Ignore any splats with an alpha less than the specified
     *                                     value (valid range: 0 - 255), defaults to 1
     *
     *         position (Array<number>):   Position of the scene, acts as an offset from its default position, defaults to [0, 0, 0]
     *
     *         rotation (Array<number>):   Rotation of the scene represented as a quaternion, defaults to [0, 0, 0, 1]
     *
     *         scale (Array<number>):      Scene's scale, defaults to [1, 1, 1]
     *
     * }
     * @param {boolean} keepSceneTransforms For a scene that already exists and is being overwritten, this flag
     *                                      says to keep the transform from the existing scene.
     * @param {boolean} finalBuild Will the splat mesh be in its final state after this build?
     * @param {function} onSplatTreeIndexesUpload Function to be called when the upload of splat centers to the splat tree
     *                                            builder worker starts and finishes.
     * @param {function} onSplatTreeConstruction Function to be called when the conversion of the local splat tree from
     *                                           the format produced by the splat tree builder worker starts and ends.
     * @return {object} Object containing info about the splats that are updated
     */
    build(splatBuffers, sceneOptions, keepSceneTransforms = true, finalBuild = false,
          onSplatTreeIndexesUpload, onSplatTreeConstruction, preserveVisibleRegion = true, processingProfile = null) {

        const buildStartTime = performance.now();

        const fastSceneCount = splatBuffers.filter((buffer) => buffer?.isCompactGSSceneData).length;
        if (fastSceneCount > 0) {
            if (fastSceneCount !== splatBuffers.length || splatBuffers.length !== 1) {
                throw new Error('COMPACTGS_V1 supports exactly one scene and cannot be mixed with profile 1 scenes.');
            }
            return this.buildCompactGS(splatBuffers[0], sceneOptions?.[0] || {}, finalBuild,
                                    onSplatTreeIndexesUpload, onSplatTreeConstruction,
                                    processingProfile, buildStartTime);
        }
        this.compactGaussianMode = false;
        this.compactGaussianDistanceCenters = null;

        this.sceneOptions = sceneOptions;
        this.finalBuild = finalBuild;

        const compressedTexturePayloads = splatBuffers.map((buffer) =>
            buffer.compressedTextureData || buffer.astcData || null
        );
        this.useCompressedTexture = splatBuffers.some(b => b.hasCompressedTexture || b.hasAstc);
        if (this.useCompressedTexture && compressedTexturePayloads.some((payload) => !payload)) {
            throw new Error('Compressed-texture scenes cannot be mixed with uncompressed scenes in one SplatMesh.');
        }
        if (this.useCompressedTexture &&
            compressedTexturePayloads.some((payload) => payload !== compressedTexturePayloads[0])) {
            throw new Error('A SplatMesh cannot combine scenes backed by different compressed texture payloads.');
        }
        this.useASTC = this.useCompressedTexture;

        const maxSplatCount = SplatMesh.getTotalMaxSplatCountForSplatBuffers(splatBuffers);

        const newScenes = SplatMesh.buildScenes(this, splatBuffers, sceneOptions);
        if (keepSceneTransforms) {
            for (let i = 0; i < this.scenes.length && i < newScenes.length; i++) {
                const newScene = newScenes[i];
                const existingScene = this.getScene(i);
                newScene.copyTransformData(existingScene);
            }
        }
        this.scenes = newScenes;

        let minSphericalHarmonicsDegree = 3;
        for (let splatBuffer of splatBuffers) {
            let bufferDegree = splatBuffer.getMinSphericalHarmonicsDegree();

            const compressedTextureData = splatBuffer.compressedTextureData || splatBuffer.astcData;
            if (compressedTextureData?.shDegree !== undefined) {
                bufferDegree = compressedTextureData.shDegree;
            }
            if (bufferDegree < minSphericalHarmonicsDegree) {
                minSphericalHarmonicsDegree = bufferDegree;
            }
        }
        this.minSphericalHarmonicsDegree = Math.min(minSphericalHarmonicsDegree, this.sphericalHarmonicsDegree);
        let splatBuffersChanged = false;
        if (splatBuffers.length !== this.lastBuildScenes.length) {
            splatBuffersChanged = true;
        } else {
            for (let i = 0; i < splatBuffers.length; i++) {
                const splatBuffer = splatBuffers[i];
                if (splatBuffer !== this.lastBuildScenes[i].splatBuffer) {
                    splatBuffersChanged = true;
                    break;
                }
            }
        }

        let isUpdateBuild = true;
        if (this.scenes.length !== 1 ||
            this.lastBuildSceneCount !== this.scenes.length ||
            this.lastBuildMaxSplatCount !== maxSplatCount ||
            splatBuffersChanged) {
                isUpdateBuild = false;
       }

       if (!isUpdateBuild) {
            this.boundingBox = new THREE.Box3();
            if (!preserveVisibleRegion) {
                this.maxSplatDistanceFromSceneCenter = 0;
                this.visibleRegionBufferRadius = 0;
                this.visibleRegionRadius = 0;
                this.visibleRegionFadeStartRadius = 0;
                this.firstRenderTime = -1;
            }
            this.lastBuildScenes = [];
            this.lastBuildSplatCount = 0;
            this.lastBuildMaxSplatCount = 0;
            this.disposeMeshData();
            this.geometry = SplatGeometry.build(maxSplatCount);
            if (this.splatRenderMode === SplatRenderMode.ThreeD) {
                this.material = SplatMaterial3D.build(this.dynamicMode, this.enableOptionalEffects, this.antialiased,
                                                      this.maxScreenSpaceSplatSize, this.splatScale, this.pointCloudModeEnabled,
                                                      this.minSphericalHarmonicsDegree, this.kernel2DSize, this.useCompressedTexture);
            } else {
                this.material = SplatMaterial2D.build(this.dynamicMode, this.enableOptionalEffects,
                                                      this.splatScale, this.pointCloudModeEnabled, this.minSphericalHarmonicsDegree);
            }

            const indexMaps = SplatMesh.buildSplatIndexMaps(splatBuffers);
            this.globalSplatIndexToLocalSplatIndexMap = indexMaps.localSplatIndexMap;
            this.globalSplatIndexToSceneIndexMap = indexMaps.sceneIndexMap;
        }

        const splatBufferSplatCount = this.getSplatCount(true);
        if (this.enableDistancesComputationOnGPU) this.setupDistancesComputationTransformFeedback();
        const dataUpdateResults = this.refreshGPUDataFromSplatBuffers(isUpdateBuild, processingProfile);

        for (let i = 0; i < this.scenes.length; i++) {
            this.lastBuildScenes[i] = this.scenes[i];
        }
        this.lastBuildSplatCount = splatBufferSplatCount;
        this.lastBuildMaxSplatCount = this.getMaxSplatCount();
        this.lastBuildSceneCount = this.scenes.length;

        let splatTreePromise = null;
        if (finalBuild && this.scenes.length > 0) {
            const splatTreeStartTime = performance.now();
            splatTreePromise = this.buildSplatTree(sceneOptions.map(options => options.splatAlphaRemovalThreshold || 1),
                                                   onSplatTreeIndexesUpload, onSplatTreeConstruction)
            .then(() => {
                recordProcessingDuration(processingProfile, 'splatTreeAsyncMs', splatTreeStartTime);
                if (this.onSplatTreeReadyCallback) this.onSplatTreeReadyCallback(this.splatTree);
                this.onSplatTreeReadyCallback = null;
            });
        }

        this.visible = (this.scenes.length > 0);

        if (processingProfile) {
            processingProfile.meshBuildTotalMs = performance.now() - buildStartTime;
            processingProfile.meshRebuild = !isUpdateBuild;
            processingProfile.meshSplatCount = splatBufferSplatCount;
            processingProfile.meshMaxSplatCount = this.getMaxSplatCount();
            processingProfile.splatTreeAsyncStarted = !!(finalBuild && this.scenes.length > 0);
            dataUpdateResults.processingProfile = processingProfile;
        }
        dataUpdateResults.splatTreePromise = splatTreePromise;

        return dataUpdateResults;
    }

    buildCompactGS(fastData, sceneOptions, finalBuild, onSplatTreeIndexesUpload,
                onSplatTreeConstruction, processingProfile, buildStartTime) {
        if (this.dynamicMode || this.splatRenderMode !== SplatRenderMode.ThreeD) {
            throw new Error('COMPACTGS_V1 v1 supports only static 3D rendering.');
        }
        const position = sceneOptions.position || [0, 0, 0];
        const rotation = sceneOptions.rotation || [0, 0, 0, 1];
        const scale = sceneOptions.scale || [1, 1, 1];
        if (position.some((value) => value !== 0) ||
            rotation.some((value, index) => value !== (index === 3 ? 1 : 0)) ||
            scale.some((value) => value !== 1)) {
            throw new Error('COMPACTGS_V1 v1 requires an identity scene transform.');
        }

        this.disposeSplatTree();
        this.disposeMeshData();
        this.disposeTextures();
        for (const scene of this.scenes) this.remove(scene);
        this.compactGaussianMode = true;
        this.finalBuild = finalBuild;
        this.sceneOptions = [sceneOptions];
        this.scenes = SplatMesh.buildScenes(this, [fastData], [sceneOptions]);
        this.useCompressedTexture = !!fastData.compressedTextureData;
        this.useASTC = fastData.compressedTextureData?.format === 'astc';
        this.minSphericalHarmonicsDegree = Math.min(3, fastData.shDegree, this.sphericalHarmonicsDegree);
        const count = fastData.getSplatCount();
        // COMPACTGS renders a block of 128 splats per hardware instance. The
        // source index for each local quad is supplied by the order texture.
        this.geometry = SplatGeometry.build(count, true, 128);
        this.material = SplatMaterial3D.buildCompactGS(
            this.maxScreenSpaceSplatSize, this.splatScale, this.pointCloudModeEnabled,
            this.minSphericalHarmonicsDegree, this.kernel2DSize, this.useCompressedTexture
        );

        const { descriptor, planes } = fastData;
        const width = descriptor.pointMapWidth;
        const height = descriptor.pointMapHeight;
        const capacity = width * height;
        if (![2, 3].includes(descriptor.layoutVersion) || !(planes.positionLow instanceof Uint8Array) ||
            !(planes.positionHigh instanceof Uint8Array) || !(planes.colorRgb instanceof Uint8Array) ||
            !(planes.shape instanceof Uint8Array) || planes.positionLow.length !== capacity * 3 ||
            planes.positionHigh.length !== capacity * 3 || planes.colorRgb.length !== capacity * 3 ||
            planes.shape.length !== capacity * (descriptor.layoutVersion >= 3 ? 16 : 12) || !(descriptor.shMin instanceof Float32Array) ||
            !(descriptor.shStep instanceof Float32Array) || descriptor.shMin.length !== 45 ||
            descriptor.shStep.length !== 45) {
            throw new Error('COMPACTGS_V1 typed planes or quantization tables are invalid.');
        }
        if (descriptor.layoutVersion === 2 &&
            (!(fastData.covariances instanceof Float32Array) || fastData.covariances.length !== count * 6)) {
            throw new Error('COMPACTGS layout 2 covariance data is missing or invalid.');
        }
        if (descriptor.layoutVersion >= 3) {
            const groupCount = capacity > 0 ? Math.ceil(fastData.numPoints / descriptor.positionGroupSize) : 0;
            if (!(descriptor.covarianceGroupMin instanceof Float32Array) ||
                !(descriptor.covarianceGroupStep instanceof Float32Array) ||
                descriptor.covarianceGroupMin.length !== groupCount * 6 ||
                descriptor.covarianceGroupStep.length !== groupCount * 6) {
                throw new Error('COMPACTGS covariance quantization tables are invalid.');
            }
        }
        const makeTexture = (data, textureWidth, textureHeight, format, type, internalFormat = null) => {
            const texture = new THREE.DataTexture(data, textureWidth, textureHeight, format, type);
            if (internalFormat) texture.internalFormat = internalFormat;
            texture.minFilter = THREE.NearestFilter;
            texture.magFilter = THREE.NearestFilter;
            texture.generateMipmaps = false;
            texture.flipY = false;
            texture.unpackAlignment = 1;
            texture.needsUpdate = true;
            return texture;
        };
        // Three.js r160 no longer exposes RGB/RGBInteger upload formats. Keep
        // the wire planes RGB and expand only the GPU upload backing to RGBA;
        // this remains byte-quantized and never materializes float centers.
        const expandRgb = (source, ArrayType, alpha) => {
            const target = new ArrayType(source.length / 3 * 4);
            for (let src = 0, dst = 0; src < source.length; src += 3, dst += 4) {
                target[dst] = source[src];
                target[dst + 1] = source[src + 1];
                target[dst + 2] = source[src + 2];
                target[dst + 3] = alpha;
            }
            return target;
        };
        const positionLowUpload = expandRgb(planes.positionLow, Uint8Array, 0);
        const positionHighUpload = expandRgb(planes.positionHigh, Uint8Array, 0);
        // Decode quantized centers exactly once during mesh construction. The
        // original low/high planes remain available for sorting and diagnostics,
        // while rendering uses this float texture directly.
        const decodedPositions = new Float32Array(capacity * 4);
        const positionGroupSize = descriptor.positionGroupSize;
        for (let point = 0; point < count; point++) {
            const source = point * 3;
            const groupBase = Math.floor(point / positionGroupSize) * 3;
            const qx = (planes.positionHigh[source] << 7) | planes.positionLow[source];
            const qy = (planes.positionHigh[source + 1] << 7) | planes.positionLow[source + 1];
            const qz = (planes.positionHigh[source + 2] << 7) | planes.positionLow[source + 2];
            const target = point * 4;
            decodedPositions[target] = descriptor.positionGroupMin[groupBase] +
                descriptor.positionGroupStep[groupBase] * qx;
            decodedPositions[target + 1] = descriptor.positionGroupMin[groupBase + 1] +
                descriptor.positionGroupStep[groupBase + 1] * qy;
            decodedPositions[target + 2] = descriptor.positionGroupMin[groupBase + 2] +
                descriptor.positionGroupStep[groupBase + 2] * qz;
            decodedPositions[target + 3] = 1.0;
        }
        const colorUpload = expandRgb(planes.colorRgb, Uint8Array, 255);
        const groupMinUpload = expandRgb(descriptor.positionGroupMin, Float32Array, 0);
        const groupStepUpload = expandRgb(descriptor.positionGroupStep, Float32Array, 0);
        const positionLow = makeTexture(positionLowUpload, width, height,
                                        THREE.RGBAIntegerFormat, THREE.UnsignedByteType, 'RGBA8UI');
        const positionHigh = makeTexture(positionHighUpload, width, height,
                                         THREE.RGBAIntegerFormat, THREE.UnsignedByteType, 'RGBA8UI');
        const decodedPositionTexture = makeTexture(decodedPositions, width, height,
                                     THREE.RGBAFormat, THREE.FloatType, 'RGBA32F');
        const color = makeTexture(colorUpload, width, height, THREE.RGBAFormat, THREE.UnsignedByteType, 'RGBA8');
        // Keep color in a regular RGBA8 texture. Packing a 24-bit color into
        // the alpha lane of an RGBA32F position texture is not portable on
        // mobile GPUs (notably Apple's Metal/WebGL implementation).
        let positionColorTexture = null;
        let positionColorUpload = null;
        // The shader intentionally uses fastPositionTexture + fastColorTexture
        // for all platforms; this avoids implementation-dependent integer
        // conversion in the vertex shader and fixes Apple color expansion.
        this.material.uniforms.fastUsePositionColorTexture.value = 0;
        const shape = makeTexture(planes.shape, width * 4,
                                 height * (descriptor.layoutVersion >= 3 ? 4 : 3),
                                 THREE.RedFormat, THREE.UnsignedByteType, 'R8');
        const groupCount = descriptor.positionGroupMin.length / 3;
        const groupMin = makeTexture(groupMinUpload, groupCount, 1, THREE.RGBAFormat, THREE.FloatType, 'RGBA32F');
        const groupStep = makeTexture(groupStepUpload, groupCount, 1, THREE.RGBAFormat, THREE.FloatType, 'RGBA32F');
        this.material.uniforms.fastPositionLowTexture.value = positionLow;
        this.material.uniforms.fastPositionHighTexture.value = positionHigh;
        this.material.uniforms.fastPositionTexture.value = decodedPositionTexture;
        this.material.uniforms.fastColorTexture.value = color;
        this.material.uniforms.fastShapeTexture.value = shape;
        this.material.uniforms.fastPositionGroupMinTexture.value = groupMin;
        this.material.uniforms.fastPositionGroupStepTexture.value = groupStep;
        // Order texture dimensions are deliberately modest to remain valid on
        // WebGL implementations with a 4096 texture-size limit.
        const maxTextureSize = Math.max(1, Math.min(4096, this.renderer?.capabilities?.maxTextureSize || 4096));
        const orderWidth = Math.min(Math.max(count, 1), maxTextureSize);
        const orderHeight = Math.max(1, Math.ceil(Math.max(count, 1) / orderWidth));
        const orderData = new Uint32Array(orderWidth * orderHeight);
        for (let i = 0; i < count; i++) orderData[i] = i;
        const orderTexture = makeTexture(orderData, orderWidth, orderHeight,
                                         THREE.RedIntegerFormat, THREE.UnsignedIntType, 'R32UI');
        this.material.uniforms.fastOrderTexture.value = orderTexture;
        this.material.uniforms.fastOrderTextureSize.value.set(orderWidth, orderHeight);
        this.material.uniforms.fastBatched.value = 1;
        this.material.uniforms.fastBatchSize.value = 128;
        this.material.uniforms.fastRenderSplatCount.value = count;
        if (descriptor.layoutVersion >= 3) {
            const covMin = new Float32Array((descriptor.covarianceGroupMin.length / 6) * 8);
            const covStep = new Float32Array(covMin.length);
            for (let g = 0; g < descriptor.covarianceGroupMin.length / 6; g++) {
                covMin.set(descriptor.covarianceGroupMin.slice(g * 6, g * 6 + 4), g * 8);
                covMin.set(descriptor.covarianceGroupMin.slice(g * 6 + 4, g * 6 + 6), g * 8 + 4);
                covStep.set(descriptor.covarianceGroupStep.slice(g * 6, g * 6 + 4), g * 8);
                covStep.set(descriptor.covarianceGroupStep.slice(g * 6 + 4, g * 6 + 6), g * 8 + 4);
            }
            const covarianceGroupWidth = (descriptor.covarianceGroupMin.length / 6) * 2;
            this.material.uniforms.fastCovarianceGroupMinTexture.value = makeTexture(
                covMin, covarianceGroupWidth, 1, THREE.RGBAFormat, THREE.FloatType, 'RGBA32F');
            this.material.uniforms.fastCovarianceGroupStepTexture.value = makeTexture(
                covStep, covarianceGroupWidth, 1, THREE.RGBAFormat, THREE.FloatType, 'RGBA32F');
            this.material.uniforms.fastCovarianceMode.value = 1;
        } else {
            const rendererSupportsHalf = !!(this.renderer &&
                (this.renderer.capabilities?.isWebGL2 ||
                 this.renderer.extensions?.has?.('OES_texture_half_float')));
            // Keep RGBA32F as the safe default when capability information is
            // unavailable (for example, during headless builds).
            const useHalfCovariance = !!this.halfPrecisionCovariancesOnGPU && rendererSupportsHalf;
            const covarianceUpload = useHalfCovariance ? new Uint16Array(capacity * 8) : new Float32Array(capacity * 8);
            if (!(fastData.covariances instanceof Float32Array) ||
                fastData.covariances.length !== fastData.getSplatCount() * 6) {
                throw new Error('COMPACTGS layout 2 covariance data is missing or has an invalid length.');
            }
            for (let point = 0; point < count; point++) {
                for (let component = 0; component < 6; component++) {
                    const value = fastData.covariances[point * 6 + component];
                    covarianceUpload[point * 8 + component] = useHalfCovariance ?
                        THREE.DataUtils.toHalfFloat(value) : value;
                }
            }
            const covarianceTexture = makeTexture(covarianceUpload, width * 2, height,
                                                   THREE.RGBAFormat,
                                                   useHalfCovariance ? THREE.HalfFloatType : THREE.FloatType,
                                                   useHalfCovariance ? 'RGBA16F' : 'RGBA32F');
            this.material.uniforms.fastCovarianceTexture.value = covarianceTexture;
            this.material.uniforms.fastCovarianceTextureSize.value.set(width, height);
            // Layout 2 uploads direct covariance values as two RGBA texels per
            // splat. Use the direct path so the shader does not reconstruct
            // covariance from scale/quaternion values for every vertex.
            this.material.uniforms.fastCovarianceMode.value = 2;
        }
        this.material.uniforms.fastPointMapSize.value.set(width, height);
        this.material.uniforms.fastShapeMin.value.set(descriptor.shapeMin);
        this.material.uniforms.fastShapeStep.value.set(descriptor.shapeStep);
        this.material.uniforms.fastImportanceMin.value = descriptor.importanceMin;
        this.material.uniforms.fastImportanceStep.value = descriptor.importanceStep;
        this.material.uniforms.fastShMin.value.set(descriptor.shMin);
        this.material.uniforms.fastShStep.value.set(descriptor.shStep);
        this.material.uniforms.fastMinimumAlpha.value = Number(sceneOptions.splatAlphaRemovalThreshold ?? fastData.minimumAlpha ?? 0);

        let shTexture;
        let shPackedTexture = null;
        let shTextureSize = new THREE.Vector2(width * 4, height * 4);
        if (this.useCompressedTexture) {
            const payload = fastData.compressedTextureData;
            const textureDescriptor = validateCompressedTexturePayload(payload);
            if (textureDescriptor.width !== width * 4 || textureDescriptor.height !== height * 4 || textureDescriptor.layers !== 1) {
                throw new Error('COMPACTGS_V1 compressed SH atlas dimensions do not match the point map.');
            }
            shTexture = new THREE.CompressedArrayTexture(
                [{ data: textureDescriptor.raw, width: textureDescriptor.width,
                   height: textureDescriptor.height, depth: 1 }],
                textureDescriptor.width, textureDescriptor.height, 1, textureDescriptor.threeFormat
            );
            shTexture.minFilter = THREE.NearestFilter;
            shTexture.magFilter = THREE.NearestFilter;
            shTexture.generateMipmaps = false;
            shTexture.needsUpdate = true;
            this.material.uniforms.fastShCompressedTexture.value = shTexture;
        } else {
            if (!(planes.shAtlas instanceof Uint8Array) || planes.shAtlas.length !== width * height * 48) {
                throw new Error('COMPACTGS_V1 CPU SH atlas is missing or has an invalid length.');
            }
            if (this.minSphericalHarmonicsDegree === 2) {
                // Repack the first eight RGB coefficient vectors into six
                // RGBA texels per point. This changes only the GPU upload
                // layout; the encoded SH plane remains untouched.
                const packed = new Uint8Array(capacity * 6 * 4);
                const sourceWidth = width * 4;
                for (let point = 0; point < capacity; point++) {
                    const x = point % width;
                    const y = Math.floor(point / width);
                    let dst = 0;
                    for (let coefficient = 0; coefficient < 8; coefficient++) {
                        const tileX = coefficient & 3;
                        const tileY = coefficient >> 2;
                        const srcBase = ((tileY * height + y) * sourceWidth + tileX * width + x) * 3;
                        for (let channel = 0; channel < 3; channel++) {
                            const texel = Math.floor(dst / 4);
                            const component = dst & 3;
                            const dstBase = (y * width * 6 + texel * width + x) * 4 + component;
                            packed[dstBase] = planes.shAtlas[srcBase + channel];
                            dst++;
                        }
                    }
                }
                shPackedTexture = makeTexture(packed, width * 6, height,
                                              THREE.RGBAFormat, THREE.UnsignedByteType, 'RGBA8');
                this.material.uniforms.fastShPackedTexture.value = shPackedTexture;
                this.material.uniforms.fastUsePackedSH.value = 1;
                shTexture = shPackedTexture;
                shTextureSize.set(width * 6, height);
            } else {
                const shUpload = expandRgb(planes.shAtlas, Uint8Array, 0);
                shTexture = makeTexture(shUpload, width * 4, height * 4,
                                        THREE.RGBAFormat, THREE.UnsignedByteType, 'RGBA8');
                this.material.uniforms.fastShTexture.value = shTexture;
            }
        }
        this.material.uniforms.fadeInComplete.value = 1;
        this.material.uniforms.sceneCenter.value.copy(fastData.sceneCenter);
        this.material.uniformsNeedUpdate = true;
        this.splatDataTextures = {
            fastPositionLow: { data: positionLowUpload, texture: positionLow, size: new THREE.Vector2(width, height) },
            fastPositionHigh: { data: positionHighUpload, texture: positionHigh, size: new THREE.Vector2(width, height) },
            fastPosition: { data: decodedPositions, texture: decodedPositionTexture, size: new THREE.Vector2(width, height) },
            fastPositionColor: { data: positionColorUpload, texture: positionColorTexture, size: new THREE.Vector2(width, height) },
            fastColor: { data: colorUpload, texture: color, size: new THREE.Vector2(width, height) },
            fastOrder: { data: orderData, texture: orderTexture, size: new THREE.Vector2(orderWidth, orderHeight) },
            fastShape: { data: planes.shape, texture: shape,
                         size: new THREE.Vector2(width * 4, height * (descriptor.layoutVersion >= 3 ? 4 : 3)) },
            fastGroupMin: { data: descriptor.positionGroupMin, texture: groupMin, size: new THREE.Vector2(groupCount, 1) },
            fastGroupStep: { data: descriptor.positionGroupStep, texture: groupStep, size: new THREE.Vector2(groupCount, 1) },
            fastSH: { data: planes.shAtlas || fastData.compressedTextureData.raw, texture: shTexture,
                      size: shTextureSize },
            // Keep the setup-worker diagnostics API stable without allocating legacy textures.
            covariances: { size: new THREE.Vector2(width * 4, height * (descriptor.layoutVersion >= 3 ? 4 : 3)) },
            centerColors: { size: new THREE.Vector2(width, height) }
        };
        if (descriptor.layoutVersion === 2) {
            this.splatDataTextures.covariances = {
                data: fastData.covariances,
                texture: this.material.uniforms.fastCovarianceTexture.value,
                size: new THREE.Vector2(width * 2, height)
            };
        }
        if (descriptor.layoutVersion >= 3) {
            this.splatDataTextures.fastCovarianceGroupMin = { texture: this.material.uniforms.fastCovarianceGroupMinTexture.value };
            this.splatDataTextures.fastCovarianceGroupStep = { texture: this.material.uniforms.fastCovarianceGroupStepTexture.value };
        }
        this.boundingBox = this.computeCompactGSBoundingBox(fastData);
        this.calculatedSceneCenter.copy(fastData.sceneCenter);
        this.maxSplatDistanceFromSceneCenter = this.boundingBox.max.clone().sub(this.calculatedSceneCenter).length();
        this.visibleRegionRadius = this.visibleRegionBufferRadius = this.maxSplatDistanceFromSceneCenter;
        this.lastBuildScenes = [...this.scenes];
        this.lastBuildSplatCount = count;
        this.lastBuildMaxSplatCount = count;
        this.lastBuildSceneCount = 1;
        this.globalSplatIndexToLocalSplatIndexMap = [];
        this.globalSplatIndexToSceneIndexMap = [];
        this.visible = count > 0;

        // Initialize the GPU distance buffer from the same centers used by the
        // render shader. This is done once at load time; camera changes only
        // update the transform-feedback matrix.
        if (count > 0) {
            let distanceCenters = decodedPositions.subarray(0, count * 4);
            if (this.integerBasedDistancesComputation) {
                distanceCenters = new Int32Array(count * 4);
                for (let i = 0; i < count; i++) {
                    const base = i * 4;
                    distanceCenters[base] = Math.round(decodedPositions[base] * 1000.0);
                    distanceCenters[base + 1] = Math.round(decodedPositions[base + 1] * 1000.0);
                    distanceCenters[base + 2] = Math.round(decodedPositions[base + 2] * 1000.0);
                    distanceCenters[base + 3] = 1000;
                }
            }
            this.compactGaussianDistanceCenters = distanceCenters;
            if (this.renderer && this.enableDistancesComputationOnGPU) {
                this.setupDistancesComputationTransformFeedback();
                this.refreshGPUBuffersForDistancesComputation(distanceCenters, EMPTY_UINT32_ARRAY, false);
            }
        } else {
            this.compactGaussianDistanceCenters = null;
        }

        let splatTreePromise = null;
        if (finalBuild && count > 0) {
            const splatTreeStartTime = performance.now();
            splatTreePromise = this.buildSplatTree(
                [sceneOptions.splatAlphaRemovalThreshold || 1],
                onSplatTreeIndexesUpload, onSplatTreeConstruction
            ).then(() => {
                recordProcessingDuration(processingProfile, 'splatTreeAsyncMs', splatTreeStartTime);
                if (this.onSplatTreeReadyCallback) this.onSplatTreeReadyCallback(this.splatTree);
                this.onSplatTreeReadyCallback = null;
            });
        }

        const compactGaussianQuantizedPositions = {
            positionLow: planes.positionLow.subarray(0, count * 3),
            positionHigh: planes.positionHigh.subarray(0, count * 3),
            groupMin: descriptor.positionGroupMin,
            groupStep: descriptor.positionGroupStep,
            groupSize: descriptor.positionGroupSize
        };
        this.compactGaussianQuantizedPositions = compactGaussianQuantizedPositions;

        const result = {
            from: 0, to: count - 1, count,
            centers: EMPTY_UINT32_ARRAY,
            sceneIndexes: EMPTY_UINT32_ARRAY,
            compactGaussianQuantizedPositions,
            splatTreePromise
        };
        if (processingProfile) {
            processingProfile.meshBuildTotalMs = performance.now() - buildStartTime;
            processingProfile.meshRebuild = true;
            processingProfile.meshSplatCount = count;
            processingProfile.meshMaxSplatCount = count;
            processingProfile.splatTreeAsyncStarted = !!splatTreePromise;
            result.processingProfile = processingProfile;
        }
        return result;
    }

    computeCompactGSBoundingBox(fastData) {
        const mins = fastData.descriptor.positionGroupMin;
        const steps = fastData.descriptor.positionGroupStep;
        const min = new THREE.Vector3(Infinity, Infinity, Infinity);
        const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
        for (let group = 0; group < mins.length / 3; group++) {
            min.x = Math.min(min.x, mins[group * 3]);
            min.y = Math.min(min.y, mins[group * 3 + 1]);
            min.z = Math.min(min.z, mins[group * 3 + 2]);
            max.x = Math.max(max.x, mins[group * 3] + steps[group * 3] * 32767.0);
            max.y = Math.max(max.y, mins[group * 3 + 1] + steps[group * 3 + 1] * 32767.0);
            max.z = Math.max(max.z, mins[group * 3 + 2] + steps[group * 3 + 2] * 32767.0);
        }
        return new THREE.Box3(min, max);
    }

    freeIntermediateSplatData() {

        if (this.compactGaussianMode) {
            // FAST textures are already direct views of the decoded planes.  Do
            // not drop their images before the renderer has completed upload.
            return;
        }

        const deleteTextureData = (texture) => {
            delete texture.source.data;
            delete texture.image;
            texture.onUpdate = null;
        };

        delete this.splatDataTextures.baseData.covariances;
        delete this.splatDataTextures.baseData.centers;
        delete this.splatDataTextures.baseData.colors;
        delete this.splatDataTextures.baseData.sphericalHarmonics;

        delete this.splatDataTextures.centerColors.data;
        delete this.splatDataTextures.covariances.data;
        if (this.splatDataTextures.sphericalHarmonics) {
            delete this.splatDataTextures.sphericalHarmonics.data;
        }
        if (this.splatDataTextures.sceneIndexes) {
            delete this.splatDataTextures.sceneIndexes.data;
        }
        if (this.splatDataTextures.compressedTexture) {
            delete this.splatDataTextures.compressedTexture.data;
        }
        if (this.splatDataTextures.compressedTextureUV) {
            delete this.splatDataTextures.compressedTextureUV.data;
        }

        this.splatDataTextures.centerColors.texture.needsUpdate = true;
        this.splatDataTextures.centerColors.texture.onUpdate = () => {
            deleteTextureData(this.splatDataTextures.centerColors.texture);
        };

        this.splatDataTextures.covariances.texture.needsUpdate = true;
        this.splatDataTextures.covariances.texture.onUpdate = () => {
            deleteTextureData(this.splatDataTextures.covariances.texture);
        };

        if (this.splatDataTextures.sphericalHarmonics) {
            if (this.splatDataTextures.sphericalHarmonics.texture) {
                this.splatDataTextures.sphericalHarmonics.texture.needsUpdate = true;
                this.splatDataTextures.sphericalHarmonics.texture.onUpdate = () => {
                    deleteTextureData(this.splatDataTextures.sphericalHarmonics.texture);
                };
            } else {
                this.splatDataTextures.sphericalHarmonics.textures.forEach((texture) => {
                    texture.needsUpdate = true;
                    texture.onUpdate = () => {
                        deleteTextureData(texture);
                    };
                });
            }
        }
        if (this.splatDataTextures.sceneIndexes) {
            this.splatDataTextures.sceneIndexes.texture.needsUpdate = true;
            this.splatDataTextures.sceneIndexes.texture.onUpdate = () => {
                deleteTextureData(this.splatDataTextures.sceneIndexes.texture);
            };
        }
        if (this.splatDataTextures.compressedTexture) {
            const textureDescriptor = this.splatDataTextures.compressedTexture;
            const compressedTexture = textureDescriptor.texture;
            compressedTexture.needsUpdate = true;
            compressedTexture.onUpdate = () => {
                console.log(
                    `[GaussianSlimLoader Timing] Compressed texture upload: ` +
                    `${(performance.now() - textureDescriptor.uploadQueuedAt).toFixed(2)} ms, ` +
                    `bytes=${textureDescriptor.byteLength}`
                );
                deleteTextureData(compressedTexture);
            };
        }
        if (this.splatDataTextures.compressedTextureUV) {
            const compressedTextureUV = this.splatDataTextures.compressedTextureUV.texture;
            compressedTextureUV.needsUpdate = true;
            compressedTextureUV.onUpdate = () => {
                deleteTextureData(compressedTextureUV);
            };
        }
    }
    /**
     * Dispose all resources held by the splat mesh
     */
    dispose() {
        this.disposeMeshData();
        this.disposeTextures();
        this.disposeSplatTree();
        if (this.enableDistancesComputationOnGPU) {
            if (this.computeDistancesOnGPUSyncTimeout) {
                clearTimeout(this.computeDistancesOnGPUSyncTimeout);
                this.computeDistancesOnGPUSyncTimeout = null;
            }
            this.disposeDistancesComputationGPUResources();
        }
        this.scenes = [];
        this.distancesTransformFeedback = {
            'id': null,
            'vertexShader': null,
            'fragmentShader': null,
            'program': null,
            'centersBuffer': null,
            'sceneIndexesBuffer': null,
            'outDistancesBuffer': null,
            'centersLoc': -1,
            'modelViewProjLoc': -1,
            'sceneIndexesLoc': -1,
            'transformsLocs': []
        };
        this.renderer = null;

        this.globalSplatIndexToLocalSplatIndexMap = [];
        this.globalSplatIndexToSceneIndexMap = [];

        this.lastBuildSplatCount = 0;
        this.lastBuildScenes = [];
        this.lastBuildMaxSplatCount = 0;
        this.lastBuildSceneCount = 0;
        this.firstRenderTime = -1;
        this.finalBuild = false;

        this.webGLUtils = null;

        this.boundingBox = new THREE.Box3();
        this.calculatedSceneCenter = new THREE.Vector3();
        this.maxSplatDistanceFromSceneCenter = 0;
        this.visibleRegionBufferRadius = 0;
        this.visibleRegionRadius = 0;
        this.visibleRegionFadeStartRadius = 0;
        this.visibleRegionChanging = false;

        this.splatScale = 1.0;
        this.pointCloudModeEnabled = false;
        this.compactGaussianMode = false;
        this.compactGaussianQuantizedPositions = null;
        this.compactGaussianDistanceCenters = null;
        this.useCompressedTexture = false;
        this.useASTC = false;

        this.disposed = true;
        this.lastRenderer = null;
        this.visible = false;
    }

    /**
     * Dispose of only the Three.js mesh resources (geometry, material, and texture)
     */
    disposeMeshData() {
        if (this.geometry && this.geometry !== dummyGeometry) {
            this.geometry.dispose();
            this.geometry = null;
        }
        if (this.material) {
            this.material.dispose();
            this.material = null;
        }
    }

    disposeTextures() {
        for (let textureKey in this.splatDataTextures) {
            if (this.splatDataTextures.hasOwnProperty(textureKey)) {
                const textureContainer = this.splatDataTextures[textureKey];
                if (textureContainer?.texture) {
                    textureContainer.texture.dispose();
                    textureContainer.texture = null;
                }
            }
        }
        this.splatDataTextures = null;
    }

    disposeSplatTree() {
        if (this.splatTree) {
            this.splatTree.dispose();
            this.splatTree = null;
        }
        if (this.baseSplatTree) {
            this.baseSplatTree.dispose();
            this.baseSplatTree = null;
        }
    }

    getSplatTree() {
        return this.splatTree;
    }

    onSplatTreeReady(callback) {
        this.onSplatTreeReadyCallback = callback;
    }

    /**
     * Get copies of data that are necessary for splat distance computation: splat center positions and splat
     * scene indexes (necessary for applying dynamic scene transformations during distance computation)
     * @param {*} start The index at which to start copying data
     * @param {*} end  The index at which to stop copying data
     * @return {object}
     */
    getDataForDistancesComputation(start, end) {
        const centers = this.getCentersForDistancesComputation(start, end, true);
        const sceneIndexes = this.dynamicMode ? this.getSceneIndexes(start, end) : EMPTY_UINT32_ARRAY;
        return {
            centers,
            sceneIndexes
        };
    }

    getCentersForDistancesComputation(start, end, padFour = false) {
        const baseCenters = this.splatDataTextures?.baseData?.centers;
        if (!baseCenters) {
            return this.integerBasedDistancesComputation ?
                   this.getIntegerCenters(start, end, padFour) :
                   this.getFloatCenters(start, end, padFour);
        }

        const splatCount = end - start + 1;
        const componentCount = padFour ? 4 : 3;
        const sourceBase = start * 3;

        if (this.integerBasedDistancesComputation) {
            const integerCenters = new Int32Array(splatCount * componentCount);
            let srcOffset = sourceBase;
            let dstOffset = 0;
            for (let i = 0; i < splatCount; i++) {
                integerCenters[dstOffset++] = Math.round(baseCenters[srcOffset++] * 1000.0);
                integerCenters[dstOffset++] = Math.round(baseCenters[srcOffset++] * 1000.0);
                integerCenters[dstOffset++] = Math.round(baseCenters[srcOffset++] * 1000.0);
                if (padFour) integerCenters[dstOffset++] = 1000;
            }
            return integerCenters;
        }

        if (!padFour) return baseCenters.slice(sourceBase, sourceBase + splatCount * 3);

        const paddedFloatCenters = new Float32Array(splatCount * 4);
        let srcOffset = sourceBase;
        let dstOffset = 0;
        for (let i = 0; i < splatCount; i++) {
            paddedFloatCenters[dstOffset++] = baseCenters[srcOffset++];
            paddedFloatCenters[dstOffset++] = baseCenters[srcOffset++];
            paddedFloatCenters[dstOffset++] = baseCenters[srcOffset++];
            paddedFloatCenters[dstOffset++] = 1.0;
        }
        return paddedFloatCenters;
    }

    /**
     * Refresh data textures and GPU buffers with splat data from the splat buffers belonging to this mesh.
     * @param {boolean} sinceLastBuildOnly Specify whether or not to only update for splats that have been added since the last build.
     * @return {object}
     */
    refreshGPUDataFromSplatBuffers(sinceLastBuildOnly, processingProfile = null) {
        const refreshStartTime = performance.now();
        const splatCount = this.getSplatCount(true);
        this.refreshDataTexturesFromSplatBuffers(sinceLastBuildOnly, processingProfile);
        const updateStart = sinceLastBuildOnly ? this.lastBuildSplatCount : 0;
        const distanceDataStartTime = performance.now();
        const { centers, sceneIndexes } = this.getDataForDistancesComputation(updateStart, splatCount - 1);
        recordProcessingDuration(processingProfile, 'meshSortDataPrepMs', distanceDataStartTime);
        if (this.enableDistancesComputationOnGPU) {
            const gpuBufferRefreshStartTime = performance.now();
            this.refreshGPUBuffersForDistancesComputation(centers, sceneIndexes, sinceLastBuildOnly);
            recordProcessingDuration(processingProfile, 'meshGpuDistanceBufferUploadMs', gpuBufferRefreshStartTime);
        }
        recordProcessingDuration(processingProfile, 'meshRefreshGpuDataMs', refreshStartTime);
        return {
            'from': updateStart,
            'to': splatCount - 1,
            'count': splatCount - updateStart,
            'centers': centers,
            'sceneIndexes': sceneIndexes
        };
    }

    /**
     * Update the GPU buffers that are used for computing splat distances on the GPU.
     * @param {Array<number>} centers Splat center positions
     * @param {Array<number>} sceneIndexes Indexes of the scene to which each splat belongs
     * @param {boolean} sinceLastBuildOnly Specify whether or not to only update for splats that have been added since the last build.
     */
    refreshGPUBuffersForDistancesComputation(centers, sceneIndexes, sinceLastBuildOnly = false) {
        const offset = sinceLastBuildOnly ? this.lastBuildSplatCount : 0;
        this.updateGPUCentersBufferForDistancesComputation(sinceLastBuildOnly, centers, offset);
        this.updateGPUTransformIndexesBufferForDistancesComputation(sinceLastBuildOnly, sceneIndexes, offset);
    }

    /**
     * Refresh data textures with data from the splat buffers for this mesh.
     * @param {boolean} sinceLastBuildOnly Specify whether or not to only update for splats that have been added since the last build.
     */
    refreshDataTexturesFromSplatBuffers(sinceLastBuildOnly, processingProfile = null) {
        const refreshStartTime = performance.now();
        const splatCount = this.getSplatCount(true);
        const fromSplat = this.lastBuildSplatCount;
        const toSplat = splatCount - 1;
        const deferVisibleRegionUpdate = !!processingProfile?.deferVisibleRegion && !sinceLastBuildOnly;

        if (!sinceLastBuildOnly) {
            const setupStartTime = performance.now();
            this.setupDataTextures();
            recordProcessingDuration(processingProfile, 'meshSetupDataTexturesMs', setupStartTime);
            this.updateBaseDataFromSplatBuffers(0, splatCount-1, processingProfile);
        } else {

            this.updateBaseDataFromSplatBuffers(fromSplat, toSplat, processingProfile);
        }

        this.updateDataTexturesFromBaseData(fromSplat, toSplat, processingProfile);
        if (deferVisibleRegionUpdate) {
            const visibleRegionPrimeStartTime = performance.now();
            this.prepareVisibleRegionForImmediateRender();
            recordProcessingDuration(processingProfile, 'meshPrimeVisibleRegionMs', visibleRegionPrimeStartTime);
        } else {
            const visibleRegionStartTime = performance.now();
            this.updateVisibleRegion(sinceLastBuildOnly);
            recordProcessingDuration(processingProfile, 'meshUpdateVisibleRegionMs', visibleRegionStartTime);
        }
        recordProcessingDuration(processingProfile, 'meshRefreshDataTexturesTotalMs', refreshStartTime);
    }

    setupDataTextures() {
        const maxSplatCount = this.getMaxSplatCount();
        const splatCount = this.getSplatCount(true);

        this.disposeTextures();

        const computeDataTextureSize = (elementsPerTexel, elementsPerSplat) => {
            const texSize = new THREE.Vector2(4096, 1024);
            while (texSize.x * texSize.y * elementsPerTexel < maxSplatCount * elementsPerSplat) texSize.y *= 2;
            return texSize;
        };

        const getCovariancesElementsPertexelStored = (compressionLevel) => {
            return compressionLevel >= 1 ? COVARIANCES_ELEMENTS_PER_TEXEL_COMPRESSED_STORED : COVARIANCES_ELEMENTS_PER_TEXEL_STORED;
        };

        const getCovariancesInitialTextureSpecs = (compressionLevel) => {
            const elementsPerTexelStored = getCovariancesElementsPertexelStored(compressionLevel);
            const texSize = computeDataTextureSize(elementsPerTexelStored, 6);
            return {elementsPerTexelStored, texSize};
        };

        let covarianceCompressionLevel = this.getTargetCovarianceCompressionLevel();
        const scaleRotationCompressionLevel = 0;
        const shCompressionLevel = this.getTargetSphericalHarmonicsCompressionLevel();

        let covariances;
        let scales;
        let rotations;
        if (this.splatRenderMode === SplatRenderMode.ThreeD) {
            const initialCovTexSpecs = getCovariancesInitialTextureSpecs(covarianceCompressionLevel);
            if (initialCovTexSpecs.texSize.x * initialCovTexSpecs.texSize.y > MAX_TEXTURE_TEXELS && covarianceCompressionLevel === 0) {
                covarianceCompressionLevel = 1;
            }
            covariances = new Float32Array(maxSplatCount * COVARIANCES_ELEMENTS_PER_SPLAT);
        } else {
            scales = new Float32Array(maxSplatCount * 3);
            rotations = new Float32Array(maxSplatCount * 4);
        }

        const centers = new Float32Array(maxSplatCount * 3);
        const colors = new Uint8Array(maxSplatCount * 4);

        let SphericalHarmonicsArrayType = Float32Array;
        if (shCompressionLevel === 1) SphericalHarmonicsArrayType = Uint16Array;
        else if (shCompressionLevel === 2) SphericalHarmonicsArrayType = Uint8Array;
        const shComponentCount = getSphericalHarmonicsComponentCountForDegree(this.minSphericalHarmonicsDegree);
        const shData = this.minSphericalHarmonicsDegree ? new SphericalHarmonicsArrayType(maxSplatCount * shComponentCount) : undefined;

        // set up centers/colors data texture
        const centersColsTexSize = computeDataTextureSize(CENTER_COLORS_ELEMENTS_PER_TEXEL, 4);
        const paddedCentersCols = new Uint32Array(centersColsTexSize.x * centersColsTexSize.y * CENTER_COLORS_ELEMENTS_PER_TEXEL);
        SplatMesh.updateCenterColorsPaddedData(0, splatCount - 1, centers, colors, paddedCentersCols);

        const centersColsTex = new THREE.DataTexture(paddedCentersCols, centersColsTexSize.x, centersColsTexSize.y,
                                                     THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
        centersColsTex.internalFormat = 'RGBA32UI';
        centersColsTex.needsUpdate = true;
        this.material.uniforms.centersColorsTexture.value = centersColsTex;
        this.material.uniforms.centersColorsTextureSize.value.copy(centersColsTexSize);
        this.material.uniformsNeedUpdate = true;

        this.splatDataTextures = {
            'baseData': {
                'covariances': covariances,
                'scales': scales,
                'rotations': rotations,
                'centers': centers,
                'colors': colors,
                'sphericalHarmonics': shData
            },
            'centerColors': {
                'data': paddedCentersCols,
                'texture': centersColsTex,
                'size': centersColsTexSize
            }
        };

        if (this.splatRenderMode === SplatRenderMode.ThreeD) {
            // set up covariances data texture

            const covTexSpecs = getCovariancesInitialTextureSpecs(covarianceCompressionLevel);
            const covariancesElementsPerTexelStored = covTexSpecs.elementsPerTexelStored;
            const covTexSize = covTexSpecs.texSize;

            let CovariancesDataType = covarianceCompressionLevel >= 1 ? Uint32Array : Float32Array;
            const covariancesElementsPerTexelAllocated = covarianceCompressionLevel >= 1 ?
                                                         COVARIANCES_ELEMENTS_PER_TEXEL_COMPRESSED_ALLOCATED :
                                                         COVARIANCES_ELEMENTS_PER_TEXEL_ALLOCATED;
            const covariancesTextureData = new CovariancesDataType(covTexSize.x * covTexSize.y * covariancesElementsPerTexelAllocated);

            if (covarianceCompressionLevel === 0) {
                covariancesTextureData.set(covariances);
            } else {
                SplatMesh.updatePaddedCompressedCovariancesTextureData(covariances, covariancesTextureData, 0, 0, covariances.length);
            }

            let covTex;
            if (covarianceCompressionLevel >= 1) {
                covTex = new THREE.DataTexture(covariancesTextureData, covTexSize.x, covTexSize.y,
                                               THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
                covTex.internalFormat = 'RGBA32UI';
                this.material.uniforms.covariancesTextureHalfFloat.value = covTex;
            } else {
                covTex = new THREE.DataTexture(covariancesTextureData, covTexSize.x, covTexSize.y, THREE.RGBAFormat, THREE.FloatType);
                this.material.uniforms.covariancesTexture.value = covTex;

                // For some reason a usampler2D needs to have a valid texture attached or WebGL complains
                const dummyTex = new THREE.DataTexture(new Uint32Array(32), 2, 2, THREE.RGBAIntegerFormat, THREE.UnsignedIntType);
                dummyTex.internalFormat = 'RGBA32UI';
                this.material.uniforms.covariancesTextureHalfFloat.value = dummyTex;
                dummyTex.needsUpdate = true;
            }
            covTex.needsUpdate = true;

            this.material.uniforms.covariancesAreHalfFloat.value = (covarianceCompressionLevel >= 1) ? 1 : 0;
            this.material.uniforms.covariancesTextureSize.value.copy(covTexSize);

            this.splatDataTextures['covariances'] = {
                'data': covariancesTextureData,
                'texture': covTex,
                'size': covTexSize,
                'compressionLevel': covarianceCompressionLevel,
                'elementsPerTexelStored': covariancesElementsPerTexelStored,
                'elementsPerTexelAllocated': covariancesElementsPerTexelAllocated
            };
        } else {
            // set up scale & rotations data texture
            const elementsPerSplat = 6;
            const scaleRotationsTexSize = computeDataTextureSize(SCALES_ROTATIONS_ELEMENTS_PER_TEXEL, elementsPerSplat);
            let ScaleRotationsDataType = scaleRotationCompressionLevel >= 1 ? Uint16Array : Float32Array;
            let scaleRotationsTextureType = scaleRotationCompressionLevel >= 1 ? THREE.HalfFloatType : THREE.FloatType;
            const paddedScaleRotations = new ScaleRotationsDataType(scaleRotationsTexSize.x * scaleRotationsTexSize.y *
                                                                    SCALES_ROTATIONS_ELEMENTS_PER_TEXEL);

            SplatMesh.updateScaleRotationsPaddedData(0, splatCount - 1, scales, rotations, paddedScaleRotations);

            const scaleRotationsTex = new THREE.DataTexture(paddedScaleRotations, scaleRotationsTexSize.x, scaleRotationsTexSize.y,
                                                            THREE.RGBAFormat, scaleRotationsTextureType);
            scaleRotationsTex.needsUpdate = true;
            this.material.uniforms.scaleRotationsTexture.value = scaleRotationsTex;
            this.material.uniforms.scaleRotationsTextureSize.value.copy(scaleRotationsTexSize);

            this.splatDataTextures['scaleRotations'] = {
                'data': paddedScaleRotations,
                'texture': scaleRotationsTex,
                'size': scaleRotationsTexSize,
                'compressionLevel': scaleRotationCompressionLevel
            };
        }
        if (this.useCompressedTexture) {
            const splatBuffer = this.scenes[0].splatBuffer;
            let payload = splatBuffer.compressedTextureData || splatBuffer.astcData;
            if (payload && !payload.format && splatBuffer.astcData === payload) {
                payload = { ...payload, format: 'astc' };
            }
            if (!payload) throw new Error('SplatMesh is missing its compressed texture payload.');
            {
                const descriptor = validateCompressedTexturePayload(payload);
                const { format, raw, width, height, layers, singleWidth, singleHeight, threeFormat } = descriptor;
                const { shnMin, shnMax } = payload;
                const uploadQueuedAt = performance.now();
                console.log(
                    `[Compressed Texture Upload] Format: ${format}, Width: ${width}, Height: ${height}, ` +
                    `Layers: ${layers}, RawSize: ${raw.byteLength}`
                );

                const compressedTexture = new THREE.CompressedArrayTexture(
                    [{ data: raw, width, height, depth: layers }],
                    width,
                    height,
                    layers,
                    threeFormat
                );
                compressedTexture.minFilter = THREE.LinearFilter;
                compressedTexture.magFilter = THREE.LinearFilter;

                compressedTexture.wrapS = THREE.ClampToEdgeWrapping;
                compressedTexture.wrapT = THREE.ClampToEdgeWrapping;
                compressedTexture.needsUpdate = true;

                this.material.uniforms.astcTextures.value = compressedTexture;
                this.material.uniforms.astcSingleWidth.value = singleWidth;
                this.material.uniforms.astcSingleHeight.value = singleHeight;
                this.material.uniforms.astcTextureWidth.value = width;
                this.splatDataTextures['compressedTexture'] = {
                    'data': raw,
                    'texture': compressedTexture,
                    'size': new THREE.Vector3(width, height, layers),
                    'format': format,
                    'uploadQueuedAt': uploadQueuedAt,
                    'byteLength': raw.byteLength
                };

                // Set the dequantization parameters (minimum and range).
                const range = new THREE.Vector3(shnMax - shnMin, shnMax - shnMin, shnMax - shnMin);
                const min = new THREE.Vector3(shnMin, shnMin, shnMin);
                this.material.uniforms.astcQuantRange.value.copy(range);
                this.material.uniforms.astcQuantMin.value.copy(min);

                // 4. Create the UV lookup texture (RG32UI).
                const shElementsPerTexel = 4; // Four RGBA channels.

                // This must be 4 because each splat occupies one complete RGBA texel.
                // Even though U and V use only R and G, a stride of 4 prevents allocating a half-sized texture.
                const paddedSHComponentCount = 4;

                const computeDataTextureSize = (elementsPerTexel, elementsPerSplat) => {
                    const texSize = new THREE.Vector2(4096, 1024);
                    while (texSize.x * texSize.y * elementsPerTexel < maxSplatCount * elementsPerSplat) texSize.y *= 2;
                    return texSize;
                };
                let uvTexSize = computeDataTextureSize(shElementsPerTexel, paddedSHComponentCount);

                const paddedUVArray = new Uint32Array(uvTexSize.x * uvTexSize.y * 4);

                const astcUVTexture = new THREE.DataTexture(
                    paddedUVArray, uvTexSize.x, uvTexSize.y, THREE.RGBAIntegerFormat, THREE.UnsignedIntType
                );
                astcUVTexture.internalFormat = 'RGBA32UI';

                astcUVTexture.minFilter = THREE.NearestFilter;
                astcUVTexture.magFilter = THREE.NearestFilter;

                astcUVTexture.needsUpdate = true;

                this.material.uniforms.astcUVTexture.value = astcUVTexture;
                this.material.uniforms.astcUVTextureSize.value.copy(uvTexSize);

                this.splatDataTextures['compressedTextureUV'] = {
                    'data': paddedUVArray,
                    'texture': astcUVTexture,
                    'size': uvTexSize,
                    'elementsPerTexel': 4
                };

                this.splatDataTextures['sphericalHarmonics'] = null;
            }
        } else {
            if (shData) {
                const shTextureType = shCompressionLevel === 2 ? THREE.UnsignedByteType : THREE.HalfFloatType;

                let paddedSHComponentCount = shComponentCount;
                if (paddedSHComponentCount % 2 !== 0) paddedSHComponentCount++;
                const shElementsPerTexel = 4;
                const texelFormat = shElementsPerTexel === 4 ? THREE.RGBAFormat : THREE.RGFormat;
                let shTexSize = computeDataTextureSize(shElementsPerTexel, paddedSHComponentCount);

                // Use one texture for all spherical harmonics data
                if (shTexSize.x * shTexSize.y <= MAX_TEXTURE_TEXELS) {
                    const paddedSHArraySize = shTexSize.x * shTexSize.y * shElementsPerTexel;
                    const paddedSHArray = new SphericalHarmonicsArrayType(paddedSHArraySize);
                    for (let c = 0; c < splatCount; c++) {
                        const srcBase = shComponentCount * c;
                        const destBase = paddedSHComponentCount * c;
                        for (let i = 0; i < shComponentCount; i++) {
                            paddedSHArray[destBase + i] = shData[srcBase + i];
                        }
                    }

                    const shTexture = new THREE.DataTexture(paddedSHArray, shTexSize.x, shTexSize.y, texelFormat, shTextureType);
                    shTexture.needsUpdate = true;
                    this.material.uniforms.sphericalHarmonicsTexture.value = shTexture;
                    this.splatDataTextures['sphericalHarmonics'] = {
                        'componentCount': shComponentCount,
                        'paddedComponentCount': paddedSHComponentCount,
                        'data': paddedSHArray,
                        'textureCount': 1,
                        'texture': shTexture,
                        'size': shTexSize,
                        'compressionLevel': shCompressionLevel,
                        'elementsPerTexel': shElementsPerTexel
                    };
                // Use three textures for spherical harmonics data, one per color channel
                } else {
                    const shComponentCountPerChannel = shComponentCount / 3;
                    paddedSHComponentCount = shComponentCountPerChannel;
                    if (paddedSHComponentCount % 2 !== 0) paddedSHComponentCount++;
                    shTexSize = computeDataTextureSize(shElementsPerTexel, paddedSHComponentCount);

                    const paddedSHArraySize = shTexSize.x * shTexSize.y * shElementsPerTexel;
                    const textureUniforms = [this.material.uniforms.sphericalHarmonicsTextureR,
                                             this.material.uniforms.sphericalHarmonicsTextureG,
                                             this.material.uniforms.sphericalHarmonicsTextureB];
                    const paddedSHArrays = [];
                    const shTextures = [];
                    for (let t = 0; t < 3; t++) {
                        const paddedSHArray = new SphericalHarmonicsArrayType(paddedSHArraySize);
                        paddedSHArrays.push(paddedSHArray);
                        for (let c = 0; c < splatCount; c++) {
                            const srcBase = shComponentCount * c;
                            const destBase = paddedSHComponentCount * c;
                            if (shComponentCountPerChannel >= 3) {
                                for (let i = 0; i < 3; i++) paddedSHArray[destBase + i] = shData[srcBase + t * 3 + i];
                                if (shComponentCountPerChannel >= 8) {
                                    for (let i = 0; i < 5; i++) paddedSHArray[destBase + 3 + i] = shData[srcBase + 9 + t * 5 + i];
                                }
                            }
                        }

                        const shTexture = new THREE.DataTexture(paddedSHArray, shTexSize.x, shTexSize.y, texelFormat, shTextureType);
                        shTextures.push(shTexture);
                        shTexture.needsUpdate = true;
                        textureUniforms[t].value = shTexture;
                    }

                    this.material.uniforms.sphericalHarmonicsMultiTextureMode.value = 1;
                    this.splatDataTextures['sphericalHarmonics'] = {
                        'componentCount': shComponentCount,
                        'componentCountPerChannel': shComponentCountPerChannel,
                        'paddedComponentCount': paddedSHComponentCount,
                        'data': paddedSHArrays,
                        'textureCount': 3,
                        'textures': shTextures,
                        'size': shTexSize,
                        'compressionLevel': shCompressionLevel,
                        'elementsPerTexel': shElementsPerTexel
                    };
                }

                this.material.uniforms.sphericalHarmonicsTextureSize.value.copy(shTexSize);
                this.material.uniforms.sphericalHarmonics8BitMode.value = shCompressionLevel === 2 ? 1 : 0;
                for (let s = 0; s < this.scenes.length; s++) {
                    const splatBuffer = this.scenes[s].splatBuffer;
                    this.material.uniforms.sphericalHarmonics8BitCompressionRangeMin.value[s] =
                        splatBuffer.minSphericalHarmonicsCoeff;
                    this.material.uniforms.sphericalHarmonics8BitCompressionRangeMax.value[s] =
                        splatBuffer.maxSphericalHarmonicsCoeff;
                }
                this.material.uniformsNeedUpdate = true;
            }
        }


        const sceneIndexesTexSize = computeDataTextureSize(SCENE_INDEXES_ELEMENTS_PER_TEXEL, 4);
        const paddedTransformIndexes = new Uint32Array(sceneIndexesTexSize.x *
                                                       sceneIndexesTexSize.y * SCENE_INDEXES_ELEMENTS_PER_TEXEL);
        for (let c = 0; c < splatCount; c++) paddedTransformIndexes[c] = this.globalSplatIndexToSceneIndexMap[c];
        const sceneIndexesTexture = new THREE.DataTexture(paddedTransformIndexes, sceneIndexesTexSize.x, sceneIndexesTexSize.y,
                                                          THREE.RedIntegerFormat, THREE.UnsignedIntType);
        sceneIndexesTexture.internalFormat = 'R32UI';
        sceneIndexesTexture.needsUpdate = true;
        this.material.uniforms.sceneIndexesTexture.value = sceneIndexesTexture;
        this.material.uniforms.sceneIndexesTextureSize.value.copy(sceneIndexesTexSize);
        this.material.uniformsNeedUpdate = true;
        this.splatDataTextures['sceneIndexes'] = {
            'data': paddedTransformIndexes,
            'texture': sceneIndexesTexture,
            'size': sceneIndexesTexSize
        };
        this.material.uniforms.sceneCount.value = this.scenes.length;
    }

    updateBaseDataFromSplatBuffers(fromSplat, toSplat, processingProfile = null) {
        console.log(`[updateBaseDataFromSplatBuffers] srcFrome:${fromSplat}, srcEnd:${toSplat}`);
        const covarancesTextureDesc = this.splatDataTextures['covariances'];
        const covarianceCompressionLevel = covarancesTextureDesc ? covarancesTextureDesc.compressionLevel : undefined;
        const scaleRotationsTextureDesc = this.splatDataTextures['scaleRotations'];
        const scaleRotationCompressionLevel = scaleRotationsTextureDesc ? scaleRotationsTextureDesc.compressionLevel : undefined;
        const shITextureDesc = this.splatDataTextures['sphericalHarmonics'];
        const shCompressionLevel = shITextureDesc ? shITextureDesc.compressionLevel : 0;

        // Select the SH data destination.
        // In ASTC mode, fillSplatDataArrays must not process SH, so use null.
        // Otherwise, preserve the existing path and target the sphericalHarmonics array.
        let shDataTarget = this.useCompressedTexture ? null : this.splatDataTextures.baseData.sphericalHarmonics;

        // In ASTC mode, ensure baseData has storage for UV coordinates.
        if (this.useCompressedTexture) {
            if (!this.splatDataTextures.baseData.compressedTextureUVs) {
                const maxSplatCount = this.getMaxSplatCount();
                this.splatDataTextures.baseData.compressedTextureUVs = new Uint32Array(maxSplatCount * 2);
            }
        }

        if (this.useCompressedTexture && this.fillDirectCompressedTextureDataArrays(
            fromSplat, toSplat, covarianceCompressionLevel, scaleRotationCompressionLevel
        )) {
            return;
        }

        // Invoke the common fill function.
        // The sixth argument is now shDataTarget, which may be null.
        const fillBaseDataStartTime = performance.now();
        this.fillSplatDataArrays(
            this.splatDataTextures.baseData.covariances,
            this.splatDataTextures.baseData.scales,
            this.splatDataTextures.baseData.rotations,
            this.splatDataTextures.baseData.centers,
            this.splatDataTextures.baseData.colors,
            shDataTarget, // Pass null in ASTC mode.
            undefined,
            covarianceCompressionLevel,
            scaleRotationCompressionLevel,
            shCompressionLevel,
            fromSplat,
            toSplat,
            fromSplat
        );
        recordProcessingDuration(processingProfile, 'meshFillBaseArraysMs', fillBaseDataStartTime);

        // Extract UV data explicitly in ASTC mode.
        if (this.useCompressedTexture) {
            const fillAstcUvStartTime = performance.now();
            this.fillSplatCompressedTextureUVs(
                this.splatDataTextures.baseData.compressedTextureUVs,
                fromSplat,
                toSplat,
                fromSplat
            );
            recordProcessingDuration(processingProfile, 'meshFillAstcUvsMs', fillAstcUvStartTime);
        }
    }

    updateDataTexturesFromBaseData(fromSplat, toSplat, processingProfile = null) {
        const updateTexturesStartTime = performance.now();
        const covarancesTextureDesc = this.splatDataTextures['covariances'];
        const covarianceCompressionLevel = covarancesTextureDesc ? covarancesTextureDesc.compressionLevel : undefined;
        const scaleRotationsTextureDesc = this.splatDataTextures['scaleRotations'];
        const scaleRotationCompressionLevel = scaleRotationsTextureDesc ? scaleRotationsTextureDesc.compressionLevel : undefined;
        const shTextureDesc = this.splatDataTextures['sphericalHarmonics'];
        const shCompressionLevel = shTextureDesc ? shTextureDesc.compressionLevel : 0;

        // Update center & color data texture
        const centerColorsTextureStartTime = performance.now();
        const centerColorsTextureDescriptor = this.splatDataTextures['centerColors'];
        const paddedCenterColors = centerColorsTextureDescriptor.data;
        const centerColorsTexture = centerColorsTextureDescriptor.texture;
        SplatMesh.updateCenterColorsPaddedData(fromSplat, toSplat, this.splatDataTextures.baseData.centers,
                                               this.splatDataTextures.baseData.colors, paddedCenterColors);
        const centerColorsTextureProps = this.renderer ? this.renderer.properties.get(centerColorsTexture) : null;
        if (!centerColorsTextureProps || !centerColorsTextureProps.__webglTexture) {
            centerColorsTexture.needsUpdate = true;
        } else {
            this.updateDataTexture(paddedCenterColors, centerColorsTextureDescriptor.texture, centerColorsTextureDescriptor.size,
                                   centerColorsTextureProps, CENTER_COLORS_ELEMENTS_PER_TEXEL, CENTER_COLORS_ELEMENTS_PER_SPLAT, 4,
                                   fromSplat, toSplat);
        }
        recordProcessingDuration(processingProfile, 'meshUpdateCenterColorsTextureMs', centerColorsTextureStartTime);

        // update covariance data texture
        if (covarancesTextureDesc) {
            const covariancesTextureStartTime = performance.now();
            const covariancesTexture = covarancesTextureDesc.texture;
            const covarancesStartElement = fromSplat * COVARIANCES_ELEMENTS_PER_SPLAT;
            const covariancesEndElement = toSplat * COVARIANCES_ELEMENTS_PER_SPLAT;

            if (covarianceCompressionLevel === 0) {
                for (let i = covarancesStartElement; i <= covariancesEndElement; i++) {
                    const covariance = this.splatDataTextures.baseData.covariances[i];
                    covarancesTextureDesc.data[i] = covariance;
                }
            } else {
                SplatMesh.updatePaddedCompressedCovariancesTextureData(this.splatDataTextures.baseData.covariances,
                                                                       covarancesTextureDesc.data,
                                                                       fromSplat * covarancesTextureDesc.elementsPerTexelAllocated,
                                                                       covarancesStartElement, covariancesEndElement);
            }

            const covariancesTextureProps = this.renderer ? this.renderer.properties.get(covariancesTexture) : null;
            if (!covariancesTextureProps || !covariancesTextureProps.__webglTexture) {
                covariancesTexture.needsUpdate = true;
            } else {
                if (covarianceCompressionLevel === 0) {
                    this.updateDataTexture(covarancesTextureDesc.data, covarancesTextureDesc.texture, covarancesTextureDesc.size,
                                           covariancesTextureProps, covarancesTextureDesc.elementsPerTexelStored,
                                           COVARIANCES_ELEMENTS_PER_SPLAT, 4, fromSplat, toSplat);
                } else {
                    this.updateDataTexture(covarancesTextureDesc.data, covarancesTextureDesc.texture, covarancesTextureDesc.size,
                                           covariancesTextureProps, covarancesTextureDesc.elementsPerTexelAllocated,
                                           covarancesTextureDesc.elementsPerTexelAllocated, 2, fromSplat, toSplat);
                }
            }
            recordProcessingDuration(processingProfile, 'meshUpdateCovariancesTextureMs', covariancesTextureStartTime);
        }

        // update scale and rotation data texture
        if (scaleRotationsTextureDesc) {
            const scaleRotationsTextureStartTime = performance.now();
            const paddedScaleRotations = scaleRotationsTextureDesc.data;
            const scaleRotationsTexture = scaleRotationsTextureDesc.texture;
            const elementsPerSplat = 6;
            const bytesPerElement = scaleRotationCompressionLevel === 0 ? 4 : 2;

            SplatMesh.updateScaleRotationsPaddedData(fromSplat, toSplat, this.splatDataTextures.baseData.scales,
                                                     this.splatDataTextures.baseData.rotations, paddedScaleRotations);
            const scaleRotationsTextureProps = this.renderer ? this.renderer.properties.get(scaleRotationsTexture) : null;
            if (!scaleRotationsTextureProps || !scaleRotationsTextureProps.__webglTexture) {
                scaleRotationsTexture.needsUpdate = true;
            } else {
                this.updateDataTexture(paddedScaleRotations, scaleRotationsTextureDesc.texture, scaleRotationsTextureDesc.size,
                                       scaleRotationsTextureProps, SCALES_ROTATIONS_ELEMENTS_PER_TEXEL, elementsPerSplat, bytesPerElement,
                                       fromSplat, toSplat);
            }
            recordProcessingDuration(processingProfile, 'meshUpdateScaleRotationsTextureMs', scaleRotationsTextureStartTime);
        }

        if (this.useCompressedTexture) {
            const uvTexDesc = this.splatDataTextures['compressedTextureUV'];
            if (uvTexDesc) {
                const astcUvTextureStartTime = performance.now();
                const paddedUVArray = uvTexDesc.data; // RGBA array created by setupDataTextures.
                const sourceUVs = this.splatDataTextures.baseData.compressedTextureUVs;

                // Expand compact U,V pairs into the padded four-element RGBA array.
                // The WebGL texture upload requires this alignment.
                for (let c = fromSplat; c <= toSplat; c++) {
                    const srcIdx = c * 2;
                    const destIdx = c * 4; // Texture is RGBA32UI
                    paddedUVArray[destIdx] = sourceUVs[srcIdx]; // R = U
                    paddedUVArray[destIdx + 1] = sourceUVs[srcIdx + 1]; // G = V
                    paddedUVArray[destIdx + 2] = 0;
                    paddedUVArray[destIdx + 3] = 0;
                }

                const uvTexture = uvTexDesc.texture;
                const uvTextureProps = this.renderer ? this.renderer.properties.get(uvTexture) : null;

                if (!uvTextureProps || !uvTextureProps.__webglTexture) {
                    uvTexture.needsUpdate = true;
                } else {
                    // Reuse updateDataTexture.
                    this.updateDataTexture(paddedUVArray, uvTexture, uvTexDesc.size,
                        uvTextureProps, uvTexDesc.elementsPerTexel, // 4
                        4, // elementsPerSplat (RGBA)
                        4, // bytesPerElement (Uint32 = 4 bytes)
                        fromSplat, toSplat);
                }
                recordProcessingDuration(processingProfile, 'meshUpdateAstcUvTextureMs', astcUvTextureStartTime);
            }
        } else {
            // update spherical harmonics data texture
            const shData = this.splatDataTextures.baseData.sphericalHarmonics;
            if (shData) {
                const sphericalHarmonicsTextureStartTime = performance.now();
                let shBytesPerElement = 4;
                if (shCompressionLevel === 1) shBytesPerElement = 2;
                else if (shCompressionLevel === 2) shBytesPerElement = 1;

                const updateTexture = (shTexture, shTextureSize, elementsPerTexel, paddedSHArray, paddedSHComponentCount) => {
                    const shTextureProps = this.renderer ? this.renderer.properties.get(shTexture) : null;
                    if (!shTextureProps || !shTextureProps.__webglTexture) {
                        shTexture.needsUpdate = true;
                    } else {
                        this.updateDataTexture(paddedSHArray, shTexture, shTextureSize, shTextureProps, elementsPerTexel,
                            paddedSHComponentCount, shBytesPerElement, fromSplat, toSplat);
                    }
                };

                const shComponentCount = shTextureDesc.componentCount;
                const paddedSHComponentCount = shTextureDesc.paddedComponentCount;

                // Update for the case of a single texture for all spherical harmonics data
                if (shTextureDesc.textureCount === 1) {
                    const paddedSHArray = shTextureDesc.data;
                    for (let c = fromSplat; c <= toSplat; c++) {
                        const srcBase = shComponentCount * c;
                        const destBase = paddedSHComponentCount * c;
                        for (let i = 0; i < shComponentCount; i++) {
                            paddedSHArray[destBase + i] = shData[srcBase + i];
                        }
                    }
                    updateTexture(shTextureDesc.texture, shTextureDesc.size,
                        shTextureDesc.elementsPerTexel, paddedSHArray, paddedSHComponentCount);
                    // Update for the case of spherical harmonics data split among three textures, one for each color channel
                } else {
                    const shComponentCountPerChannel = shTextureDesc.componentCountPerChannel;
                    for (let t = 0; t < 3; t++) {
                        const paddedSHArray = shTextureDesc.data[t];
                        for (let c = fromSplat; c <= toSplat; c++) {
                            const srcBase = shComponentCount * c;
                            const destBase = paddedSHComponentCount * c;
                            if (shComponentCountPerChannel >= 3) {
                                for (let i = 0; i < 3; i++) paddedSHArray[destBase + i] = shData[srcBase + t * 3 + i];
                                if (shComponentCountPerChannel >= 8) {
                                    for (let i = 0; i < 5; i++) paddedSHArray[destBase + 3 + i] = shData[srcBase + 9 + t * 5 + i];
                                }
                            }
                        }
                        updateTexture(shTextureDesc.textures[t], shTextureDesc.size,
                            shTextureDesc.elementsPerTexel, paddedSHArray, paddedSHComponentCount);
                    }
                }
                recordProcessingDuration(processingProfile, 'meshUpdateSphericalHarmonicsTextureMs', sphericalHarmonicsTextureStartTime);
            }
        }

        // update scene index & transform data
        const sceneIndexesTextureStartTime = performance.now();
        const sceneIndexesTexDesc = this.splatDataTextures['sceneIndexes'];
        const paddedSceneIndexes = sceneIndexesTexDesc.data;
        for (let c = this.lastBuildSplatCount; c <= toSplat; c++) {
            paddedSceneIndexes[c] = this.globalSplatIndexToSceneIndexMap[c];
        }
        const sceneIndexesTexture = sceneIndexesTexDesc.texture;
        const sceneIndexesTextureProps = this.renderer ? this.renderer.properties.get(sceneIndexesTexture) : null;
        if (!sceneIndexesTextureProps || !sceneIndexesTextureProps.__webglTexture) {
            sceneIndexesTexture.needsUpdate = true;
        } else {
            this.updateDataTexture(paddedSceneIndexes, sceneIndexesTexDesc.texture, sceneIndexesTexDesc.size,
                                   sceneIndexesTextureProps, 1, 1, 1, this.lastBuildSplatCount, toSplat);
        }
        recordProcessingDuration(processingProfile, 'meshUpdateSceneIndexesTextureMs', sceneIndexesTextureStartTime);
        recordProcessingDuration(processingProfile, 'meshUpdateDataTexturesMs', updateTexturesStartTime);
    }

    getTargetCovarianceCompressionLevel() {
        return this.halfPrecisionCovariancesOnGPU ? 1 : 0;
    }

    getTargetSphericalHarmonicsCompressionLevel() {
        return Math.max(1, this.getMaximumSplatBufferCompressionLevel());
    }

    getMaximumSplatBufferCompressionLevel() {
        let maxCompressionLevel;
        for (let i = 0; i < this.scenes.length; i++) {
            const scene = this.getScene(i);
            const splatBuffer = scene.splatBuffer;
            if (i === 0 || splatBuffer.compressionLevel > maxCompressionLevel) {
                maxCompressionLevel = splatBuffer.compressionLevel;
            }
        }
        return maxCompressionLevel;
    }

    getMinimumSplatBufferCompressionLevel() {
        let minCompressionLevel;
        for (let i = 0; i < this.scenes.length; i++) {
            const scene = this.getScene(i);
            const splatBuffer = scene.splatBuffer;
            if (i === 0 || splatBuffer.compressionLevel < minCompressionLevel) {
                minCompressionLevel = splatBuffer.compressionLevel;
            }
        }
        return minCompressionLevel;
    }

    static computeTextureUpdateRegion(startSplat, endSplat, textureWidth, elementsPerTexel, elementsPerSplat) {
        const texelsPerSplat = elementsPerSplat / elementsPerTexel;

        const startSplatTexels = startSplat * texelsPerSplat;
        const startRow = Math.floor(startSplatTexels / textureWidth);
        const startRowElement = startRow * textureWidth * elementsPerTexel;

        const endSplatTexels = endSplat * texelsPerSplat;
        const endRow = Math.floor(endSplatTexels / textureWidth);
        const endRowEndElement = endRow * textureWidth * elementsPerTexel + (textureWidth * elementsPerTexel);

        return {
            'dataStart': startRowElement,
            'dataEnd': endRowEndElement,
            'startRow': startRow,
            'endRow': endRow
        };
    }

    updateDataTexture(paddedData, texture, textureSize, textureProps, elementsPerTexel, elementsPerSplat, bytesPerElement, from, to) {
        const gl = this.renderer.getContext();
        const updateRegion = SplatMesh.computeTextureUpdateRegion(from, to, textureSize.x, elementsPerTexel, elementsPerSplat);
        const updateElementCount = updateRegion.dataEnd - updateRegion.dataStart;
        const updateDataView = new paddedData.constructor(paddedData.buffer,
                                                          updateRegion.dataStart * bytesPerElement, updateElementCount);
        const updateHeight = updateRegion.endRow - updateRegion.startRow + 1;
        const glType = this.webGLUtils.convert(texture.type);
        const glFormat = this.webGLUtils.convert(texture.format, texture.colorSpace);
        const currentTexture = gl.getParameter(gl.TEXTURE_BINDING_2D);
        gl.bindTexture(gl.TEXTURE_2D, textureProps.__webglTexture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, updateRegion.startRow,
                         textureSize.x, updateHeight, glFormat, glType, updateDataView);
        gl.bindTexture(gl.TEXTURE_2D, currentTexture);
    }

    static updatePaddedCompressedCovariancesTextureData(sourceData, textureData, textureDataStartIndex, fromElement, toElement) {
        let textureDataView = new DataView(textureData.buffer);
        let textureDataIndex = textureDataStartIndex;
        let sequentialCount = 0;
        for (let i = fromElement; i <= toElement; i+=2) {
            textureDataView.setUint16(textureDataIndex * 2, sourceData[i], true);
            textureDataView.setUint16(textureDataIndex * 2 + 2, sourceData[i + 1], true);
            textureDataIndex += 2;
            sequentialCount++;
            if (sequentialCount >= 3) {
                textureDataIndex += 2;
                sequentialCount = 0;
            }
        }
    }

    static updateCenterColorsPaddedData(from, to, centers, colors, paddedCenterColors) {
        for (let c = from; c <= to; c++) {
            const colorsBase = c * 4;
            const centersBase = c * 3;
            const centerColorsBase = c * 4;
            paddedCenterColors[centerColorsBase] = rgbaArrayToInteger(colors, colorsBase);
            paddedCenterColors[centerColorsBase + 1] = uintEncodedFloat(centers[centersBase]);
            paddedCenterColors[centerColorsBase + 2] = uintEncodedFloat(centers[centersBase + 1]);
            paddedCenterColors[centerColorsBase + 3] = uintEncodedFloat(centers[centersBase + 2]);
        }
    }

    static updateScaleRotationsPaddedData(from, to, scales, rotations, paddedScaleRotations) {
        const combinedSize = 6;
        for (let c = from; c <= to; c++) {
            const scaleBase = c * 3;
            const rotationBase = c * 4;
            const scaleRotationsBase = c * combinedSize;

            paddedScaleRotations[scaleRotationsBase] = scales[scaleBase];
            paddedScaleRotations[scaleRotationsBase + 1] = scales[scaleBase + 1];
            paddedScaleRotations[scaleRotationsBase + 2] = scales[scaleBase + 2];

            paddedScaleRotations[scaleRotationsBase + 3] = rotations[rotationBase];
            paddedScaleRotations[scaleRotationsBase + 4] = rotations[rotationBase + 1];
            paddedScaleRotations[scaleRotationsBase + 5] = rotations[rotationBase + 2];
        }
    }

    prepareVisibleRegionForImmediateRender() {
        if (this.scenes.length > 0) {
            const avgCenter = new THREE.Vector3();
            this.scenes.forEach((scene) => {
                avgCenter.add(scene.splatBuffer.sceneCenter);
            });
            avgCenter.multiplyScalar(1.0 / this.scenes.length);
            this.calculatedSceneCenter.copy(avgCenter);
        }

        this.material.uniforms.sceneCenter.value.copy(this.calculatedSceneCenter);
        this.material.uniforms.visibleRegionFadeStartRadius.value = this.visibleRegionFadeStartRadius;
        this.material.uniforms.visibleRegionRadius.value = this.visibleRegionRadius;
        this.material.uniforms.firstRenderTime.value = this.firstRenderTime;
        this.material.uniforms.currentTime.value = performance.now();
        this.material.uniforms.fadeInComplete.value = 1;
        this.material.uniformsNeedUpdate = true;
        this.visibleRegionChanging = false;
    }

    updateVisibleRegion(sinceLastBuildOnly, sceneRevealMode = SceneRevealMode.Default) {
        const splatCount = this.getSplatCount(true);
        const tempCenter = new THREE.Vector3();
        if (!sinceLastBuildOnly) {
            const avgCenter = new THREE.Vector3();
            this.scenes.forEach((scene) => {
                avgCenter.add(scene.splatBuffer.sceneCenter);
            });
            avgCenter.multiplyScalar(1.0 / this.scenes.length);
            this.calculatedSceneCenter.copy(avgCenter);
            this.material.uniforms.sceneCenter.value.copy(this.calculatedSceneCenter);
            this.material.uniformsNeedUpdate = true;
        }

        const startSplatFormMaxDistanceCalc = sinceLastBuildOnly ? this.lastBuildSplatCount : 0;
        for (let i = startSplatFormMaxDistanceCalc; i < splatCount; i++) {
            this.getSplatCenter(i, tempCenter, true);
            const distFromCSceneCenter = tempCenter.sub(this.calculatedSceneCenter).length();
            if (distFromCSceneCenter > this.maxSplatDistanceFromSceneCenter) this.maxSplatDistanceFromSceneCenter = distFromCSceneCenter;
        }

        if (this.maxSplatDistanceFromSceneCenter - this.visibleRegionBufferRadius > VISIBLE_REGION_EXPANSION_DELTA) {
            this.visibleRegionBufferRadius = this.maxSplatDistanceFromSceneCenter;
            this.visibleRegionRadius = Math.max(this.visibleRegionBufferRadius - VISIBLE_REGION_EXPANSION_DELTA, 0.0);
        }
        if (this.finalBuild) this.visibleRegionRadius = this.visibleRegionBufferRadius = this.maxSplatDistanceFromSceneCenter;
        this.updateVisibleRegionFadeDistance(sceneRevealMode);
    }

    updateVisibleRegionFadeDistance(sceneRevealMode = SceneRevealMode.Default) {
        const fastFadeRate = SCENE_FADEIN_RATE_FAST * this.sceneFadeInRateMultiplier;
        const gradualFadeRate = SCENE_FADEIN_RATE_GRADUAL * this.sceneFadeInRateMultiplier;
        const defaultFadeInRate = this.finalBuild ? fastFadeRate : gradualFadeRate;
        const fadeInRate = sceneRevealMode === SceneRevealMode.Default ? defaultFadeInRate : gradualFadeRate;
        this.visibleRegionFadeStartRadius = (this.visibleRegionRadius - this.visibleRegionFadeStartRadius) *
                                             fadeInRate + this.visibleRegionFadeStartRadius;
        const fadeInPercentage = (this.visibleRegionBufferRadius > 0) ?
                                 (this.visibleRegionFadeStartRadius / this.visibleRegionBufferRadius) : 0;
        const fadeInComplete = fadeInPercentage > 0.99;
        const shaderFadeInComplete = (fadeInComplete || sceneRevealMode === SceneRevealMode.Instant) ? 1 : 0;

        this.material.uniforms.visibleRegionFadeStartRadius.value = this.visibleRegionFadeStartRadius;
        this.material.uniforms.visibleRegionRadius.value = this.visibleRegionRadius;
        this.material.uniforms.firstRenderTime.value = this.firstRenderTime;
        this.material.uniforms.currentTime.value = performance.now();
        this.material.uniforms.fadeInComplete.value = shaderFadeInComplete;
        this.material.uniformsNeedUpdate = true;
        this.visibleRegionChanging = !fadeInComplete;
    }

    /**
     * Set the indexes of splats that should be rendered; should be sorted in desired render order.
     * @param {Uint32Array} globalIndexes Sorted index list of splats to be rendered
     * @param {number} renderSplatCount Total number of splats to be rendered. Necessary because we may not want to render
     *                                  every splat.
     */
    updateRenderIndexes(globalIndexes, renderSplatCount) {
        const geometry = this.geometry;
        if (this.compactGaussianMode && geometry.userData.fastBatched && this.splatDataTextures.fastOrder) {
            const order = this.splatDataTextures.fastOrder.data;
            order.fill(0);
            const indexes = globalIndexes.subarray ?
                globalIndexes.subarray(0, Math.min(renderSplatCount, order.length)) :
                globalIndexes.slice(0, Math.min(renderSplatCount, order.length));
            order.set(indexes);
            this.splatDataTextures.fastOrder.texture.needsUpdate = true;
            this.material.uniforms.fastRenderSplatCount.value = renderSplatCount;
        } else {
            geometry.attributes.splatIndex.set(globalIndexes);
            geometry.attributes.splatIndex.needsUpdate = true;
        }
        if (renderSplatCount > 0 && this.firstRenderTime === -1) this.firstRenderTime = performance.now();
        if (geometry.userData.fastBatched) {
            geometry.instanceCount = Math.ceil(renderSplatCount / (geometry.userData.fastBatchSize || 128));
            geometry.setDrawRange(0, geometry.index.count);
        } else {
            geometry.instanceCount = renderSplatCount;
            geometry.setDrawRange(0, renderSplatCount);
        }
    }

    /**
     * Update the transforms for each scene in this splat mesh from their individual components (position,
     * quaternion, and scale)
     */
    updateTransforms() {
        for (let i = 0; i < this.scenes.length; i++) {
            const scene = this.getScene(i);
            scene.updateTransform(this.dynamicMode);
        }
    }

    updateUniforms = function() {

        const viewport = new THREE.Vector2();

        return function(renderDimensions, cameraFocalLengthX, cameraFocalLengthY,
                        orthographicMode, orthographicZoom, inverseFocalAdjustment) {
            const splatCount = this.getSplatCount();
            if (splatCount > 0) {
                viewport.set(renderDimensions.x * this.devicePixelRatio,
                             renderDimensions.y * this.devicePixelRatio);
                this.material.uniforms.viewport.value.copy(viewport);
                this.material.uniforms.basisViewport.value.set(1.0 / viewport.x, 1.0 / viewport.y);
                this.material.uniforms.focal.value.set(cameraFocalLengthX, cameraFocalLengthY);
                this.material.uniforms.orthographicMode.value = orthographicMode ? 1 : 0;
                this.material.uniforms.orthoZoom.value = orthographicZoom;
                this.material.uniforms.inverseFocalAdjustment.value = inverseFocalAdjustment;
                if (this.dynamicMode) {
                    for (let i = 0; i < this.scenes.length; i++) {
                        this.material.uniforms.transforms.value[i].copy(this.getScene(i).transform);
                    }
                }
                if (this.enableOptionalEffects) {
                    for (let i = 0; i < this.scenes.length; i++) {
                        this.material.uniforms.sceneOpacity.value[i] = clamp(this.getScene(i).opacity, 0.0, 1.0);
                        this.material.uniforms.sceneVisibility.value[i] = this.getScene(i).visible ? 1 : 0;
                        this.material.uniformsNeedUpdate = true;
                    }
                }
                this.material.uniformsNeedUpdate = true;
            }
        };

    }();

    setSplatScale(splatScale = 1) {
        this.splatScale = splatScale;
        this.material.uniforms.splatScale.value = splatScale;
        this.material.uniformsNeedUpdate = true;
    }

    getSplatScale() {
        return this.splatScale;
    }

    setPointCloudModeEnabled(enabled) {
        this.pointCloudModeEnabled = enabled;
        this.material.uniforms.pointCloudModeEnabled.value = enabled ? 1 : 0;
        this.material.uniformsNeedUpdate = true;
    }

    getPointCloudModeEnabled() {
        return this.pointCloudModeEnabled;
    }

    getSplatDataTextures() {
        return this.splatDataTextures;
    }

    getSplatCount(includeSinceLastBuild = false) {
        if (!includeSinceLastBuild) return this.lastBuildSplatCount;
        else return SplatMesh.getTotalSplatCountForScenes(this.scenes);
    }

    static getTotalSplatCountForScenes(scenes) {
        let totalSplatCount = 0;
        for (let scene of scenes) {
            if (scene && scene.splatBuffer) totalSplatCount += scene.splatBuffer.getSplatCount();
        }
        return totalSplatCount;
    }

    static getTotalSplatCountForSplatBuffers(splatBuffers) {
        let totalSplatCount = 0;
        for (let splatBuffer of splatBuffers) totalSplatCount += splatBuffer.getSplatCount();
        return totalSplatCount;
    }

    getMaxSplatCount() {
        return SplatMesh.getTotalMaxSplatCountForScenes(this.scenes);
    }

    static getTotalMaxSplatCountForScenes(scenes) {
        let totalSplatCount = 0;
        for (let scene of scenes) {
            if (scene && scene.splatBuffer) totalSplatCount += scene.splatBuffer.getMaxSplatCount();
        }
        return totalSplatCount;
    }

    static getTotalMaxSplatCountForSplatBuffers(splatBuffers) {
        let totalSplatCount = 0;
        for (let splatBuffer of splatBuffers) totalSplatCount += splatBuffer.getMaxSplatCount();
        return totalSplatCount;
    }

    disposeDistancesComputationGPUResources() {

        if (!this.renderer) return;

        const gl = this.renderer.getContext();

        if (this.distancesTransformFeedback.vao) {
            gl.deleteVertexArray(this.distancesTransformFeedback.vao);
            this.distancesTransformFeedback.vao = null;
        }
        if (this.distancesTransformFeedback.program) {
            gl.deleteProgram(this.distancesTransformFeedback.program);
            gl.deleteShader(this.distancesTransformFeedback.vertexShader);
            gl.deleteShader(this.distancesTransformFeedback.fragmentShader);
            this.distancesTransformFeedback.program = null;
            this.distancesTransformFeedback.vertexShader = null;
            this.distancesTransformFeedback.fragmentShader = null;
        }
        this.disposeDistancesComputationGPUBufferResources();
        if (this.distancesTransformFeedback.id) {
            gl.deleteTransformFeedback(this.distancesTransformFeedback.id);
            this.distancesTransformFeedback.id = null;
        }
    }

    disposeDistancesComputationGPUBufferResources() {

        if (!this.renderer) return;

        const gl = this.renderer.getContext();

        if (this.distancesTransformFeedback.centersBuffer) {
            this.distancesTransformFeedback.centersBuffer = null;
            gl.deleteBuffer(this.distancesTransformFeedback.centersBuffer);
        }
        if (this.distancesTransformFeedback.outDistancesBuffer) {
            gl.deleteBuffer(this.distancesTransformFeedback.outDistancesBuffer);
            this.distancesTransformFeedback.outDistancesBuffer = null;
        }
    }

    /**
     * Set the Three.js renderer used by this splat mesh
     * @param {THREE.WebGLRenderer} renderer Instance of THREE.WebGLRenderer
     */
    setRenderer(renderer) {
        if (renderer !== this.renderer) {
            this.renderer = renderer;
            const gl = this.renderer.getContext();
            const extensions = new WebGLExtensions(gl);
            const capabilities = new WebGLCapabilities(gl, extensions, {});
            extensions.init(capabilities);
            this.webGLUtils = new THREE.WebGLUtils(gl, extensions, capabilities);
            if (this.enableDistancesComputationOnGPU && this.getSplatCount() > 0) {
                this.setupDistancesComputationTransformFeedback();
                const distanceData = this.compactGaussianMode && this.compactGaussianDistanceCenters ? null :
                    this.getDataForDistancesComputation(0, this.getSplatCount() - 1);
                const centers = this.compactGaussianMode && this.compactGaussianDistanceCenters ?
                    this.compactGaussianDistanceCenters : distanceData.centers;
                const sceneIndexes = this.compactGaussianMode ? EMPTY_UINT32_ARRAY : distanceData.sceneIndexes;
                this.refreshGPUBuffersForDistancesComputation(centers, sceneIndexes);
            }
        }
    }

    setupDistancesComputationTransformFeedback = function() {

        let currentMaxSplatCount;

        return function() {
            const maxSplatCount = this.getMaxSplatCount();

            if (!this.renderer) return;

            const rebuildGPUObjects = (this.lastRenderer !== this.renderer);
            const rebuildBuffers = currentMaxSplatCount !== maxSplatCount;

            if (!rebuildGPUObjects && !rebuildBuffers) return;

            if (rebuildGPUObjects) {
                this.disposeDistancesComputationGPUResources();
            } else if (rebuildBuffers) {
                this.disposeDistancesComputationGPUBufferResources();
            }

            const gl = this.renderer.getContext();

            const createShader = (gl, type, source) => {
                const shader = gl.createShader(type);
                if (!shader) {
                    console.error('Fatal error: gl could not create a shader object.');
                    return null;
                }

                gl.shaderSource(shader, source);
                gl.compileShader(shader);

                const compiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
                if (!compiled) {
                    let typeName = 'unknown';
                    if (type === gl.VERTEX_SHADER) typeName = 'vertex shader';
                    else if (type === gl.FRAGMENT_SHADER) typeName = 'fragement shader';
                    const errors = gl.getShaderInfoLog(shader);
                    console.error('Failed to compile ' + typeName + ' with these errors:' + errors);
                    gl.deleteShader(shader);
                    return null;
                }

                return shader;
            };

            let vsSource;
            if (this.integerBasedDistancesComputation) {
                vsSource =
                `#version 300 es
                in ivec4 center;
                flat out int distance;`;
                if (this.dynamicMode) {
                    vsSource += `
                        in uint sceneIndex;
                        uniform ivec4 transforms[${Constants.MaxScenes}];
                        void main(void) {
                            ivec4 transform = transforms[sceneIndex];
                            distance = center.x * transform.x + center.y * transform.y + center.z * transform.z + transform.w * center.w;
                        }
                    `;
                } else {
                    vsSource += `
                        uniform ivec3 modelViewProj;
                        void main(void) {
                            distance = center.x * modelViewProj.x + center.y * modelViewProj.y + center.z * modelViewProj.z;
                        }
                    `;
                }
            } else {
                vsSource =
                `#version 300 es
                in vec4 center;
                flat out float distance;`;
                if (this.dynamicMode) {
                    vsSource += `
                        in uint sceneIndex;
                        uniform mat4 transforms[${Constants.MaxScenes}];
                        void main(void) {
                            vec4 transformedCenter = transforms[sceneIndex] * vec4(center.xyz, 1.0);
                            distance = transformedCenter.z;
                        }
                    `;
                } else {
                    vsSource += `
                        uniform vec3 modelViewProj;
                        void main(void) {
                            distance = center.x * modelViewProj.x + center.y * modelViewProj.y + center.z * modelViewProj.z;
                        }
                    `;
                }
            }

            const fsSource =
            `#version 300 es
                precision lowp float;
                out vec4 fragColor;
                void main(){}
            `;

            const currentVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
            const currentProgram = gl.getParameter(gl.CURRENT_PROGRAM);
            const currentProgramDeleted = currentProgram ? gl.getProgramParameter(currentProgram, gl.DELETE_STATUS) : false;

            if (rebuildGPUObjects) {
                this.distancesTransformFeedback.vao = gl.createVertexArray();
            }

            gl.bindVertexArray(this.distancesTransformFeedback.vao);

            if (rebuildGPUObjects) {
                const program = gl.createProgram();
                const vertexShader = createShader(gl, gl.VERTEX_SHADER, vsSource);
                const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fsSource);
                if (!vertexShader || !fragmentShader) {
                    throw new Error('Could not compile shaders for distances computation on GPU.');
                }
                gl.attachShader(program, vertexShader);
                gl.attachShader(program, fragmentShader);
                gl.transformFeedbackVaryings(program, ['distance'], gl.SEPARATE_ATTRIBS);
                gl.linkProgram(program);

                const linked = gl.getProgramParameter(program, gl.LINK_STATUS);
                if (!linked) {
                    const error = gl.getProgramInfoLog(program);
                    console.error('Fatal error: Failed to link program: ' + error);
                    gl.deleteProgram(program);
                    gl.deleteShader(fragmentShader);
                    gl.deleteShader(vertexShader);
                    throw new Error('Could not link shaders for distances computation on GPU.');
                }

                this.distancesTransformFeedback.program = program;
                this.distancesTransformFeedback.vertexShader = vertexShader;
                this.distancesTransformFeedback.vertexShader = fragmentShader;
            }

            gl.useProgram(this.distancesTransformFeedback.program);

            this.distancesTransformFeedback.centersLoc =
                gl.getAttribLocation(this.distancesTransformFeedback.program, 'center');
            if (this.dynamicMode) {
                this.distancesTransformFeedback.sceneIndexesLoc =
                    gl.getAttribLocation(this.distancesTransformFeedback.program, 'sceneIndex');
                for (let i = 0; i < this.scenes.length; i++) {
                    this.distancesTransformFeedback.transformsLocs[i] =
                        gl.getUniformLocation(this.distancesTransformFeedback.program, `transforms[${i}]`);
                }
            } else {
                this.distancesTransformFeedback.modelViewProjLoc =
                    gl.getUniformLocation(this.distancesTransformFeedback.program, 'modelViewProj');
            }

            if (rebuildGPUObjects || rebuildBuffers) {
                this.distancesTransformFeedback.centersBuffer = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, this.distancesTransformFeedback.centersBuffer);
                gl.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc);
                if (this.integerBasedDistancesComputation) {
                    gl.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc, 4, gl.INT, 0, 0);
                } else {
                    gl.vertexAttribPointer(this.distancesTransformFeedback.centersLoc, 4, gl.FLOAT, false, 0, 0);
                }

                if (this.dynamicMode) {
                    this.distancesTransformFeedback.sceneIndexesBuffer = gl.createBuffer();
                    gl.bindBuffer(gl.ARRAY_BUFFER, this.distancesTransformFeedback.sceneIndexesBuffer);
                    gl.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc);
                    gl.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc, 1, gl.UNSIGNED_INT, 0, 0);
                }
            }

            if (rebuildGPUObjects || rebuildBuffers) {
                this.distancesTransformFeedback.outDistancesBuffer = gl.createBuffer();
            }
            gl.bindBuffer(gl.ARRAY_BUFFER, this.distancesTransformFeedback.outDistancesBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, maxSplatCount * 4, gl.STATIC_READ);

            if (rebuildGPUObjects) {
                this.distancesTransformFeedback.id = gl.createTransformFeedback();
            }
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.distancesTransformFeedback.id);
            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.distancesTransformFeedback.outDistancesBuffer);

            if (currentProgram && currentProgramDeleted !== true) gl.useProgram(currentProgram);
            if (currentVao) gl.bindVertexArray(currentVao);

            this.lastRenderer = this.renderer;
            currentMaxSplatCount = maxSplatCount;
        };

    }();

    /**
     * Refresh GPU buffers used for computing splat distances with centers data from the scenes for this mesh.
     * @param {boolean} isUpdate Specify whether or not to update the GPU buffer or to initialize & fill
     * @param {Array<number>} centers The splat centers data
     * @param {number} offsetSplats Offset in the GPU buffer at which to start updating data, specified in splats
     */
    updateGPUCentersBufferForDistancesComputation(isUpdate, centers, offsetSplats) {

        if (!this.renderer) return;

        const gl = this.renderer.getContext();

        const currentVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
        gl.bindVertexArray(this.distancesTransformFeedback.vao);

        const ArrayType = this.integerBasedDistancesComputation ? Uint32Array : Float32Array;
        const attributeBytesPerCenter = 16;
        const subBufferOffset = offsetSplats * attributeBytesPerCenter;

        gl.bindBuffer(gl.ARRAY_BUFFER, this.distancesTransformFeedback.centersBuffer);

        if (isUpdate) {
            gl.bufferSubData(gl.ARRAY_BUFFER, subBufferOffset, centers);
        } else {
            const maxArray = new ArrayType(this.getMaxSplatCount() * attributeBytesPerCenter);
            maxArray.set(centers);
            gl.bufferData(gl.ARRAY_BUFFER, maxArray, gl.STATIC_DRAW);
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        if (currentVao) gl.bindVertexArray(currentVao);
    }

    /**
     * Refresh GPU buffers used for pre-computing splat distances with centers data from the scenes for this mesh.
     * @param {boolean} isUpdate Specify whether or not to update the GPU buffer or to initialize & fill
     * @param {Array<number>} sceneIndexes The splat scene indexes
     * @param {number} offsetSplats Offset in the GPU buffer at which to start updating data, specified in splats
     */
    updateGPUTransformIndexesBufferForDistancesComputation(isUpdate, sceneIndexes, offsetSplats) {

        if (!this.renderer || !this.dynamicMode) return;

        const gl = this.renderer.getContext();

        const currentVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
        gl.bindVertexArray(this.distancesTransformFeedback.vao);

        const subBufferOffset = offsetSplats * 4;

        gl.bindBuffer(gl.ARRAY_BUFFER, this.distancesTransformFeedback.sceneIndexesBuffer);

        if (isUpdate) {
            gl.bufferSubData(gl.ARRAY_BUFFER, subBufferOffset, sceneIndexes);
        } else {
            const maxArray = new Uint32Array(this.getMaxSplatCount() * 4);
            maxArray.set(sceneIndexes);
            gl.bufferData(gl.ARRAY_BUFFER, maxArray, gl.STATIC_DRAW);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        if (currentVao) gl.bindVertexArray(currentVao);
    }

    /**
     * Get a typed array containing a mapping from global splat indexes to their scene index.
     * @param {number} start Starting splat index to store
     * @param {number} end Ending splat index to store
     * @return {Uint32Array}
     */
    getSceneIndexes(start, end) {

        const fillCount = end - start + 1;
        const sceneIndexes = new Uint32Array(fillCount);
        for (let i = 0; i < fillCount; i++) {
            sceneIndexes[i] = this.globalSplatIndexToSceneIndexMap[start + i];
        }

        return sceneIndexes;
    }

    /**
     * Fill 'array' with the transforms for each scene in this splat mesh.
     * @param {Array} array Empty array to be filled with scene transforms. If not empty, contents will be overwritten.
     */
    fillTransformsArray = function() {

        const tempArray = [];

        return function(array) {
            if (tempArray.length !== array.length) tempArray.length = array.length;
            for (let i = 0; i < this.scenes.length; i++) {
                const sceneTransform = this.getScene(i).transform;
                const sceneTransformElements = sceneTransform.elements;
                for (let j = 0; j < 16; j++) {
                    tempArray[i * 16 + j] = sceneTransformElements[j];
                }
            }
            array.set(tempArray);
        };

    }();

    computeDistancesOnGPU = function() {

        const tempMatrix = new THREE.Matrix4();

        return function(modelViewProjMatrix, outComputedDistances) {
            if (!this.renderer) return;

            // console.time("gpu_compute_distances");
            const gl = this.renderer.getContext();

            const currentVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
            const currentProgram = gl.getParameter(gl.CURRENT_PROGRAM);
            const currentProgramDeleted = currentProgram ? gl.getProgramParameter(currentProgram, gl.DELETE_STATUS) : false;

            gl.bindVertexArray(this.distancesTransformFeedback.vao);
            gl.useProgram(this.distancesTransformFeedback.program);

            gl.enable(gl.RASTERIZER_DISCARD);

            if (this.dynamicMode) {
                for (let i = 0; i < this.scenes.length; i++) {
                    tempMatrix.copy(this.getScene(i).transform);
                    tempMatrix.premultiply(modelViewProjMatrix);

                    if (this.integerBasedDistancesComputation) {
                        const iTempMatrix = SplatMesh.getIntegerMatrixArray(tempMatrix);
                        const iTransform = [iTempMatrix[2], iTempMatrix[6], iTempMatrix[10], iTempMatrix[14]];
                        gl.uniform4i(this.distancesTransformFeedback.transformsLocs[i], iTransform[0], iTransform[1],
                                                                                        iTransform[2], iTransform[3]);
                    } else {
                        gl.uniformMatrix4fv(this.distancesTransformFeedback.transformsLocs[i], false, tempMatrix.elements);
                    }
                }
            } else {
                if (this.integerBasedDistancesComputation) {
                    const iViewProjMatrix = SplatMesh.getIntegerMatrixArray(modelViewProjMatrix);
                    const iViewProj = [iViewProjMatrix[2], iViewProjMatrix[6], iViewProjMatrix[10]];
                    gl.uniform3i(this.distancesTransformFeedback.modelViewProjLoc, iViewProj[0], iViewProj[1], iViewProj[2]);
                } else {
                    const viewProj = [modelViewProjMatrix.elements[2], modelViewProjMatrix.elements[6], modelViewProjMatrix.elements[10]];
                    gl.uniform3f(this.distancesTransformFeedback.modelViewProjLoc, viewProj[0], viewProj[1], viewProj[2]);
                }
            }

            gl.bindBuffer(gl.ARRAY_BUFFER, this.distancesTransformFeedback.centersBuffer);
            gl.enableVertexAttribArray(this.distancesTransformFeedback.centersLoc);
            if (this.integerBasedDistancesComputation) {
                gl.vertexAttribIPointer(this.distancesTransformFeedback.centersLoc, 4, gl.INT, 0, 0);
            } else {
                gl.vertexAttribPointer(this.distancesTransformFeedback.centersLoc, 4, gl.FLOAT, false, 0, 0);
            }

            if (this.dynamicMode) {
                gl.bindBuffer(gl.ARRAY_BUFFER, this.distancesTransformFeedback.sceneIndexesBuffer);
                gl.enableVertexAttribArray(this.distancesTransformFeedback.sceneIndexesLoc);
                gl.vertexAttribIPointer(this.distancesTransformFeedback.sceneIndexesLoc, 1, gl.UNSIGNED_INT, 0, 0);
            }

            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.distancesTransformFeedback.id);
            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.distancesTransformFeedback.outDistancesBuffer);

            gl.beginTransformFeedback(gl.POINTS);
            gl.drawArrays(gl.POINTS, 0, this.getSplatCount());
            gl.endTransformFeedback();

            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

            gl.disable(gl.RASTERIZER_DISCARD);

            const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
            gl.flush();

            const promise = new Promise((resolve) => {
                const checkSync = () => {
                    if (this.disposed) {
                        resolve();
                    } else {
                        const timeout = 0;
                        const bitflags = 0;
                        const status = gl.clientWaitSync(sync, bitflags, timeout);
                        switch (status) {
                            case gl.TIMEOUT_EXPIRED:
                                this.computeDistancesOnGPUSyncTimeout = setTimeout(checkSync);
                                return this.computeDistancesOnGPUSyncTimeout;
                            case gl.WAIT_FAILED:
                                throw new Error('should never get here');
                            default:
                                this.computeDistancesOnGPUSyncTimeout = null;
                                gl.deleteSync(sync);
                                const currentVao = gl.getParameter(gl.VERTEX_ARRAY_BINDING);
                                gl.bindVertexArray(this.distancesTransformFeedback.vao);
                                gl.bindBuffer(gl.ARRAY_BUFFER, this.distancesTransformFeedback.outDistancesBuffer);
                                gl.getBufferSubData(gl.ARRAY_BUFFER, 0, outComputedDistances);
                                gl.bindBuffer(gl.ARRAY_BUFFER, null);

                                if (currentVao) gl.bindVertexArray(currentVao);

                                // console.timeEnd("gpu_compute_distances");

                                resolve();
                        }
                    }
                };
                this.computeDistancesOnGPUSyncTimeout = setTimeout(checkSync);
            });

            if (currentProgram && currentProgramDeleted !== true) gl.useProgram(currentProgram);
            if (currentVao) gl.bindVertexArray(currentVao);

            return promise;
        };

    }();

    /**
     * Given a global splat index, return corresponding local data (splat buffer, index of splat in that splat
     * buffer, and the corresponding transform)
     * @param {number} globalIndex Global splat index
     * @param {object} paramsObj Object in which to store local data
     * @param {boolean} returnSceneTransform By default, the transform of the scene to which the splat at 'globalIndex' belongs will be
     *                                       returned via the 'sceneTransform' property of 'paramsObj' only if the splat mesh is static.
     *                                       If 'returnSceneTransform' is true, the 'sceneTransform' property will always contain the scene
     *                                       transform, and if 'returnSceneTransform' is false, the 'sceneTransform' property will always
     *                                       be null.
     */
    getLocalSplatParameters(globalIndex, paramsObj, returnSceneTransform) {
        if (returnSceneTransform === undefined || returnSceneTransform === null) {
            returnSceneTransform = this.dynamicMode ? false : true;
        }
        paramsObj.splatBuffer = this.getSplatBufferForSplat(globalIndex);
        paramsObj.localIndex = this.getSplatLocalIndex(globalIndex);
        paramsObj.sceneTransform = returnSceneTransform ? this.getSceneTransformForSplat(globalIndex) : null;
    }

    /**
     * Fill arrays with splat data and apply transforms if appropriate. Each array is optional.
     * @param {Float32Array} covariances Target storage for splat covariances
     * @param {Float32Array} scales Target storage for splat scales
     * @param {Float32Array} rotations Target storage for splat rotations
     * @param {Float32Array} centers Target storage for splat centers
     * @param {Uint8Array} colors Target storage for splat colors
     * @param {Float32Array} sphericalHarmonics Target storage for spherical harmonics
     * @param {boolean} applySceneTransform By default, scene transforms are applied to relevant splat data only if the splat mesh is
     *                                      static. If 'applySceneTransform' is true, scene transforms will always be applied and if
     *                                      it is false, they will never be applied. If undefined, the default behavior will apply.
     * @param {number} covarianceCompressionLevel The compression level for covariances in the destination array
     * @param {number} sphericalHarmonicsCompressionLevel The compression level for spherical harmonics in the destination array
     * @param {number} srcStart The start location from which to pull source data
     * @param {number} srcEnd The end location from which to pull source data
     * @param {number} destStart The start location from which to write data
     */
    fillSplatDataArrays(covariances, scales, rotations, centers, colors, sphericalHarmonics, applySceneTransform,
                        covarianceCompressionLevel = 0, scaleRotationCompressionLevel = 0, sphericalHarmonicsCompressionLevel = 1,
                        srcStart, srcEnd, destStart = 0, sceneIndex) {
        const scaleOverride = new THREE.Vector3();
        scaleOverride.x = undefined;
        scaleOverride.y = undefined;
        if (this.splatRenderMode === SplatRenderMode.ThreeD) {
            scaleOverride.z = undefined;
        } else {
            scaleOverride.z = 1;
        }
        const tempTransform = new THREE.Matrix4();

        let startSceneIndex = 0;
        let endSceneIndex = this.scenes.length - 1;
        if (sceneIndex !== undefined && sceneIndex !== null && sceneIndex >= 0 && sceneIndex <= this.scenes.length) {
            startSceneIndex = sceneIndex;
            endSceneIndex = sceneIndex;
        }
        for (let i = startSceneIndex; i <= endSceneIndex; i++) {
            if (applySceneTransform === undefined || applySceneTransform === null) {
                applySceneTransform = this.dynamicMode ? false : true;
            }

            const scene = this.getScene(i);
            const splatBuffer = scene.splatBuffer;
            let sceneTransform;
            if (applySceneTransform) {
                this.getSceneTransform(i, tempTransform);
                sceneTransform = tempTransform;
            }
            if (covariances) {
                splatBuffer.fillSplatCovarianceArray(covariances, sceneTransform, srcStart, srcEnd, destStart, covarianceCompressionLevel);
            }
            if (scales || rotations) {
                if (!scales || !rotations) {
                    throw new Error('SplatMesh::fillSplatDataArrays() -> "scales" and "rotations" must both be valid.');
                }
                splatBuffer.fillSplatScaleRotationArray(scales, rotations, sceneTransform,
                                                        srcStart, srcEnd, destStart, scaleRotationCompressionLevel, scaleOverride);
            }
            if (centers) splatBuffer.fillSplatCenterArray(centers, sceneTransform, srcStart, srcEnd, destStart);
            if (colors) splatBuffer.fillSplatColorArray(colors, scene.minimumAlpha, srcStart, srcEnd, destStart);
            if (sphericalHarmonics) {
                splatBuffer.fillSphericalHarmonicsArray(sphericalHarmonics, this.minSphericalHarmonicsDegree,
                                                        sceneTransform, srcStart, srcEnd, destStart, sphericalHarmonicsCompressionLevel);
            }
            destStart += splatBuffer.getSplatCount();
        }
    }

    fillSplatCompressedTextureUVs(outUVArray, srcFrom, srcEnd, destStart) {
        let currentDest = destStart * 2;

        for (let i = srcFrom; i <= srcEnd; i++) {
            const localIndex = this.getSplatLocalIndex(i);
            const splatBuffer = this.getSplatBufferForSplat(i);
            const sectionIndex = splatBuffer.globalSplatIndexToSectionMap[localIndex];
            const section = splatBuffer.sections[sectionIndex];
            const inSectionIndex = localIndex - section.splatCountOffset;

            const compressionLevel = splatBuffer.compressionLevel;
            const shOffset = splatBuffer.constructor.CompressionLevels[compressionLevel].SphericalHarmonicsOffsetBytes;
            const baseOffset = section.dataBase + inSectionIndex * section.bytesPerSplat + shOffset;

            const dataView = new DataView(splatBuffer.bufferData, baseOffset, 8);
            const u = dataView.getUint32(0, true);
            const v = dataView.getUint32(4, true);

            outUVArray[currentDest] = u;
            outUVArray[currentDest + 1] = v;

            currentDest += 2;
        }
    }

    fillSplatASTCUVs(outUVArray, srcFrom, srcEnd, destStart) {
        return this.fillSplatCompressedTextureUVs(outUVArray, srcFrom, srcEnd, destStart);
    }

    fillDirectCompressedTextureDataArrays = function() {

        const scale = new THREE.Vector3();
        const rotation = new THREE.Quaternion();
        const tempMatrix4 = new THREE.Matrix4();
        const scaleMatrix = new THREE.Matrix3();
        const rotationMatrix = new THREE.Matrix3();
        const covarianceMatrix = new THREE.Matrix3();
        const transformedCovariance = new THREE.Matrix3();
        const toHalfFloat = THREE.DataUtils.toHalfFloat.bind(THREE.DataUtils);

        const writeCovariance = (scale, rotation, outCovariance, outOffset, desiredOutputCompressionLevel) => {
            tempMatrix4.makeScale(scale.x, scale.y, scale.z);
            scaleMatrix.setFromMatrix4(tempMatrix4);
            tempMatrix4.makeRotationFromQuaternion(rotation);
            rotationMatrix.setFromMatrix4(tempMatrix4);
            covarianceMatrix.copy(rotationMatrix).multiply(scaleMatrix);
            transformedCovariance.copy(covarianceMatrix).transpose().premultiply(covarianceMatrix);

            if (desiredOutputCompressionLevel >= 1) {
                outCovariance[outOffset] = toHalfFloat(transformedCovariance.elements[0]);
                outCovariance[outOffset + 1] = toHalfFloat(transformedCovariance.elements[3]);
                outCovariance[outOffset + 2] = toHalfFloat(transformedCovariance.elements[6]);
                outCovariance[outOffset + 3] = toHalfFloat(transformedCovariance.elements[4]);
                outCovariance[outOffset + 4] = toHalfFloat(transformedCovariance.elements[7]);
                outCovariance[outOffset + 5] = toHalfFloat(transformedCovariance.elements[8]);
            } else {
                outCovariance[outOffset] = transformedCovariance.elements[0];
                outCovariance[outOffset + 1] = transformedCovariance.elements[3];
                outCovariance[outOffset + 2] = transformedCovariance.elements[6];
                outCovariance[outOffset + 3] = transformedCovariance.elements[4];
                outCovariance[outOffset + 4] = transformedCovariance.elements[7];
                outCovariance[outOffset + 5] = transformedCovariance.elements[8];
            }
        };

        return function(fromSplat, toSplat, covarianceCompressionLevel = 0, scaleRotationCompressionLevel = 0) {
            if (!this.useCompressedTexture || this.dynamicMode || this.scenes.length !== 1) return false;

            const scene = this.scenes[0];
            if (scene.position.lengthSq() !== 0 ||
                scene.quaternion.x !== 0 || scene.quaternion.y !== 0 || scene.quaternion.z !== 0 || scene.quaternion.w !== 1 ||
                scene.scale.x !== 1 || scene.scale.y !== 1 || scene.scale.z !== 1) {
                return false;
            }

            const directData = scene.splatBuffer.directCompressedTextureData || scene.splatBuffer.directASTCData;
            if (!directData) return false;

            const covariances = this.splatDataTextures.baseData.covariances;
            const scales = this.splatDataTextures.baseData.scales;
            const rotations = this.splatDataTextures.baseData.rotations;
            const centers = this.splatDataTextures.baseData.centers;
            const colors = this.splatDataTextures.baseData.colors;
            const compressedTextureUVs = this.splatDataTextures.baseData.compressedTextureUVs;
            const directPositions = directData.positions;
            const directScales = directData.scales;
            const directRotations = directData.rotations;
            const directColors = directData.colors;
            const directCompressedTextureUVs = directData.uvs || directData.astcUVs;
            const scaleConversion = scaleRotationCompressionLevel >= 1 ? toHalfFloat : (v) => v;

            for (let i = fromSplat; i <= toSplat; i++) {
                const centerBase = i * 3;
                const rotationBase = i * 4;
                const uvBase = i * 2;
                const covarianceBase = i * 6;

                centers[centerBase] = directPositions[centerBase];
                centers[centerBase + 1] = directPositions[centerBase + 1];
                centers[centerBase + 2] = directPositions[centerBase + 2];

                scale.set(directScales[centerBase], directScales[centerBase + 1], directScales[centerBase + 2]);
                rotation.set(directRotations[rotationBase + 1],
                             directRotations[rotationBase + 2],
                             directRotations[rotationBase + 3],
                             directRotations[rotationBase]).normalize();
                const flip = rotation.w < 0 ? -1 : 1;

                if (scales) {
                    scales[centerBase] = scaleConversion(scale.x);
                    scales[centerBase + 1] = scaleConversion(scale.y);
                    scales[centerBase + 2] = scaleConversion(scale.z);
                }
                if (rotations) {
                    rotations[rotationBase] = scaleConversion(rotation.x * flip);
                    rotations[rotationBase + 1] = scaleConversion(rotation.y * flip);
                    rotations[rotationBase + 2] = scaleConversion(rotation.z * flip);
                    rotations[rotationBase + 3] = scaleConversion(rotation.w * flip);
                }

                colors[rotationBase] = directColors[rotationBase];
                colors[rotationBase + 1] = directColors[rotationBase + 1];
                colors[rotationBase + 2] = directColors[rotationBase + 2];
                colors[rotationBase + 3] = directColors[rotationBase + 3];

                compressedTextureUVs[uvBase] = directCompressedTextureUVs[uvBase];
                compressedTextureUVs[uvBase + 1] = directCompressedTextureUVs[uvBase + 1];

                writeCovariance(scale, rotation, covariances, covarianceBase, covarianceCompressionLevel);
            }

            return true;
        };

    }();

    fillDirectASTCDataArrays(...args) {
        return this.fillDirectCompressedTextureDataArrays(...args);
    }

    /**
     * Convert splat centers, which are floating point values, to an array of integers and multiply
     * each by 1000. Centers will get transformed as appropriate before conversion to integer.
     * @param {number} start The index at which to start retrieving data
     * @param {number} end The index at which to stop retrieving data
     * @param {boolean} padFour Enforce alignment of 4 by inserting a 1 after every 3 values
     * @return {Int32Array}
     */
    getIntegerCenters(start, end, padFour = false) {
        const splatCount = end - start + 1;
        const floatCenters = new Float32Array(splatCount * 3);
        this.fillSplatDataArrays(null, null, null, floatCenters, null, null, undefined, undefined, undefined, undefined, start);
        let intCenters;
        let componentCount = padFour ? 4 : 3;
        intCenters = new Int32Array(splatCount * componentCount);
        for (let i = 0; i < splatCount; i++) {
            for (let t = 0; t < 3; t++) {
                intCenters[i * componentCount + t] = Math.round(floatCenters[i * 3 + t] * 1000.0);
            }
            if (padFour) intCenters[i * componentCount + 3] = 1000;
        }
        return intCenters;
    }

    /**
     * Returns an array of splat centers, transformed as appropriate, optionally padded.
     * @param {number} start The index at which to start retrieving data
     * @param {number} end The index at which to stop retrieving data
     * @param {boolean} padFour Enforce alignment of 4 by inserting a 1 after every 3 values
     * @return {Float32Array}
     */
    getFloatCenters(start, end, padFour = false) {
        const splatCount = end - start + 1;
        const floatCenters = new Float32Array(splatCount * 3);
        this.fillSplatDataArrays(null, null, null, floatCenters, null, null, undefined, undefined, undefined, undefined, start);
        if (!padFour) return floatCenters;
        let paddedFloatCenters = new Float32Array(splatCount * 4);
        for (let i = 0; i < splatCount; i++) {
            for (let t = 0; t < 3; t++) {
                paddedFloatCenters[i * 4 + t] = floatCenters[i * 3 + t];
            }
            paddedFloatCenters[i * 4 + 3] = 1.0;
        }
        return paddedFloatCenters;
    }

    /**
     * Get the center for a splat, transformed as appropriate.
     * @param {number} globalIndex Global index of splat
     * @param {THREE.Vector3} outCenter THREE.Vector3 instance in which to store splat center
     * @param {boolean} applySceneTransform By default, if the splat mesh is static, the transform of the scene to which the splat at
     *                                      'globalIndex' belongs will be applied to the splat center. If 'applySceneTransform' is true,
     *                                      the scene transform will always be applied and if 'applySceneTransform' is false, the
     *                                      scene transform will never be applied. If undefined, the default behavior will apply.
     */
    getSplatCenter = function() {

        const paramsObj = {};

        return function(globalIndex, outCenter, applySceneTransform) {
            this.getLocalSplatParameters(globalIndex, paramsObj, applySceneTransform);
            paramsObj.splatBuffer.getSplatCenter(paramsObj.localIndex, outCenter, paramsObj.sceneTransform);
        };

    }();

    /**
     * Get the scale and rotation for a splat, transformed as appropriate.
     * @param {number} globalIndex Global index of splat
     * @param {THREE.Vector3} outScale THREE.Vector3 instance in which to store splat scale
     * @param {THREE.Quaternion} outRotation THREE.Quaternion instance in which to store splat rotation
     * @param {boolean} applySceneTransform By default, if the splat mesh is static, the transform of the scene to which the splat at
     *                                      'globalIndex' belongs will be applied to the splat scale and rotation. If
     *                                      'applySceneTransform' is true, the scene transform will always be applied and if
     *                                      'applySceneTransform' is false, the scene transform will never be applied. If undefined,
     *                                      the default behavior will apply.
     */
    getSplatScaleAndRotation = function() {

        const paramsObj = {};
        const scaleOverride = new THREE.Vector3();

        return function(globalIndex, outScale, outRotation, applySceneTransform) {
            this.getLocalSplatParameters(globalIndex, paramsObj, applySceneTransform);
            scaleOverride.x = undefined;
            scaleOverride.y = undefined;
            scaleOverride.z = undefined;
            if (this.splatRenderMode === SplatRenderMode.TwoD) scaleOverride.z = 0;
            paramsObj.splatBuffer.getSplatScaleAndRotation(paramsObj.localIndex, outScale, outRotation,
                                                           paramsObj.sceneTransform, scaleOverride);
        };

    }();

    /**
     * Get the color for a splat.
     * @param {number} globalIndex Global index of splat
     * @param {THREE.Vector4} outColor THREE.Vector4 instance in which to store splat color
     */
    getSplatColor = function() {

        const paramsObj = {};

        return function(globalIndex, outColor) {
            this.getLocalSplatParameters(globalIndex, paramsObj);
            paramsObj.splatBuffer.getSplatColor(paramsObj.localIndex, outColor);
        };

    }();

    /**
     * Store the transform of the scene at 'sceneIndex' in 'outTransform'.
     * @param {number} sceneIndex Index of the desired scene
     * @param {THREE.Matrix4} outTransform Instance of THREE.Matrix4 in which to store the scene's transform
     */
    getSceneTransform(sceneIndex, outTransform) {
        const scene = this.getScene(sceneIndex);
        scene.updateTransform(this.dynamicMode);
        outTransform.copy(scene.transform);
    }

    /**
     * Get the scene at 'sceneIndex'.
     * @param {number} sceneIndex Index of the desired scene
     * @return {SplatScene}
     */
    getScene(sceneIndex) {
        if (sceneIndex < 0 || sceneIndex >= this.scenes.length) {
            throw new Error('SplatMesh::getScene() -> Invalid scene index.');
        }
        return this.scenes[sceneIndex];
    }

    getSceneCount() {
        return this.scenes.length;
    }

    getSplatBufferForSplat(globalIndex) {
        if (this.compactGaussianMode) return this.getScene(0).splatBuffer;
        return this.getScene(this.globalSplatIndexToSceneIndexMap[globalIndex]).splatBuffer;
    }

    getSceneIndexForSplat(globalIndex) {
        if (this.compactGaussianMode) return 0;
        return this.globalSplatIndexToSceneIndexMap[globalIndex];
    }

    getSceneTransformForSplat(globalIndex) {
        if (this.compactGaussianMode) return this.getScene(0).transform;
        return this.getScene(this.globalSplatIndexToSceneIndexMap[globalIndex]).transform;
    }

    getSplatLocalIndex(globalIndex) {
        if (this.compactGaussianMode) return globalIndex;
        return this.globalSplatIndexToLocalSplatIndexMap[globalIndex];
    }

    static getIntegerMatrixArray(matrix) {
        const matrixElements = matrix.elements;
        const intMatrixArray = [];
        for (let i = 0; i < 16; i++) {
            intMatrixArray[i] = Math.round(matrixElements[i] * 1000.0);
        }
        return intMatrixArray;
    }

    computeBoundingBox(applySceneTransforms = false, sceneIndex) {
        if (this.compactGaussianMode) return this.boundingBox.clone();
        let splatCount = this.getSplatCount();
        if (sceneIndex !== undefined && sceneIndex !== null) {
            if (sceneIndex < 0 || sceneIndex >= this.scenes.length) {
                throw new Error('SplatMesh::computeBoundingBox() -> Invalid scene index.');
            }
            splatCount = this.scenes[sceneIndex].splatBuffer.getSplatCount();
        }

        const floatCenters = new Float32Array(splatCount * 3);
        this.fillSplatDataArrays(null, null, null, floatCenters, null, null, applySceneTransforms,
                                 undefined, undefined, undefined, undefined, sceneIndex);

        const min = new THREE.Vector3();
        const max = new THREE.Vector3();
        for (let i = 0; i < splatCount; i++) {
            const offset = i * 3;
            const x = floatCenters[offset];
            const y = floatCenters[offset + 1];
            const z = floatCenters[offset + 2];
            if (i === 0 || x < min.x) min.x = x;
            if (i === 0 || y < min.y) min.y = y;
            if (i === 0 || z < min.z) min.z = z;
            if (i === 0 || x > max.x) max.x = x;
            if (i === 0 || y > max.y) max.y = y;
            if (i === 0 || z > max.z) max.z = z;
        }

        return new THREE.Box3(min, max);
    }
}
