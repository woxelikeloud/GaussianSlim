import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';
import { PlyLoader } from './loaders/ply/PlyLoader.js';
import { SplatLoader } from './loaders/splat/SplatLoader.js';
import { KSplatLoader } from './loaders/ksplat/KSplatLoader.js';
import { GaussianSlimLoader } from './loaders/compactLoader/GaussianSlimLoader.js';
import { SpzLoader } from './loaders/spz/SpzLoader.js';
import { sceneFormatFromPath } from './loaders/Utils.js';
import { LoadingSpinner } from './ui/LoadingSpinner.js';
import { LoadingProgressBar } from './ui/LoadingProgressBar.js';
import { InfoPanel } from './ui/InfoPanel.js';
import { SceneHelper } from './SceneHelper.js';
import { Raycaster } from './raycaster/Raycaster.js';
import { SplatMesh } from './splatmesh/SplatMesh.js';
import { createSortWorker } from './worker/SortWorker.js';
import { Constants } from './Constants.js';
import { getCurrentTime, isIOS, getIOSSemever, clamp } from './Util.js';
import { AbortablePromise, AbortedPromiseError } from './AbortablePromise.js';
import { SceneFormat } from './loaders/SceneFormat.js';
import { WebXRMode } from './webxr/WebXRMode.js';
import { VRButton } from './webxr/VRButton.js';
import { ARButton } from './webxr/ARButton.js';
import { delayedExecute, abortablePromiseWithExtractedComponents } from './Util.js';
import { LoaderStatus } from './loaders/LoaderStatus.js';
import { DirectLoadError } from './loaders/DirectLoadError.js';
import { RenderMode } from './RenderMode.js';
import { LogLevel } from './LogLevel.js';
import { SceneRevealMode } from './SceneRevealMode.js';
import { SplatRenderMode } from './SplatRenderMode.js';

const THREE_CAMERA_FOV = 50;
const MINIMUM_DISTANCE_TO_NEW_FOCAL_POINT = .75;
const MIN_SPLAT_COUNT_TO_SHOW_SPLAT_TREE_LOADING_SPINNER = 1500000;
const FOCUS_MARKER_FADE_IN_SPEED = 10.0;
const FOCUS_MARKER_FADE_OUT_SPEED = 2.5;
const CONSECUTIVE_RENDERED_FRAMES_FOR_FPS_CALCULATION = 60;
const TIMING_REPORT_SCHEMA = 'compactgs.timing.report.v1';
const TIMING_REPORT_TIMEOUT_MS = 120000;
const SCREENSHOT_CAPTURE_TIMEOUT_MS = 15000;
const TIMING_REPORT_TASK_NAMES = [
    'firstSplatFrame',
    'firstFullSort',
    'deferredVisibleRegion',
    'splatTree',
    'processingProfile'
];

const roundTimingMs = (value) => Number.isFinite(value) ? Number(value.toFixed(2)) : undefined;

const createJsonSafeSnapshot = (value) => {
    const ancestors = new WeakSet();
    const maxDepth = 24;
    const maxArrayEntries = 4096;
    const maxObjectEntries = 4096;

    const snapshot = (current, depth) => {
        if (current === null || typeof current === 'string' || typeof current === 'boolean') return current;
        if (typeof current === 'number') {
            if (Number.isFinite(current)) return current;
            if (Number.isNaN(current)) return 'NaN';
            return current > 0 ? 'Infinity' : '-Infinity';
        }
        if (typeof current === 'bigint') return current.toString();
        if (typeof current === 'undefined' || typeof current === 'function' || typeof current === 'symbol') return undefined;
        if (depth >= maxDepth) return {'type': 'Truncated', 'reason': 'maximum depth reached'};

        if (typeof ArrayBuffer !== 'undefined' && current instanceof ArrayBuffer) {
            return {'type': 'ArrayBuffer', 'byteLength': current.byteLength};
        }
        if (typeof ArrayBuffer !== 'undefined' && ArrayBuffer.isView(current)) {
            return {
                'type': current.constructor?.name || 'ArrayBufferView',
                'byteLength': current.byteLength,
                ...(Number.isFinite(current.length) ? {'length': current.length} : {})
            };
        }
        if (typeof WebAssembly !== 'undefined' && WebAssembly.Module && current instanceof WebAssembly.Module) {
            return {'type': 'WebAssembly.Module'};
        }
        if (typeof Blob !== 'undefined' && current instanceof Blob) {
            return {'type': current.constructor?.name || 'Blob', 'size': current.size, 'mimeType': current.type || ''};
        }
        if (current instanceof Date) return Number.isNaN(current.getTime()) ? null : current.toISOString();
        if (current instanceof Error) {
            return {
                'type': current.name || 'Error',
                'message': current.message || String(current)
            };
        }
        if (ancestors.has(current)) return {'type': 'CircularReference'};
        ancestors.add(current);
        try {
            if (Array.isArray(current)) {
                const result = current.slice(0, maxArrayEntries).map((entry) => {
                    const safeEntry = snapshot(entry, depth + 1);
                    return safeEntry === undefined ? null : safeEntry;
                });
                if (current.length > maxArrayEntries) {
                    result.push({'type': 'Truncated', 'omittedEntries': current.length - maxArrayEntries});
                }
                return result;
            }

            const result = {};
            const keys = Object.keys(current);
            for (const key of keys.slice(0, maxObjectEntries)) {
                let safeValue;
                try {
                    safeValue = snapshot(current[key], depth + 1);
                } catch (error) {
                    safeValue = {'type': 'SnapshotError', 'message': error?.message || String(error)};
                }
                if (safeValue !== undefined) result[key] = safeValue;
            }
            if (keys.length > maxObjectEntries) {
                result.__truncated__ = {'omittedEntries': keys.length - maxObjectEntries};
            }
            return result;
        } finally {
            ancestors.delete(current);
        }
    };

    try {
        return snapshot(value, 0);
    } catch (error) {
        return {'type': 'SnapshotError', 'message': error?.message || String(error)};
    }
};

const collectTimingEnvironment = () => {
    const environment = {};
    try {
        if (typeof navigator !== 'undefined') {
            environment.userAgent = navigator.userAgent;
            environment.platform = navigator.platform;
            environment.language = navigator.language;
            environment.hardwareConcurrency = navigator.hardwareConcurrency;
        }
        if (typeof window !== 'undefined') {
            environment.crossOriginIsolated = window.crossOriginIsolated;
            environment.devicePixelRatio = window.devicePixelRatio;
            environment.viewport = {'width': window.innerWidth, 'height': window.innerHeight};
            if (window.screen) {
                environment.screen = {
                    'width': window.screen.width,
                    'height': window.screen.height,
                    'availWidth': window.screen.availWidth,
                    'availHeight': window.screen.availHeight,
                    'colorDepth': window.screen.colorDepth,
                    'pixelDepth': window.screen.pixelDepth
                };
            }
        }
    } catch (_) {
        // Environment diagnostics must never interrupt rendering.
    }
    return environment;
};

const getDebugLoggingEnabled = () => {
    if (typeof window === 'undefined' || typeof URLSearchParams === 'undefined') return false;
    try {
        return new URLSearchParams(window.location?.search || '').has('debug');
    } catch (_) {
        return false;
    }
};

const emitTimingTable = (label, scalarRows, collapsed = false) => {
    if (typeof console === 'undefined') return;

    const rows = scalarRows.filter((row) => Number.isFinite(row.ms)).map((row) => ({
        phase: row.phase,
        ms: roundTimingMs(row.ms),
        ...(row.note ? { note: row.note } : {})
    }));
    let groupOpened = false;
    try {
        const group = collapsed ? console.groupCollapsed : console.group;
        if (typeof group === 'function') {
            group.call(console, label);
            groupOpened = true;
        } else if (typeof console.log === 'function') {
            console.log(label);
        }
        if (rows.length > 0) {
            if (typeof console.table === 'function') console.table(rows);
            else if (typeof console.log === 'function') console.log(rows);
        }
    } catch (_) {
        // Timing diagnostics must never interrupt rendering.
    } finally {
        if (groupOpened && typeof console.groupEnd === 'function') {
            try {
                console.groupEnd();
            } catch (_) {}
        }
    }
};

/**
 * Viewer: Manages the rendering of splat scenes. Manages an instance of SplatMesh as well as a web worker
 * that performs the sort for its splats.
 */
export class Viewer {

    constructor(options = {}) {

        // The natural 'up' vector for viewing the scene (only has an effect when used with orbit controls and
        // when the viewer uses its own camera).
        if (!options.cameraUp) options.cameraUp = [0, 1, 0];
        this.cameraUp = new THREE.Vector3().fromArray(options.cameraUp);

        // The camera's initial position (only used when the viewer uses its own camera).
        if (!options.initialCameraPosition) options.initialCameraPosition = [0, 10, 15];
        this.initialCameraPosition = new THREE.Vector3().fromArray(options.initialCameraPosition);

        // The initial focal point of the camera and center of the camera's orbit (only used when the viewer uses its own camera).
        if (!options.initialCameraLookAt) options.initialCameraLookAt = [0, 0, 0];
        this.initialCameraLookAt = new THREE.Vector3().fromArray(options.initialCameraLookAt);

        // 'dropInMode' is a flag that is used internally to support the usage of the viewer as a Three.js scene object
        this.dropInMode = options.dropInMode || false;

        // If 'selfDrivenMode' is true, the viewer manages its own update/animation loop via requestAnimationFrame()
        if (options.selfDrivenMode === undefined || options.selfDrivenMode === null) options.selfDrivenMode = true;
        this.selfDrivenMode = options.selfDrivenMode && !this.dropInMode;
        this.selfDrivenUpdateFunc = this.selfDrivenUpdate.bind(this);

        // If 'useBuiltInControls' is true, the viewer will create its own instance of OrbitControls and attach to the camera
        if (options.useBuiltInControls === undefined) options.useBuiltInControls = true;
        this.useBuiltInControls = options.useBuiltInControls;

        // parent element of the Three.js renderer canvas
        this.rootElement = options.rootElement;
        this.ownsRootElement = false;

        // Use an explicitly requested pixel ratio when provided.
        if (options.ignoreDevicePixelRatio === undefined || options.ignoreDevicePixelRatio === null) {
            options.ignoreDevicePixelRatio = true;
        }
        this.ignoreDevicePixelRatio = options.ignoreDevicePixelRatio;
        const devicePixelRatio = Number(options.devicePixelRatio);
        this.devicePixelRatio = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0 ?
            devicePixelRatio : (this.ignoreDevicePixelRatio ? 1 : (window.devicePixelRatio || 1));

        // Tells the viewer to use 16-bit floating point values when storing splat covariance data in textures, instead of 32-bit
        if (options.halfPrecisionCovariancesOnGPU === undefined || options.halfPrecisionCovariancesOnGPU === null) {
            options.halfPrecisionCovariancesOnGPU = true;
        }
        this.halfPrecisionCovariancesOnGPU = options.halfPrecisionCovariancesOnGPU;

        // If 'threeScene' is valid, it will be rendered by the viewer along with the splat mesh
        this.threeScene = options.threeScene;
        // Allows for usage of an external Three.js renderer
        this.renderer = options.renderer;
        // Allows for usage of an external Three.js camera
        this.camera = options.camera;

        // If 'gpuAcceleratedSort' is true, a partially GPU-accelerated approach to sorting splats will be used.
        // Currently this means pre-computing splat distances from the camera on the GPU
        this.gpuAcceleratedSort = options.gpuAcceleratedSort || false;
        this.debugRenderTimings = options.debugRenderTimings === true;
        this.debugRenderTimingFrame = 0;
        this.debugRenderTimingLastSortMs = 0;
        this.debugRenderTimingGpuQueries = [];
        this.debugRenderTimingSamples = [];
        this.debugRenderTimingDownloaded = false;

        // if 'integerBasedSort' is true, the integer version of splat centers as well as other values used to calculate
        // splat distances are used instead of the float version. This speeds up computation, but introduces the possibility of
        // overflow in larger scenes.
        if (options.integerBasedSort === undefined || options.integerBasedSort === null) {
            options.integerBasedSort = true;
        }
        this.integerBasedSort = options.integerBasedSort;

        // Retained as a compatibility field. Sorting always uses ordinary ArrayBuffers.
        this.sharedMemoryForWorkers = false;

        // if 'dynamicScene' is true, it tells the viewer to assume scene elements are not stationary or that the number of splats in the
        // scene may change. This prevents optimizations that depend on a static scene from being made. Additionally, if 'dynamicScene' is
        // true it tells the splat mesh to not apply scene tranforms to splat data that is returned by functions like
        // SplatMesh.getSplatCenter() by default.
        this.dynamicScene = !!options.dynamicScene;

        // When true, will perform additional steps during rendering to address artifacts caused by the rendering of gaussians at a
        // substantially different resolution than that at which they were rendered during training. This will only work correctly
        // for models that were trained using a process that utilizes this compensation calculation. For more details:
        // https://github.com/nerfstudio-project/gsplat/pull/117
        // https://github.com/graphdeco-inria/gaussian-splatting/issues/294#issuecomment-1772688093
        this.antialiased = options.antialiased || false;

        // This constant is added to the projected 2D screen-space splat scales
        this.kernel2DSize = (options.kernel2DSize === undefined) ? 0.18 : options.kernel2DSize;

        this.webXRMode = options.webXRMode || WebXRMode.None;
        if (this.webXRMode !== WebXRMode.None) {
            this.gpuAcceleratedSort = false;
        }
        this.webXRActive = false;

        this.webXRSessionInit = options.webXRSessionInit || {};

        // if 'renderMode' is RenderMode.Always, then the viewer will rrender the scene on every update. If it is RenderMode.OnChange,
        // it will only render when something in the scene has changed.
        this.renderMode = options.renderMode || RenderMode.Always;

        // SceneRevealMode.Default results in a nice, slow fade-in effect for progressively loaded scenes,
        // and a fast fade-in for non progressively loaded scenes.
        // SceneRevealMode.Gradual will force a slow fade-in for all scenes.
        // SceneRevealMode.Instant will force all loaded scene data to be immediately visible.
        this.sceneRevealMode = options.sceneRevealMode || SceneRevealMode.Default;

        // Hacky, experimental, non-scientific parameter for tweaking focal length related calculations. For scenes with very
        // small gaussians, small details, and small dimensions -- increasing this value can help improve visual quality.
        this.focalAdjustment = options.focalAdjustment || 1.0;

        // Specify the maximum screen-space splat size, can help deal with large splats that get too unwieldy
        this.maxScreenSpaceSplatSize = options.maxScreenSpaceSplatSize || 324;

        // The verbosity of console logging
        this.logLevel = options.logLevel || LogLevel.None;

        // Degree of spherical harmonics to utilize in rendering splats (assuming the data is present in the splat scene).
        // Valid values are 0 - 2. Default value is 0.
        this.sphericalHarmonicsDegree = options.sphericalHarmonicsDegree || 0;

        // When true, allows for usage of extra properties and attributes during rendering for effects such as opacity adjustment.
        // Default is false for performance reasons. These properties are separate from transform properties (scale, rotation, position)
        // that are enabled by the 'dynamicScene' parameter.
        this.enableOptionalEffects = options.enableOptionalEffects || false;

        // Enable the usage of SIMD WebAssembly instructions for the splat sort
        if (options.enableSIMDInSort === undefined || options.enableSIMDInSort === null) options.enableSIMDInSort = true;
        this.enableSIMDInSort = options.enableSIMDInSort;

        // Level to compress non KSPLAT files when loading them for direct rendering
        if (options.inMemoryCompressionLevel === undefined || options.inMemoryCompressionLevel === null) {
            options.inMemoryCompressionLevel = 0;
        }
        this.inMemoryCompressionLevel = options.inMemoryCompressionLevel;

        // Reorder splat data in memory after loading is complete to optimize cache utilization. Default is true.
        // Does not apply if splat scene is progressively loaded.
        if (options.optimizeSplatData === undefined || options.optimizeSplatData === null) {
            options.optimizeSplatData = true;
        }
        this.optimizeSplatData = options.optimizeSplatData;

        // When true, the intermediate splat data that is the result of decompressing splat bufffer(s) and is used to
        // populate the data textures will be freed. This will reduces memory usage, but if that data needs to be modified
        // it will need to be re-populated from the splat buffer(s). Default is false.
        if (options.freeIntermediateSplatData === undefined || options.freeIntermediateSplatData === null) {
            options.freeIntermediateSplatData = false;
        }
        this.freeIntermediateSplatData = options.freeIntermediateSplatData;

        // It appears that for certain iOS versions, special actions need to be taken with the
        // usage of SIMD instructions
        if (isIOS()) {
            const semver = getIOSSemever();
            if (semver.major < 17) {
                this.enableSIMDInSort = false;
            }
        }

        // Tell the viewer how to render the splats
        if (options.splatRenderMode === undefined || options.splatRenderMode === null) {
            options.splatRenderMode = SplatRenderMode.ThreeD;
        }
        this.splatRenderMode = options.splatRenderMode;

        // Customize the speed at which the scene is revealed
        this.sceneFadeInRateMultiplier = options.sceneFadeInRateMultiplier || 1.0;

        // Set the range for the depth map for the counting sort used to sort the splats
        this.splatSortDistanceMapPrecision = options.splatSortDistanceMapPrecision || Constants.DefaultSplatSortDistanceMapPrecision;
        const maxPrecision = this.integerBasedSort ? 20 : 24;
        this.splatSortDistanceMapPrecision = clamp(this.splatSortDistanceMapPrecision, 10, maxPrecision);

        // Require a larger camera change before re-sorting to reduce worker traffic while moving through dense scenes.
        this.splatSortRotationThreshold = options.splatSortRotationThreshold ?? 0.2;
        this.splatSortPositionThreshold = options.splatSortPositionThreshold ?? 2.0;
        this.enableProgressiveSort = options.enableProgressiveSort ?? false;

        this.onSplatMeshChangedCallback = null;
        this.createSplatMesh();

        this.controls = null;
        this.perspectiveControls = null;
        this.orthographicControls = null;

        this.orthographicCamera = null;
        this.perspectiveCamera = null;

        this.showMeshCursor = false;
        this.showControlPlane = false;
        this.showInfo = !!options.showInfo;

        this.sceneHelper = null;

        this.sortWorker = null;
        this.sortRunning = false;
        this.splatRenderCount = 0;
        this.splatSortCount = 0;
        this.lastSplatSortCount = 0;
        this.identityRenderIndexes = null;
        this.deferredVisibleRegionUpdatePending = false;
        this.sortWorkerIndexesToSort = null;
        this.sortWorkerPrecomputedDistances = null;
        this.sortWorkerTransforms = null;
        this.preSortMessages = [];
        this.runAfterNextSort = [];

        this.selfDrivenModeRunning = false;
        this.splatRenderReady = false;

        this.raycaster = new Raycaster();

        this.infoPanel = null;

        this.startInOrthographicMode = false;

        this.currentFPS = 0;
        this.lastSortTime = 0;
        this.consecutiveRenderFrames = 0;
        this.firstFrameStartTimeMs = options.firstFrameStartTimeMs;
        this.firstFrameElapsedMs = null;
        this.firstFrameLabel = options.firstFrameLabel || 'First splat frame';
        this.firstFrameBreakdown = options.firstFrameBreakdown || null;
        this.timingReportSession = null;
        this.pendingScreenshotCapture = null;

        this.previousCameraTarget = new THREE.Vector3();
        this.nextCameraTarget = new THREE.Vector3();

        this.mousePosition = new THREE.Vector2();
        this.mouseDownPosition = new THREE.Vector2();
        this.mouseDownTime = null;

        this.resizeObserver = null;
        this.resizeRendererToObservedDimensions = null;
        this.mouseMoveListener = null;
        this.mouseDownListener = null;

        this.debugRenderTimingCpuStart = 0;
        this.mouseUpListener = null;
        this.keyDownListener = null;

        this.sortPromise = null;
        this.sortPromiseResolver = null;
        this.activeSortTimingCallbacks = null;
        this.lastProcessingProfile = null;
        this.splatSceneDownloadPromises = {};
        this.splatSceneDownloadAndBuildPromise = null;
        this.splatSceneRemovalPromise = null;

        this.loadingSpinner = new LoadingSpinner(null, this.rootElement || document.body);
        this.loadingSpinner.hide();
        this.loadingProgressBar = new LoadingProgressBar(this.rootElement || document.body);
        this.loadingProgressBar.hide();
        this.infoPanel = new InfoPanel(this.rootElement || document.body, () => this.captureScreenshot());
        this.infoPanel.setFirstFrameTime(this.firstFrameLabel, this.firstFrameElapsedMs);
        if (this.showInfo) {
            this.infoPanel.show();
        } else {
            this.infoPanel.hide();
        }
        if (options.timingReport !== undefined && options.timingReport !== null) {
            this.createTimingReportSession(options.timingReport);
        }

        this.usingExternalCamera = (this.dropInMode || this.camera) ? true : false;
        this.usingExternalRenderer = (this.dropInMode || this.renderer) ? true : false;

        this.initialized = false;
        this.disposing = false;
        this.disposed = false;
        this.disposePromise = null;
        if (!this.dropInMode) this.init();
    }

    createSplatMesh() {
        this.splatMesh = new SplatMesh(this.splatRenderMode, this.dynamicScene, this.enableOptionalEffects,
                                       this.halfPrecisionCovariancesOnGPU, this.devicePixelRatio, this.gpuAcceleratedSort,
                                       this.integerBasedSort, this.antialiased, this.maxScreenSpaceSplatSize, this.logLevel,
                                       this.sphericalHarmonicsDegree, this.sceneFadeInRateMultiplier, this.kernel2DSize);
        this.splatMesh.frustumCulled = false;
        if (this.onSplatMeshChangedCallback) this.onSplatMeshChangedCallback();
    }

    createTimingReportSession(timingReportContext) {
        const startedAt = Number.isFinite(this.firstFrameStartTimeMs) ? this.firstFrameStartTimeMs : performance.now();
        const context = createJsonSafeSnapshot(timingReportContext) || {};
        const session = {
            startedAt,
            context,
            tasks: {},
            taskPromises: [],
            warnings: [],
            buildAttached: false,
            finalized: false,
            firstFrameMetrics: null,
            processingProfile: null,
            splatBufferTimings: []
        };

        for (const taskName of TIMING_REPORT_TASK_NAMES) {
            let resolveTask;
            const promise = new Promise((resolve) => {
                resolveTask = resolve;
            });
            session.tasks[taskName] = {
                name: taskName,
                status: 'pending',
                startedElapsedMs: performance.now() - startedAt,
                resolve: resolveTask
            };
            session.taskPromises.push(promise);
        }

        if (!context.warmupDiagnostics) {
            session.warnings.push('Warmup diagnostics were unavailable when the Viewer timing session was created.');
        }

        this.timingReportSession = session;
        this.infoPanel.setTimingReportPending();
        session.timeoutId = window.setTimeout(() => {
            for (const taskName of TIMING_REPORT_TASK_NAMES) {
                if (session.tasks[taskName].status === 'pending') {
                    this.settleTimingReportTask(session, taskName, 'timed_out', 'Timing collection exceeded 120 seconds.');
                }
            }
        }, TIMING_REPORT_TIMEOUT_MS);
        Promise.all(session.taskPromises).then(() => this.finalizeTimingReportSession(session));
    }

    settleTimingReportTask(session, taskName, status, detail = '') {
        if (!session || session !== this.timingReportSession || session.finalized) return false;
        const task = session.tasks[taskName];
        if (!task || task.status !== 'pending') return false;

        const endedElapsedMs = performance.now() - session.startedAt;
        task.status = status;
        task.endedElapsedMs = endedElapsedMs;
        task.durationMs = endedElapsedMs - task.startedElapsedMs;
        if (detail) task.detail = detail;
        const resolveTask = task.resolve;
        delete task.resolve;
        resolveTask();
        return true;
    }

    cancelTimingReportSession(reason) {
        const session = this.timingReportSession;
        if (!session || session.finalized) return;
        for (const taskName of TIMING_REPORT_TASK_NAMES) {
            this.settleTimingReportTask(session, taskName, 'canceled', reason || 'Viewer disposed during timing collection.');
        }
    }

    finalizeTimingReportSession(session) {
        if (!session || session !== this.timingReportSession || session.finalized) return;
        session.finalized = true;
        if (session.timeoutId !== undefined) window.clearTimeout(session.timeoutId);

        const tasks = {};
        let complete = session.warnings.length === 0;
        for (const taskName of TIMING_REPORT_TASK_NAMES) {
            const task = session.tasks[taskName];
            tasks[taskName] = createJsonSafeSnapshot(task);
            if (task.status !== 'complete' && task.status !== 'skipped') {
                complete = false;
                session.warnings.push(`${taskName}: ${task.status}${task.detail ? ` (${task.detail})` : ''}`);
            }
        }

        const context = session.context || {};
        const input = context.input || null;
        const report = {
            'schema': TIMING_REPORT_SCHEMA,
            'generatedAt': new Date().toISOString(),
            'status': complete ? 'complete' : 'partial',
            'completionPolicy': {
                'timeoutMs': TIMING_REPORT_TIMEOUT_MS,
                'requiredTasks': TIMING_REPORT_TASK_NAMES.slice(),
                'successfulTaskStatuses': ['complete', 'skipped'],
                'elapsedOrigin': 'View click / firstFrameStartTimeMs'
            },
            'tasks': tasks,
            'environment': createJsonSafeSnapshot(collectTimingEnvironment()),
            'input': createJsonSafeSnapshot(input),
            'config': createJsonSafeSnapshot(context.config || null),
            'warmupDiagnostics': createJsonSafeSnapshot(context.warmupDiagnostics || null),
            'loading': {
                'firstFrameMetrics': createJsonSafeSnapshot(session.firstFrameMetrics),
                'rawBreakdown': createJsonSafeSnapshot(this.firstFrameBreakdown)
            },
            'splatBuffer': {
                'compactgsLoadTimings': createJsonSafeSnapshot(session.splatBufferTimings[0] || null),
                'allCompressedLoadTimings': createJsonSafeSnapshot(session.splatBufferTimings)
            },
            'processingProfile': createJsonSafeSnapshot(session.processingProfile || this.lastProcessingProfile),
            'validation': {
                'warnings': createJsonSafeSnapshot(session.warnings)
            }
        };

        let serializedReport;
        try {
            serializedReport = JSON.stringify(report, null, 2);
        } catch (error) {
            report.status = 'partial';
            report.validation.warnings.push(`JSON serialization fallback: ${error?.message || String(error)}`);
            serializedReport = JSON.stringify(createJsonSafeSnapshot(report), null, 2);
        }

        const inputName = typeof input?.name === 'string' ? input.name.replace(/\.[^.]*$/, '') : 'scene';
        const timestamp = report.generatedAt.replace(/[:.]/g, '-');
        this.infoPanel.setTimingReportReady(serializedReport, `${inputName}-${timestamp}-timing.json`, report.status === 'complete');
    }

    logProcessingProfile(processingProfile) {
        if (!processingProfile) return;

        const splatCount = processingProfile.splatCount ?? processingProfile.meshSplatCount ?? 0;
        emitTimingTable(`[Viewer Timing] Mesh / sort processing (${splatCount} splats)`, [
            { phase: 'Total', ms: processingProfile.totalMs },
            { phase: 'Mesh build', ms: processingProfile.meshBuildTotalMs },
            { phase: 'GPU data refresh', ms: processingProfile.meshRefreshGpuDataMs },
            { phase: 'Data textures', ms: processingProfile.meshRefreshDataTexturesTotalMs },
            { phase: 'Sort data preparation', ms: processingProfile.meshSortDataPrepMs },
            { phase: 'Sort-worker setup', ms: processingProfile.sortWorkerSetupMs },
            { phase: 'First-sort dispatch', ms: processingProfile.firstSortDispatchMs },
            { phase: 'First-sort end-to-end', ms: processingProfile.firstSortEndToEndMs }
        ]);

        if (!getDebugLoggingEnabled()) return;

        emitTimingTable('[Viewer Timing] Detailed processing phases', [
            { phase: 'total', ms: processingProfile.totalMs },
            { phase: 'delayedExecuteWait', ms: processingProfile.delayedExecuteWaitMs },
            { phase: 'meshBuildTotal', ms: processingProfile.meshBuildTotalMs },
            { phase: 'meshRefreshGpuData', ms: processingProfile.meshRefreshGpuDataMs },
            { phase: 'meshRefreshDataTextures', ms: processingProfile.meshRefreshDataTexturesTotalMs },
            { phase: 'meshSetupDataTextures', ms: processingProfile.meshSetupDataTexturesMs },
            { phase: 'meshFillBaseArrays', ms: processingProfile.meshFillBaseArraysMs },
            { phase: 'meshFillAstcUvs', ms: processingProfile.meshFillAstcUvsMs },
            { phase: 'meshUpdateDataTextures', ms: processingProfile.meshUpdateDataTexturesMs },
            { phase: 'meshUpdateCenterColorsTexture', ms: processingProfile.meshUpdateCenterColorsTextureMs },
            { phase: 'meshUpdateCovariancesTexture', ms: processingProfile.meshUpdateCovariancesTextureMs },
            { phase: 'meshUpdateScaleRotationsTexture', ms: processingProfile.meshUpdateScaleRotationsTextureMs },
            { phase: 'meshUpdateAstcUvTexture', ms: processingProfile.meshUpdateAstcUvTextureMs },
            { phase: 'meshUpdateSphericalHarmonicsTexture', ms: processingProfile.meshUpdateSphericalHarmonicsTextureMs },
            { phase: 'meshUpdateSceneIndexesTexture', ms: processingProfile.meshUpdateSceneIndexesTextureMs },
            { phase: 'meshUpdateVisibleRegion', ms: processingProfile.meshUpdateVisibleRegionMs },
            { phase: 'meshPrimeVisibleRegion', ms: processingProfile.meshPrimeVisibleRegionMs },
            { phase: 'meshSortDataPrep', ms: processingProfile.meshSortDataPrepMs },
            { phase: 'meshGpuDistanceBufferUpload', ms: processingProfile.meshGpuDistanceBufferUploadMs },
            { phase: 'sortWorkerSetup', ms: processingProfile.sortWorkerSetupMs },
            { phase: 'firstSortEndToEnd', ms: processingProfile.firstSortEndToEndMs },
            { phase: 'firstSortWorker', ms: processingProfile.firstSortWorkerMs },
            { phase: 'firstSortDispatch', ms: processingProfile.firstSortDispatchMs },
            { phase: 'splatTreeAsync', ms: processingProfile.splatTreeAsyncMs }
        ], true);
        if (typeof console !== 'undefined' && typeof console.log === 'function') {
            try {
                console.log('[Viewer Timing] Raw processing profile', processingProfile);
            } catch (_) {}
        }
    }

    logUserFirstFrameTiming(firstFrameAt) {
        const breakdown = this.firstFrameBreakdown;
        if (!breakdown || breakdown.logged) return;
        breakdown.logged = true;

        const processingProfile = this.lastProcessingProfile;
        const compactgsLoadTimings = breakdown.compactgsLoadTimings || {};
        const workerTimings = compactgsLoadTimings.workerTimings || {};
        const wall = workerTimings.wall || {};
        const prepare = workerTimings.prepare || {};
        const shards = workerTimings.shards || {};
        const loaderRequest = workerTimings.loaderRequest || {};
        const renderReadyAt = processingProfile?.firstRenderReadyAt ?? processingProfile?.completedAt;
        const processingToRenderReadyMs = Number.isFinite(renderReadyAt) && Number.isFinite(processingProfile?.startedAt) ?
            renderReadyAt - processingProfile.startedAt : undefined;
        const renderReadyToFirstFrameMs = Number.isFinite(renderReadyAt) ? firstFrameAt - renderReadyAt : undefined;
        const wasmSubstreamsMs = prepare.decodeSubstreamsMs ?? (
            (prepare.decodeNonVideoSubstreamsMs || 0) +
            (prepare.astcTextureDecodeMs || 0) +
            (prepare.bcTextureEncodeMs || 0) +
            (prepare.webCodecsMs || 0) +
            (prepare.ffmpegFallbackWallMs || 0)
        );
        const substreams = prepare.substreams || {};
        const reconstructionToSplatBufferReadyWallMs =
            Number.isFinite(wall.reconstructionStartAbsMs) &&
            Number.isFinite(compactgsLoadTimings.splatBufferBuildEndAbsMs) &&
            compactgsLoadTimings.splatBufferBuildEndAbsMs >= wall.reconstructionStartAbsMs ?
                compactgsLoadTimings.splatBufferBuildEndAbsMs - wall.reconstructionStartAbsMs : undefined;
        const workerDataAssemblyCandidateMs = Number.isFinite(reconstructionToSplatBufferReadyWallMs) &&
            Number.isFinite(wall.reconstructionWallMs) && Number.isFinite(compactgsLoadTimings.splatBufferBuildMs) ?
                reconstructionToSplatBufferReadyWallMs - wall.reconstructionWallMs -
                    compactgsLoadTimings.splatBufferBuildMs : undefined;
        const workerDataAssemblyMs = Number.isFinite(workerDataAssemblyCandidateMs) &&
            workerDataAssemblyCandidateMs >= 0 ? workerDataAssemblyCandidateMs : undefined;

        const metricRows = [
            { phase: 'totalClickToFirstSplatFrame', ms: this.firstFrameElapsedMs,
              note: 'From clicking View to completion of the first rendered splat frame' },
            { phase: 'clickToFileReadStart', ms: breakdown.fileReadStartAt - breakdown.clickAt,
              note: 'Post-click validation, UI update, and texture strategy readiness until FileReader starts' },
            { phase: 'fileReaderReadAsArrayBuffer', ms: breakdown.fileReadMs,
              note: 'Browser FileReader reads the local file into an ArrayBuffer' },
            { phase: 'workerMessageDispatch', ms: wall.requestToDecodeStartMs,
              note: 'From the main-thread decode postMessage to decoder worker processing' },
            { phase: 'wasmSubstreamsTotal', ms: wasmSubstreamsMs,
              note: 'Total WASM substream decoding; see the substream rows below for details' },
            { phase: 'substreamNonVideo', ms: prepare.decodeNonVideoSubstreamsMs,
              note: 'Non-video substream decoding' },
            { phase: 'substreamAstcTexture', ms: prepare.astcTextureDecodeMs,
              note: 'ASTC texture substream decoding' },
            { phase: 'substreamBcTextureEncode', ms: prepare.bcTextureEncodeMs,
              note: 'BC texture encode' },
            { phase: 'substream0', ms: substreams['0']?.primaryDecodeMs,
              note: 'Substream 0 (non-video), single-stream WASM wall-clock elapsed time' },
            { phase: 'substream1', ms: substreams['1']?.primaryDecodeMs,
              note: 'Substream 1 (non-video), single-stream WASM wall-clock elapsed time' },
            { phase: 'substream2', ms: substreams['2']?.primaryDecodeMs,
              note: `Substream 2 (video), primary decode time on the ` +
                    `${substreams['2']?.actualPath || prepare.videoDecoderPath || 'unknown'} path` },
            { phase: 'substream3', ms: substreams['3']?.primaryDecodeMs,
              note: 'Substream 3 (texture), overall wall-clock elapsed time for ASTC passthrough or BC transcoding' },
            { phase: 'substream4', ms: substreams['4']?.primaryDecodeMs,
              note: 'Substream 4 (non-video), single-stream WASM wall-clock elapsed time' },
            { phase: 'substreamWebCodecsVideo', ms: prepare.webCodecsMs,
              note: 'WebCodecs video substream decode wall-clock elapsed time' },
            { phase: 'substreamWebCodecsPlaneCopy', ms: prepare.webCodecsCopyMs,
              note: 'WebCodecs plane copy' },
            { phase: 'substreamDecodedVideoJsToWasm', ms: prepare.jsToWasmInjectMs,
              note: 'Inject WebCodecs decode results into WASM' },
            { phase: 'substreamFfmpegFallbackVideo', ms: prepare.ffmpegFallbackWallMs,
              note: 'FFmpeg WASM fallback video substream decode wall-clock elapsed time' },
            { phase: 'reconstructionToSplatBufferReadyWall', ms: reconstructionToSplatBufferReadyWallMs,
              note: 'From first reconstruction shard dispatch to main-thread SplatBuffer completion; ' +
                    'continuous cross-thread wall-clock elapsed time' },
            { phase: 'reconstructionWall', ms: wall.reconstructionWallMs,
              note: 'End-to-end wall-clock elapsed time for parallel reconstruction by shard workers' },
            { phase: 'reconstructionStartSinceTraceOrigin', ms: wall.reconstructionStartSinceTraceOriginMs,
              note: 'From Loader parse trace origin to the first reconstruction shard dispatch' },
            { phase: 'reconstructionEndSinceTraceOrigin', ms: wall.reconstructionEndSinceTraceOriginMs,
              note: 'From Loader parse trace origin to final shard merge completion' },
            { phase: 'reconstructionKernelTotalSum', ms: shards.kernelTotalMs,
              note: 'Sum of kernelTotalMs across all shards; not wall-clock elapsed time' },
            { phase: 'workerDataAssembly', ms: workerDataAssemblyMs,
              note: 'Cross-thread interval from reconstruction completion to SplatBuffer start, including compaction, ' +
                    'result assembly, transfer, postMessage, and main-thread receipt' },
            { phase: 'splatBufferBuild', ms: compactgsLoadTimings.splatBufferBuildMs,
              note: 'Build SplatBuffer from worker result data' },
            { phase: 'renderPrepToFirstFrame', ms: (processingToRenderReadyMs || 0) + (renderReadyToFirstFrameMs || 0),
              note: 'From Viewer/SplatMesh rendering preparation to the first rendered frame' },
            { phase: 'processingToRenderReady', ms: processingToRenderReadyMs,
              note: 'SplatMesh, baseData, GPU texture, and index setup until render-ready' },
            { phase: 'renderReadyToFirstFrame', ms: renderReadyToFirstFrameMs,
              note: 'From render-ready to the next WebGL render, which is the first rendered frame' },
            { phase: 'loaderWorkerRoundTrip', ms: compactgsLoadTimings.workerRoundTripMs,
              note: 'From the main thread sending the worker request to receiving decodeResult' },
            { phase: 'postMessageReturn', ms: loaderRequest.postMessageReturnMs,
              note: 'From decoder worker postMessage to main-thread onmessage receipt; estimated with Date.now' }
        ];
        emitTimingTable(`[UserFirstFrameTiming] total=${roundTimingMs(this.firstFrameElapsedMs)} ms`, metricRows);

        const timingSession = this.timingReportSession;
        if (timingSession && !timingSession.finalized) {
            timingSession.firstFrameMetrics = {
                firstFrameAt,
                totalClickToFirstSplatFrameMs: this.firstFrameElapsedMs,
                rows: metricRows.filter((row) => Number.isFinite(row.ms)).map((row) => ({
                    'phase': row.phase,
                    'ms': roundTimingMs(row.ms),
                    ...(row.note ? {'note': row.note} : {})
                }))
            };
        }

        if (typeof console !== 'undefined' && typeof console.log === 'function') {
            try {
                console.log('[UserFirstFrameTiming] raw', {
                    firstFrameAt,
                    fileBytes: breakdown.fileBytes,
                    loader: compactgsLoadTimings,
                    processing: processingProfile
                });
            } catch (_) {}
        }
    }

    ensureIdentityRenderIndexes(splatCount) {
        if (!this.identityRenderIndexes || this.identityRenderIndexes.length < splatCount) {
            const previousLength = this.identityRenderIndexes ? this.identityRenderIndexes.length : 0;
            const nextIdentityIndexes = new Uint32Array(splatCount);
            if (this.identityRenderIndexes) nextIdentityIndexes.set(this.identityRenderIndexes);
            for (let i = previousLength; i < splatCount; i++) {
                nextIdentityIndexes[i] = i;
            }
            this.identityRenderIndexes = nextIdentityIndexes;
        }
        return this.identityRenderIndexes;
    }

    scheduleDeferredVisibleRegionUpdate(timingSession = null) {
        if (this.deferredVisibleRegionUpdatePending) {
            this.settleTimingReportTask(timingSession, 'deferredVisibleRegion', 'skipped',
                                        'A deferred visible-region update was already pending.');
            return;
        }
        if (!this.splatMesh) {
            this.settleTimingReportTask(timingSession, 'deferredVisibleRegion', 'canceled', 'SplatMesh was unavailable.');
            return;
        }
        this.deferredVisibleRegionUpdatePending = true;
        window.setTimeout(() => {
            this.deferredVisibleRegionUpdatePending = false;
            const processingProfile = timingSession?.processingProfile;
            if (!this.splatMesh || !this.initialized || this.isDisposingOrDisposed()) {
                if (processingProfile) processingProfile.deferredVisibleRegionStatus = 'canceled';
                this.settleTimingReportTask(timingSession, 'deferredVisibleRegion', 'canceled',
                                            'Viewer or SplatMesh was disposed before the deferred update.');
                return;
            }
            const updateStartTime = performance.now();
            let elapsedMs;
            try {
                this.splatMesh.updateVisibleRegion(false, SceneRevealMode.Instant);
                elapsedMs = performance.now() - updateStartTime;
                if (processingProfile) {
                    processingProfile.deferredVisibleRegionStatus = 'complete';
                    processingProfile.deferredVisibleRegionMs = elapsedMs;
                }
                this.settleTimingReportTask(timingSession, 'deferredVisibleRegion', 'complete');
            } catch (error) {
                elapsedMs = performance.now() - updateStartTime;
                if (processingProfile) {
                    processingProfile.deferredVisibleRegionStatus = 'failed';
                    processingProfile.deferredVisibleRegionMs = elapsedMs;
                }
                this.settleTimingReportTask(timingSession, 'deferredVisibleRegion', 'failed', error?.message || String(error));
                if (typeof console !== 'undefined' && typeof console.error === 'function') console.error(error);
                return;
            }
            if (getDebugLoggingEnabled() && typeof console !== 'undefined' && typeof console.log === 'function') {
                try {
                    console.log(`[DeferredVisibleRegion] ${elapsedMs.toFixed(2)} ms`);
                } catch (_) {}
            }
            this.forceRenderNextFrame();
        }, 0);
    }

    init() {

        if (this.initialized) return;

        if (!this.rootElement) {
            if (!this.usingExternalRenderer) {
                this.rootElement = document.createElement('div');
                this.rootElement.style.position = 'fixed';
                this.rootElement.style.inset = '0';
                this.rootElement.style.overflow = 'hidden';
                document.body.appendChild(this.rootElement);
                this.ownsRootElement = true;
            } else {
                this.rootElement = this.renderer.domElement || document.body;
            }
        }

        this.setupCamera();
        this.setupRenderer();
        this.setupWebXR(this.webXRSessionInit);
        this.setupControls();
        this.setupEventHandlers();

        this.threeScene = this.threeScene || new THREE.Scene();
        this.sceneHelper = new SceneHelper(this.threeScene);
        this.sceneHelper.setupMeshCursor();
        this.sceneHelper.setupFocusMarker();
        this.sceneHelper.setupControlPlane();

        this.loadingProgressBar.setContainer(this.rootElement);
        this.loadingSpinner.setContainer(this.rootElement);
        this.infoPanel.setContainer(this.rootElement);

        this.initialized = true;
    }

    setupCamera() {
        if (!this.usingExternalCamera) {
            const renderDimensions = new THREE.Vector2();
            this.getRenderDimensions(renderDimensions);

            this.perspectiveCamera = new THREE.PerspectiveCamera(THREE_CAMERA_FOV, renderDimensions.x / renderDimensions.y, 0.1, 1000);
            this.orthographicCamera = new THREE.OrthographicCamera(renderDimensions.x / -2, renderDimensions.x / 2,
                                                                   renderDimensions.y / 2, renderDimensions.y / -2, 0.1, 1000 );
            this.camera = this.startInOrthographicMode ? this.orthographicCamera : this.perspectiveCamera;
            this.camera.position.copy(this.initialCameraPosition);
            this.camera.up.copy(this.cameraUp).normalize();
            this.camera.lookAt(this.initialCameraLookAt);
        }
    }

    setupRenderer() {
        if (!this.usingExternalRenderer) {
            const renderDimensions = new THREE.Vector2();
            const normalizeDimension = (dimension) => Math.max(1, Math.round(Number.isFinite(dimension) ? dimension : 0));
            this.getRenderDimensions(renderDimensions);
            renderDimensions.set(normalizeDimension(renderDimensions.x), normalizeDimension(renderDimensions.y));

            this.renderer = new THREE.WebGLRenderer({
                antialias: false,
                precision: 'highp'
            });
            this.renderer.setPixelRatio(this.devicePixelRatio);
            this.renderer.autoClear = true;
            this.renderer.setClearColor(new THREE.Color( 0x000000 ), 0.0);
            this.renderer.setSize(renderDimensions.x, renderDimensions.y);
            this.initDebugRenderTiming();
            this.renderer.domElement.style.display = 'block';
            this.renderer.domElement.style.width = '100%';
            this.renderer.domElement.style.height = '100%';

            const observedDimensions = renderDimensions.clone();
            const currentRendererDimensions = new THREE.Vector2();
            this.resizeRendererToObservedDimensions = () => {
                if (!this.renderer || this.renderer.xr.isPresenting) return;
                this.renderer.getSize(currentRendererDimensions);
                if (currentRendererDimensions.equals(observedDimensions)) return;

                this.renderer.setSize(observedDimensions.x, observedDimensions.y, false);
                this.forceRenderNextFrame();
            };

            this.resizeObserver = new ResizeObserver((entries) => {
                const entry = entries && entries[0];
                if (!entry || !entry.contentRect) return;

                const width = normalizeDimension(entry.contentRect.width);
                const height = normalizeDimension(entry.contentRect.height);
                observedDimensions.set(width, height);
                this.resizeRendererToObservedDimensions();
            });
            this.rootElement.appendChild(this.renderer.domElement);
            this.resizeObserver.observe(this.rootElement);
        }

    }

    setupWebXR(webXRSessionInit) {
        if (this.webXRMode) {
            if (this.webXRMode === WebXRMode.VR) {
                this.rootElement.appendChild(VRButton.createButton(this.renderer, webXRSessionInit));
            } else if (this.webXRMode === WebXRMode.AR) {
                this.rootElement.appendChild(ARButton.createButton(this.renderer, webXRSessionInit));
            }
            this.renderer.xr.addEventListener('sessionstart', (e) => {
                this.webXRActive = true;
            });
            this.renderer.xr.addEventListener('sessionend', (e) => {
                this.webXRActive = false;
                if (this.resizeRendererToObservedDimensions) {
                    this.resizeRendererToObservedDimensions();
                }
            });
            this.renderer.xr.enabled = true;
            this.camera.position.copy(this.initialCameraPosition);
            this.camera.up.copy(this.cameraUp).normalize();
            this.camera.lookAt(this.initialCameraLookAt);
        }
    }

    setupControls() {
        if (this.useBuiltInControls && this.webXRMode === WebXRMode.None) {
            if (!this.usingExternalCamera) {
                this.perspectiveControls = new OrbitControls(this.perspectiveCamera, this.renderer.domElement);
                this.orthographicControls = new OrbitControls(this.orthographicCamera, this.renderer.domElement);
            } else {
                if (this.camera.isOrthographicCamera) {
                    this.orthographicControls = new OrbitControls(this.camera, this.renderer.domElement);
                } else {
                    this.perspectiveControls = new OrbitControls(this.camera, this.renderer.domElement);
                }
            }
            for (let controls of [this.orthographicControls, this.perspectiveControls,]) {
                if (controls) {
                    controls.listenToKeyEvents(window);
                    controls.rotateSpeed = 0.5;
                    controls.maxPolarAngle = Math.PI * .75;
                    controls.minPolarAngle = 0.1;
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    controls.target.copy(this.initialCameraLookAt);
                    controls.update();
                }
            }
            this.controls = this.camera.isOrthographicCamera ? this.orthographicControls : this.perspectiveControls;
            this.controls.update();
        }
    }

    setupEventHandlers() {
        if (this.useBuiltInControls && this.webXRMode === WebXRMode.None) {
            this.mouseMoveListener = this.onMouseMove.bind(this);
            this.renderer.domElement.addEventListener('pointermove', this.mouseMoveListener, false);
            this.mouseDownListener = this.onMouseDown.bind(this);
            this.renderer.domElement.addEventListener('pointerdown', this.mouseDownListener, false);
            this.mouseUpListener = this.onMouseUp.bind(this);
            this.renderer.domElement.addEventListener('pointerup', this.mouseUpListener, false);
            this.keyDownListener = this.onKeyDown.bind(this);
            window.addEventListener('keydown', this.keyDownListener, false);
        }
    }

    removeEventHandlers() {
        if (this.useBuiltInControls) {
            this.renderer.domElement.removeEventListener('pointermove', this.mouseMoveListener);
            this.mouseMoveListener = null;
            this.renderer.domElement.removeEventListener('pointerdown', this.mouseDownListener);
            this.mouseDownListener = null;
            this.renderer.domElement.removeEventListener('pointerup', this.mouseUpListener);
            this.mouseUpListener = null;
            window.removeEventListener('keydown', this.keyDownListener);
            this.keyDownListener = null;
        }
    }

    setRenderMode(renderMode) {
        this.renderMode = renderMode;
    }

    setActiveSphericalHarmonicsDegrees(activeSphericalHarmonicsDegrees) {
        this.splatMesh.material.uniforms.sphericalHarmonicsDegree.value = activeSphericalHarmonicsDegrees;
        this.splatMesh.material.uniformsNeedUpdate = true;
    }

    onSplatMeshChanged(callback) {
        this.onSplatMeshChangedCallback = callback;
    }

    onKeyDown = function() {

        const forward = new THREE.Vector3();
        const tempMatrixLeft = new THREE.Matrix4();
        const tempMatrixRight = new THREE.Matrix4();

        return function(e) {
            forward.set(0, 0, -1);
            forward.transformDirection(this.camera.matrixWorld);
            tempMatrixLeft.makeRotationAxis(forward, Math.PI / 128);
            tempMatrixRight.makeRotationAxis(forward, -Math.PI / 128);
            switch (e.code) {
                case 'KeyG':
                    this.focalAdjustment += 0.02;
                    this.forceRenderNextFrame();
                break;
                case 'KeyF':
                    this.focalAdjustment -= 0.02;
                    this.forceRenderNextFrame();
                break;
                case 'ArrowLeft':
                    this.camera.up.transformDirection(tempMatrixLeft);
                break;
                case 'ArrowRight':
                    this.camera.up.transformDirection(tempMatrixRight);
                break;
                case 'KeyC':
                    this.showMeshCursor = !this.showMeshCursor;
                break;
                case 'KeyU':
                    this.showControlPlane = !this.showControlPlane;
                break;
                case 'KeyI':
                    this.showInfo = !this.showInfo;
                    if (this.showInfo) {
                        this.infoPanel.show();
                    } else {
                        this.infoPanel.hide();
                    }
                break;
                case 'KeyO':
                    if (!this.usingExternalCamera) {
                        this.setOrthographicMode(!this.camera.isOrthographicCamera);
                    }
                break;
                case 'KeyP':
                    if (!this.usingExternalCamera) {
                        this.splatMesh.setPointCloudModeEnabled(!this.splatMesh.getPointCloudModeEnabled());
                    }
                break;
                case 'Equal':
                    if (!this.usingExternalCamera) {
                        this.splatMesh.setSplatScale(this.splatMesh.getSplatScale() + 0.05);
                    }
                break;
                case 'Minus':
                    if (!this.usingExternalCamera) {
                        this.splatMesh.setSplatScale(Math.max(this.splatMesh.getSplatScale() - 0.05, 0.0));
                    }
                break;
            }
        };

    }();

    setMousePositionFromEvent(mouse) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        if (Number.isFinite(mouse.clientX) && Number.isFinite(mouse.clientY)) {
            this.mousePosition.set(mouse.clientX - rect.left, mouse.clientY - rect.top);
        } else {
            this.mousePosition.set(mouse.offsetX || 0, mouse.offsetY || 0);
        }
    }

    onMouseMove(mouse) {
        this.setMousePositionFromEvent(mouse);
    }

    onMouseDown(mouse) {
        this.setMousePositionFromEvent(mouse);
        this.mouseDownPosition.copy(this.mousePosition);
        this.mouseDownTime = getCurrentTime();
    }

    onMouseUp = function() {

        const clickOffset = new THREE.Vector2();

        return function(mouse) {
            this.setMousePositionFromEvent(mouse);
            clickOffset.copy(this.mousePosition).sub(this.mouseDownPosition);
            const mouseUpTime = getCurrentTime();
            const wasClick = mouseUpTime - this.mouseDownTime < 0.5 && clickOffset.length() < 2;
            if (wasClick) {
                this.onMouseClick(mouse);
            }
        };

    }();

    onMouseClick(mouse) {
        this.setMousePositionFromEvent(mouse);
        this.checkForFocalPointChange();
    }

    checkForFocalPointChange = function() {

        const renderDimensions = new THREE.Vector2();
        const toNewFocalPoint = new THREE.Vector3();
        const outHits = [];

        return function() {
            if (!this.transitioningCameraTarget) {
                this.getRenderDimensions(renderDimensions);
                outHits.length = 0;
                this.raycaster.setFromCameraAndScreenPosition(this.camera, this.mousePosition, renderDimensions);
                this.raycaster.intersectSplatMesh(this.splatMesh, outHits);
                if (outHits.length > 0) {
                    const hit = outHits[0];
                    const intersectionPoint = hit.origin;
                    toNewFocalPoint.copy(intersectionPoint).sub(this.camera.position);
                    if (toNewFocalPoint.length() > MINIMUM_DISTANCE_TO_NEW_FOCAL_POINT) {
                        this.previousCameraTarget.copy(this.controls.target);
                        this.nextCameraTarget.copy(intersectionPoint);
                        this.transitioningCameraTarget = true;
                        this.transitioningCameraTargetStartTime = getCurrentTime();
                    }
                }
            }
        };

    }();

    getRenderDimensions(outDimensions) {
        if (this.renderer && !this.usingExternalRenderer) {
            this.renderer.getSize(outDimensions);
        } else if (this.rootElement) {
            outDimensions.x = this.rootElement.offsetWidth;
            outDimensions.y = this.rootElement.offsetHeight;
        } else {
            this.renderer.getSize(outDimensions);
        }
    }

    setOrthographicMode(orthographicMode) {
        if (orthographicMode === this.camera.isOrthographicCamera) return;
        const fromCamera = this.camera;
        const toCamera = orthographicMode ? this.orthographicCamera : this.perspectiveCamera;
        toCamera.position.copy(fromCamera.position);
        toCamera.up.copy(fromCamera.up);
        toCamera.rotation.copy(fromCamera.rotation);
        toCamera.quaternion.copy(fromCamera.quaternion);
        toCamera.matrix.copy(fromCamera.matrix);
        this.camera = toCamera;

        if (this.controls) {

            const resetControls = (controls) => {
                controls.saveState();
                controls.reset();
            };

            const fromControls = this.controls;
            const toControls = orthographicMode ? this.orthographicControls : this.perspectiveControls;

            resetControls(toControls);
            resetControls(fromControls);

            toControls.target.copy(fromControls.target);
            if (orthographicMode) {
                Viewer.setCameraZoomFromPosition(toCamera, fromCamera, fromControls);
            } else {
                Viewer.setCameraPositionFromZoom(toCamera, fromCamera, toControls);
            }
            this.controls = toControls;
            this.camera.lookAt(this.controls.target);
        }
    }

    static setCameraPositionFromZoom = function() {

        const tempVector = new THREE.Vector3();

        return function(positionCamera, zoomedCamera, controls) {
            const toLookAtDistance = 1 / (zoomedCamera.zoom * 0.001);
            tempVector.copy(controls.target).sub(positionCamera.position).normalize().multiplyScalar(toLookAtDistance).negate();
            positionCamera.position.copy(controls.target).add(tempVector);
        };

    }();


    static setCameraZoomFromPosition = function() {

        const tempVector = new THREE.Vector3();

        return function(zoomCamera, positionZamera, controls) {
            const toLookAtDistance = tempVector.copy(controls.target).sub(positionZamera.position).length();
            zoomCamera.zoom = 1 / (toLookAtDistance * .001);
        };

    }();

    updateSplatMesh = function() {

        const renderDimensions = new THREE.Vector2();

        return function() {
            if (!this.splatMesh) return;
            const splatCount = this.splatMesh.getSplatCount();
            if (splatCount > 0) {
                this.splatMesh.updateVisibleRegionFadeDistance(this.sceneRevealMode);
                this.splatMesh.updateTransforms();
                this.getRenderDimensions(renderDimensions);
                const focalLengthX = this.camera.projectionMatrix.elements[0] * 0.5 *
                                     this.devicePixelRatio * renderDimensions.x;
                const focalLengthY = this.camera.projectionMatrix.elements[5] * 0.5 *
                                     this.devicePixelRatio * renderDimensions.y;

                const focalMultiplier = this.camera.isOrthographicCamera ? (1.0 / this.devicePixelRatio) : 1.0;
                const focalAdjustment = this.focalAdjustment * focalMultiplier;
                const inverseFocalAdjustment = 1.0 / focalAdjustment;

                this.adjustForWebXRStereo(renderDimensions);
                this.splatMesh.updateUniforms(renderDimensions, focalLengthX * focalAdjustment, focalLengthY * focalAdjustment,
                                              this.camera.isOrthographicCamera, this.camera.zoom || 1.0, inverseFocalAdjustment);
            }
        };

    }();

    adjustForWebXRStereo(renderDimensions) {
        // TODO: Figure out a less hacky way to determine if stereo rendering is active
        if (this.camera && this.webXRActive) {
            const xrCamera = this.renderer.xr.getCamera();
            const xrCameraProj00 = xrCamera.projectionMatrix.elements[0];
            const cameraProj00 = this.camera.projectionMatrix.elements[0];
            renderDimensions.x *= (cameraProj00 / xrCameraProj00);
        }
    }

    isLoadingOrUnloading() {
        return Object.keys(this.splatSceneDownloadPromises).length > 0 || this.splatSceneDownloadAndBuildPromise !== null ||
                           this.splatSceneRemovalPromise !== null;
    }

    isDisposingOrDisposed() {
        return this.disposing || this.disposed;
    }

    addSplatSceneDownloadPromise(promise) {
        this.splatSceneDownloadPromises[promise.id] = promise;
    }

    removeSplatSceneDownloadPromise(promise) {
        delete this.splatSceneDownloadPromises[promise.id];
    }

    setSplatSceneDownloadAndBuildPromise(promise) {
        this.splatSceneDownloadAndBuildPromise = promise;
    }

    clearSplatSceneDownloadAndBuildPromise() {
        this.splatSceneDownloadAndBuildPromise = null;
    }

    /**
     * Add a splat scene to the viewer and display any loading UI if appropriate.
     * @param {string} path Path to splat scene to be loaded
     * @param {object} options {
     *
     *         splatAlphaRemovalThreshold: Ignore any splats with an alpha less than the specified
     *                                     value (valid range: 0 - 255), defaults to 1
     *
     *         showLoadingUI:         Display a loading spinner while the scene is loading, defaults to true
     *
     *         position (Array<number>):   Position of the scene, acts as an offset from its default position, defaults to [0, 0, 0]
     *
     *         rotation (Array<number>):   Rotation of the scene represented as a quaternion, defaults to [0, 0, 0, 1]
     *
     *         scale (Array<number>):      Scene's scale, defaults to [1, 1, 1]
     *
     *         onProgress:                 Function to be called as file data are received, or other processing occurs
     *
     *         headers:                    Optional HTTP headers to be sent along with splat requests
     * }
     * @return {AbortablePromise}
     */
    addSplatScene(path, options = {}) {

        if (this.isLoadingOrUnloading()) {
            throw new Error('Cannot add splat scene while another load or unload is already in progress.');
        }

        if (this.isDisposingOrDisposed()) {
            throw new Error('Cannot add splat scene after dispose() is called.');
        }

        if (options.progressiveLoad && this.splatMesh.scenes && this.splatMesh.scenes.length > 0) {
            console.log('addSplatScene(): "progressiveLoad" option ignore because there are multiple splat scenes');
            options.progressiveLoad = false;
        }

        const format = (options.format !== undefined && options.format !== null) ? options.format : sceneFormatFromPath(path);
        const progressiveLoad = Viewer.isProgressivelyLoadable(format) && options.progressiveLoad;
        const showLoadingUI = (options.showLoadingUI !== undefined && options.showLoadingUI !== null) ? options.showLoadingUI : true;

        let loadingUITaskId = null;
        if (showLoadingUI) {
            this.loadingSpinner.removeAllTasks();
            loadingUITaskId = this.loadingSpinner.addTask('Downloading...');
        }
        const hideLoadingUI = () => {
            this.loadingProgressBar.hide();
            this.loadingSpinner.removeAllTasks();
        };

        const onProgressUIUpdate = (percentComplete, percentCompleteLabel, loaderStatus) => {
            if (showLoadingUI) {
                if (loaderStatus === LoaderStatus.Downloading) {
                    if (percentComplete == 100) {
                        this.loadingSpinner.setMessageForTask(loadingUITaskId, 'Download complete!');
                    } else {
                        if (progressiveLoad) {
                            this.loadingSpinner.setMessageForTask(loadingUITaskId, 'Downloading splats...');
                        } else {
                            const suffix = percentCompleteLabel ? `: ${percentCompleteLabel}` : `...`;
                            this.loadingSpinner.setMessageForTask(loadingUITaskId, `Downloading${suffix}`);
                        }
                    }
                } else if (loaderStatus === LoaderStatus.Processing) {
                    this.loadingSpinner.setMessageForTask(loadingUITaskId, 'Processing splats...');
                }
            }
        };

        let downloadDone = false;
        let downloadedPercentage = 0;
        const splatBuffersAddedUIUpdate = (firstBuild, finalBuild) => {
            if (showLoadingUI) {
                if (firstBuild && progressiveLoad || finalBuild && !progressiveLoad) {
                    this.loadingSpinner.removeTask(loadingUITaskId);
                    if (!finalBuild && !downloadDone) this.loadingProgressBar.show();
                }
                if (progressiveLoad) {
                    if (finalBuild) {
                        downloadDone = true;
                        this.loadingProgressBar.hide();
                    } else {
                        this.loadingProgressBar.setProgress(downloadedPercentage);
                    }
                }
            }
        };

        const onProgress = (percentComplete, percentCompleteLabel, loaderStatus) => {
            downloadedPercentage = percentComplete;
            onProgressUIUpdate(percentComplete, percentCompleteLabel, loaderStatus);
            if (options.onProgress) options.onProgress(percentComplete, percentCompleteLabel, loaderStatus);
        };

        const buildSection = (splatBuffer, firstBuild, finalBuild) => {
            if (!progressiveLoad && options.onProgress) options.onProgress(0, '0%', LoaderStatus.Processing);
            const addSplatBufferOptions = {
                'rotation': options.rotation || options.orientation,
                'position': options.position,
                'scale': options.scale,
                'splatAlphaRemovalThreshold': options.splatAlphaRemovalThreshold,
            };
            return this.addSplatBuffers([splatBuffer], [addSplatBufferOptions],
                                         finalBuild, firstBuild && showLoadingUI, showLoadingUI,
                                         progressiveLoad, progressiveLoad).then(() => {
                if (!progressiveLoad && options.onProgress) options.onProgress(100, '100%', LoaderStatus.Processing);
                splatBuffersAddedUIUpdate(firstBuild, finalBuild);
            });
        };

        const loadFunc = progressiveLoad ? this.downloadAndBuildSingleSplatSceneProgressiveLoad.bind(this) :
                                           this.downloadAndBuildSingleSplatSceneStandardLoad.bind(this);
        return loadFunc(path, format, options.splatAlphaRemovalThreshold, buildSection.bind(this),
                        onProgress, hideLoadingUI.bind(this), options.headers);
    }

    /**
     * Download a single splat scene, convert to splat buffer and then rebuild the viewer's splat mesh
     * by calling 'buildFunc' -- all before displaying the scene. Also sets/clears relevant instance synchronization objects,
     * and calls appropriate functions on success or failure.
     * @param {string} path Path to splat scene to be loaded
     * @param {SceneFormat} format Format of the splat scene file
     * @param {number} splatAlphaRemovalThreshold Ignore any splats with an alpha less than the specified value (valid range: 0 - 255)
     * @param {function} buildFunc Function to build the viewer's splat mesh with the downloaded splat buffer
     * @param {function} onProgress Function to be called as file data are received, or other processing occurs
     * @param {function} onException Function to be called when exception occurs
     * @param {object} headers Optional HTTP headers to pass to use for downloading splat scene
     * @return {AbortablePromise}
     */
    downloadAndBuildSingleSplatSceneStandardLoad(path, format, splatAlphaRemovalThreshold, buildFunc, onProgress, onException, headers) {

        const downloadPromise = this.downloadSplatSceneToSplatBuffer(path, splatAlphaRemovalThreshold, onProgress, false,
                                                                     undefined, format, headers);
        const downloadAndBuildPromise = abortablePromiseWithExtractedComponents(downloadPromise.abortHandler);

        downloadPromise.then((splatBuffer) => {
            this.removeSplatSceneDownloadPromise(downloadPromise);
            return buildFunc(splatBuffer, true, true).then(() => {
                downloadAndBuildPromise.resolve();
                this.clearSplatSceneDownloadAndBuildPromise();
            });
        })
        .catch((e) => {
            if (onException) onException();
            this.clearSplatSceneDownloadAndBuildPromise();
            this.removeSplatSceneDownloadPromise(downloadPromise);
            downloadAndBuildPromise.reject(this.updateError(e, `Viewer::addSplatScene -> Could not load file ${path}`));
        });

        this.addSplatSceneDownloadPromise(downloadPromise);
        this.setSplatSceneDownloadAndBuildPromise(downloadAndBuildPromise.promise);

        return downloadAndBuildPromise.promise;
    }

    /**
     * Download a single splat scene and convert to splat buffer in a progressive manner, allowing rendering as the file downloads.
     * As each section is downloaded, the viewer's splat mesh is rebuilt by calling 'buildFunc'
     * Also sets/clears relevant instance synchronization objects, and calls appropriate functions on success or failure.
     * @param {string} path Path to splat scene to be loaded
     * @param {SceneFormat} format Format of the splat scene file
     * @param {number} splatAlphaRemovalThreshold Ignore any splats with an alpha less than the specified value (valid range: 0 - 255)
     * @param {function} buildFunc Function to rebuild the viewer's splat mesh after a new splat buffer section is downloaded
     * @param {function} onDownloadProgress Function to be called as file data are received
     * @param {function} onDownloadException Function to be called when exception occurs at any point during the full download
     * @param {object} headers Optional HTTP headers to pass to use for downloading splat scene
     * @return {AbortablePromise}
     */
    downloadAndBuildSingleSplatSceneProgressiveLoad(path, format, splatAlphaRemovalThreshold, buildFunc,
                                                    onDownloadProgress, onDownloadException, headers) {
        let progressiveLoadedSectionBuildCount = 0;
        let progressiveLoadedSectionBuilding = false;
        const queuedProgressiveLoadSectionBuilds = [];

        const checkAndBuildProgressiveLoadSections = () => {
            if (queuedProgressiveLoadSectionBuilds.length > 0 &&
                !progressiveLoadedSectionBuilding &&
                !this.isDisposingOrDisposed()) {
                progressiveLoadedSectionBuilding = true;
                const queuedBuild = queuedProgressiveLoadSectionBuilds.shift();
                buildFunc(queuedBuild.splatBuffer, queuedBuild.firstBuild, queuedBuild.finalBuild)
                .then(() => {
                    progressiveLoadedSectionBuilding = false;
                    if (queuedBuild.firstBuild) {
                        progressiveLoadFirstSectionBuildPromise.resolve();
                    } else if (queuedBuild.finalBuild) {
                        splatSceneDownloadAndBuildPromise.resolve();
                        this.clearSplatSceneDownloadAndBuildPromise();
                    }
                    if (queuedProgressiveLoadSectionBuilds.length > 0) {
                        delayedExecute(() => checkAndBuildProgressiveLoadSections());
                    }
                });
            }
        };

        const onProgressiveLoadSectionProgress = (splatBuffer, finalBuild) => {
            if (!this.isDisposingOrDisposed()) {
                if (finalBuild || queuedProgressiveLoadSectionBuilds.length === 0 ||
                    splatBuffer.getSplatCount() > queuedProgressiveLoadSectionBuilds[0].splatBuffer.getSplatCount()) {
                    queuedProgressiveLoadSectionBuilds.push({
                        splatBuffer,
                        firstBuild: progressiveLoadedSectionBuildCount === 0,
                        finalBuild
                    });
                    progressiveLoadedSectionBuildCount++;
                    checkAndBuildProgressiveLoadSections();
                }
            }
        };

        const splatSceneDownloadPromise = this.downloadSplatSceneToSplatBuffer(path, splatAlphaRemovalThreshold, onDownloadProgress, true,
                                                                               onProgressiveLoadSectionProgress, format, headers);

        const progressiveLoadFirstSectionBuildPromise = abortablePromiseWithExtractedComponents(splatSceneDownloadPromise.abortHandler);
        const splatSceneDownloadAndBuildPromise = abortablePromiseWithExtractedComponents();

        this.addSplatSceneDownloadPromise(splatSceneDownloadPromise);
        this.setSplatSceneDownloadAndBuildPromise(splatSceneDownloadAndBuildPromise.promise);

        splatSceneDownloadPromise.then(() => {
            this.removeSplatSceneDownloadPromise(splatSceneDownloadPromise);
        })
        .catch((e) => {
            this.clearSplatSceneDownloadAndBuildPromise();
            this.removeSplatSceneDownloadPromise(splatSceneDownloadPromise);
            const error = this.updateError(e, `Viewer::addSplatScene -> Could not load one or more scenes`);
            progressiveLoadFirstSectionBuildPromise.reject(error);
            if (onDownloadException) onDownloadException(error);
        });

        return progressiveLoadFirstSectionBuildPromise.promise;
    }

    /**
     * Add multiple splat scenes to the viewer and display any loading UI if appropriate.
     * @param {Array<object>} sceneOptions Array of per-scene options: {
     *
     *         path: Path to splat scene to be loaded
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
     *         headers:                    Optional HTTP headers to be sent along with splat requests
     *
     *         format (SceneFormat)        Optional, the format of the scene data (.ply, .ksplat, .splat). If not present, the
     *                                     file extension in 'path' will be used to determine the format (if it is present)
     * }
     * @param {boolean} showLoadingUI Display a loading spinner while the scene is loading, defaults to true
     * @param {function} onProgress Function to be called as file data are received
     * @return {AbortablePromise}
     */
    addSplatScenes(sceneOptions, showLoadingUI = true, onProgress = undefined) {

        if (this.isLoadingOrUnloading()) {
            throw new Error('Cannot add splat scene while another load or unload is already in progress.');
        }

        if (this.isDisposingOrDisposed()) {
            throw new Error('Cannot add splat scene after dispose() is called.');
        }

        const fileCount = sceneOptions.length;
        const percentComplete = [];

        let loadingUITaskId;
        if (showLoadingUI) {
            this.loadingSpinner.removeAllTasks();
            loadingUITaskId = this.loadingSpinner.addTask('Downloading...');
        }

        const onLoadProgress = (fileIndex, percent, percentLabel, loaderStatus) => {
            percentComplete[fileIndex] = percent;
            let totalPercent = 0;
            for (let i = 0; i < fileCount; i++) totalPercent += percentComplete[i] || 0;
            totalPercent = totalPercent / fileCount;
            percentLabel = `${totalPercent.toFixed(2)}%`;
            if (showLoadingUI) {
                if (loaderStatus === LoaderStatus.Downloading) {
                    this.loadingSpinner.setMessageForTask(loadingUITaskId, totalPercent == 100 ?
                                                          `Download complete!` : `Downloading: ${percentLabel}`);
                }
            }
            if (onProgress) onProgress(totalPercent, percentLabel, loaderStatus);
        };

        const baseDownloadPromises = [];
        const nativeDownloadPromises = [];
        for (let i = 0; i < sceneOptions.length; i++) {
            const options = sceneOptions[i];
            const format = (options.format !== undefined && options.format !== null) ? options.format : sceneFormatFromPath(options.path);
            const baseDownloadPromise = this.downloadSplatSceneToSplatBuffer(options.path, options.splatAlphaRemovalThreshold,
                                                                             onLoadProgress.bind(this, i), false, undefined,
                                                                             format, options.headers);
            baseDownloadPromises.push(baseDownloadPromise);
            nativeDownloadPromises.push(baseDownloadPromise.promise);
        }

        const downloadAndBuildPromise = new AbortablePromise((resolve, reject) => {
            Promise.all(nativeDownloadPromises)
            .then((splatBuffers) => {
                if (showLoadingUI) this.loadingSpinner.removeTask(loadingUITaskId);
                if (onProgress) onProgress(0, '0%', LoaderStatus.Processing);
                this.addSplatBuffers(splatBuffers, sceneOptions, true, showLoadingUI, showLoadingUI, false, false).then(() => {
                    if (onProgress) onProgress(100, '100%', LoaderStatus.Processing);
                    this.clearSplatSceneDownloadAndBuildPromise();
                    resolve();
                });
            })
            .catch((e) => {
                if (showLoadingUI) this.loadingSpinner.removeTask(loadingUITaskId);
                this.clearSplatSceneDownloadAndBuildPromise();
                reject(this.updateError(e, `Viewer::addSplatScenes -> Could not load one or more splat scenes.`));
            })
            .finally(() => {
                this.removeSplatSceneDownloadPromise(downloadAndBuildPromise);
            });
        }, (reason) => {
            for (let baseDownloadPromise of baseDownloadPromises) {
                baseDownloadPromise.abort(reason);
            }
        });
        this.addSplatSceneDownloadPromise(downloadAndBuildPromise);
        this.setSplatSceneDownloadAndBuildPromise(downloadAndBuildPromise);
        return downloadAndBuildPromise;
    }

    /**
     * Download a splat scene and convert to SplatBuffer instance.
     * @param {string} path Path to splat scene to be loaded
     * @param {number} splatAlphaRemovalThreshold Ignore any splats with an alpha less than the specified
     *                                            value (valid range: 0 - 255), defaults to 1
     *
     * @param {function} onProgress Function to be called as file data are received
     * @param {boolean} progressiveBuild Construct file sections into splat buffers as they are downloaded
     * @param {function} onSectionBuilt Function to be called when new section is added to the file
     * @param {string} format File format of the scene
     * @param {object} headers Optional HTTP headers to pass to use for downloading splat scene
     * @return {AbortablePromise}
     */
    downloadSplatSceneToSplatBuffer(path, splatAlphaRemovalThreshold = 1, onProgress = undefined,
                                    progressiveBuild = false, onSectionBuilt = undefined, format, headers) {
        try {
            if (format === SceneFormat.Splat || format === SceneFormat.KSplat || format === SceneFormat.Ply) {
                const optimizeSplatData = progressiveBuild ? false : this.optimizeSplatData;
                if (format === SceneFormat.Splat) {
                    return SplatLoader.loadFromURL(path, onProgress, progressiveBuild, onSectionBuilt, splatAlphaRemovalThreshold,
                                                   this.inMemoryCompressionLevel, optimizeSplatData, headers);
                } else if (format === SceneFormat.KSplat) {
                    return KSplatLoader.loadFromURL(path, onProgress, progressiveBuild, onSectionBuilt, headers);
                } else if (format === SceneFormat.Ply) {
                    return PlyLoader.loadFromURL(path, onProgress, progressiveBuild, onSectionBuilt, splatAlphaRemovalThreshold,
                                                 this.inMemoryCompressionLevel, optimizeSplatData, this.sphericalHarmonicsDegree, headers);
                } else if (format === SceneFormat.GaussianSlim) {
                    return GaussianSlimLoader.loadFromURL(
                        path,
                        onProgress,
                        progressiveBuild, // GaussianSlimLoader is currently implemented as a non-streaming path.
                        onSectionBuilt,
                        splatAlphaRemovalThreshold,
                        this.inMemoryCompressionLevel,
                        optimizeSplatData,
                        this.sphericalHarmonicsDegree,
                        headers
                    );
                }
            } else if (format === SceneFormat.Spz) {
                return SpzLoader.loadFromURL(path, onProgress, splatAlphaRemovalThreshold, this.inMemoryCompressionLevel,
                                             this.optimizeSplatData, this.sphericalHarmonicsDegree, headers);
            }
        } catch (e) {
            throw this.updateError(e, null);
        }

        throw new Error(`Viewer::downloadSplatSceneToSplatBuffer -> File format not supported: ${path}`);
    }

    static isProgressivelyLoadable(format) {
        return format === SceneFormat.Splat || format === SceneFormat.KSplat || format === SceneFormat.Ply;
    }

    /**
     * Add one or more instances of SplatBuffer to the SplatMesh instance managed by the viewer and set up the sorting web worker.
     * This function will terminate the existing sort worker (if there is one).
     */
    addSplatBuffers = function() {

        return function(splatBuffers, splatBufferOptions = [], finalBuild = true, showLoadingUI = true,
                        showLoadingUIForSplatTreeBuild = true, replaceExisting = false,
                        enableRenderBeforeFirstSort = true, preserveVisibleRegion = true) {

            if (this.isDisposingOrDisposed()) return Promise.resolve();

            const processingProfile = {
                startedAt: performance.now(),
                finalBuild,
                replaceExisting,
                preserveVisibleRegion,
                enableRenderBeforeFirstSort,
                deferVisibleRegion: this.firstFrameElapsedMs === null && finalBuild
            };
            this.lastProcessingProfile = processingProfile;
            const timingSession = this.timingReportSession && !this.timingReportSession.buildAttached ?
                this.timingReportSession : null;
            if (timingSession) {
                timingSession.buildAttached = true;
                timingSession.processingProfile = processingProfile;
                timingSession.splatBufferTimings = splatBuffers.map((splatBuffer) => splatBuffer?.compactgsLoadTimings || null);
            }

            let splatProcessingTaskId = null;
            const removeSplatProcessingTask = () => {
                if (splatProcessingTaskId !== null) {
                    this.loadingSpinner.removeTask(splatProcessingTaskId);
                    splatProcessingTaskId = null;
                }
            };
            const finalizeProcessingProfile = (status = 'complete', detail = '') => {
                if (processingProfile.completedAt !== undefined) return;
                processingProfile.completedAt = performance.now();
                processingProfile.totalMs = processingProfile.completedAt - processingProfile.startedAt;
                this.lastProcessingProfile = processingProfile;
                this.logProcessingProfile(processingProfile);
                this.settleTimingReportTask(timingSession, 'processingProfile', status, detail);
            };
            const markFirstRenderReady = () => {
                if (!Number.isFinite(processingProfile.firstRenderReadyAt)) {
                    processingProfile.firstRenderReadyAt = performance.now();
                    processingProfile.firstRenderReadyMs = processingProfile.firstRenderReadyAt - processingProfile.startedAt;
                }
            };

            this.splatRenderReady = false;
            return new Promise((resolve) => {
                if (showLoadingUI) {
                    splatProcessingTaskId = this.loadingSpinner.addTask('Processing splats...');
                }
                const delayedExecuteStartTime = performance.now();
                delayedExecute(() => {
                    processingProfile.delayedExecuteWaitMs = performance.now() - delayedExecuteStartTime;
                    if (this.isDisposingOrDisposed()) {
                        finalizeProcessingProfile('canceled', 'Viewer was disposed before mesh processing.');
                        resolve();
                    } else {
                        const buildResults = this.addSplatBuffersToMesh(splatBuffers, splatBufferOptions, finalBuild,
                                                                        showLoadingUIForSplatTreeBuild, replaceExisting,
                                                                        preserveVisibleRegion, processingProfile);
                        if (buildResults.processingProfile) {
                            Object.assign(processingProfile, buildResults.processingProfile);
                        }
                        if (timingSession) {
                            if (buildResults.splatTreePromise) {
                                buildResults.splatTreePromise.then(() => {
                                    if (this.isDisposingOrDisposed() || !this.splatMesh?.getSplatTree()) {
                                        this.settleTimingReportTask(timingSession, 'splatTree', 'canceled',
                                                                    'SplatTree build ended after mesh disposal.');
                                    } else {
                                        this.settleTimingReportTask(timingSession, 'splatTree', 'complete');
                                    }
                                    // The first sort may have started before
                                    // the asynchronous tree was ready. Re-sort
                                    // once so subsequent camera changes use the
                                    // culled node list immediately.
                                    if (this.splatMesh?.compactGaussianMode && !this.isDisposingOrDisposed()) {
                                        const resort = () => this.runSplatSort(true, false);
                                        if (this.sortRunning) this.runAfterNextSort.push(resort);
                                        else resort();
                                    }
                                }).catch((error) => {
                                    this.settleTimingReportTask(timingSession, 'splatTree', 'failed',
                                                                error?.message || String(error));
                                });
                            } else {
                                this.settleTimingReportTask(timingSession, 'splatTree', 'skipped',
                                                            finalBuild ? 'No splats required a SplatTree.' : 'Build was not final.');
                            }
                        }
                        if (!timingSession && buildResults.splatTreePromise) {
                            buildResults.splatTreePromise.then(() => {
                                if (this.splatMesh?.compactGaussianMode && !this.isDisposingOrDisposed()) {
                                    const resort = () => this.runSplatSort(true, false);
                                    if (this.sortRunning) this.runAfterNextSort.push(resort);
                                    else resort();
                                }
                            });
                        }

                        const maxSplatCount = this.splatMesh.getMaxSplatCount();
                        processingProfile.splatCount = this.splatMesh.getSplatCount();
                        processingProfile.maxSplatCount = maxSplatCount;
                        const shouldRenderBeforeSortWorkerSetup = this.firstFrameElapsedMs === null &&
                                                                  !this.sortWorker &&
                                                                  maxSplatCount > 0;
                        if (this.sortWorker && this.sortWorker.maxSplatCount !== maxSplatCount) this.disposeSortWorker();
                        // If we aren't calculating the splat distances from the center on the GPU, the sorting worker needs
                        // splat centers and transform indexes so that it can calculate those distance values.
                        if (!this.gpuAcceleratedSort && !buildResults.compactGaussianQuantizedPositions) {
                            this.preSortMessages.push({
                                'centers': buildResults.centers.buffer,
                                'sceneIndexes': buildResults.sceneIndexes.buffer,
                                'range': {
                                    'from': buildResults.from,
                                    'to': buildResults.to,
                                    'count': buildResults.count
                                }
                            });
                        }
                        const sortWorkerSetupStartTime = performance.now();
                        const sortWorkerSetupPromise = (!this.sortWorker && maxSplatCount > 0) ?
                                                         this.setupSortWorker(this.splatMesh) : Promise.resolve();
                        let initialSortWorker = null;
                        let pendingInitialSortProfileCallback = null;
                        let initialSortProfileFinalized = false;
                        const cancelInitialSortProfile = () => {
                            if (initialSortProfileFinalized) return;
                            initialSortProfileFinalized = true;
                            this.settleTimingReportTask(timingSession, 'firstFullSort', 'canceled',
                                                        'The initial full-sort worker was replaced or canceled.');
                            if (pendingInitialSortProfileCallback) {
                                const callbackIndex = this.runAfterNextSort.indexOf(pendingInitialSortProfileCallback);
                                if (callbackIndex >= 0) this.runAfterNextSort.splice(callbackIndex, 1);
                                pendingInitialSortProfileCallback = null;
                            }
                            if (initialSortWorker?._cancelTimingProfile === cancelInitialSortProfile) {
                                initialSortWorker._cancelTimingProfile = null;
                            }
                            processingProfile.firstSortEndToEndMs = 0;
                            processingProfile.firstSortWorkerMs = 0;
                            finalizeProcessingProfile();
                        };
                        if (shouldRenderBeforeSortWorkerSetup) {
                            const identityRenderIndexes = this.ensureIdentityRenderIndexes(processingProfile.splatCount);
                            this.splatMesh.updateRenderIndexes(identityRenderIndexes, processingProfile.splatCount);
                            this.splatRenderCount = processingProfile.splatCount;
                            markFirstRenderReady();
                            this.splatRenderReady = true;
                            removeSplatProcessingTask();
                            resolve();
                        }
                        sortWorkerSetupPromise.then(() => {
                            processingProfile.sortWorkerSetupMs = performance.now() - sortWorkerSetupStartTime;
                            if (shouldRenderBeforeSortWorkerSetup && !initialSortWorker) {
                                initialSortWorker = this.sortWorker;
                                if (initialSortWorker) initialSortWorker._cancelTimingProfile = cancelInitialSortProfile;
                            }
                            if (this.isDisposingOrDisposed()) {
                                if (shouldRenderBeforeSortWorkerSetup) {
                                    this.settleTimingReportTask(timingSession, 'firstFullSort', 'canceled',
                                                                'Viewer was disposed before the first full sort.');
                                    cancelInitialSortProfile();
                                } else {
                                    finalizeProcessingProfile('canceled', 'Viewer was disposed before the first full sort.');
                                    removeSplatProcessingTask();
                                    resolve();
                                }
                                return;
                            }
                            if (shouldRenderBeforeSortWorkerSetup &&
                                (initialSortProfileFinalized || this.sortWorker !== initialSortWorker)) {
                                cancelInitialSortProfile();
                                return;
                            }
                            const firstSortStartTime = performance.now();
                            this.runSplatSort(true, true, () => {
                                this.settleTimingReportTask(timingSession, 'firstFullSort', 'complete');
                            }, () => {
                                this.settleTimingReportTask(timingSession, 'firstFullSort', 'canceled',
                                                            'The initial full sort was canceled.');
                                if (shouldRenderBeforeSortWorkerSetup) {
                                    cancelInitialSortProfile();
                                } else {
                                    removeSplatProcessingTask();
                                    finalizeProcessingProfile('canceled', 'The initial full sort was canceled.');
                                    resolve();
                                }
                            }).then((sortRunning) => {
                                processingProfile.firstSortDispatchMs = performance.now() - firstSortStartTime;
                                if (shouldRenderBeforeSortWorkerSetup) {
                                    if (initialSortProfileFinalized) return;
                                    if (this.isDisposingOrDisposed() || this.sortWorker !== initialSortWorker || !sortRunning) {
                                        this.settleTimingReportTask(timingSession, 'firstFullSort',
                                                                    maxSplatCount > 0 ? 'canceled' : 'skipped',
                                                                    maxSplatCount > 0 ? 'The initial full sort was not dispatched.' :
                                                                        'There were no splats to sort.');
                                        cancelInitialSortProfile();
                                    } else {
                                        const finalizeAfterInitialSort = () => {
                                            if (initialSortProfileFinalized) return;
                                            if (this.sortWorker !== initialSortWorker) {
                                                pendingInitialSortProfileCallback = null;
                                                cancelInitialSortProfile();
                                                return;
                                            }
                                            initialSortProfileFinalized = true;
                                            pendingInitialSortProfileCallback = null;
                                            if (initialSortWorker._cancelTimingProfile === cancelInitialSortProfile) {
                                                initialSortWorker._cancelTimingProfile = null;
                                            }
                                            processingProfile.firstSortEndToEndMs = performance.now() - firstSortStartTime;
                                            processingProfile.firstSortWorkerMs = this.lastSortTime;
                                            this.settleTimingReportTask(timingSession, 'firstFullSort', 'complete');
                                            finalizeProcessingProfile();
                                        };
                                        pendingInitialSortProfileCallback = finalizeAfterInitialSort;
                                        this.runAfterNextSort.push(finalizeAfterInitialSort);
                                    }
                                    return;
                                }
                                if (!this.sortWorker || !sortRunning) {
                                    processingProfile.firstSortEndToEndMs = 0;
                                    processingProfile.firstSortWorkerMs = 0;
                                    markFirstRenderReady();
                                    this.splatRenderReady = true;
                                    removeSplatProcessingTask();
                                    this.settleTimingReportTask(timingSession, 'firstFullSort', 'skipped',
                                                                maxSplatCount > 0 ? 'No sort worker was available.' :
                                                                    'There were no splats to sort.');
                                    if (maxSplatCount <= 0) {
                                        this.settleTimingReportTask(timingSession, 'firstSplatFrame', 'skipped',
                                                                    'There were no splats to render.');
                                        this.settleTimingReportTask(timingSession, 'deferredVisibleRegion', 'skipped',
                                                                    'There were no splats to render.');
                                    }
                                    finalizeProcessingProfile();
                                    resolve();
                                } else {
                                    if (enableRenderBeforeFirstSort) {
                                        markFirstRenderReady();
                                        this.splatRenderReady = true;
                                    } else {
                                        this.runAfterNextSort.push(() => {
                                            markFirstRenderReady();
                                            this.splatRenderReady = true;
                                        });
                                    }
                                    this.runAfterNextSort.push(() => {
                                        processingProfile.firstSortEndToEndMs = performance.now() - firstSortStartTime;
                                        processingProfile.firstSortWorkerMs = this.lastSortTime;
                                        this.settleTimingReportTask(timingSession, 'firstFullSort', 'complete');
                                        removeSplatProcessingTask();
                                        finalizeProcessingProfile();
                                        resolve();
                                    });
                                }
                            });
                        });
                    }
                }, true);
            });
        };

    }();

    /**
     * Add one or more instances of SplatBuffer to the SplatMesh instance managed by the viewer. By default, this function is additive;
     * all splat buffers contained by the viewer's splat mesh before calling this function will be preserved. This behavior can be
     * changed by passing 'true' for 'replaceExisting'.
     * @param {Array<SplatBuffer>} splatBuffers SplatBuffer instances
     * @param {Array<object>} splatBufferOptions Array of options objects: {
     *
     *         splatAlphaRemovalThreshold: Ignore any splats with an alpha less than the specified
     *                                     value (valid range: 0 - 255), defaults to 1
     *
     *         position (Array<number>):   Position of the scene, acts as an offset from its default position, defaults to [0, 0, 0]
     *
     *         rotation (Array<number>):   Rotation of the scene represented as a quaternion, defaults to [0, 0, 0, 1]
     *
     *         scale (Array<number>):      Scene's scale, defaults to [1, 1, 1]
     * }
     * @param {boolean} finalBuild Will the splat mesh be in its final state after this build?
     * @param {boolean} showLoadingUIForSplatTreeBuild Whether or not to show the loading spinner during construction of the splat tree.
     * @return {object} Object containing info about the splats that are updated
     */
    addSplatBuffersToMesh = function() {

        let splatOptimizingTaskId;

        return function(splatBuffers, splatBufferOptions, finalBuild = true, showLoadingUIForSplatTreeBuild = false,
                        replaceExisting = false, preserveVisibleRegion = true, processingProfile = null) {
            if (this.isDisposingOrDisposed()) return;
            let allSplatBuffers = [];
            let allSplatBufferOptions = [];
            if (!replaceExisting) {
                allSplatBuffers = this.splatMesh.scenes.map((scene) => scene.splatBuffer) || [];
                allSplatBufferOptions = this.splatMesh.sceneOptions ? this.splatMesh.sceneOptions.map((sceneOptions) => sceneOptions) : [];
            }
            allSplatBuffers.push(...splatBuffers);
            allSplatBufferOptions.push(...splatBufferOptions);
            if (this.renderer) this.splatMesh.setRenderer(this.renderer);
            const onSplatTreeIndexesUpload = (finished) => {
                if (this.isDisposingOrDisposed()) return;
                const splatCount = this.splatMesh.getSplatCount();
                if (showLoadingUIForSplatTreeBuild && splatCount >= MIN_SPLAT_COUNT_TO_SHOW_SPLAT_TREE_LOADING_SPINNER) {
                    if (!finished && !splatOptimizingTaskId) {
                        this.loadingSpinner.setMinimized(true, true);
                        splatOptimizingTaskId = this.loadingSpinner.addTask('Optimizing data structures...');
                    }
                }
            };
            const onSplatTreeReady = (finished) => {
                if (this.isDisposingOrDisposed()) return;
                if (finished && splatOptimizingTaskId) {
                    this.loadingSpinner.removeTask(splatOptimizingTaskId);
                    splatOptimizingTaskId = null;
                }
            };
            const buildResults = this.splatMesh.build(allSplatBuffers, allSplatBufferOptions, true, finalBuild, onSplatTreeIndexesUpload,
                                                      onSplatTreeReady, preserveVisibleRegion, processingProfile);
            if (finalBuild && this.freeIntermediateSplatData) this.splatMesh.freeIntermediateSplatData();
            return buildResults;
        };

    }();

    /**
     * Set up the splat sorting web worker.
     * @param {SplatMesh} splatMesh SplatMesh instance that contains the splats to be sorted
     * @return {Promise}
     */
    setupSortWorker(splatMesh) {
        if (this.isDisposingOrDisposed()) return;
        return new Promise((resolve) => {
            const DistancesArrayType = this.integerBasedSort ? Int32Array : Float32Array;
            const splatCount = splatMesh.getSplatCount();
            const maxSplatCount = splatMesh.getMaxSplatCount();
            this.sortWorker = createSortWorker(maxSplatCount, this.enableSIMDInSort,
                                               this.integerBasedSort, this.splatMesh.dynamicMode, this.splatSortDistanceMapPrecision,
                                               splatMesh.compactGaussianMode ? splatMesh.compactGaussianQuantizedPositions : null);
            this.sortWorker.onmessage = (e) => {
                if (e.data.sortDone) {
                    this.sortRunning = false;
                    const sortedIndexes = new Uint32Array(e.data.sortedIndexes.buffer, 0, e.data.splatRenderCount);
                    this.splatMesh.updateRenderIndexes(sortedIndexes, e.data.splatRenderCount);

                    this.lastSplatSortCount = this.splatSortCount;

                    this.lastSortTime = e.data.sortTime;
                    const timingCallbacks = this.activeSortTimingCallbacks;
                    this.activeSortTimingCallbacks = null;
                    if (timingCallbacks?.worker === this.sortWorker && timingCallbacks.onComplete) {
                        timingCallbacks.onComplete();
                    }
                    this.sortPromiseResolver();
                    this.sortPromiseResolver = null;
                    this.forceRenderNextFrame();
                    if (this.runAfterNextSort.length > 0) {
                        this.runAfterNextSort.forEach((func) => {
                            func();
                        });
                        this.runAfterNextSort.length = 0;
                    }
                } else if (e.data.sortCanceled) {
                    this.sortRunning = false;
                    const timingCallbacks = this.activeSortTimingCallbacks;
                    this.activeSortTimingCallbacks = null;
                    if (timingCallbacks?.worker === this.sortWorker && timingCallbacks.onCanceled) {
                        timingCallbacks.onCanceled();
                    }
                } else if (e.data.sortSetupPhase1Complete) {
                    if (this.logLevel >= LogLevel.Info) console.log('Sorting web worker WASM setup complete.');
                    this.sortWorkerIndexesToSort = new Uint32Array(maxSplatCount);
                    this.sortWorkerPrecomputedDistances = new DistancesArrayType(maxSplatCount);
                    this.sortWorkerTransforms = new Float32Array(Constants.MaxScenes * 16);
                    for (let i = 0; i < splatCount; i++) this.sortWorkerIndexesToSort[i] = i;
                    this.sortWorker.maxSplatCount = maxSplatCount;

                    if (this.logLevel >= LogLevel.Info) {
                        console.log('Sorting web worker ready.');
                        const splatDataTextures = this.splatMesh.getSplatDataTextures();
                        const covariancesTextureSize = splatDataTextures.covariances.size;
                        const centersColorsTextureSize = splatDataTextures.centerColors.size;
                        console.log('Covariances texture size: ' + covariancesTextureSize.x + ' x ' + covariancesTextureSize.y);
                        console.log('Centers/colors texture size: ' + centersColorsTextureSize.x + ' x ' + centersColorsTextureSize.y);
                    }

                    resolve();
                }
            };
        });
    }

    updateError(error, defaultMessage) {
        if (error instanceof AbortedPromiseError) return error;
        if (error instanceof DirectLoadError) {
            return new Error('File type or server does not support progressive loading.');
        }
        return defaultMessage ? new Error(defaultMessage) : error;
    }

    disposeSortWorker() {
        const timingCallbacks = this.activeSortTimingCallbacks;
        this.activeSortTimingCallbacks = null;
        if (timingCallbacks?.onCanceled) {
            try {
                timingCallbacks.onCanceled();
            } catch (_) {}
        }
        if (this.sortWorker) {
            const cancelTimingProfile = this.sortWorker._cancelTimingProfile;
            this.sortWorker._cancelTimingProfile = null;
            if (typeof cancelTimingProfile === 'function') {
                try {
                    cancelTimingProfile();
                } catch (_) {}
            }
            this.sortWorker.terminate();
        }
        this.sortWorker = null;
        this.sortPromise = null;
        if (this.sortPromiseResolver) {
            this.sortPromiseResolver();
            this.sortPromiseResolver = null;
        }
        this.preSortMessages = [];
        this.sortRunning = false;
    }

    removeSplatScene(indexToRemove, showLoadingUI = true) {
        return this.removeSplatScenes([indexToRemove], showLoadingUI);
    }

    removeSplatScenes(indexesToRemove, showLoadingUI = true) {
        if (this.isLoadingOrUnloading()) {
            throw new Error('Cannot remove splat scene while another load or unload is already in progress.');
        }

        if (this.isDisposingOrDisposed()) {
            throw new Error('Cannot remove splat scene after dispose() is called.');
        }

        let sortPromise;

        this.splatSceneRemovalPromise = new Promise((resolve, reject) => {
            let revmovalTaskId;

            if (showLoadingUI) {
                this.loadingSpinner.removeAllTasks();
                this.loadingSpinner.show();
                revmovalTaskId = this.loadingSpinner.addTask('Removing splat scene...');
            }

            const checkAndHideLoadingUI = () => {
                if (showLoadingUI) {
                    this.loadingSpinner.hide();
                    this.loadingSpinner.removeTask(revmovalTaskId);
                }
            };

            const onDone = (error) => {
                checkAndHideLoadingUI();
                this.splatSceneRemovalPromise = null;
                if (!error) resolve();
                else reject(error);
            };

            const checkForEarlyExit = () => {
                if (this.isDisposingOrDisposed()) {
                    onDone();
                    return true;
                }
                return false;
            };

            sortPromise = this.sortPromise || Promise.resolve();
            sortPromise.then(() => {
                if (checkForEarlyExit()) return;
                const savedSplatBuffers = [];
                const savedSceneOptions = [];
                const savedSceneTransformComponents = [];
                for (let i = 0; i < this.splatMesh.scenes.length; i++) {
                    let shouldRemove = false;
                    for (let indexToRemove of indexesToRemove) {
                        if (indexToRemove === i) {
                            shouldRemove = true;
                            break;
                        }
                    }
                    if (!shouldRemove) {
                        const scene = this.splatMesh.scenes[i];
                        savedSplatBuffers.push(scene.splatBuffer);
                        savedSceneOptions.push(this.splatMesh.sceneOptions[i]);
                        savedSceneTransformComponents.push({
                            'position': scene.position.clone(),
                            'quaternion': scene.quaternion.clone(),
                            'scale': scene.scale.clone()
                        });
                    }
                }
                this.disposeSortWorker();
                this.splatMesh.dispose();
                this.sceneRevealMode = SceneRevealMode.Instant;
                this.createSplatMesh();
                this.addSplatBuffers(savedSplatBuffers, savedSceneOptions, true, false, true)
                .then(() => {
                    if (checkForEarlyExit()) return;
                    checkAndHideLoadingUI();
                    this.splatMesh.scenes.forEach((scene, index) => {
                        scene.position.copy(savedSceneTransformComponents[index].position);
                        scene.quaternion.copy(savedSceneTransformComponents[index].quaternion);
                        scene.scale.copy(savedSceneTransformComponents[index].scale);
                    });
                    this.splatMesh.updateTransforms();
                    this.splatRenderReady = false;

                    this.runSplatSort(true)
                    .then(() => {
                        if (checkForEarlyExit()) {
                            this.splatRenderReady = true;
                            return;
                        }
                        sortPromise = this.sortPromise || Promise.resolve();
                        sortPromise.then(() => {
                            this.splatRenderReady = true;
                            onDone();
                        });
                    });
                })
                .catch((e) => {
                    onDone(e);
                });
            });
        });

        return this.splatSceneRemovalPromise;
    }

    /**
     * Start self-driven mode
     */
    start() {
        if (this.selfDrivenMode) {
            if (this.webXRMode) {
                this.renderer.setAnimationLoop(this.selfDrivenUpdateFunc);
            } else {
                this.requestFrameId = requestAnimationFrame(this.selfDrivenUpdateFunc);
            }
            this.selfDrivenModeRunning = true;
        } else {
            throw new Error('Cannot start viewer unless it is in self driven mode.');
        }
    }

    /**
     * Stop self-driven mode
     */
    stop() {
        if (this.selfDrivenMode && this.selfDrivenModeRunning) {
            if (this.webXRMode) {
                this.renderer.setAnimationLoop(null);
            } else {
                cancelAnimationFrame(this.requestFrameId);
            }
            this.selfDrivenModeRunning = false;
            this.cancelPendingScreenshotCapture('Screenshot capture was canceled because the Viewer render loop stopped.');
        }
    }

    /**
     * Capture the next complete frame rendered to this Viewer's internally-owned canvas, resampled to the
     * canvas's CSS display size at the browser's device pixel ratio and composited over opaque black.
     *
     * @returns {Promise<Blob>} A promise that resolves to a PNG blob.
     */
    captureScreenshot() {
        if (this.pendingScreenshotCapture) return this.pendingScreenshotCapture.promise;

        if (this.dropInMode || this.usingExternalRenderer) {
            return Promise.reject(new Error('Screenshots require a Viewer-owned renderer.'));
        }
        if (this.webXRMode !== WebXRMode.None || this.webXRActive || this.renderer?.xr?.isPresenting) {
            return Promise.reject(new Error('Screenshots are unavailable in WebXR mode.'));
        }
        if (this.isDisposingOrDisposed()) {
            return Promise.reject(new Error('Cannot capture a screenshot while the Viewer is being disposed.'));
        }
        if (!this.initialized || !this.renderer || !this.renderer.domElement) {
            return Promise.reject(new Error('Cannot capture a screenshot before the Viewer is initialized.'));
        }
        if (!this.splatRenderReady) {
            return Promise.reject(new Error('Cannot capture a screenshot before the scene is ready to render.'));
        }
        if (!this.selfDrivenMode || !this.selfDrivenModeRunning || this.renderMode === RenderMode.Never) {
            return Promise.reject(new Error('Cannot capture a screenshot while the Viewer render loop is stopped.'));
        }

        const canvas = this.renderer.domElement;

        let resolveCapture;
        let rejectCapture;
        const promise = new Promise((resolve, reject) => {
            resolveCapture = resolve;
            rejectCapture = reject;
        });
        const capture = {
            promise,
            resolve: resolveCapture,
            reject: rejectCapture,
            timeoutId: undefined,
            encoding: false,
            canvas
        };
        capture.timeoutId = window.setTimeout(() => {
            this.cancelPendingScreenshotCapture('Screenshot capture timed out.');
        }, SCREENSHOT_CAPTURE_TIMEOUT_MS);
        this.pendingScreenshotCapture = capture;
        this.forceRenderNextFrame();
        return promise;
    }

    cancelPendingScreenshotCapture(reason) {
        const capture = this.pendingScreenshotCapture;
        if (!capture) return;

        this.pendingScreenshotCapture = null;
        if (capture.timeoutId !== undefined) window.clearTimeout(capture.timeoutId);
        capture.reject(new Error(reason || 'Screenshot capture was canceled.'));
    }

    capturePendingScreenshotAtRenderTail() {
        const capture = this.pendingScreenshotCapture;
        if (!capture || capture.encoding) return;

        if (!this.renderer || this.renderer.domElement !== capture.canvas || this.isDisposingOrDisposed()) {
            this.cancelPendingScreenshotCapture('Screenshot capture was canceled because the Viewer is unavailable.');
            return;
        }

        capture.encoding = true;
        try {
            if (typeof capture.canvas.getBoundingClientRect !== 'function') {
                throw new Error('The screenshot canvas CSS content area cannot be measured.');
            }
            const rect = capture.canvas.getBoundingClientRect();
            if (!rect || !Number.isFinite(rect.width) || !Number.isFinite(rect.height) || rect.width <= 0 || rect.height <= 0) {
                throw new Error('The screenshot canvas has no measurable CSS content area.');
            }
            if (!Number.isFinite(capture.canvas.width) || !Number.isFinite(capture.canvas.height) ||
                capture.canvas.width <= 0 || capture.canvas.height <= 0) {
                throw new Error('The screenshot source bitmap has no pixels.');
            }

            const browserDevicePixelRatio = Number(window.devicePixelRatio);
            const outputDevicePixelRatio = Number.isFinite(browserDevicePixelRatio) && browserDevicePixelRatio > 0 ?
                browserDevicePixelRatio : 1;
            const targetWidth = Math.max(1, Math.round(rect.width * outputDevicePixelRatio));
            const targetHeight = Math.max(1, Math.round(rect.height * outputDevicePixelRatio));
            if (!Number.isFinite(targetWidth) || !Number.isFinite(targetHeight) || targetWidth <= 0 || targetHeight <= 0) {
                throw new Error('The screenshot output dimensions are unavailable.');
            }

            const ownerDocument = capture.canvas.ownerDocument;
            if (!ownerDocument || typeof ownerDocument.createElement !== 'function') {
                throw new Error('A temporary screenshot canvas could not be created.');
            }
            const outputCanvas = ownerDocument.createElement('canvas');
            outputCanvas.width = targetWidth;
            outputCanvas.height = targetHeight;
            if (outputCanvas.width !== targetWidth || outputCanvas.height !== targetHeight) {
                throw new Error('The screenshot output dimensions are unsupported by this browser.');
            }
            const outputContext = outputCanvas.getContext('2d');
            if (!outputContext) {
                throw new Error('A 2D context for the screenshot could not be created.');
            }
            if (typeof outputCanvas.toBlob !== 'function') {
                throw new Error('PNG screenshot encoding is unavailable in this browser.');
            }

            outputContext.fillStyle = '#000000';
            outputContext.fillRect(0, 0, targetWidth, targetHeight);
            outputContext.imageSmoothingEnabled = true;
            if ('imageSmoothingQuality' in outputContext) outputContext.imageSmoothingQuality = 'high';
            outputContext.drawImage(capture.canvas, 0, 0, capture.canvas.width, capture.canvas.height,
                                    0, 0, targetWidth, targetHeight);

            outputCanvas.toBlob((blob) => {
                if (this.pendingScreenshotCapture !== capture) return;

                this.pendingScreenshotCapture = null;
                if (capture.timeoutId !== undefined) window.clearTimeout(capture.timeoutId);
                if (blob instanceof Blob) {
                    capture.resolve(blob);
                } else {
                    capture.reject(new Error('The browser could not encode the screenshot as PNG.'));
                }
            }, 'image/png');
        } catch (error) {
            if (this.pendingScreenshotCapture !== capture) return;
            this.pendingScreenshotCapture = null;
            if (capture.timeoutId !== undefined) window.clearTimeout(capture.timeoutId);
            const detail = error?.message ? `: ${error.message}` : '';
            capture.reject(new Error(`The browser could not prepare or encode the screenshot as PNG${detail}`));
        }
    }

    /**
     * Dispose of all resources held directly and indirectly by this viewer.
     */
    async dispose() {
        if (this.isDisposingOrDisposed()) return this.disposePromise;

        this.cancelPendingScreenshotCapture('Screenshot capture was canceled because the Viewer is being disposed.');
        this.cancelTimingReportSession('Viewer disposed during timing collection.');

        let waitPromises = [];
        let promisesToAbort = [];
        for (let promiseKey in this.splatSceneDownloadPromises) {
            if (this.splatSceneDownloadPromises.hasOwnProperty(promiseKey)) {
                const downloadPromiseToAbort = this.splatSceneDownloadPromises[promiseKey];
                promisesToAbort.push(downloadPromiseToAbort);
                waitPromises.push(downloadPromiseToAbort.promise);
            }
        }
        if (this.sortPromise) {
            waitPromises.push(this.sortPromise);
        }

        this.disposing = true;
        this.disposePromise = Promise.all(waitPromises).finally(() => {
            this.stop();
            if (this.orthographicControls) {
                this.orthographicControls.dispose();
                this.orthographicControls = null;
            }
            if (this.perspectiveControls) {
                this.perspectiveControls.dispose();
                this.perspectiveControls = null;
            }
            this.controls = null;
            if (this.splatMesh) {
                this.splatMesh.dispose();
                this.splatMesh = null;
            }
            if (this.sceneHelper) {
                this.sceneHelper.dispose();
                this.sceneHelper = null;
            }
            if (this.resizeObserver) {
                this.resizeObserver.disconnect();
                this.resizeObserver = null;
            }
            this.resizeRendererToObservedDimensions = null;
            this.disposeSortWorker();
            this.removeEventHandlers();

            this.loadingSpinner.removeAllTasks();
            this.loadingSpinner.setContainer(null);
            this.loadingProgressBar.hide();
            this.loadingProgressBar.setContainer(null);
            this.infoPanel.setContainer(null);

            this.camera = null;
            this.threeScene = null;
            this.splatRenderReady = false;
            this.initialized = false;
            if (this.renderer) {
                if (!this.usingExternalRenderer) {
                    this.rootElement.removeChild(this.renderer.domElement);
                    this.renderer.dispose();
                }
                this.renderer = null;
            }

            if (this.ownsRootElement && this.rootElement.parentElement) {
                this.rootElement.parentElement.removeChild(this.rootElement);
            }

            this.sortWorkerIndexesToSort = null;
            this.sortWorkerPrecomputedDistances = null;
            this.sortWorkerTransforms = null;
            this.disposed = true;
            this.disposing = false;
            this.disposePromise = null;
        });
        promisesToAbort.forEach((toAbort) => {
            toAbort.abort('Scene disposed');
        });
        return this.disposePromise;
    }

    selfDrivenUpdate() {
        if (this.selfDrivenMode && !this.webXRMode) {
            this.requestFrameId = requestAnimationFrame(this.selfDrivenUpdateFunc);
        }
        this.update();
        if (this.shouldRender()) {
            this.render();
            this.consecutiveRenderFrames++;
        } else {
            this.consecutiveRenderFrames = 0;
        }
        this.renderNextFrame = false;
    }

    initDebugRenderTiming() {
        this.debugRenderTimingGl = null;
        this.debugRenderTimingExt = null;
        if (!this.debugRenderTimings || !this.renderer) return;
        const gl = this.renderer.getContext();
        const ext = gl && gl instanceof WebGL2RenderingContext ? gl.getExtension('EXT_disjoint_timer_query_webgl2') : null;
        if (ext) {
            this.debugRenderTimingGl = gl;
            this.debugRenderTimingExt = ext;
        }
        console.info('[RenderTiming] enabled', { gpuTimer: !!ext });
    }

    beginDebugGpuTimer() {
        const gl = this.debugRenderTimingGl;
        const ext = this.debugRenderTimingExt;
        if (!gl || !ext) return null;
        const query = gl.createQuery();
        gl.beginQuery(ext.TIME_ELAPSED_EXT, query);
        return query;
    }

    endDebugGpuTimer(query) {
        if (!query || !this.debugRenderTimingGl) return;
        const gl = this.debugRenderTimingGl;
        const ext = this.debugRenderTimingExt;
        gl.endQuery(ext.TIME_ELAPSED_EXT);
        gl.flush();
        this.debugRenderTimingGpuQueries.push(query);
    }

    pollDebugGpuTimers() {
        const gl = this.debugRenderTimingGl;
        const ext = this.debugRenderTimingExt;
        if (!gl || !ext) return null;
        if (gl.getParameter(ext.GPU_DISJOINT_EXT)) return null;
        let total = 0;
        const pending = [];
        for (const query of this.debugRenderTimingGpuQueries) {
            const availableEnum = ext.QUERY_RESULT_AVAILABLE_EXT ?? gl.QUERY_RESULT_AVAILABLE;
            const resultEnum = ext.QUERY_RESULT_EXT ?? gl.QUERY_RESULT;
            if (!gl.getQueryParameter(query, availableEnum)) {
                pending.push(query);
                continue;
            }
            total += Number(gl.getQueryParameter(query, resultEnum)) / 1e6;
            gl.deleteQuery(query);
        }
        this.debugRenderTimingGpuQueries = pending;
        return total > 0 ? total : null;
    }

    downloadDebugRenderTimings() {
        if (this.debugRenderTimingDownloaded || this.debugRenderTimingSamples.length === 0) return;
        this.debugRenderTimingDownloaded = true;
        const blob = new Blob([JSON.stringify({
            generatedAt: new Date().toISOString(),
            samples: this.debugRenderTimingSamples
        }, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `render-timing-${Date.now()}.json`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    forceRenderNextFrame() {
        this.renderNextFrame = true;
    }

    shouldRender = function() {

        let renderCount = 0;
        const lastCameraPosition = new THREE.Vector3();
        const lastCameraOrientation = new THREE.Quaternion();
        const changeEpsilon = 0.0001;

        return function() {
            if (!this.initialized || !this.splatRenderReady || this.isDisposingOrDisposed()) return false;

            let shouldRender = false;
            let cameraChanged = false;
            if (this.camera) {
                const cp = this.camera.position;
                const co = this.camera.quaternion;
                cameraChanged = Math.abs(cp.x - lastCameraPosition.x) > changeEpsilon ||
                                Math.abs(cp.y - lastCameraPosition.y) > changeEpsilon ||
                                Math.abs(cp.z - lastCameraPosition.z) > changeEpsilon ||
                                Math.abs(co.x - lastCameraOrientation.x) > changeEpsilon ||
                                Math.abs(co.y - lastCameraOrientation.y) > changeEpsilon ||
                                Math.abs(co.z - lastCameraOrientation.z) > changeEpsilon ||
                                Math.abs(co.w - lastCameraOrientation.w) > changeEpsilon;
            }

            shouldRender = this.renderMode !== RenderMode.Never && (renderCount === 0 || this.splatMesh.visibleRegionChanging ||
                           cameraChanged || this.renderMode === RenderMode.Always || this.dynamicMode === true || this.renderNextFrame);

            if (this.camera) {
                lastCameraPosition.copy(this.camera.position);
                lastCameraOrientation.copy(this.camera.quaternion);
            }

            renderCount++;
            return shouldRender;
        };

    }();

    render = function() {

        return function() {
            if (!this.initialized || !this.splatRenderReady || this.isDisposingOrDisposed()) return;

            const debugTiming = this.debugRenderTimings;
            const cpuStart = debugTiming ? performance.now() : 0;
            const debugGl = debugTiming ? this.debugRenderTimingGl : null;
            if (debugGl) debugGl.finish();
            const gpuWallStart = debugGl ? performance.now() : 0;

            const hasRenderables = (threeScene) => {
                for (let child of threeScene.children) {
                    if (child.visible) return true;
                }
                return false;
            };

            const savedAuoClear = this.renderer.autoClear;
            if (hasRenderables(this.threeScene)) {
                this.renderer.render(this.threeScene, this.camera);
                this.renderer.autoClear = false;
            }
            this.renderer.render(this.splatMesh, this.camera);
            if (debugGl) debugGl.finish();
            const gpuWallMs = debugGl ? performance.now() - gpuWallStart : null;
            if (this.firstFrameElapsedMs === null && this.splatRenderCount > 0) {
                const startTime = Number.isFinite(this.firstFrameStartTimeMs) ? this.firstFrameStartTimeMs : performance.now();
                const firstFrameAt = performance.now();
                this.firstFrameElapsedMs = firstFrameAt - startTime;
                this.infoPanel.setFirstFrameTime(this.firstFrameLabel, this.firstFrameElapsedMs);
                if (typeof console !== 'undefined' && typeof console.log === 'function') {
                    try {
                        console.log(`[Viewer Timing] ${this.firstFrameLabel}: ${roundTimingMs(this.firstFrameElapsedMs)} ms`);
                    } catch (_) {}
                }
                this.logUserFirstFrameTiming(firstFrameAt);
                const timingSession = this.timingReportSession?.buildAttached ? this.timingReportSession : null;
                this.settleTimingReportTask(timingSession, 'firstSplatFrame', 'complete');
                this.scheduleDeferredVisibleRegionUpdate(timingSession);
            }
            this.renderer.autoClear = false;
            if (this.sceneHelper.getFocusMarkerOpacity() > 0.0) this.renderer.render(this.sceneHelper.focusMarker, this.camera);
            if (this.showControlPlane) this.renderer.render(this.sceneHelper.controlPlane, this.camera);
            this.renderer.autoClear = savedAuoClear;
            this.capturePendingScreenshotAtRenderTail();

            if (debugTiming) {
                const cpuMs = performance.now() - cpuStart;
                this.debugRenderTimingFrame++;
                if ((this.debugRenderTimingFrame % 60) === 0) {
                    const sample = {
                        frame: this.debugRenderTimingFrame,
                        cpuFrameMs: Number(cpuMs.toFixed(3)),
                        gpuWallMs: gpuWallMs === null ? null : Number(gpuWallMs.toFixed(3)),
                        lastSortMs: Number(this.lastSortTime.toFixed(3)),
                        renderedSplats: this.splatRenderCount,
                        sortedSplats: this.splatSortCount,
                        drawSize: `${this.renderer.domElement.width}x${this.renderer.domElement.height}`
                    };
                    this.debugRenderTimingSamples.push(sample);
                    if (this.debugRenderTimingSamples.length >= 5) this.downloadDebugRenderTimings();
                }
            }
        };

    }();

    update(renderer, camera) {
        if (this.dropInMode) this.updateForDropInMode(renderer, camera);

        if (!this.initialized || !this.splatRenderReady || this.isDisposingOrDisposed()) return;

        if (this.controls) {
            this.controls.update();
            if (this.camera.isOrthographicCamera && !this.usingExternalCamera) {
                Viewer.setCameraPositionFromZoom(this.camera, this.camera, this.controls);
            }
        }
        this.runSplatSort();
        this.updateForRendererSizeChanges();
        this.updateSplatMesh();
        this.updateMeshCursor();
        this.updateFPS();
        this.timingSensitiveUpdates();
        this.updateInfoPanel();
        this.updateControlPlane();
    }

    updateForDropInMode(renderer, camera) {
        this.renderer = renderer;
        if (this.splatMesh) this.splatMesh.setRenderer(this.renderer);
        this.camera = camera;
        if (this.controls) this.controls.object = camera;
        this.init();
    }

    updateFPS = function() {

        let lastCalcTime = getCurrentTime();
        let frameCount = 0;

        return function() {
            if (this.consecutiveRenderFrames > CONSECUTIVE_RENDERED_FRAMES_FOR_FPS_CALCULATION) {
                const currentTime = getCurrentTime();
                const calcDelta = currentTime - lastCalcTime;
                if (calcDelta >= 1.0) {
                    this.currentFPS = frameCount;
                    frameCount = 0;
                    lastCalcTime = currentTime;
                } else {
                    frameCount++;
                }
            } else {
                this.currentFPS = null;
            }
        };

    }();

    updateForRendererSizeChanges = function() {

        const lastRendererSize = new THREE.Vector2();
        const currentRendererSize = new THREE.Vector2();
        let lastCameraOrthographic;

        return function() {
            if (!this.usingExternalCamera) {
                this.renderer.getSize(currentRendererSize);
                if (lastCameraOrthographic === undefined || lastCameraOrthographic !== this.camera.isOrthographicCamera ||
                    currentRendererSize.x !== lastRendererSize.x || currentRendererSize.y !== lastRendererSize.y) {
                    if (this.camera.isOrthographicCamera) {
                        this.camera.left = -currentRendererSize.x / 2.0;
                        this.camera.right = currentRendererSize.x / 2.0;
                        this.camera.top = currentRendererSize.y / 2.0;
                        this.camera.bottom = -currentRendererSize.y / 2.0;
                    } else {
                        this.camera.aspect = currentRendererSize.x / currentRendererSize.y;
                    }
                    this.camera.updateProjectionMatrix();
                    lastRendererSize.copy(currentRendererSize);
                    lastCameraOrthographic = this.camera.isOrthographicCamera;
                }
            }
        };

    }();

    timingSensitiveUpdates = function() {

        let lastUpdateTime;

        return function() {
            const currentTime = getCurrentTime();
            if (!lastUpdateTime) lastUpdateTime = currentTime;
            const timeDelta = currentTime - lastUpdateTime;

            this.updateCameraTransition(currentTime);
            this.updateFocusMarker(timeDelta);

            lastUpdateTime = currentTime;
        };

    }();

    updateCameraTransition = function() {

        let tempCameraTarget = new THREE.Vector3();
        let toPreviousTarget = new THREE.Vector3();
        let toNextTarget = new THREE.Vector3();

        return function(currentTime) {
            if (this.transitioningCameraTarget) {
                toPreviousTarget.copy(this.previousCameraTarget).sub(this.camera.position).normalize();
                toNextTarget.copy(this.nextCameraTarget).sub(this.camera.position).normalize();
                const rotationAngle = Math.acos(toPreviousTarget.dot(toNextTarget));
                const rotationSpeed = rotationAngle / (Math.PI / 3) * .65 + .3;
                const t = (rotationSpeed / rotationAngle * (currentTime - this.transitioningCameraTargetStartTime));
                tempCameraTarget.copy(this.previousCameraTarget).lerp(this.nextCameraTarget, t);
                this.camera.lookAt(tempCameraTarget);
                this.controls.target.copy(tempCameraTarget);
                if (t >= 1.0) {
                    this.transitioningCameraTarget = false;
                }
            }
        };

    }();

    updateFocusMarker = function() {

        const renderDimensions = new THREE.Vector2();
        let wasTransitioning = false;

        return function(timeDelta) {
            this.getRenderDimensions(renderDimensions);
            if (this.transitioningCameraTarget) {
                this.sceneHelper.setFocusMarkerVisibility(true);
                const currentFocusMarkerOpacity = Math.max(this.sceneHelper.getFocusMarkerOpacity(), 0.0);
                let newFocusMarkerOpacity = Math.min(currentFocusMarkerOpacity + FOCUS_MARKER_FADE_IN_SPEED * timeDelta, 1.0);
                this.sceneHelper.setFocusMarkerOpacity(newFocusMarkerOpacity);
                this.sceneHelper.updateFocusMarker(this.nextCameraTarget, this.camera, renderDimensions);
                wasTransitioning = true;
                this.forceRenderNextFrame();
            } else {
                let currentFocusMarkerOpacity;
                if (wasTransitioning) currentFocusMarkerOpacity = 1.0;
                else currentFocusMarkerOpacity = Math.min(this.sceneHelper.getFocusMarkerOpacity(), 1.0);
                if (currentFocusMarkerOpacity > 0) {
                    this.sceneHelper.updateFocusMarker(this.nextCameraTarget, this.camera, renderDimensions);
                    let newFocusMarkerOpacity = Math.max(currentFocusMarkerOpacity - FOCUS_MARKER_FADE_OUT_SPEED * timeDelta, 0.0);
                    this.sceneHelper.setFocusMarkerOpacity(newFocusMarkerOpacity);
                    if (newFocusMarkerOpacity === 0.0) this.sceneHelper.setFocusMarkerVisibility(false);
                }
                if (currentFocusMarkerOpacity > 0.0) this.forceRenderNextFrame();
                wasTransitioning = false;
            }
        };

    }();

    updateMeshCursor = function() {

        const outHits = [];
        const renderDimensions = new THREE.Vector2();

        return function() {
            if (this.showMeshCursor) {
                this.forceRenderNextFrame();
                this.getRenderDimensions(renderDimensions);
                outHits.length = 0;
                this.raycaster.setFromCameraAndScreenPosition(this.camera, this.mousePosition, renderDimensions);
                this.raycaster.intersectSplatMesh(this.splatMesh, outHits);
                if (outHits.length > 0) {
                    this.sceneHelper.setMeshCursorVisibility(true);
                    this.sceneHelper.positionAndOrientMeshCursor(outHits[0].origin, this.camera);
                } else {
                    this.sceneHelper.setMeshCursorVisibility(false);
                }
            } else {
                if (this.sceneHelper.getMeschCursorVisibility()) this.forceRenderNextFrame();
                this.sceneHelper.setMeshCursorVisibility(false);
            }
        };

    }();

    updateInfoPanel = function() {

        const renderDimensions = new THREE.Vector2();
        const renderResolution = new THREE.Vector2();

        return function() {
            if (!this.showInfo) return;
            const splatCount = this.splatMesh.getSplatCount();
            this.getRenderDimensions(renderDimensions);
            this.renderer.getDrawingBufferSize(renderResolution);
            const cameraLookAtPosition = this.controls ? this.controls.target : null;
            const meshCursorPosition = this.showMeshCursor ? this.sceneHelper.meshCursor.position : null;
            const splatRenderCountPct = splatCount > 0 ? this.splatRenderCount / splatCount * 100 : 0;
            this.infoPanel.update(renderDimensions, renderResolution, this.camera.position, cameraLookAtPosition,
                                  this.camera.up, this.camera.isOrthographicCamera, meshCursorPosition,
                                  this.currentFPS || 'N/A', splatCount, this.splatRenderCount, splatRenderCountPct,
                                  this.lastSortTime, this.focalAdjustment, this.splatMesh.getSplatScale(),
                                  this.splatMesh.getPointCloudModeEnabled());
            this.infoPanel.setFirstFrameTime(this.firstFrameLabel, this.firstFrameElapsedMs);
        };

    }();

    setFirstFrameLabel(label) {
        this.firstFrameLabel = label;
        if (this.infoPanel) {
            this.infoPanel.setFirstFrameTime(this.firstFrameLabel, this.firstFrameElapsedMs);
        }
    }

    updateControlPlane() {
        if (this.showControlPlane) {
            this.sceneHelper.setControlPlaneVisibility(true);
            this.sceneHelper.positionAndOrientControlPlane(this.controls.target, this.camera.up);
        } else {
            this.sceneHelper.setControlPlaneVisibility(false);
        }
    }

    runSplatSort = function() {

        const mvpMatrix = new THREE.Matrix4();
        const cameraPositionArray = [];
        const lastSortViewDir = new THREE.Vector3(0, 0, -1);
        const sortViewDir = new THREE.Vector3(0, 0, -1);
        const lastSortViewPos = new THREE.Vector3();
        const sortViewOffset = new THREE.Vector3();
        const queuedSorts = [];

        const partialSorts = [
            {
                'angleThreshold': 0.55,
                'sortFractions': [0.125, 0.33333, 0.75]
            },
            {
                'angleThreshold': 0.65,
                'sortFractions': [0.33333, 0.66667]
            },
            {
                'angleThreshold': 0.8,
                'sortFractions': [0.5]
            }
        ];

        return function(force = false, forceSortAll = false, onComplete = null, onCanceled = null) {
            if (!this.initialized) return Promise.resolve(false);
            if (this.sortRunning) return Promise.resolve(true);
            if (!this.sortWorker || !this.sortWorkerIndexesToSort) return Promise.resolve(false);
            if (this.splatMesh.getSplatCount() <= 0) {
                this.splatRenderCount = 0;
                return Promise.resolve(false);
            }

            let angleDiff = 0;
            let positionDiff = 0;
            let needsRefreshForRotation = false;
            let needsRefreshForPosition = false;

            sortViewDir.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
            angleDiff = Math.acos(clamp(sortViewDir.dot(lastSortViewDir), -1, 1));
            positionDiff = sortViewOffset.copy(this.camera.position).sub(lastSortViewPos).length();

            if (!force) {
                if (!this.splatMesh.dynamicMode && queuedSorts.length === 0) {
                    if (angleDiff >= this.splatSortRotationThreshold) needsRefreshForRotation = true;
                    if (positionDiff >= this.splatSortPositionThreshold) needsRefreshForPosition = true;
                    if (!needsRefreshForRotation && !needsRefreshForPosition) return Promise.resolve(false);
                }
            }

            this.sortRunning = true;
            let { splatRenderCount, shouldSortAll } = this.gatherSceneNodesForSort();
            shouldSortAll = shouldSortAll || forceSortAll;
            this.splatRenderCount = splatRenderCount;

            mvpMatrix.copy(this.camera.matrixWorld).invert();
            const mvpCamera = this.perspectiveCamera || this.camera;
            mvpMatrix.premultiply(mvpCamera.projectionMatrix);
            if (!this.splatMesh.dynamicMode) mvpMatrix.multiply(this.splatMesh.matrixWorld);

            const useGpuAcceleratedSort = this.gpuAcceleratedSort;
            let gpuAcceleratedSortPromise = Promise.resolve(true);
            if (useGpuAcceleratedSort && (queuedSorts.length <= 1 || queuedSorts.length % 2 === 0)) {
                gpuAcceleratedSortPromise = this.splatMesh.computeDistancesOnGPU(mvpMatrix, this.sortWorkerPrecomputedDistances);
            }

            gpuAcceleratedSortPromise.then(() => {
                if (queuedSorts.length === 0) {
                    if (this.splatMesh.dynamicMode || shouldSortAll || !this.enableProgressiveSort) {
                        queuedSorts.push(this.splatRenderCount);
                    } else {
                        const angleDiffDot = Math.cos(angleDiff);
                        for (let partialSort of partialSorts) {
                            if (angleDiffDot < partialSort.angleThreshold) {
                                for (let sortFraction of partialSort.sortFractions) {
                                    queuedSorts.push(Math.floor(this.splatRenderCount * sortFraction));
                                }
                                break;
                            }
                        }
                        queuedSorts.push(this.splatRenderCount);
                    }
                }
                let sortCount = Math.min(queuedSorts.shift(), this.splatRenderCount);
                this.splatSortCount = sortCount;

                cameraPositionArray[0] = this.camera.position.x;
                cameraPositionArray[1] = this.camera.position.y;
                cameraPositionArray[2] = this.camera.position.z;

                const sortMessage = {
                    'modelViewProj': mvpMatrix.elements,
                    'cameraPosition': cameraPositionArray,
                    'splatRenderCount': this.splatRenderCount,
                    'splatSortCount': sortCount,
                    'usePrecomputedDistances': useGpuAcceleratedSort
                };
                if (this.splatMesh.dynamicMode) {
                    this.splatMesh.fillTransformsArray(this.sortWorkerTransforms);
                }
                sortMessage.indexesToSort = this.sortWorkerIndexesToSort;
                sortMessage.transforms = this.sortWorkerTransforms;
                if (useGpuAcceleratedSort) {
                    sortMessage.precomputedDistances = this.sortWorkerPrecomputedDistances;
                }

                this.sortPromise = new Promise((resolve) => {
                    this.sortPromiseResolver = resolve;
                });
                this.activeSortTimingCallbacks = {
                    'worker': this.sortWorker,
                    'onComplete': typeof onComplete === 'function' ? onComplete : null,
                    'onCanceled': typeof onCanceled === 'function' ? onCanceled : null
                };

                if (this.preSortMessages.length > 0) {
                    this.preSortMessages.forEach((message) => {
                        this.sortWorker.postMessage(message);
                    });
                    this.preSortMessages = [];
                }
                this.sortWorker.postMessage({
                    'sort': sortMessage
                });

                if (queuedSorts.length === 0) {
                    lastSortViewPos.copy(this.camera.position);
                    lastSortViewDir.copy(sortViewDir);
                }

                return true;
            });

            return gpuAcceleratedSortPromise;
        };

    }();

    /**
     * Determine which splats to render by checking which are inside or close to the view frustum
     */
    gatherSceneNodesForSort = function() {

        const nodeRenderList = [];
        let allSplatsSortBuffer = null;
        const tempVectorYZ = new THREE.Vector3();
        const tempVectorXZ = new THREE.Vector3();
        const tempVector = new THREE.Vector3();
        const modelView = new THREE.Matrix4();
        const baseModelView = new THREE.Matrix4();
        const sceneTransform = new THREE.Matrix4();
        const renderDimensions = new THREE.Vector3();
        const forward = new THREE.Vector3(0, 0, -1);

        const tempMax = new THREE.Vector3();
        const nodeSize = (node) => {
            return tempMax.copy(node.max).sub(node.min).length();
        };

        return function(gatherAllNodes = false) {

            this.getRenderDimensions(renderDimensions);
            const cameraFocalLength = (renderDimensions.y / 2.0) / Math.tan(this.camera.fov / 2.0 * THREE.MathUtils.DEG2RAD);
            const fovXOver2 = Math.atan(renderDimensions.x / 2.0 / cameraFocalLength);
            const fovYOver2 = Math.atan(renderDimensions.y / 2.0 / cameraFocalLength);
            const cosFovXOver2 = Math.cos(fovXOver2);
            const cosFovYOver2 = Math.cos(fovYOver2);

            const splatTree = this.splatMesh.getSplatTree();

            if (splatTree) {
                baseModelView.copy(this.camera.matrixWorld).invert();
                if (!this.splatMesh.dynamicMode) baseModelView.multiply(this.splatMesh.matrixWorld);

                let nodeRenderCount = 0;
                let splatRenderCount = 0;

                for (let s = 0; s < splatTree.subTrees.length; s++) {
                    const subTree = splatTree.subTrees[s];
                    modelView.copy(baseModelView);
                    if (this.splatMesh.dynamicMode) {
                        this.splatMesh.getSceneTransform(s, sceneTransform);
                        modelView.multiply(sceneTransform);
                    }
                    const nodeCount = subTree.nodesWithIndexes.length;
                    for (let i = 0; i < nodeCount; i++) {
                        const node = subTree.nodesWithIndexes[i];
                        if (!node.data || !node.data.indexes || node.data.indexes.length === 0) continue;
                        tempVector.copy(node.center).applyMatrix4(modelView);

                        const distanceToNode = tempVector.length();
                        tempVector.normalize();

                        tempVectorYZ.copy(tempVector).setX(0).normalize();
                        tempVectorXZ.copy(tempVector).setY(0).normalize();

                        const cameraAngleXZDot = forward.dot(tempVectorXZ);
                        const cameraAngleYZDot = forward.dot(tempVectorYZ);

                        const ns = nodeSize(node);
                        const outOfFovY = cameraAngleYZDot < (cosFovYOver2 - .6);
                        const outOfFovX = cameraAngleXZDot < (cosFovXOver2 - .6);
                        if (!gatherAllNodes && ((outOfFovX || outOfFovY) && distanceToNode > ns)) {
                            continue;
                        }
                        splatRenderCount += node.data.indexes.length;
                        nodeRenderList[nodeRenderCount] = node;
                        node.data.distanceToNode = distanceToNode;
                        nodeRenderCount++;
                    }
                }

                nodeRenderList.length = nodeRenderCount;
                nodeRenderList.sort((a, b) => {
                    if (a.data.distanceToNode < b.data.distanceToNode) return -1;
                    else return 1;
                });

                let currentByteOffset = splatRenderCount * Constants.BytesPerInt;
                for (let i = 0; i < nodeRenderCount; i++) {
                    const node = nodeRenderList[i];
                    const windowSizeInts = node.data.indexes.length;
                    const windowSizeBytes = windowSizeInts * Constants.BytesPerInt;
                    let destView = new Uint32Array(this.sortWorkerIndexesToSort.buffer,
                                                   currentByteOffset - windowSizeBytes, windowSizeInts);
                    destView.set(node.data.indexes);
                    currentByteOffset -= windowSizeBytes;
                }

                return {
                    'splatRenderCount': splatRenderCount,
                    'shouldSortAll': false
                };
            } else {
                const totalSplatCount = this.splatMesh.getSplatCount();
                if (!allSplatsSortBuffer || allSplatsSortBuffer.length !== totalSplatCount) {
                    allSplatsSortBuffer = new Uint32Array(totalSplatCount);
                    for (let i = 0; i < totalSplatCount; i++) {
                        allSplatsSortBuffer[i] = i;
                    }
                }
                this.sortWorkerIndexesToSort.set(allSplatsSortBuffer);
                return {
                    'splatRenderCount': totalSplatCount,
                    'shouldSortAll': true
                };
            }
        };

    }();

    getSplatMesh() {
        return this.splatMesh;
    }

    /**
     * Get a reference to a splat scene.
     * @param {number} sceneIndex The index of the scene to which the reference will be returned
     * @return {SplatScene}
     */
    getSplatScene(sceneIndex) {
        return this.splatMesh.getScene(sceneIndex);
    }

    getSceneCount() {
        return this.splatMesh.getSceneCount();
    }

    isMobile() {
        return navigator.userAgent.includes('Mobi');
    }
}
