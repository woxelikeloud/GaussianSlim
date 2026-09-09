import createGaussianSlimLoaderWasm from './gaussianslim_wasm.js';
import {
    classifyWebCodecsFastPath,
    createWebCodecsCapabilityKey,
    decodeVideoStreamWithWebCodecs,
    isWebCodecsNegativeCapabilityError,
    videoBufferByteLength
} from './WebCodecsVideoDecoder.js';
import {
    getWebCodecsCapabilityCanary,
    WEB_CODECS_CANARY_DESCRIPTOR
} from './WebCodecsCapabilityCanary.js';
import {
    normalizeTextureStrategyChain,
    normalizeTextureStrategy,
    TextureBuildCapabilities,
    TextureStrategy,
    textureStrategyToNativeMode
} from './TextureStrategy.js';

const DECODER_MODULE_BODY_START_ABS_MS = absoluteNowMs();
const DECODER_CLOCK_METHOD = absoluteClockMethod();

const PROTOCOL_VERSION = 2;
const BUILD_VERSION = 20260906;
const WEB_CODECS_PREFLIGHT_TIMEOUT_MS = 5000;
const WEB_CODECS_PHASE_TIMEOUT_MS = 20000;
const WEB_CODECS_FAST_PATH_SCOPE = 'hevc-main-8bit-420';
const SHARD_WORKER_COUNT = 2;
const GPU_BLOCKS_PER_SHARD = 64;
const CPU_BLOCKS_PER_SHARD = 16;
const MODEL_TIMEOUT_MS = 60000;
const SHARD_TIMEOUT_MS = 20000;
const CONTROL_TIMEOUT_MS = 15000;
const MAX_RECONSTRUCTION_TRACE_TASKS = 512;
const COORDINATOR_WASM_URL = new URL('./gaussianslim_wasm.wasm', import.meta.url).href;
const PTHREAD_COORDINATOR_JS_URL = new URL('./gaussianslim_wasm_pthread.js', import.meta.url);
const PTHREAD_COORDINATOR_WASM_URL = new URL('./gaussianslim_wasm_pthread.wasm', import.meta.url).href;
const RECONSTRUCTION_WASM_URL = new URL('./gaussianslim_reconstruction_wasm.wasm', import.meta.url).href;

let coordinatorModule = null;
let coordinatorModulePromise = null;
let coordinator = null;
let reconstructionModulePreparation = null;
let reconstructionModulePreparationPromise = null;
let shardPool = [];
let decoderSessionPrewarmPromise = null;
let decoderSessionGeneration = 0;
let decoderSessionActive = true;
let activeWarmupProgress = null;
let activePreparation = null;
let activeModel = null;
let nextModelId = 1;
let nextShardStartupAttemptId = 1;
let verboseLog = false;
let coordinatorWasmUrl = COORDINATOR_WASM_URL;
let coordinatorFlavor = 'single-thread';
let coordinatorTextureBuildCapabilities = { ...TextureBuildCapabilities };
let pthreadFallbackLogged = false;
let webCodecsSessionPolicy = null;
let webCodecsSessionPolicyPromise = null;
let webCodecsProbeAbortController = null;
let webCodecsProbeStatus = 'not-started';
let webCodecsProbeStartedAt = 0;
const webCodecsNegativeCapabilities = new Set();

function absoluteNowMs() {
    try {
        const timeOrigin = performance.timeOrigin;
        const now = performance.now();
        const absolute = timeOrigin + now;
        if (Number.isFinite(timeOrigin) && Number.isFinite(now) && Number.isFinite(absolute)) return absolute;
    } catch (_) {}
    const fallback = Date.now();
    return fallback;
}

function absoluteClockMethod() {
    try {
        const timeOrigin = performance.timeOrigin;
        const now = performance.now();
        if (Number.isFinite(timeOrigin) && Number.isFinite(now) && Number.isFinite(timeOrigin + now)) {
            return 'performance.timeOrigin+performance.now';
        }
    } catch (_) {}
    return 'Date.now';
}

function roundTimingMs(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.round(Math.max(0, value) * 100) / 100;
}

function sessionDisposedError() {
    return new Error('GaussianSlimLoader decoder session was disposed.');
}

function isDecoderSessionActive(generation = decoderSessionGeneration) {
    return decoderSessionActive && generation === decoderSessionGeneration;
}

function assertDecoderSessionActive(generation = decoderSessionGeneration) {
    if (!isDecoderSessionActive(generation)) throw sessionDisposedError();
}

function normalizeResourceTiming(timing) {
    if (!timing || typeof timing !== 'object') return null;
    const finiteOrNull = (value) => Number.isFinite(value) ? value : null;
    return {
        url: typeof timing.url === 'string' ? timing.url : '',
        initiatorType: typeof timing.initiatorType === 'string' ? timing.initiatorType : '',
        nextHopProtocol: typeof timing.nextHopProtocol === 'string' ? timing.nextHopProtocol : '',
        timeOriginMs: finiteOrNull(timing.timeOriginMs),
        startTimeAbsMs: finiteOrNull(timing.startTimeAbsMs),
        fetchStartAbsMs: finiteOrNull(timing.fetchStartAbsMs),
        responseStartAbsMs: finiteOrNull(timing.responseStartAbsMs),
        responseEndAbsMs: finiteOrNull(timing.responseEndAbsMs),
        durationMs: finiteOrNull(timing.durationMs),
        transferSize: finiteOrNull(timing.transferSize),
        encodedBodySize: finiteOrNull(timing.encodedBodySize),
        decodedBodySize: finiteOrNull(timing.decodedBodySize)
    };
}

function getResourceTiming(resourceUrl) {
    try {
        if (typeof performance === 'undefined' || typeof performance.getEntriesByName !== 'function') return null;
        const entries = performance.getEntriesByName(resourceUrl, 'resource');
        const entry = entries.length > 0 ? entries[entries.length - 1] : null;
        if (!entry || entry.entryType !== 'resource') return null;
        const timeOriginMs = Number.isFinite(performance.timeOrigin) ? performance.timeOrigin : null;
        const absolute = (value) => Number.isFinite(timeOriginMs) && Number.isFinite(value) ?
            timeOriginMs + value : null;
        return normalizeResourceTiming({
            url: entry.name || resourceUrl,
            initiatorType: entry.initiatorType,
            nextHopProtocol: entry.nextHopProtocol,
            timeOriginMs,
            startTimeAbsMs: absolute(entry.startTime),
            fetchStartAbsMs: absolute(entry.fetchStart),
            responseStartAbsMs: absolute(entry.responseStart),
            responseEndAbsMs: absolute(entry.responseEnd),
            durationMs: entry.duration,
            transferSize: entry.transferSize,
            encodedBodySize: entry.encodedBodySize,
            decodedBodySize: entry.decodedBodySize
        });
    } catch (_) {
        return null;
    }
}

function assertProtocol(message) {
    if (message.protocolVersion !== PROTOCOL_VERSION || message.buildVersion !== BUILD_VERSION) {
        throw new Error(`Worker protocol/build mismatch: ${message.protocolVersion}/${message.buildVersion}`);
    }
}

export function extractCompressedPayload(arrayBuffer) {
    const dataView = new DataView(arrayBuffer);
    const GLB_MAGIC = 0x46546c67;
    const JSON_CHUNK = 0x4e4f534a;
    const BIN_CHUNK = 0x004e4942;
    if (dataView.byteLength < 20 || dataView.getUint32(0, true) !== GLB_MAGIC) {
        return { bytes: new Uint8Array(arrayBuffer), compressedPayload: false, reason: 'raw GSBS payload' };
    }
    if (dataView.getUint32(4, true) !== 2) throw new Error('Only GLB version 2 is supported.');

    let offset = 12;
    let json = null;
    let binOffset = 0;
    let binLength = 0;
    while (offset + 8 <= dataView.byteLength) {
        const chunkLength = dataView.getUint32(offset, true);
        const chunkType = dataView.getUint32(offset + 4, true);
        const chunkOffset = offset + 8;
        if (chunkOffset + chunkLength > dataView.byteLength) throw new Error('Invalid GLB chunk length.');
        if (chunkType === JSON_CHUNK) {
            json = JSON.parse(new TextDecoder().decode(
                new Uint8Array(arrayBuffer, chunkOffset, chunkLength)
            ).trim());
        } else if (chunkType === BIN_CHUNK) {
            binOffset = chunkOffset;
            binLength = chunkLength;
        }
        offset = chunkOffset + chunkLength;
    }
    if (!json || !binOffset || !binLength) throw new Error('GLB is missing its JSON or BIN chunk.');

    const extensionNames = new Set([
        'COMPACTGS_gaussian_splatting_compression_EGSC',
        'COMPACTGS_primitive_3DGS_compression'
    ]);
    let extension = null;
    const stack = [json];
    while (stack.length > 0 && !extension) {
        const current = stack.pop();
        if (!current || typeof current !== 'object') continue;
        for (const [key, value] of Object.entries(current)) {
            if (extensionNames.has(key) && value && typeof value === 'object') {
                extension = value;
                break;
            }
            if (value && typeof value === 'object') stack.push(value);
        }
    }
    const bufferViewIndex = extension?.bufferView ?? extension?.buffer_view ?? extension?.buffer_view_index;
    const bufferView = Number.isInteger(bufferViewIndex) ? json.bufferViews?.[bufferViewIndex] : null;
    if (!bufferView || !Number.isInteger(bufferView.byteLength)) {
        throw new Error('GLB does not contain a supported GaussianSlim compression bufferView.');
    }
    const byteOffset = bufferView.byteOffset || 0;
    if (byteOffset < 0 || bufferView.byteLength <= 0 || byteOffset + bufferView.byteLength > binLength) {
        throw new Error('GaussianSlim compression bufferView is outside the GLB BIN chunk.');
    }
    return {
        bytes: new Uint8Array(arrayBuffer, binOffset + byteOffset, bufferView.byteLength),
        compressedPayload: true,
        reason: `bufferView=${bufferViewIndex}`
    };
}

async function initializeCoordinatorModule(generation) {
    const instantiate = async (factory, wasmUrl, flavor) => {
        const module = await factory({
            locateFile: (path) => path.endsWith('.wasm') ? wasmUrl : path,
            print: (text) => console.log('[GaussianSlimLoader coordinator]', text),
            printErr: (text) => console.error('[GaussianSlimLoader coordinator]', text)
        });
        return { module, wasmUrl, flavor };
    };
    const canUsePthreads = globalThis.crossOriginIsolated === true &&
        typeof globalThis.SharedArrayBuffer === 'function';
    let selected = null;
    if (canUsePthreads) {
        try {
            const pthreadModule = await import(PTHREAD_COORDINATOR_JS_URL.href);
            selected = await instantiate(
                pthreadModule.default, PTHREAD_COORDINATOR_WASM_URL, 'pthread'
            );
        } catch (error) {
            if (!pthreadFallbackLogged) {
                pthreadFallbackLogged = true;
                console.warn('[GaussianSlimLoader coordinator] pthread artifact unavailable; using single-thread fallback.', error);
            }
        }
    }
    if (!selected) {
        selected = await instantiate(
            createGaussianSlimLoaderWasm, COORDINATOR_WASM_URL, 'single-thread'
        );
    }
    assertDecoderSessionActive(generation);
    const nextCoordinator = new selected.module.GaussianSlimLoaderCoordinatorWasm();
    const encoderMask = typeof nextCoordinator.getTextureEncoderMask === 'function' ?
        nextCoordinator.getTextureEncoderMask() : 0;
    if (!isDecoderSessionActive(generation)) {
        nextCoordinator.delete();
        throw sessionDisposedError();
    }
    coordinatorModule = selected.module;
    coordinator = nextCoordinator;
    coordinatorWasmUrl = selected.wasmUrl;
    coordinatorFlavor = selected.flavor;
    coordinatorTextureBuildCapabilities = {
        bc7: (encoderMask & 1) !== 0,
        bc3: (encoderMask & 2) !== 0
    };
    return coordinatorModule;
}

function getCoordinatorModule() {
    assertDecoderSessionActive();
    if (coordinatorModule) return Promise.resolve(coordinatorModule);
    if (coordinatorModulePromise) return coordinatorModulePromise;

    const generation = decoderSessionGeneration;
    let trackedPromise;
    trackedPromise = initializeCoordinatorModule(generation).finally(() => {
        if (coordinatorModulePromise === trackedPromise) coordinatorModulePromise = null;
    });
    coordinatorModulePromise = trackedPromise;
    return trackedPromise;
}

function postWarmupProgress(phase, message, details = {}) {
    const progress = activeWarmupProgress;
    if (!progress || !isDecoderSessionActive(progress.generation)) return;
    postProgress(progress.requestId, message, {
        progressKind: 'warmup',
        phase,
        elapsedMs: roundTimingMs(performance.now() - progress.startedAt),
        details
    });
}

async function compileReconstructionModule(generation) {
    const startedAt = performance.now();
    const compileBeginAbsMs = absoluteNowMs();
    let compiledModule = null;
    let source = 'array-buffer-compile';
    let fallbackReason = null;

    if (typeof WebAssembly.compileStreaming === 'function') {
        try {
            const response = await fetch(RECONSTRUCTION_WASM_URL, { credentials: 'same-origin' });
            if (!response.ok) {
                throw new Error(`Reconstruction WASM fetch failed: ${response.status} ${response.statusText}`);
            }
            compiledModule = await WebAssembly.compileStreaming(response);
            source = 'compile-streaming';
        } catch (error) {
            assertDecoderSessionActive(generation);
            fallbackReason = error?.message || String(error);
        }
    }

    if (!compiledModule) {
        try {
            const response = await fetch(RECONSTRUCTION_WASM_URL, { credentials: 'same-origin' });
            if (!response.ok) {
                throw new Error(`Reconstruction WASM fetch failed: ${response.status} ${response.statusText}`);
            }
            compiledModule = await WebAssembly.compile(await response.arrayBuffer());
            source = fallbackReason ? 'array-buffer-compile-fallback' : 'array-buffer-compile';
        } catch (error) {
            assertDecoderSessionActive(generation);
            const compileError = error?.message || String(error);
            fallbackReason = fallbackReason ? `${fallbackReason} <- ${compileError}` : compileError;
            source = 'worker-self-load-compile-fallback';
        }
    }

    assertDecoderSessionActive(generation);
    const compileEndAbsMs = absoluteNowMs();
    reconstructionModulePreparation = {
        module: compiledModule,
        source,
        compileMs: roundTimingMs(performance.now() - startedAt),
        compileBeginAbsMs,
        compileEndAbsMs,
        fallbackReason,
        resourceTiming: getResourceTiming(RECONSTRUCTION_WASM_URL)
    };
    return reconstructionModulePreparation;
}

function getReconstructionModulePreparation(generation = decoderSessionGeneration) {
    assertDecoderSessionActive(generation);
    if (reconstructionModulePreparation) return Promise.resolve(reconstructionModulePreparation);
    if (reconstructionModulePreparationPromise) return reconstructionModulePreparationPromise;

    let trackedPromise;
    trackedPromise = compileReconstructionModule(generation).finally(() => {
        if (reconstructionModulePreparationPromise === trackedPromise) {
            reconstructionModulePreparationPromise = null;
        }
    });
    reconstructionModulePreparationPromise = trackedPromise;
    trackedPromise.catch(() => {});
    return trackedPromise;
}

function postShardWorkerInit(record, preparation) {
    assertDecoderSessionActive(record.sessionGeneration);
    if (!record.alive || !record.worker) throw new Error(`Shard worker ${record.workerId} was terminated before init.`);

    const baseMessage = {
        type: 'init',
        protocolVersion: PROTOCOL_VERSION,
        buildVersion: BUILD_VERSION,
        workerId: record.workerId,
        startupAttemptId: record.startupAttemptId
    };
    record.initPostedAt = performance.now();
    record.readyTimeoutId = setTimeout(() => {
        if (record.readyReject) {
            handleShardWorkerCrash(record, new Error(`Shard worker ${record.workerId} startup timed out.`));
        }
    }, CONTROL_TIMEOUT_MS);

    if (preparation.module) {
        const attempt = {
            source: 'shared-compiled-module',
            postBeginAbsMs: absoluteNowMs(),
            success: false
        };
        record.initAttempts.push(attempt);
        try {
            record.worker.postMessage({
                ...baseMessage,
                reconstructionWasmModule: preparation.module
            });
            attempt.postReturnAbsMs = absoluteNowMs();
            attempt.success = true;
            record.initPostBeginAbsMs = attempt.postBeginAbsMs;
            record.initPostReturnAbsMs = attempt.postReturnAbsMs;
            record.moduleSource = 'shared-compiled-module';
            return;
        } catch (error) {
            attempt.errorAbsMs = absoluteNowMs();
            attempt.error = error?.message || String(error);
            if (error?.name !== 'DataCloneError' && !/clone/i.test(error?.message || '')) throw error;
            record.moduleSource = 'worker-self-load-clone-fallback';
            record.moduleFallbackReason = error?.message || String(error);
            postWarmupProgress(
                'reconstruction-module-fallback',
                `Failed to share the reconstruction module; shard ${record.workerId + 1} will load independently`,
                { workerId: record.workerId, reason: record.moduleFallbackReason }
            );
        }
    } else {
        record.moduleSource = 'worker-self-load-compile-fallback';
        record.moduleFallbackReason = preparation.fallbackReason;
    }

    const fallbackAttempt = {
        source: record.moduleSource,
        fallbackReason: record.moduleFallbackReason,
        postBeginAbsMs: absoluteNowMs(),
        success: false
    };
    record.initAttempts.push(fallbackAttempt);
    try {
        record.worker.postMessage({
            ...baseMessage,
            moduleCloneFallbackReason: record.moduleFallbackReason
        });
        fallbackAttempt.postReturnAbsMs = absoluteNowMs();
        fallbackAttempt.success = true;
        record.initPostBeginAbsMs = fallbackAttempt.postBeginAbsMs;
        record.initPostReturnAbsMs = fallbackAttempt.postReturnAbsMs;
    } catch (error) {
        fallbackAttempt.errorAbsMs = absoluteNowMs();
        fallbackAttempt.error = error?.message || String(error);
        throw error;
    }
}

function createShardWorker(workerId, poolIndex, generation = decoderSessionGeneration) {
    assertDecoderSessionActive(generation);
    const createdAt = performance.now();
    const createBeginAbsMs = absoluteNowMs();
    const worker = new Worker(new URL('./SplatReconstruction.worker.js', import.meta.url), { type: 'module' });
    const createEndAbsMs = absoluteNowMs();
    const record = {
        workerId,
        poolIndex,
        startupAttemptId: nextShardStartupAttemptId++,
        sessionGeneration: generation,
        worker,
        alive: true,
        busy: false,
        inflight: null,
        modelReady: null,
        readyResolve: null,
        readyReject: null,
        readyPromise: null,
        readyTimeoutId: null,
        ready: false,
        moduleSource: 'pending',
        moduleFallbackReason: null,
        initPromise: null,
        createdAt,
        initPostedAt: null,
        moduleInitMs: 0,
        startupMs: 0,
        roundTripMs: 0,
        resourceTiming: null,
        clockMethod: DECODER_CLOCK_METHOD,
        createBeginAbsMs,
        createEndAbsMs,
        initPostBeginAbsMs: undefined,
        initPostReturnAbsMs: undefined,
        readyReceiveAbsMs: undefined,
        initAttempts: [],
        workerWarmupTrace: null
    };
    record.readyPromise = new Promise((resolve, reject) => {
        record.readyResolve = resolve;
        record.readyReject = reject;
    });

    worker.onmessage = (event) => {
        const handlerReceiveAbsMs = absoluteNowMs();
        handleShardWorkerMessage(record, event.data || {}, handlerReceiveAbsMs);
    };
    worker.onerror = (error) => handleShardWorkerCrash(record, error);
    worker.onmessageerror = (error) => handleShardWorkerCrash(record, error);
    record.initPromise = getReconstructionModulePreparation(generation)
        .then((preparation) => postShardWorkerInit(record, preparation))
        .catch((error) => handleShardWorkerCrash(record, error));
    record.initPromise.catch(() => {});
    return record;
}

function terminateShardRecord(record, error = null) {
    if (!record) return;
    record.alive = false;
    record.ready = false;
    if (record.readyTimeoutId) {
        clearTimeout(record.readyTimeoutId);
        record.readyTimeoutId = null;
    }
    if (record.inflight?.timeoutId) clearTimeout(record.inflight.timeoutId);
    const terminationError = error || new Error(`Shard worker ${record.workerId} was terminated.`);
    if (record.readyReject) {
        const reject = record.readyReject;
        record.readyResolve = null;
        record.readyReject = null;
        reject(terminationError);
    }
    if (record.modelReady) {
        clearTimeout(record.modelReady.timeoutId);
        const reject = record.modelReady.reject;
        record.modelReady = null;
        reject(terminationError);
    }
    if (record.worker) {
        record.worker.onmessage = null;
        record.worker.onerror = null;
        record.worker.onmessageerror = null;
        record.worker.terminate();
        record.worker = null;
    }
}

function shardWorkerDiagnostics(record) {
    return {
        workerId: record.workerId,
        poolIndex: record.poolIndex,
        ready: record.ready,
        moduleSource: record.moduleSource,
        moduleFallbackReason: record.moduleFallbackReason,
        moduleInitMs: roundTimingMs(record.moduleInitMs),
        startupMs: roundTimingMs(record.startupMs),
        roundTripMs: roundTimingMs(record.roundTripMs),
        resourceTiming: normalizeResourceTiming(record.resourceTiming),
        warmupTrace: {
            schema: 'compactgs.warmup.trace.v1',
            workerId: record.workerId,
            poolIndex: record.poolIndex,
            startupAttemptId: record.startupAttemptId,
            status: record.ready ? 'ready' : 'pending',
            moduleSource: record.moduleSource,
            moduleFallbackReason: record.moduleFallbackReason,
            parent: {
                clockMethod: record.clockMethod,
                createBeginAbsMs: record.createBeginAbsMs,
                createEndAbsMs: record.createEndAbsMs,
                initAttempts: record.initAttempts,
                initPostBeginAbsMs: record.initPostBeginAbsMs,
                initPostReturnAbsMs: record.initPostReturnAbsMs,
                readyReceiveAbsMs: record.readyReceiveAbsMs
            },
            worker: record.workerWarmupTrace
        }
    };
}

async function ensureShardWorkerReady(poolIndex, generation = decoderSessionGeneration) {
    assertDecoderSessionActive(generation);
    let record = shardPool[poolIndex];
    if (!record?.alive || record.sessionGeneration !== generation) {
        if (record) terminateShardRecord(record);
        record = createShardWorker(poolIndex, poolIndex, generation);
        shardPool[poolIndex] = record;
    }
    try {
        await record.readyPromise;
        assertDecoderSessionActive(generation);
        if (shardPool[poolIndex] !== record || !record.alive || !record.ready) {
            throw new Error(`Shard worker ${poolIndex} became unavailable during startup.`);
        }
        return record;
    } catch (error) {
        if (shardPool[poolIndex] === record) {
            terminateShardRecord(record, error);
            shardPool[poolIndex] = null;
        }
        throw error;
    }
}

async function ensureTargetShardPool(generation = decoderSessionGeneration) {
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt++) {
        assertDecoderSessionActive(generation);
        try {
            const records = await Promise.all(
                Array.from({ length: SHARD_WORKER_COUNT },
                    (_, poolIndex) => ensureShardWorkerReady(poolIndex, generation))
            );
            return records;
        } catch (error) {
            lastError = error;
            if (!isDecoderSessionActive(generation)) throw error;
        }
    }
    throw new Error(`Failed to initialize the ${SHARD_WORKER_COUNT}-worker shard pool after retry: ` +
        `${lastError?.message || String(lastError)}`);
}

function webCodecsCapabilitySnapshot() {
    if (webCodecsSessionPolicy) return { ...webCodecsSessionPolicy };
    return {
        status: webCodecsProbeStatus,
        supported: null,
        scope: WEB_CODECS_FAST_PATH_SCOPE,
        format: null,
        probeMs: webCodecsProbeStatus === 'pending' ?
            roundTimingMs(performance.now() - webCodecsProbeStartedAt) : 0
    };
}

function probeWebCodecsSessionPolicy() {
    assertDecoderSessionActive();
    if (webCodecsSessionPolicy) return Promise.resolve(webCodecsSessionPolicy);
    if (webCodecsSessionPolicyPromise) return webCodecsSessionPolicyPromise;

    const generation = decoderSessionGeneration;
    const startedAt = performance.now();
    const abortController = new AbortController();
    webCodecsProbeAbortController = abortController;
    webCodecsProbeStartedAt = startedAt;
    webCodecsProbeStatus = 'pending';
    const probePromise = (async () => {
        let policy;
        try {
            const result = await decodeVideoStreamWithWebCodecs(
                WEB_CODECS_CANARY_DESCRIPTOR,
                getWebCodecsCapabilityCanary(),
                {
                    signal: abortController.signal,
                    timeoutMs: WEB_CODECS_PREFLIGHT_TIMEOUT_MS
                }
            );
            if (result.layout.format !== 'I420' && result.layout.format !== 'NV12') {
                throw new Error(`Canary returned unsupported VideoFrame format ${result.layout.format || 'unknown'}.`);
            }
            policy = {
                status: 'complete',
                supported: true,
                scope: WEB_CODECS_FAST_PATH_SCOPE,
                format: result.layout.format,
                acceleration: result.config.hardwareAcceleration,
                probeMs: roundTimingMs(performance.now() - startedAt)
            };
        } catch (error) {
            assertDecoderSessionActive(generation);
            policy = {
                status: 'complete',
                supported: false,
                scope: WEB_CODECS_FAST_PATH_SCOPE,
                format: null,
                reason: normalizeFallbackReason(error),
                probeMs: roundTimingMs(performance.now() - startedAt)
            };
        }
        assertDecoderSessionActive(generation);
        webCodecsSessionPolicy = policy;
        webCodecsProbeStatus = 'complete';
        return policy;
    })();

    let trackedPromise;
    trackedPromise = probePromise.finally(() => {
        if (webCodecsSessionPolicyPromise === trackedPromise) {
            webCodecsSessionPolicyPromise = null;
            webCodecsProbeAbortController = null;
        }
    });
    webCodecsSessionPolicyPromise = trackedPromise;
    trackedPromise.catch(() => {});
    return trackedPromise;
}

function warmupEntryState(ready, inflight) {
    if (ready) return 'ready';
    if (inflight) return 'inflight';
    return 'cold';
}

function startWarmupBranch(decoderTrace, name, entryState, factory) {
    const branch = {
        entryState,
        beginAbsMs: absoluteNowMs(),
        endAbsMs: undefined
    };
    decoderTrace.branches[name] = branch;
    let promise;
    try {
        promise = factory();
    } catch (error) {
        branch.endAbsMs = absoluteNowMs();
        branch.error = error?.message || String(error);
        return Promise.reject(error);
    }
    return Promise.resolve(promise).then((result) => {
        branch.endAbsMs = absoluteNowMs();
        return result;
    }, (error) => {
        branch.endAbsMs = absoluteNowMs();
        branch.error = error?.message || String(error);
        throw error;
    });
}

function createDecoderWarmupTrace(handlerReceiveAbsMs, requestId = 0, bootstrapTrace = null) {
    return {
        schema: 'compactgs.warmup.trace.v1',
        attempt: {
            requestId,
            lifecycleGeneration: decoderSessionGeneration,
            status: 'in-progress'
        },
        decoder: {
            clockMethod: DECODER_CLOCK_METHOD,
            moduleBodyStartAbsMs: DECODER_MODULE_BODY_START_ABS_MS,
            moduleBodyMeaning: 'first module-body point after static imports; excludes static import internals',
            handlerReceiveAbsMs,
            fullReadyBeginAbsMs: undefined,
            fullReadyEndAbsMs: undefined,
            resultPostBeginAbsMs: undefined,
            branches: {},
            reconstructionCompile: null
        },
        bootstrap: bootstrapTrace && typeof bootstrapTrace === 'object' ? bootstrapTrace : null,
        shardWorkers: []
    };
}

function prewarmDecoderSession(requestId = 0, requestWarmupTrace = null) {
    assertDecoderSessionActive();
    if (decoderSessionPrewarmPromise) return decoderSessionPrewarmPromise;

    const generation = decoderSessionGeneration;
    const startedAt = performance.now();
    const warmupTrace = requestWarmupTrace || createDecoderWarmupTrace(undefined, requestId);
    const decoderTrace = warmupTrace.decoder;
    decoderTrace.fullReadyBeginAbsMs = absoluteNowMs();
    activeWarmupProgress = requestId > 0 ? { requestId, generation, startedAt } : null;

    const coordinatorStartedAt = startedAt;
    const coordinatorEntryState = warmupEntryState(!!coordinatorModule, !!coordinatorModulePromise);
    const coordinatorInitPromise = startWarmupBranch(
        decoderTrace, 'coordinator', coordinatorEntryState, () => getCoordinatorModule()
    ).then(() => ({
        durationMs: performance.now() - coordinatorStartedAt,
        resourceTiming: getResourceTiming(coordinatorWasmUrl),
        flavor: coordinatorFlavor
    })).then((result) => {
        postWarmupProgress(
            'coordinator-ready',
            `Coordinator ready (${result.flavor})`,
            { flavor: result.flavor, durationMs: roundTimingMs(result.durationMs) }
        );
        return result;
    });

    // Reconstruction is intentionally lazy: COMPACTGS never needs shard workers,
    // and profile-1 decodes create the pool only when a reconstruction shard
    // is actually requested.
    const reconstructionEntryState = 'skipped';
    const reconstructionPreparationPromise = Promise.resolve({
        module: null, source: 'lazy', compileMs: 0,
        compileBeginAbsMs: null, compileEndAbsMs: null, fallbackReason: null
    });

    const shardInitPromise = Promise.resolve({ durationMs: 0, records: [] });

    const webCodecsEntryState = warmupEntryState(
        !!webCodecsSessionPolicy, !!webCodecsSessionPolicyPromise || webCodecsProbeStatus === 'in-progress'
    );
    const webCodecsPolicyPromise = startWarmupBranch(
        decoderTrace, 'webCodecs', webCodecsEntryState, () => probeWebCodecsSessionPolicy()
    ).then((policy) => {
        postWarmupProgress(
            'hevc-complete',
            policy.supported ?
                `HEVC probe complete (WebCodecs ${policy.format})` :
                'HEVC probe complete (using FFmpeg fallback)',
            {
                supported: policy.supported,
                format: policy.format,
                reason: policy.reason,
                probeMs: policy.probeMs
            }
        );
        return policy;
    });

    const initializationPromise = (async () => {
        const [coordinatorResult, reconstructionResult, shardResult, webCodecsCapability] = await Promise.all([
            coordinatorInitPromise,
            reconstructionPreparationPromise,
            shardInitPromise,
            webCodecsPolicyPromise
        ]);
        decoderTrace.fullReadyEndAbsMs = absoluteNowMs();
        assertDecoderSessionActive(generation);
        const fullReadyMs = performance.now() - startedAt;
        const serialEquivalentMs = coordinatorResult.durationMs + shardResult.durationMs +
            webCodecsCapability.probeMs;
        decoderTrace.reconstructionCompile = {
            beginAbsMs: reconstructionResult.compileBeginAbsMs,
            endAbsMs: reconstructionResult.compileEndAbsMs,
            source: reconstructionResult.source,
            execution: 'skipped-lazy'
        };
        const shardDiagnostics = shardResult.records.map(shardWorkerDiagnostics);
        warmupTrace.shardWorkers = shardDiagnostics.map((record) => record.warmupTrace);
        const bootstrapResources = Array.isArray(warmupTrace.bootstrap?.javaScriptResourceTiming) ?
            warmupTrace.bootstrap.javaScriptResourceTiming : [];
        warmupTrace.resources = [
            ...bootstrapResources.map((timing, index) => ({
                name: `Decoder JS ${index + 1}`,
                owner: 'decoder-bootstrap-module-graph',
                kind: 'javascript',
                source: 'bootstrap-resource-timing',
                timing: normalizeResourceTiming({ ...timing, durationMs: timing.duration })
            })),
            {
                name: 'Coordinator selected WASM',
                owner: `coordinator-${coordinatorResult.flavor}`,
                kind: 'wasm',
                source: coordinatorResult.flavor,
                timing: coordinatorResult.resourceTiming
            },
            {
                name: 'Reconstruction WASM compile',
                owner: 'decoder-reconstruction-compile',
                kind: 'wasm',
                source: reconstructionResult.source,
                timing: reconstructionResult.resourceTiming
            },
            ...shardDiagnostics.map((record) => ({
                name: `Shard ${record.workerId} reconstruction resource`,
                owner: `shard-${record.workerId}`,
                kind: 'worker-or-wasm',
                source: record.moduleSource,
                entryUrl: new URL('./SplatReconstruction.worker.js', import.meta.url).href,
                timing: record.resourceTiming
            }))
        ];
        warmupTrace.attempt.status = 'success';
        const diagnostics = {
            coordinatorModuleMs: roundTimingMs(coordinatorResult.durationMs),
            coordinatorFlavor: coordinatorResult.flavor,
            reconstructionModuleMs: reconstructionResult.compileMs,
            reconstructionModuleSource: reconstructionResult.source,
            reconstructionModuleFallbackReason: reconstructionResult.fallbackReason,
            reconstructionModuleShared: !!reconstructionResult.module,
            shardWarmupMs: roundTimingMs(shardResult.durationMs),
            shardWorkerCount: SHARD_WORKER_COUNT,
            targetShardWorkerCount: SHARD_WORKER_COUNT,
            fullReadyMs: roundTimingMs(fullReadyMs),
            parallelInitMs: roundTimingMs(fullReadyMs),
            totalMs: roundTimingMs(fullReadyMs),
            serialEquivalentMs: roundTimingMs(serialEquivalentMs),
            estimatedOverlapMs: roundTimingMs(Math.max(0, serialEquivalentMs - fullReadyMs)),
            textureBuildCapabilities: coordinatorTextureBuildCapabilities,
            webCodecsCapability,
            webCodecsProbeStatus: webCodecsCapability.status,
            webCodecsProbeMs: webCodecsCapability.probeMs,
            coordinatorResourceTiming: coordinatorResult.resourceTiming,
            reconstructionResourceTiming: reconstructionResult.resourceTiming,
            shardWorkers: shardDiagnostics,
            warmupTrace
        };
        postWarmupProgress(
            'full-ready',
            'Coordinator ready (reconstruction workers lazy)',
            { shardWorkerCount: 0, targetShardWorkerCount: SHARD_WORKER_COUNT, fullReadyMs: diagnostics.fullReadyMs }
        );
        if (activeWarmupProgress?.generation === generation) activeWarmupProgress = null;
        return diagnostics;
    })();

    let trackedPromise;
    trackedPromise = initializationPromise.catch((error) => {
        if (!Number.isFinite(decoderTrace.fullReadyEndAbsMs)) {
            decoderTrace.failureAbsMs = absoluteNowMs();
            decoderTrace.failure = error?.message || String(error);
        }
        warmupTrace.attempt.status = 'error';
        warmupTrace.attempt.failureStage = 'prewarmDecoderSession';
        if (decoderSessionPrewarmPromise === trackedPromise) decoderSessionPrewarmPromise = null;
        if (activeWarmupProgress?.generation === generation) activeWarmupProgress = null;
        throw error;
    });
    decoderSessionPrewarmPromise = trackedPromise;
    return trackedPromise;
}

function beginModelOnWorker(record, model) {
    if (!record.alive || !record.ready ||
        !isDecoderSessionActive(record.sessionGeneration)) {
        return Promise.reject(new Error(`Shard worker ${record.workerId} is unavailable.`));
    }
    return new Promise((resolve, reject) => {
        const metadata = model.metadata.slice(0);
        const timeoutId = setTimeout(() => {
            if (record.modelReady?.modelId !== model.modelId) return;
            handleShardWorkerCrash(
                record,
                new Error(`Shard worker ${record.workerId} model initialization timed out.`)
            );
        }, CONTROL_TIMEOUT_MS);
        record.modelReady = { modelId: model.modelId, requestId: model.requestId, resolve, reject, timeoutId };
        try {
            record.worker.postMessage({
                type: 'beginModel',
                protocolVersion: PROTOCOL_VERSION,
                buildVersion: BUILD_VERSION,
                workerId: record.workerId,
                modelId: model.modelId,
                requestId: model.requestId,
                traceOriginAbsMs: model.traceOriginAbsMs,
                reconstructionTraceLevel: model.reconstructionTraceLevel,
                metadata
            }, [metadata]);
        } catch (error) {
            handleShardWorkerCrash(record, error);
        }
    });
}

function postProgress(requestId, message, extra = {}) {
    self.postMessage({
        type: 'progress',
        protocolVersion: PROTOCOL_VERSION,
        buildVersion: BUILD_VERSION,
        requestId,
        message,
        ...extra
    });
}

function assertPreparationActive(preparation) {
    if (activePreparation !== preparation || preparation.abortController.signal.aborted) {
        const reason = preparation.abortController.signal.reason;
        throw reason instanceof Error ? reason : new Error('GaussianSlimLoader decode preparation was canceled.');
    }
}

function readPendingVideo(Module, pendingIndex) {
    const descriptor = coordinator.getPendingVideo(pendingIndex);
    if (!descriptor.success) {
        throw new Error(coordinator.getLastError() || `Failed to read pending video ${pendingIndex}.`);
    }
    if (!Number.isInteger(descriptor.frameWidth) || descriptor.frameWidth <= 0 ||
        !Number.isInteger(descriptor.frameHeight) || descriptor.frameHeight <= 0 ||
        !Number.isInteger(descriptor.frameCount) || descriptor.frameCount <= 0 ||
        descriptor.encodedSize <= 0 || descriptor.encodedPtr <= 0) {
        throw new Error(`Pending video ${pendingIndex} has an invalid bridge descriptor.`);
    }
    return {
        streamIndex: descriptor.streamIndex,
        codecId: descriptor.codecId,
        frameWidth: descriptor.frameWidth,
        frameHeight: descriptor.frameHeight,
        frameCount: descriptor.frameCount,
        encoded: Module.HEAPU8.slice(descriptor.encodedPtr, descriptor.encodedPtr + descriptor.encodedSize)
    };
}

function injectDecodedVideo(Module, descriptor, result) {
    const decoded = result?.decoded;
    const pixelFormat = result?.pixelFormat;
    const expectedSize = videoBufferByteLength(
        pixelFormat, descriptor.frameWidth, descriptor.frameHeight, descriptor.frameCount
    );
    if (!(decoded instanceof Uint8Array) || decoded.byteLength !== expectedSize ||
        result.layout?.width !== descriptor.frameWidth || result.layout?.height !== descriptor.frameHeight ||
        result.layout?.frameCount !== descriptor.frameCount) {
        throw new Error(`WebCodecs returned an invalid byte layout for video substream ${descriptor.streamIndex}.`);
    }
    const startedAt = performance.now();
    const decodedPtr = Module._malloc(decoded.byteLength);
    if (!decodedPtr) throw new Error(`Failed to allocate ${decoded.byteLength} decoded video bytes.`);
    try {
        Module.HEAPU8.set(decoded, decodedPtr);
        if (!coordinator.injectDecodedVideo(
            descriptor.streamIndex, decodedPtr, decoded.byteLength, pixelFormat
        )) {
            throw new Error(coordinator.getLastError() ||
                `Failed to inject video substream ${descriptor.streamIndex}.`);
        }
    } finally {
        Module._free(decodedPtr);
    }
    return performance.now() - startedAt;
}

function normalizeFallbackReason(error) {
    const reasons = [];
    const visited = new Set();
    let current = error;
    while (current && !visited.has(current) && reasons.length < 3) {
        visited.add(current);
        const name = typeof current.name === 'string' && current.name ? current.name : '';
        const message = typeof current.message === 'string' && current.message ? current.message : String(current);
        const reason = name && name !== 'Error' ? `${name}: ${message}` : message;
        if (reason && reasons[reasons.length - 1] !== reason) reasons.push(reason);
        current = current.cause;
    }
    const normalized = reasons.join(' <- ') || 'Unknown WebCodecs failure';
    return normalized.length > 500 ? `${normalized.slice(0, 497)}...` : normalized;
}

export function composeVideoDecoderPath(rawVideoStreamCount, pendingVideoCount, encodedPath = 'none') {
    const rawCount = Number.isInteger(rawVideoStreamCount) && rawVideoStreamCount > 0 ?
        rawVideoStreamCount : 0;
    const pendingCount = Number.isInteger(pendingVideoCount) && pendingVideoCount > 0 ?
        pendingVideoCount : 0;
    if (rawCount === 0) return pendingCount === 0 ? 'none' : encodedPath;
    if (pendingCount === 0) return 'raw';
    return `raw+${encodedPath}`;
}

async function preparePendingVideos(Module, staged, preparation) {
    const pendingVideoCount = staged.pendingVideoCount;
    const rawVideoStreamCount = staged.rawVideoStreamCount || 0;
    const timings = {
        videoDecoderPath: composeVideoDecoderPath(
            rawVideoStreamCount, pendingVideoCount, pendingVideoCount > 0 ? 'ffmpeg' : 'none'
        ),
        pendingVideoCount,
        rawVideoStreamCount,
        rawVideoInputBytes: staged.rawVideoInputBytes || 0,
        rawVideoAdoptMs: staged.rawVideoAdoptMs || 0,
        webCodecsEligibleStreamCount: 0,
        webCodecsStreamCount: 0,
        webCodecsPolicySkipCount: 0,
        webCodecsIneligibleStreamCount: 0,
        webCodecsNegativeCacheHitCount: 0,
        ffmpegFallbackStreamCount: 0,
        webCodecsMs: 0,
        webCodecsCopyMs: 0,
        jsToWasmInjectMs: 0,
        webCodecsLayouts: [],
        streams: [],
        ffmpegFallbackWallMs: 0,
        webCodecsSessionPolicy: webCodecsCapabilitySnapshot()
    };
    if (pendingVideoCount === 0) return timings;

    assertPreparationActive(preparation);
    const sessionPolicy = await probeWebCodecsSessionPolicy();
    assertPreparationActive(preparation);
    timings.webCodecsSessionPolicy = sessionPolicy;
    const webCodecsEnabled = sessionPolicy.supported === true;
    const webCodecsDeadline = performance.now() + WEB_CODECS_PHASE_TIMEOUT_MS;
    const preparationSignal = preparation.abortController.signal;
    const phaseAbortController = webCodecsEnabled ? new AbortController() : null;
    const abortPhaseFromPreparation = () => phaseAbortController?.abort(preparationSignal.reason);
    if (preparationSignal.aborted) abortPhaseFromPreparation();
    if (phaseAbortController) {
        preparationSignal.addEventListener('abort', abortPhaseFromPreparation, { once: true });
    }
    const phaseTimeoutId = phaseAbortController ? setTimeout(() => {
        phaseAbortController.abort(new Error(
            `WebCodecs phase timed out after ${WEB_CODECS_PHASE_TIMEOUT_MS} ms.`
        ));
    }, WEB_CODECS_PHASE_TIMEOUT_MS) : null;
    const phaseAbortPromise = phaseAbortController ? new Promise((_, reject) => {
        if (phaseAbortController.signal.aborted) {
            reject(phaseAbortController.signal.reason);
            return;
        }
        phaseAbortController.signal.addEventListener('abort', () => {
            reject(phaseAbortController.signal.reason);
        }, { once: true });
    }) : null;
    phaseAbortPromise?.catch(() => {});
    const fallbackReasons = [];
    const webCodecsFailureReasons = [];
    const streamDetails = new Map();
    try {
        for (let pendingIndex = 0; pendingIndex < pendingVideoCount; pendingIndex++) {
            let streamIndex = pendingIndex;
            let capabilityKey = '';
            let attemptedWebCodecs = false;
            let attemptStartedAt = 0;
            let streamTiming = null;
            try {
                assertPreparationActive(preparation);
                const descriptor = readPendingVideo(Module, pendingIndex);
                streamIndex = descriptor.streamIndex;
                streamTiming = {
                    streamIndex,
                    role: 'video',
                    decoderPath: 'ffmpeg',
                    actualPath: 'ffmpeg'
                };
                streamDetails.set(streamIndex, streamTiming);
                const eligibility = classifyWebCodecsFastPath(descriptor, descriptor.encoded);
                if (!eligibility.eligible) {
                    timings.webCodecsIneligibleStreamCount++;
                    fallbackReasons.push(`stream ${streamIndex}: direct FFmpeg route: ${eligibility.reason}`);
                    continue;
                }
                timings.webCodecsEligibleStreamCount++;
                if (!webCodecsEnabled) {
                    timings.webCodecsPolicySkipCount++;
                    fallbackReasons.push(
                        `stream ${streamIndex}: session policy selected FFmpeg: ` +
                        `${sessionPolicy.reason || 'WebCodecs preflight was not available'}`
                    );
                    continue;
                }
                if (phaseAbortController.signal.aborted) throw phaseAbortController.signal.reason;
                capabilityKey = createWebCodecsCapabilityKey(
                    descriptor, descriptor.encoded, eligibility.sps
                );
                if (webCodecsNegativeCapabilities.has(capabilityKey)) {
                    timings.webCodecsNegativeCacheHitCount++;
                    fallbackReasons.push(`stream ${streamIndex}: cached incompatible WebCodecs output layout`);
                    continue;
                }
                const remainingTimeoutMs = Math.max(1, Math.ceil(webCodecsDeadline - performance.now()));
                attemptedWebCodecs = true;
                attemptStartedAt = performance.now();
                const decodePromise = decodeVideoStreamWithWebCodecs(descriptor, descriptor.encoded, {
                    signal: phaseAbortController.signal,
                    timeoutMs: remainingTimeoutMs,
                    accelerationPreference: sessionPolicy.acceleration
                });
                decodePromise.catch(() => {});
                const result = await Promise.race([decodePromise, phaseAbortPromise]);
                assertPreparationActive(preparation);
                if (phaseAbortController.signal.aborted) throw phaseAbortController.signal.reason;
                const injectToWasmMs = injectDecodedVideo(Module, descriptor, result);
                timings.jsToWasmInjectMs += injectToWasmMs;
                timings.webCodecsCopyMs += result.timing?.copyMs || 0;
                Object.assign(streamTiming, {
                    decoderPath: 'webcodecs',
                    actualPath: 'webcodecs',
                    decodeAndFlushMs: result.timing?.decodeAndFlushMs,
                    totalMs: result.timing?.totalMs,
                    copyMs: result.timing?.copyMs,
                    injectToWasmMs
                });
                timings.webCodecsLayouts.push({
                    streamIndex,
                    format: result.layout.format,
                    frameByteLength: result.layout.frameByteLength,
                    planeLayouts: result.layout.observedPlaneLayouts
                });
                timings.webCodecsStreamCount++;
            } catch (error) {
                assertPreparationActive(preparation);
                if (capabilityKey && isWebCodecsNegativeCapabilityError(error)) {
                    webCodecsNegativeCapabilities.add(capabilityKey);
                }
                const reason = `stream ${streamIndex}: ${normalizeFallbackReason(error)}`;
                fallbackReasons.push(reason);
                if (attemptedWebCodecs) webCodecsFailureReasons.push(reason);
                if (phaseAbortController?.signal.aborted || performance.now() >= webCodecsDeadline) break;
            } finally {
                if (attemptStartedAt > 0) {
                    const attemptWallMs = performance.now() - attemptStartedAt;
                    timings.webCodecsMs += attemptWallMs;
                    if (streamTiming) streamTiming.attemptWallMs = attemptWallMs;
                }
            }
        }
    } finally {
        if (phaseTimeoutId) clearTimeout(phaseTimeoutId);
        if (phaseAbortController) {
            preparationSignal.removeEventListener('abort', abortPhaseFromPreparation);
        }
    }
    if (fallbackReasons.length === 0) {
        timings.streams = [...streamDetails.values()];
        timings.videoDecoderPath = composeVideoDecoderPath(
            rawVideoStreamCount, pendingVideoCount, 'webcodecs'
        );
        return timings;
    }

    timings.fallbackReason = normalizeFallbackReason(fallbackReasons.join('; '));
    if (webCodecsFailureReasons.length > 0) {
        console.warn(`[GaussianSlimLoader coordinator] WebCodecs failed for ${webCodecsFailureReasons.length} stream(s); ` +
            'continuing with FFmpeg fallback.');
    } else if (verboseLog) {
        console.info(`[GaussianSlimLoader coordinator] Routed ${fallbackReasons.length} video stream(s) directly to FFmpeg.`);
    }

    const fallbackStartedAt = performance.now();
    if (!coordinator.decodePendingVideosWithFallback()) {
        throw new Error(coordinator.getLastError() || 'FFmpeg video fallback failed.');
    }
    timings.ffmpegFallbackWallMs = performance.now() - fallbackStartedAt;
    timings.ffmpegFallbackStreamCount = pendingVideoCount - timings.webCodecsStreamCount;
    timings.streams = [...streamDetails.values()];
    timings.videoDecoderPath = composeVideoDecoderPath(
        rawVideoStreamCount, pendingVideoCount,
        timings.webCodecsStreamCount > 0 ? 'mixed' : 'ffmpeg'
    );
    return timings;
}

function buildSubstreamTimings(prepared, videoTimings) {
    const isCompactGS = prepared.profileIdc === 2;
    const roles = isCompactGS ?
        ['position-low', 'position-high', 'color-rgba', 'shape-video', 'sh-atlas'] :
        ['non-video', 'non-video', 'video', 'texture', 'non-video'];
    const videoStreamIndex = isCompactGS ? 3 : 2;
    const videoByStream = new Map((videoTimings?.streams || []).map((stream) => [stream.streamIndex, stream]));
    return Object.fromEntries(roles.map((role, streamIndex) => {
        const wasmDecodeMs = prepared[`substream${streamIndex}Ms`];
        const entry = { streamIndex, role, wasmDecodeMs };
        if (streamIndex !== videoStreamIndex) {
            entry.primaryDecodeMs = wasmDecodeMs;
            return [String(streamIndex), entry];
        }

        const detail = videoByStream.get(streamIndex);
        const actualPath = detail?.actualPath || detail?.decoderPath || videoTimings?.videoDecoderPath || 'unknown';
        Object.assign(entry, detail || {}, {
            streamIndex,
            role,
            decoderPath: actualPath,
            actualPath
        });
        if (actualPath === 'webcodecs') {
            entry.primaryDecodeMs = Number.isFinite(detail?.decodeAndFlushMs) ?
                detail.decodeAndFlushMs : detail?.totalMs;
            entry.wasmDecodeMeaning = 'WebCodecs decode runs in JavaScript; wasmDecodeMs records only the C++ injection marker.';
        } else if (actualPath === 'ffmpeg') {
            entry.primaryDecodeMs = wasmDecodeMs;
            entry.ffmpegRawDecodeMs = wasmDecodeMs;
            entry.ffmpegOuterWallMs = videoTimings?.ffmpegFallbackWallMs;
        }
        return [String(streamIndex), entry];
    }));
}

export function copyCompactGSPreparedResult(Module, prepared, compressedTextureData, videoTimings) {
    const width = prepared.pointMapWidth;
    const height = prepared.pointMapHeight;
    const capacity = width * height;
    const shapeComponents = prepared.shapeComponentCount === 7 ? 7 : 6;
    const shapeRows = prepared.layoutVersion >= 3 ? 4 : (shapeComponents === 7 ? 3 : 2);
    const expected = {
        positionLowSize: capacity * 3,
        positionHighSize: capacity * 3,
        colorRgbSize: capacity * 3,
        shapePlaneSize: width * 4 * height * shapeRows
    };
    if (prepared.profileIdc !== 2 || ![2, 3].includes(prepared.layoutVersion) ||
        !Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 ||
        width % 4 !== 0 || height % 4 !== 0 || width * 4 > 8192 || height * 4 > 8192 ||
        prepared.pointCount > capacity || prepared.positionGroupSize !== 256) {
        throw new Error('Coordinator returned an invalid COMPACTGS_V1 descriptor.');
    }
    for (const [field, byteLength] of Object.entries(expected)) {
        if (prepared[field] !== byteLength) {
            throw new Error(`Coordinator returned invalid COMPACTGS_V1 ${field}: ${prepared[field]}/${byteLength}.`);
        }
    }
    const groupCount = Math.ceil(prepared.pointCount / prepared.positionGroupSize);
    const expectedShapeComponents = prepared.layoutVersion >= 3 ? 6 : 7;
    if (prepared.positionGroupFloatCount !== groupCount * 3 || prepared.shapeComponentCount !== expectedShapeComponents ||
        prepared.fastShComponentCount !== 45 || !Number.isFinite(prepared.importanceMin) ||
        !Number.isFinite(prepared.importanceStep) || prepared.importanceStep < 0) {
        throw new Error('Coordinator returned malformed COMPACTGS_V1 quantization tables.');
    }
    const copyU8 = (pointer, byteLength) => {
        if (!Number.isInteger(pointer) || pointer <= 0 || !Number.isInteger(byteLength) || byteLength <= 0) {
            throw new Error('Coordinator returned an invalid COMPACTGS_V1 plane pointer.');
        }
        return Module.HEAPU8.slice(pointer, pointer + byteLength);
    };
    const copyF32 = (pointer, count) => {
        if (!Number.isInteger(pointer) || pointer <= 0 || !Number.isInteger(count) || count <= 0) {
            throw new Error('Coordinator returned an invalid COMPACTGS_V1 table pointer.');
        }
        return Module.HEAPF32.slice(pointer >>> 2, (pointer >>> 2) + count);
    };
    const shAtlasExpected = width * 4 * height * 4 * 3;
    const shAtlas = prepared.decodeFeaturesRest ? copyU8(prepared.shPlanePtr, prepared.shPlaneSize) : null;
    if (prepared.decodeFeaturesRest && prepared.shPlaneSize !== shAtlasExpected) {
        throw new Error(`Coordinator returned invalid COMPACTGS_V1 CPU SH atlas length: ` +
            `${prepared.shPlaneSize}/${shAtlasExpected}.`);
    }
    if (!prepared.decodeFeaturesRest && !compressedTextureData) {
        throw new Error('COMPACTGS_V1 requires either a compressed texture or a CPU-decoded SH atlas.');
    }
    const positionGroupMin = copyF32(prepared.positionGroupMinPtr, prepared.positionGroupFloatCount);
    const positionGroupStep = copyF32(prepared.positionGroupStepPtr, prepared.positionGroupFloatCount);
    const shapeMin = copyF32(prepared.shapeMinPtr, shapeComponents);
    const shapeStep = copyF32(prepared.shapeStepPtr, shapeComponents);
    const covarianceGroupMin = prepared.layoutVersion >= 3 ? copyF32(prepared.covarianceGroupMinPtr, groupCount * 6) : null;
    const covarianceGroupStep = prepared.layoutVersion >= 3 ? copyF32(prepared.covarianceGroupStepPtr, groupCount * 6) : null;
    if (prepared.layoutVersion >= 3 && (prepared.covarianceGroupFloatCount !== groupCount * 6 ||
        !covarianceGroupMin || !covarianceGroupStep)) {
        throw new Error('Coordinator returned malformed COMPACTGS covariance tables.');
    }
    const shMin = copyF32(prepared.fastShMinPtr, 45);
    const shStep = copyF32(prepared.fastShStepPtr, 45);
    if (![positionGroupMin, shapeMin, shMin, ...(covarianceGroupMin ? [covarianceGroupMin] : [])].every((table) => table.every(Number.isFinite)) ||
        ![positionGroupStep, shapeStep, shStep].every((table) =>
            table.every((value) => Number.isFinite(value) && value >= 0))) {
        throw new Error('Coordinator returned non-finite or negative COMPACTGS_V1 quantization metadata.');
    }
    // Layout 2 keeps lossy scale/quaternion tiles in the video plane. Rebuild
    // covariance once on this worker so the render shader can consume a float
    // covariance texture directly.
    let covariances = null;
    let compactgsCovarianceMs = 0;
    if (prepared.layoutVersion === 2) {
        const startedAt = performance.now();
        covariances = new Float32Array(prepared.pointCount * 6);
        const shape = copyU8(prepared.shapePlanePtr, prepared.shapePlaneSize);
        const readQ = (index, tile) => {
            const x = index % width;
            const y = Math.floor(index / width);
            const tileX = tile & 3;
            const tileY = tile >> 2;
            return shape[(tileY * height + y) * (width * 4) + tileX * width + x];
        };
        for (let i = 0; i < prepared.pointCount; i++) {
            const values = new Float32Array(7);
            for (let c = 0; c < 7; c++) values[c] = shapeMin[c] + shapeStep[c] * readQ(i, c);
            const sx = Math.exp(values[0]), sy = Math.exp(values[1]), sz = Math.exp(values[2]);
            let qw = values[3], qx = values[4], qy = values[5], qz = values[6];
            const qnorm = Math.hypot(qw, qx, qy, qz) || 1;
            qw /= qnorm; qx /= qnorm; qy /= qnorm; qz /= qnorm;
            const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qz * qw), r02 = 2 * (qx * qz + qy * qw);
            const r10 = 2 * (qx * qy + qz * qw), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qx * qw);
            const r20 = 2 * (qx * qz - qy * qw), r21 = 2 * (qy * qz + qx * qw), r22 = 1 - 2 * (qx * qx + qy * qy);
            const d0 = sx * sx, d1 = sy * sy, d2 = sz * sz;
            const base = i * 6;
            covariances[base] = r00 * r00 * d0 + r01 * r01 * d1 + r02 * r02 * d2;
            covariances[base + 1] = r00 * r10 * d0 + r01 * r11 * d1 + r02 * r12 * d2;
            covariances[base + 2] = r00 * r20 * d0 + r01 * r21 * d1 + r02 * r22 * d2;
            covariances[base + 3] = r10 * r10 * d0 + r11 * r11 * d1 + r12 * r12 * d2;
            covariances[base + 4] = r10 * r20 * d0 + r11 * r21 * d1 + r12 * r22 * d2;
            covariances[base + 5] = r20 * r20 * d0 + r21 * r21 * d1 + r22 * r22 * d2;
        }
        compactgsCovarianceMs = performance.now() - startedAt;
    }
    return {
        variant: 'compactgs',
        profileIdc: 2,
        numPoints: prepared.pointCount,
        shDegree: prepared.shDegree,
        descriptor: {
            layoutVersion: prepared.layoutVersion,
            pointMapWidth: width,
            pointMapHeight: height,
            pointCapacity: capacity,
            positionBits: 15,
            positionLowBits: 7,
            positionGroupSize: prepared.positionGroupSize,
            positionGroupMin,
            positionGroupStep,
            shapeEncoding: prepared.layoutVersion >= 3 ? 'COVARIANCE_Q16_GROUPED' : 'SCALE_ROTATION_Q8',
            shapeMin,
            shapeStep,
            covarianceGroupMin,
            covarianceGroupStep,
            encodedMinimumAlpha: prepared.encodedMinimumAlpha,
            importanceMin: prepared.importanceMin,
            importanceStep: prepared.importanceStep,
            shMin,
            shStep
        },
        planes: {
            positionLow: copyU8(prepared.positionLowPtr, prepared.positionLowSize),
            positionHigh: copyU8(prepared.positionHighPtr, prepared.positionHighSize),
            colorRgb: copyU8(prepared.colorRgbPtr, prepared.colorRgbSize),
            shape: copyU8(prepared.shapePlanePtr, prepared.shapePlaneSize),
            shAtlas
        },
        covariances,
        compressedTextureData,
        timings: {
            prepare: {
                parseMs: prepared.parseMs,
                decodeNonVideoSubstreamsMs: prepared.decodeNonVideoSubstreamsMs,
                astcTextureDecodeMs: prepared.astcTextureDecodeMs,
                bcTextureEncodeMs: prepared.bcTextureEncodeMs,
                textureInputBytes: prepared.textureInputBytes,
                textureOutputBytes: prepared.textureOutputBytes,
                decodeVideoFallbackMs: prepared.decodeVideoFallbackMs,
                decodeSubstreamsMs: prepared.decodeSubstreamsMs,
                totalMs: prepared.totalMs,
                substreams: buildSubstreamTimings(prepared, videoTimings),
                ...videoTimings
            },
            compactgsPlaneExportMs: 0,
            compactgsCovarianceMs,
            wall: {
                ...(prepared.wallTimings || {}),
                reconstructionWallMs: null,
                reconstructionApplicable: false,
                clockMethod: absoluteClockMethod()
            }
        }
    };
}

export function copyCompressedTextureUvs(Module, prepared, textureStrategy) {
    if (prepared.profileIdc === 2) {
        if (prepared.astcUvCount !== 0) {
            throw new Error(`COMPACTGS_V1 must not export per-point ${textureStrategy} UVs.`);
        }
        return null;
    }
    if (!Number.isInteger(prepared.astcUvCount) ||
        prepared.astcUvCount !== prepared.pointCount * 2 ||
        !Number.isInteger(prepared.astcUvPtr) || prepared.astcUvPtr <= 0) {
        throw new Error(`Coordinator returned an invalid ${textureStrategy} UV count.`);
    }
    return Module.HEAPU32.slice(prepared.astcUvPtr >>> 2,
        (prepared.astcUvPtr >>> 2) + prepared.astcUvCount);
}

function failModel(model, error) {
    if (!model || model.settled) return;
    model.settled = true;
    model.canceled = true;
    if (model.deadlineId) clearTimeout(model.deadlineId);
    for (const record of shardPool) {
        if (!record) continue;
        if (record.inflight?.timeoutId) clearTimeout(record.inflight.timeoutId);
        record.inflight = null;
        record.busy = false;
        if (record.modelReady?.modelId === model.modelId) {
            clearTimeout(record.modelReady.timeoutId);
            const reject = record.modelReady.reject;
            record.modelReady = null;
            reject(error);
        }
        if (record.alive) {
            try {
                record.worker.postMessage({
                    type: 'cancelModel',
                    protocolVersion: PROTOCOL_VERSION,
                    buildVersion: BUILD_VERSION,
                    workerId: record.workerId,
                    modelId: model.modelId,
                    requestId: model.requestId
                });
            } catch (postError) {
                decoderSessionPrewarmPromise = null;
                terminateShardRecord(record, postError);
            }
        }
    }
    model.reject(error);
}

function dispatchRange(record, model, range) {
    if (model.canceled || model.settled || !record.alive) return;
    let packed;
    const exportStartedAt = performance.now();
    const traceEnabled = model.reconstructionTraceLevel === 'js';
    const packExportBeginAbsMs = traceEnabled ? absoluteNowMs() : undefined;
    try {
        packed = coordinator.packShard(range.startBlock, range.blockCount);
        if (!packed.success) throw new Error(coordinator.getLastError() || 'Coordinator failed to pack a shard.');
        if (packed.startPoint !== range.startPoint || packed.pointCount !== range.pointCount) {
            throw new Error('Coordinator returned a mismatched packed shard range.');
        }
        const packet = coordinatorModule.HEAPU8.slice(packed.packetPtr, packed.packetPtr + packed.packetSize).buffer;
        coordinator.releasePackedShard();
        model.timings.taskExportCopyMs += performance.now() - exportStartedAt;
        model.timings.packShardMs += packed.packMs;
        model.timings.unpackMs += packed.unpackMs || 0;
        const packExportEndAbsMs = traceEnabled ? absoluteNowMs() : undefined;

        const shardId = model.nextShardId++;
        const timeoutId = setTimeout(() => {
            if (record.inflight?.shardId !== shardId) return;
            recoverShardWorker(record, new Error(`Shard ${shardId} timed out.`)).catch((error) => {
                if (activeModel && !activeModel.settled) failModel(activeModel, error);
            });
        }, SHARD_TIMEOUT_MS);
        const trace = traceEnabled ? {
            workerId: record.workerId,
            modelId: model.modelId,
            requestId: model.requestId,
            shardId,
            startBlock: range.startBlock,
            blockCount: range.blockCount,
            startPoint: range.startPoint,
            pointCount: range.pointCount,
            retry: range.retries || 0,
            packExportBeginAbsMs,
            packExportEndAbsMs
        } : null;
        record.busy = true;
        record.inflight = { ...range, shardId, timeoutId, trace };
        const dispatchBeginAbsMs = traceEnabled ? absoluteNowMs() : undefined;
        const dispatchAbsMs = Number.isFinite(dispatchBeginAbsMs) ? dispatchBeginAbsMs : absoluteNowMs();
        if (trace) trace.dispatchBeginAbsMs = dispatchBeginAbsMs;
        record.worker.postMessage({
            type: 'reconstructShard',
            protocolVersion: PROTOCOL_VERSION,
            buildVersion: BUILD_VERSION,
            workerId: record.workerId,
            modelId: model.modelId,
            requestId: model.requestId,
            shardId,
            startBlock: range.startBlock,
            blockCount: range.blockCount,
            startPoint: range.startPoint,
            pointCount: range.pointCount,
            dispatchedAt: performance.now(),
            dispatchAbsMs,
            traceOriginAbsMs: model.traceOriginAbsMs,
            reconstructionTraceLevel: model.reconstructionTraceLevel,
            ...(trace ? { trace } : {}),
            packet
        }, [packet]);
        if (trace) trace.dispatchReturnAbsMs = absoluteNowMs();
    } catch (error) {
        try {
            coordinator.releasePackedShard();
        } catch (_) {}
        failModel(model, error);
    }
}

function dispatchNext(record, model) {
    if (model.canceled || model.settled || record.busy || !record.alive) return;
    if (model.nextBlock >= model.blockCount) {
        if (model.completedBlocks === model.blockCount && shardPool.every((item) => !item?.busy)) finishModel(model);
        return;
    }

    const startBlock = model.nextBlock;
    const blockCount = Math.min(model.blocksPerShard, model.blockCount - startBlock);
    const startPoint = startBlock * model.pointsPerBlock;
    const pointCount = Math.min(model.pointCount - startPoint, blockCount * model.pointsPerBlock);
    model.nextBlock += blockCount;
    dispatchRange(record, model, { startBlock, blockCount, startPoint, pointCount, retries: 0 });
}

function accumulateShardTimings(target, source) {
    if (!source) return;
    for (const [name, value] of Object.entries(source)) {
        if (typeof value === 'number' && Number.isFinite(value)) target[name] = (target[name] || 0) + value;
    }
}

function traceDurationMs(start, end) {
    return Number.isFinite(start) && Number.isFinite(end) ? Math.max(0, end - start) : 0;
}

function recordCompletedShardTrace(model, range, message, coordinatorReceiveAbsMs,
    mergeBeginAbsMs, mergeEndAbsMs) {
    const shardTrace = model.timings.shardTrace;
    if (!shardTrace) return;
    try {
        const workerTrace = message.trace || {};
        const coordinatorTrace = range.trace || {};
        const task = {
            workerId: message.workerId,
            modelId: model.modelId,
            requestId: model.requestId,
            shardId: message.shardId,
            startBlock: range.startBlock,
            blockCount: range.blockCount,
            startPoint: range.startPoint,
            pointCount: range.pointCount,
            retry: range.retries || 0,
            packExportBeginAbsMs: coordinatorTrace.packExportBeginAbsMs,
            packExportEndAbsMs: coordinatorTrace.packExportEndAbsMs,
            dispatchBeginAbsMs: coordinatorTrace.dispatchBeginAbsMs,
            dispatchReturnAbsMs: coordinatorTrace.dispatchReturnAbsMs,
            workerReceiveAbsMs: workerTrace.workerReceiveAbsMs,
            wasmInputCopyBeginAbsMs: workerTrace.wasmInputCopyBeginAbsMs,
            kernelCallBeginAbsMs: workerTrace.kernelCallBeginAbsMs,
            kernelCallEndAbsMs: workerTrace.kernelCallEndAbsMs,
            resultCopyBeginAbsMs: workerTrace.resultCopyBeginAbsMs,
            resultCopyEndAbsMs: workerTrace.resultCopyEndAbsMs,
            resultPostBeginAbsMs: workerTrace.resultPostBeginAbsMs,
            coordinatorReceiveAbsMs,
            mergeBeginAbsMs,
            mergeEndAbsMs
        };

        let summary = shardTrace.workerSummaries.find((item) => item.workerId === task.workerId);
        if (!summary) {
            summary = {
                workerId: task.workerId,
                taskCount: 0,
                blockCount: 0,
                pointCount: 0,
                firstWorkerReceiveAbsMs: undefined,
                lastWorkerPostAbsMs: undefined,
                activeTaskMs: 0,
                kernelBoundaryMs: 0,
                resultCopyMs: 0,
                spanMs: 0
            };
            shardTrace.workerSummaries.push(summary);
        }
        summary.taskCount++;
        summary.blockCount += task.blockCount;
        summary.pointCount += task.pointCount;
        if (Number.isFinite(task.workerReceiveAbsMs)) {
            summary.firstWorkerReceiveAbsMs = Number.isFinite(summary.firstWorkerReceiveAbsMs) ?
                Math.min(summary.firstWorkerReceiveAbsMs, task.workerReceiveAbsMs) : task.workerReceiveAbsMs;
        }
        if (Number.isFinite(task.resultPostBeginAbsMs)) {
            summary.lastWorkerPostAbsMs = Number.isFinite(summary.lastWorkerPostAbsMs) ?
                Math.max(summary.lastWorkerPostAbsMs, task.resultPostBeginAbsMs) : task.resultPostBeginAbsMs;
        }
        summary.activeTaskMs += traceDurationMs(task.workerReceiveAbsMs, task.resultPostBeginAbsMs);
        summary.kernelBoundaryMs += traceDurationMs(task.kernelCallBeginAbsMs, task.kernelCallEndAbsMs);
        summary.resultCopyMs += traceDurationMs(task.resultCopyBeginAbsMs, task.resultCopyEndAbsMs);
        summary.spanMs = traceDurationMs(summary.firstWorkerReceiveAbsMs, summary.lastWorkerPostAbsMs);

        if (shardTrace.tasks.length < MAX_RECONSTRUCTION_TRACE_TASKS) {
            shardTrace.tasks.push(task);
        } else {
            shardTrace.droppedTaskCount++;
            shardTrace.truncated = true;
        }
    } catch (_) {
        // Optional trace collection must never interrupt reconstruction.
    }
}

function handleShardResult(record, message, coordinatorReceiveAbsMs) {
    const model = activeModel;
    if (!model || model.canceled || model.settled || shardPool[record.poolIndex] !== record) return;
    if (message.modelId !== model.modelId || message.requestId !== model.requestId) return;
    const range = record.inflight;
    if (!range || message.shardId !== range.shardId) return;
    try {
        assertProtocol(message);
        if (message.startBlock !== range.startBlock || message.blockCount !== range.blockCount ||
            message.startPoint !== range.startPoint || message.pointCount !== range.pointCount) {
            throw new Error('Shard worker returned a mismatched range.');
        }
        if (message.positions.length !== range.pointCount * 3 || message.scales.length !== range.pointCount * 3 ||
            message.rotations.length !== range.pointCount * 4 || message.colors.length !== range.pointCount * 4 ||
            message.validity.length !== range.pointCount) {
            throw new Error('Shard worker returned invalid typed-array lengths.');
        }
        if (model.featuresRest && (!message.featuresRest || message.featuresRest.length !== range.pointCount * 45)) {
            throw new Error('CPU reconstruction shard is missing features_rest data.');
        }

        const mergeStartedAt = performance.now();
        const mergeBeginAbsMs = model.timings.shardTrace ? absoluteNowMs() : undefined;
        model.positions.set(message.positions, range.startPoint * 3);
        model.scales.set(message.scales, range.startPoint * 3);
        model.rotations.set(message.rotations, range.startPoint * 4);
        model.colors.set(message.colors, range.startPoint * 4);
        model.validity.set(message.validity, range.startPoint);
        if (model.featuresRest) model.featuresRest.set(message.featuresRest, range.startPoint * 45);
        model.timings.resultMergeMs += performance.now() - mergeStartedAt;
        const mergeEndAbsMs = model.timings.shardTrace ? absoluteNowMs() : undefined;
        accumulateShardTimings(model.timings.shards, message.timings);
        recordCompletedShardTrace(
            model, range, message, coordinatorReceiveAbsMs, mergeBeginAbsMs, mergeEndAbsMs
        );

        clearTimeout(range.timeoutId);
        record.inflight = null;
        record.busy = false;
        model.completedBlocks += range.blockCount;
        model.completedPoints += range.pointCount;
        dispatchNext(record, model);
        if (model.completedBlocks === model.blockCount && shardPool.every((item) => !item?.busy)) finishModel(model);
    } catch (error) {
        failModel(model, error);
    }
}

function compactModel(model) {
    const pruneStartedAt = performance.now();
    let validCount = 0;
    for (let point = 0; point < model.pointCount; point++) validCount += model.validity[point] ? 1 : 0;
    if (validCount === model.pointCount) {
        model.timings.pruneMs += performance.now() - pruneStartedAt;
        return;
    }

    const positions = new Float32Array(validCount * 3);
    const scales = new Float32Array(validCount * 3);
    const rotations = new Float32Array(validCount * 4);
    const colors = new Uint8Array(validCount * 4);
    const featuresRest = model.featuresRest ? new Float32Array(validCount * 45) : null;
    const compressedTextureUVs = model.compressedTextureData?.uvs ? new Uint32Array(validCount * 2) : null;
    let destination = 0;
    for (let source = 0; source < model.pointCount; source++) {
        if (!model.validity[source]) continue;
        positions.set(model.positions.subarray(source * 3, source * 3 + 3), destination * 3);
        scales.set(model.scales.subarray(source * 3, source * 3 + 3), destination * 3);
        rotations.set(model.rotations.subarray(source * 4, source * 4 + 4), destination * 4);
        colors.set(model.colors.subarray(source * 4, source * 4 + 4), destination * 4);
        if (featuresRest) featuresRest.set(model.featuresRest.subarray(source * 45, source * 45 + 45), destination * 45);
        if (compressedTextureUVs) {
            compressedTextureUVs.set(
                model.compressedTextureData.uvs.subarray(source * 2, source * 2 + 2), destination * 2
            );
        }
        destination++;
    }
    model.pointCount = validCount;
    model.positions = positions;
    model.scales = scales;
    model.rotations = rotations;
    model.colors = colors;
    model.featuresRest = featuresRest;
    if (model.compressedTextureData) model.compressedTextureData.uvs = compressedTextureUVs;
    model.timings.pruneMs += performance.now() - pruneStartedAt;
}

function finishModel(model) {
    if (!model || model.settled || model.canceled) return;
    const reconstructionEndedAt = performance.now();
    const reconstructionEndAbsMs = absoluteNowMs();
    try {
        model.timings.wall.reconstructionEndAbsMs = reconstructionEndAbsMs;
        model.timings.wall.reconstructionEndSinceTraceOriginMs =
            reconstructionEndAbsMs - model.timings.wall.traceOriginAbsMs;
        if (Number.isFinite(model.timings.wall.reconstructionStartedAt)) {
            model.timings.wall.reconstructionWallMs =
                reconstructionEndedAt - model.timings.wall.reconstructionStartedAt;
            delete model.timings.wall.reconstructionStartedAt;
        }
        const compactStartedAt = performance.now();
        compactModel(model);
        model.timings.wall.compactModelMs = performance.now() - compactStartedAt;
        const resultAssemblyStartedAt = performance.now();
        model.settled = true;
        clearTimeout(model.deadlineId);
        const result = {
            numPoints: model.pointCount,
            positions: model.positions,
            scales: model.scales,
            rotations: model.rotations,
            colors: model.colors,
            featuresRest: model.featuresRest,
            compressedTextureData: model.compressedTextureData,
            shDegree: model.shDegree,
            timings: model.timings
        };
        model.timings.wall.resultAssemblyMs = performance.now() - resultAssemblyStartedAt;
        model.resolve(result);
    } catch (error) {
        failModel(model, error);
    }
}

async function recoverShardWorker(record, error) {
    const model = activeModel;
    if (!model || model.canceled || model.settled || shardPool[record.poolIndex] !== record) return;
    const failedRange = record.inflight;
    if (!failedRange || failedRange.retries >= 1) {
        decoderSessionPrewarmPromise = null;
        terminateShardRecord(record);
        failModel(model, error);
        return;
    }
    if (failedRange.timeoutId) clearTimeout(failedRange.timeoutId);
    decoderSessionPrewarmPromise = null;
    terminateShardRecord(record, error);
    try {
        const generation = decoderSessionGeneration;
        assertDecoderSessionActive(generation);
        const replacement = createShardWorker(record.workerId, record.poolIndex, generation);
        shardPool[record.poolIndex] = replacement;
        await replacement.readyPromise;
        if (!isDecoderSessionActive(generation) ||
            model !== activeModel || model.canceled || model.settled) return;
        await beginModelOnWorker(replacement, model);
        if (!isDecoderSessionActive(generation) ||
            model !== activeModel || model.canceled || model.settled) return;
        dispatchRange(replacement, model, { ...failedRange, retries: failedRange.retries + 1, timeoutId: undefined });
    } catch (replacementError) {
        decoderSessionPrewarmPromise = null;
        failModel(model, new Error(`${error.message} Replacement worker failed: ${replacementError.message}`));
    }
}

function handleShardWorkerCrash(record, error) {
    if (!record.alive) return;
    const failure = error instanceof Error ? error : new Error(error?.message || String(error));
    const wasReady = record.ready;
    record.alive = false;
    record.ready = false;
    if (wasReady || !activeWarmupProgress) decoderSessionPrewarmPromise = null;
    if (record.readyTimeoutId) {
        clearTimeout(record.readyTimeoutId);
        record.readyTimeoutId = null;
    }
    if (record.readyReject) {
        record.readyReject(failure);
        record.readyResolve = null;
        record.readyReject = null;
    }
    if (record.modelReady) {
        clearTimeout(record.modelReady.timeoutId);
        record.modelReady.reject(failure);
        record.modelReady = null;
    }
    const shouldRecover = !!record.inflight;
    terminateShardRecord(record, failure);
    if (shouldRecover) {
        recoverShardWorker(record, failure).catch((recoveryError) => {
            if (activeModel && !activeModel.settled) failModel(activeModel, recoveryError);
        });
    }
}

function handleShardWorkerMessage(record, message, handlerReceiveAbsMs) {
    if (!isDecoderSessionActive(record.sessionGeneration) ||
        shardPool[record.poolIndex] !== record || !record.alive) {
        return;
    }
    try {
        assertProtocol(message);
        if (message.workerId !== record.workerId) {
            throw new Error(`Shard worker identity mismatch: ${message.workerId}/${record.workerId}`);
        }
        if (message.type === 'ready') {
            const readyAt = performance.now();
            record.readyReceiveAbsMs = handlerReceiveAbsMs;
            record.workerWarmupTrace = message.warmupTrace || null;
            if (record.readyTimeoutId) {
                clearTimeout(record.readyTimeoutId);
                record.readyTimeoutId = null;
            }
            record.ready = true;
            record.moduleSource = message.moduleSource || record.moduleSource;
            record.moduleFallbackReason = message.moduleFallbackReason || record.moduleFallbackReason;
            record.moduleInitMs = roundTimingMs(message.moduleInitMs);
            record.startupMs = roundTimingMs(readyAt - record.createdAt);
            record.roundTripMs = roundTimingMs(readyAt - (record.initPostedAt ?? record.createdAt));
            record.resourceTiming = normalizeResourceTiming(message.resourceTiming);
            const resolve = record.readyResolve;
            record.readyResolve = null;
            record.readyReject = null;
            resolve(message);
            const readyWorkerCount = shardPool.reduce(
                (count, item) => count + (item?.alive && item.ready ? 1 : 0),
                0
            );
            const usesSharedModule = record.moduleSource === 'shared-compiled-module';
            postWarmupProgress(
                `shard-${readyWorkerCount}-ready`,
                `Shard workers ready ${readyWorkerCount}/${SHARD_WORKER_COUNT}` +
                    ` (${usesSharedModule ? 'shared precompiled module' : 'independent-load fallback'})`,
                {
                    workerId: record.workerId,
                    readyWorkerCount,
                    moduleSource: record.moduleSource,
                    moduleFallbackReason: record.moduleFallbackReason
                }
            );
        } else if (message.type === 'modelReady') {
            if (record.modelReady && record.modelReady.modelId === message.modelId &&
                record.modelReady.requestId === message.requestId) {
                clearTimeout(record.modelReady.timeoutId);
                const resolve = record.modelReady.resolve;
                record.modelReady = null;
                resolve(message);
            }
        } else if (message.type === 'shardResult') {
            handleShardResult(record, message, handlerReceiveAbsMs);
        } else if (message.type === 'workerError') {
            const error = new Error(message.error || `Shard worker ${record.workerId} failed.`);
            error.details = message.details;
            if (record.readyReject) {
                handleShardWorkerCrash(record, error);
                return;
            }
            if (record.modelReady) {
                if (message.modelId !== record.modelReady.modelId ||
                    message.requestId !== record.modelReady.requestId) return;
                clearTimeout(record.modelReady.timeoutId);
                record.modelReady.reject(error);
                record.modelReady = null;
            }
            if (activeModel && message.modelId === activeModel.modelId &&
                message.requestId === activeModel.requestId) {
                failModel(activeModel, error);
            }
        }
    } catch (error) {
        handleShardWorkerCrash(record, error);
    }
}

function createModel(prepared, metadata, auxiliary, requestId, videoTimings,
    traceOriginAbsMs, reconstructionTraceLevel) {
    const modelId = nextModelId++;
    const normalizedTraceOriginAbsMs = Number.isFinite(traceOriginAbsMs) ? traceOriginAbsMs : absoluteNowMs();
    const normalizedTraceLevel = reconstructionTraceLevel === 'js' ? 'js' : 'off';
    const model = {
        modelId,
        requestId,
        traceOriginAbsMs: normalizedTraceOriginAbsMs,
        reconstructionTraceLevel: normalizedTraceLevel,
        pointCount: prepared.pointCount,
        shDegree: prepared.shDegree,
        blockCount: prepared.blockCount,
        pointsPerBlock: prepared.pointsPerBlock,
        blocksPerShard: prepared.decodeFeaturesRest ? CPU_BLOCKS_PER_SHARD : GPU_BLOCKS_PER_SHARD,
        metadata,
        positions: new Float32Array(prepared.pointCount * 3),
        scales: new Float32Array(prepared.pointCount * 3),
        rotations: new Float32Array(prepared.pointCount * 4),
        colors: new Uint8Array(prepared.pointCount * 4),
        validity: new Uint8Array(prepared.pointCount),
        featuresRest: prepared.decodeFeaturesRest ? new Float32Array(prepared.pointCount * 45) : null,
        ...auxiliary,
        nextBlock: 0,
        nextShardId: 1,
        completedBlocks: 0,
        completedPoints: 0,
        canceled: false,
        settled: false,
        deadlineId: null,
        resolve: null,
        reject: null,
        completion: null,
        timings: {
            prepare: {
                parseMs: prepared.parseMs,
                decodeNonVideoSubstreamsMs: prepared.decodeNonVideoSubstreamsMs,
                astcTextureDecodeMs: prepared.astcTextureDecodeMs,
                bcTextureEncodeMs: prepared.bcTextureEncodeMs,
                textureInputBytes: prepared.textureInputBytes,
                textureOutputBytes: prepared.textureOutputBytes,
                decodeVideoFallbackMs: prepared.decodeVideoFallbackMs,
                decodeSubstreamsMs: prepared.decodeSubstreamsMs,
                totalMs: prepared.totalMs,
                substreams: buildSubstreamTimings(prepared, videoTimings),
                ...videoTimings
            },
            packShardMs: 0,
            unpackMs: 0,
            taskExportCopyMs: 0,
            resultMergeMs: 0,
            pruneMs: 0,
            wall: {
                ...(prepared.wallTimings || {}),
                clockMethod: absoluteClockMethod(),
                traceOriginAbsMs: normalizedTraceOriginAbsMs
            },
            shards: {},
            ...(normalizedTraceLevel === 'js' ? {
                shardTrace: {
                    schema: 'compactgs.reconstruction.shard-trace.v1',
                    clock: 'performance.timeOrigin+performance.now; fallback=Date.now',
                    origin: normalizedTraceOriginAbsMs,
                    originAbsMs: normalizedTraceOriginAbsMs,
                    level: normalizedTraceLevel,
                    tasks: [],
                    droppedTaskCount: 0,
                    truncated: false,
                    workerSummaries: []
                }
            } : {})
        }
    };
    model.completion = new Promise((resolve, reject) => {
        model.resolve = resolve;
        model.reject = reject;
    });
    model.deadlineId = setTimeout(() => failModel(model, new Error('GaussianSlimLoader model decode timed out.')), MODEL_TIMEOUT_MS);
    return model;
}

async function decodeTextureStrategy(request, textureStrategyValue) {
    const decodeStartedAt = performance.now();
    const textureStrategy = normalizeTextureStrategy(textureStrategyValue);
    if (textureStrategy === TextureStrategy.BC7 && !coordinatorTextureBuildCapabilities.bc7) {
        throw new Error('BC7 texture strategy requested, but this worker build has no BC7 encoder.');
    }
    if (textureStrategy === TextureStrategy.BC3 && !coordinatorTextureBuildCapabilities.bc3) {
        throw new Error('BC3 texture strategy requested, but this worker build has no BC3 encoder.');
    }

    const payloadExtractStartedAt = performance.now();
    const payload = extractCompressedPayload(request.buffer);
    const input = new Uint8Array(payload.bytes);
    const payloadExtractMs = performance.now() - payloadExtractStartedAt;
    postProgress(request.requestId, `payload ${payload.reason}, bytes=${input.byteLength}`);
    const moduleStartedAt = performance.now();
    const Module = await getCoordinatorModule();
    const coordinatorModuleWaitMs = performance.now() - moduleStartedAt;
    const preparation = {
        requestId: request.requestId,
        abortController: new AbortController()
    };
    activePreparation = preparation;
    let model = null;
    try {
        const inputCopyStartedAt = performance.now();
        const inputPtr = Module._malloc(input.byteLength);
        if (!inputPtr) throw new Error(`Failed to allocate ${input.byteLength} coordinator input bytes.`);
        let staged;
        try {
            Module.HEAPU8.set(input, inputPtr);
            const beginPrepareStartedAt = performance.now();
            staged = coordinator.beginPrepare(
                inputPtr, input.byteLength, textureStrategyToNativeMode(textureStrategy), payload.compressedPayload
            );
            staged.wallTimings = {
                requestToDecodeStartMs: Number.isFinite(request.postedEpochMs) ? Date.now() - request.postedEpochMs : 0,
                payloadExtractMs,
                coordinatorModuleWaitMs,
                inputHeapCopyAndMallocMs: beginPrepareStartedAt - inputCopyStartedAt,
                beginPrepareWallMs: performance.now() - beginPrepareStartedAt
            };
        } finally {
            Module._free(inputPtr);
        }
        if (!staged.success) {
            throw new Error(coordinator.getLastError() || 'Coordinator staged prepare failed.');
        }
        if (staged.protocolVersion !== PROTOCOL_VERSION || staged.buildVersion !== BUILD_VERSION) {
            throw new Error('Coordinator staged artifact protocol/build mismatch.');
        }

        const videoPrepareStartedAt = performance.now();
        const videoTimings = await preparePendingVideos(Module, staged, preparation);
        const videoPrepareWallMs = performance.now() - videoPrepareStartedAt;
        assertPreparationActive(preparation);
        const finishPrepareStartedAt = performance.now();
        const prepared = coordinator.finishPrepare();
        prepared.wallTimings = {
            ...(staged.wallTimings || {}),
            videoPrepareWallMs,
            finishPrepareWallMs: performance.now() - finishPrepareStartedAt
        };
        if (!prepared.success) {
            throw new Error(coordinator.getLastError() || 'Coordinator finish prepare failed.');
        }
        if (prepared.protocolVersion !== PROTOCOL_VERSION || prepared.buildVersion !== BUILD_VERSION) {
            throw new Error('Coordinator artifact protocol/build mismatch.');
        }
        if (prepared.pointCount === 0 || prepared.blockCount === 0 || prepared.pointsPerBlock === 0) {
            throw new Error('Coordinator returned an invalid decode descriptor.');
        }
        const expectsCpuFeatures = textureStrategy === TextureStrategy.CPU;
        const requestedNativeMode = textureStrategyToNativeMode(textureStrategy);
        if (Number.isFinite(prepared.textureOutputMode) && prepared.textureOutputMode !== requestedNativeMode) {
            throw new Error(
                `Coordinator returned texture mode ${prepared.textureOutputMode}; requested ${requestedNativeMode}.`
            );
        }
        if (!!prepared.decodeFeaturesRest !== expectsCpuFeatures) {
            throw new Error(
                `Coordinator returned an output mode inconsistent with ${textureStrategy} texture strategy.`
            );
        }

        const auxiliaryCopyStartedAt = performance.now();
        const metadata = Module.HEAPU8.slice(prepared.metadataPtr, prepared.metadataPtr + prepared.metadataSize).buffer;
        let compressedTextureData = null;
        if (textureStrategy !== TextureStrategy.CPU) {
            const raw = prepared.astcRawSize > 0 ?
                Module.HEAPU8.slice(prepared.astcRawPtr, prepared.astcRawPtr + prepared.astcRawSize) : null;
            const uvs = copyCompressedTextureUvs(Module, prepared, textureStrategy);
            const metas = prepared.astcMetasCount > 0 ?
                Module.HEAPU32.slice(prepared.astcMetasPtr >>> 2,
                    (prepared.astcMetasPtr >>> 2) + prepared.astcMetasCount) : null;
            if (!raw || !metas || metas.length < 6 || metas.length % 6 !== 0) {
                throw new Error(`Coordinator returned an incomplete ${textureStrategy} texture payload.`);
            }
            const blockWidth = metas[0];
            const blockHeight = metas[0];
            const width = metas[1];
            const height = metas[2];
            const layers = metas.length / 6;
            const singleWidth = metas[3];
            const regionPixelCount = metas[4];
            const singleHeight = singleWidth > 0 ? regionPixelCount / singleWidth : 0;
            const validAstcBlocks = [4, 5, 6, 8, 10, 12];
            if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 ||
                width > 16384 || height > 16384 || layers !== 1 || prepared.textureNum !== layers ||
                !Number.isInteger(singleWidth) || !Number.isInteger(singleHeight) ||
                singleWidth <= 0 || singleHeight <= 0 || singleWidth > width || singleHeight > height ||
                regionPixelCount < prepared.pointCount ||
                textureStrategy === TextureStrategy.ASTC && !validAstcBlocks.includes(blockWidth) ||
                textureStrategy !== TextureStrategy.ASTC && (blockWidth !== 4 || blockHeight !== 4) ||
                !Number.isFinite(prepared.shnMin) || !Number.isFinite(prepared.shnMax) ||
                prepared.shnMax < prepared.shnMin) {
                throw new Error(`Coordinator returned invalid ${textureStrategy} texture metadata.`);
            }
            let metadataBytes = 0;
            for (let layer = 0; layer < layers; layer++) {
                const offset = layer * 6;
                if (metas[offset] !== blockWidth || metas[offset + 1] !== width || metas[offset + 2] !== height ||
                    metas[offset + 3] !== singleWidth || metas[offset + 4] !== regionPixelCount) {
                    throw new Error(`Coordinator returned inconsistent ${textureStrategy} texture layers.`);
                }
                metadataBytes += metas[offset + 5];
            }
            const expectedBytes = Math.ceil(width / blockWidth) * Math.ceil(height / blockHeight) * 16 * layers;
            if (raw.byteLength !== metadataBytes || raw.byteLength !== expectedBytes) {
                throw new Error(`Coordinator returned an invalid ${textureStrategy} texture byte count.`);
            }
            compressedTextureData = {
                format: textureStrategy,
                raw,
                uvs,
                metas,
                width,
                height,
                layers,
                blockWidth,
                blockHeight,
                singleWidth,
                singleHeight,
                shnMin: prepared.shnMin,
                shnMax: prepared.shnMax,
                textureNum: prepared.textureNum
            };
        }
        prepared.wallTimings.auxiliaryCopyMs = performance.now() - auxiliaryCopyStartedAt;
        if (prepared.profileIdc === 2) {
            const exportStartedAt = performance.now();
            const result = copyCompactGSPreparedResult(Module, prepared, compressedTextureData, videoTimings);
            result.textureStrategy = textureStrategy;
            result.timings.compactgsPlaneExportMs = performance.now() - exportStartedAt;
            result.timings.wall.totalDecodeFunctionMs = performance.now() - decodeStartedAt;
            activePreparation = null;
            postProgress(request.requestId,
                `COMPACTGS_V1 prepared points=${result.numPoints}, map=${prepared.pointMapWidth}x${prepared.pointMapHeight}, ` +
                `video=${videoTimings.videoDecoderPath}`);
            return result;
        }
        const auxiliary = { compressedTextureData };
        const shardPoolWaitStartedAt = performance.now();
        const readyShardPool = await ensureTargetShardPool();
        prepared.wallTimings.shardPoolWaitMs = performance.now() - shardPoolWaitStartedAt;
        assertPreparationActive(preparation);

        model = createModel(
            prepared, metadata, auxiliary, request.requestId, videoTimings,
            request.traceOriginAbsMs, request.reconstructionTraceLevel
        );
        activePreparation = null;
        activeModel = model;
        postProgress(request.requestId,
            `prepared points=${model.pointCount}, blocks=${model.blockCount}, ` +
            `video=${videoTimings.videoDecoderPath}, shardWorkers=${SHARD_WORKER_COUNT}`);
        try {
            const beginShardWorkersStartedAt = performance.now();
            await Promise.all(readyShardPool.map((record) => beginModelOnWorker(record, model)));
            model.timings.wall.beginShardWorkersMs = performance.now() - beginShardWorkersStartedAt;
            model.timings.wall.reconstructionStartedAt = performance.now();
            model.timings.wall.reconstructionStartAbsMs = absoluteNowMs();
            model.timings.wall.reconstructionStartSinceTraceOriginMs =
                model.timings.wall.reconstructionStartAbsMs - model.timings.wall.traceOriginAbsMs;
            readyShardPool.forEach((record) => dispatchNext(record, model));
        } catch (error) {
            failModel(model, error);
        }
        const result = await model.completion;
        result.timings.wall.totalDecodeFunctionMs = performance.now() - decodeStartedAt;
        return result;
    } finally {
        if (activePreparation === preparation) activePreparation = null;
        coordinator?.release();
        if (activeModel === model) activeModel = null;
    }
}

function isTerminalTextureStrategyError(error) {
    const message = error?.message || String(error);
    return error?.name === 'AbortError' || /cancel|disposed|timed out|protocol\/build mismatch/i.test(message);
}

async function decode(request) {
    await prewarmDecoderSession();
    assertDecoderSessionActive();
    if (activePreparation || activeModel && !activeModel.settled) {
        throw new Error('A GaussianSlimLoader model is already being decoded.');
    }
    if (!(request.buffer instanceof ArrayBuffer)) throw new Error('Decode request is missing an ArrayBuffer.');

    const textureStrategies = normalizeTextureStrategyChain(
        request.textureStrategies || request.textureStrategy,
        TextureStrategy.CPU
    );
    const attempts = [];
    const failures = [];
    for (const textureStrategy of textureStrategies) {
        attempts.push(textureStrategy);
        try {
            const data = await decodeTextureStrategy(request, textureStrategy);
            data.textureStrategy = textureStrategy;
            data.textureAttempts = attempts.slice();
            data.textureFallbackErrors = failures;
            return data;
        } catch (error) {
            failures.push({ strategy: textureStrategy, message: error?.message || String(error) });
            if (isTerminalTextureStrategyError(error)) throw error;
            if (verboseLog) {
                console.warn(
                    `[GaussianSlimLoader coordinator] Texture strategy ${textureStrategy} failed; ` +
                    `trying the next fallback.`, error
                );
            }
        }
    }

    const summary = failures.map((failure) => `${failure.strategy}: ${failure.message}`).join('; ');
    const error = new Error(`All texture strategies failed (${attempts.join(' -> ')}). ${summary}`);
    error.details = { textureAttempts: attempts, textureFailures: failures };
    throw error;
}

function buildErrorDetails(error, stage) {
    return {
        name: error?.name,
        message: error?.message || String(error),
        stack: error?.stack,
        stage,
        modelId: activeModel?.modelId,
        requestId: activeModel?.requestId ?? activePreparation?.requestId,
        details: error?.details
    };
}

export function buildDecodeTransferList(data) {
    const transferList = data.variant === 'compactgs' ? [
        data.descriptor.positionGroupMin.buffer,
        data.descriptor.positionGroupStep.buffer,
        data.descriptor.shapeMin.buffer,
        data.descriptor.shapeStep.buffer,
        ...(data.descriptor.covarianceGroupMin ? [data.descriptor.covarianceGroupMin.buffer,
            data.descriptor.covarianceGroupStep.buffer] : []),
        data.descriptor.shMin.buffer,
        data.descriptor.shStep.buffer,
        data.planes.positionLow.buffer,
        data.planes.positionHigh.buffer,
        data.planes.colorRgb.buffer,
        data.planes.shape.buffer
    ] : [data.positions.buffer, data.scales.buffer, data.rotations.buffer, data.colors.buffer];
    if (data.variant === 'compactgs' && data.planes.shAtlas) transferList.push(data.planes.shAtlas.buffer);
    if (data.variant === 'compactgs' && data.covariances) transferList.push(data.covariances.buffer);
    if (data.featuresRest) transferList.push(data.featuresRest.buffer);
    if (data.compressedTextureData) {
        transferList.push(data.compressedTextureData.raw.buffer);
        if (data.compressedTextureData.uvs) {
            transferList.push(data.compressedTextureData.uvs.buffer);
        }
        transferList.push(data.compressedTextureData.metas.buffer);
    }
    return transferList;
}

function transferDecodeResult(requestId, data) {
    const transferPrepStartedAt = performance.now();
    const transferList = buildDecodeTransferList(data);
    const sentAt = performance.now();
    const sentEpochMs = Date.now();
    if (data.timings?.wall) {
        data.timings.wall.transferPrepMs = sentAt - transferPrepStartedAt;
        data.timings.wall.sentAt = sentAt;
        data.timings.wall.sentEpochMs = sentEpochMs;
    }
    self.postMessage({
        type: 'decodeResult',
        protocolVersion: PROTOCOL_VERSION,
        buildVersion: BUILD_VERSION,
        requestId,
        success: true,
        data,
        sentAt,
        sentEpochMs
    }, transferList);
}

function dispose() {
    if (!decoderSessionActive) return;
    decoderSessionActive = false;
    decoderSessionGeneration++;
    activeWarmupProgress = null;
    webCodecsProbeAbortController?.abort(sessionDisposedError());
    if (activePreparation && !activePreparation.abortController.signal.aborted) {
        activePreparation.abortController.abort(new Error('Decoder disposed.'));
    }
    if (activeModel && !activeModel.settled) failModel(activeModel, new Error('Decoder disposed.'));
    try {
        coordinator?.release();
    } catch (_) {}
    shardPool.forEach((record) => {
        if (!record) return;
        if (record.alive && record.worker) {
            try {
                record.worker.postMessage({
                    type: 'dispose',
                    protocolVersion: PROTOCOL_VERSION,
                    buildVersion: BUILD_VERSION,
                    workerId: record.workerId
                });
            } catch (_) {}
        }
        terminateShardRecord(record, sessionDisposedError());
    });
    shardPool = [];
    coordinator?.delete();
    coordinator = null;
    coordinatorModule = null;
    coordinatorModulePromise = null;
    reconstructionModulePreparation = null;
    reconstructionModulePreparationPromise = null;
    coordinatorWasmUrl = COORDINATOR_WASM_URL;
    coordinatorFlavor = 'single-thread';
    coordinatorTextureBuildCapabilities = { ...TextureBuildCapabilities };
    decoderSessionPrewarmPromise = null;
    webCodecsSessionPolicy = null;
    webCodecsSessionPolicyPromise = null;
    webCodecsProbeAbortController = null;
    webCodecsProbeStatus = 'disposed';
    webCodecsNegativeCapabilities.clear();
}

export async function handleDecoderWorkerMessage(event) {
    const handlerReceiveAbsMs = absoluteNowMs();
    const request = event.data || {};
    const requestId = request.requestId ?? 0;
    const requestWarmupTrace = request.type === 'warmup' ?
        createDecoderWarmupTrace(handlerReceiveAbsMs, requestId, request.bootstrapTrace) : null;
    verboseLog = !!request.verboseLog;
    let stage = request.type || 'unknown';
    try {
        if (request.type === 'warmup') {
            assertProtocol(request);
            const startedAt = performance.now();
            const stats = await prewarmDecoderSession(requestId, requestWarmupTrace);
            const response = {
                type: 'warmupResult',
                protocolVersion: PROTOCOL_VERSION,
                buildVersion: BUILD_VERSION,
                requestId,
                success: true,
                elapsedMs: performance.now() - startedAt,
                stats
            };
            stats.warmupTrace.decoder.resultPostBeginAbsMs = absoluteNowMs();
            self.postMessage(response);
        } else if (request.type === 'decode') {
            assertProtocol(request);
            stage = 'decode';
            const data = await decode(request);
            if (verboseLog) console.log('[GaussianSlimLoader coordinator] timings', data.timings);
            transferDecodeResult(requestId, data);
        } else if (request.type === 'cancelModel') {
            assertProtocol(request);
            if (activePreparation && (!request.requestId || activePreparation.requestId === request.requestId) &&
                !activePreparation.abortController.signal.aborted) {
                activePreparation.abortController.abort(new Error('GaussianSlimLoader decode canceled.'));
            } else if (activeModel && (!request.modelId || activeModel.modelId === request.modelId)) {
                failModel(activeModel, new Error('GaussianSlimLoader decode canceled.'));
            }
        } else if (request.type === 'dispose') {
            dispose();
            self.postMessage({
                type: 'disposeResult',
                protocolVersion: PROTOCOL_VERSION,
                buildVersion: BUILD_VERSION,
                requestId,
                success: true
            });
            self.close();
        } else {
            throw new Error(`Unknown decoder worker request type: ${request.type}`);
        }
    } catch (error) {
        if (!decoderSessionActive && request.type !== 'dispose') return;
        const details = buildErrorDetails(error, stage);
        const response = {
            type: request.type === 'warmup' ? 'warmupResult' :
                (request.type === 'dispose' ? 'disposeResult' : 'decodeResult'),
            protocolVersion: PROTOCOL_VERSION,
            buildVersion: BUILD_VERSION,
            requestId,
            success: false,
            error: details.message,
            details,
            ...(requestWarmupTrace ? { stats: { warmupTrace: requestWarmupTrace } } : {})
        };
        if (requestWarmupTrace) {
            requestWarmupTrace.decoder.resultPostBeginAbsMs = absoluteNowMs();
            requestWarmupTrace.attempt.status = 'error';
            requestWarmupTrace.attempt.failureStage = stage;
        }
        self.postMessage(response);
    }
}

if (!self.__COMPACTGS_SPLAT_DECODER_BOOTSTRAP__) {
    self.onmessage = handleDecoderWorkerMessage;
}
