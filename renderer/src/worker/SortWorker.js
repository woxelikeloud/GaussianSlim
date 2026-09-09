import SorterWasm from './sorter.wasm';
import SorterWasmNoSIMD from './sorter_no_simd.wasm';
import { Constants } from '../Constants.js';

function sortWorker(self) {

    let wasmInstance;
    let wasmMemory;
    let integerBasedSort;
    let dynamicMode;
    let splatCount;
    let indexesToSortOffset;
    let sortedIndexesOffset;
    let sceneIndexesOffset;
    let transformsOffset;
    let precomputedDistancesOffset;
    let mappedDistancesOffset;
    let frequenciesOffset;
    let centersOffset;
    let positionLowOffset;
    let positionHighOffset;
    let positionGroupMinOffset;
    let positionGroupStepOffset;
    let compactGaussianQuantizedMode;
    let compactGaussianGroupSize;
    let modelViewProjOffset;
    let countsZero;
    let sortedIndexesOut;
    let distanceMapRange;
    let uploadedSplatCount;
    let Constants;

    function sort(splatSortCount, splatRenderCount, modelViewProj,
                  usePrecomputedDistances, copyIndexesToSort, copyPrecomputedDistances, copyTransforms) {
        const sortStartTime = performance.now();

        const indexesToSort = new Uint32Array(wasmMemory, indexesToSortOffset,
                                              copyIndexesToSort.byteLength / Constants.BytesPerInt);
        indexesToSort.set(copyIndexesToSort);
        if (dynamicMode) {
            const transforms = new Float32Array(wasmMemory, transformsOffset, copyTransforms.byteLength / Constants.BytesPerFloat);
            transforms.set(copyTransforms);
        }
        if (usePrecomputedDistances) {
            let precomputedDistances;
            if (integerBasedSort) {
                precomputedDistances = new Int32Array(wasmMemory, precomputedDistancesOffset,
                                                      copyPrecomputedDistances.byteLength / Constants.BytesPerInt);
            } else {
                precomputedDistances = new Float32Array(wasmMemory, precomputedDistancesOffset,
                                                        copyPrecomputedDistances.byteLength / Constants.BytesPerFloat);
            }
            precomputedDistances.set(copyPrecomputedDistances);
        }

        if (!countsZero) countsZero = new Uint32Array(distanceMapRange);
        new Float32Array(wasmMemory, modelViewProjOffset, 16).set(modelViewProj);
        new Uint32Array(wasmMemory, frequenciesOffset, distanceMapRange).set(countsZero);
        if (compactGaussianQuantizedMode) {
            wasmInstance.exports.sortIndexesQuantized(
                indexesToSortOffset, positionLowOffset, positionHighOffset,
                positionGroupMinOffset, positionGroupStepOffset, compactGaussianGroupSize,
                mappedDistancesOffset, frequenciesOffset, modelViewProjOffset, sortedIndexesOffset,
                distanceMapRange, splatSortCount, splatRenderCount, splatCount
            );
        } else {
            wasmInstance.exports.sortIndexes(indexesToSortOffset, centersOffset, precomputedDistancesOffset,
                                             mappedDistancesOffset, frequenciesOffset, modelViewProjOffset,
                                             sortedIndexesOffset, sceneIndexesOffset, transformsOffset, distanceMapRange,
                                             splatSortCount, splatRenderCount, splatCount, usePrecomputedDistances, integerBasedSort,
                                             dynamicMode);
        }

        const sortMessage = {
            'sortDone': true,
            'splatSortCount': splatSortCount,
            'splatRenderCount': splatRenderCount,
            'sortTime': 0
        };
        const sortedIndexes = new Uint32Array(wasmMemory, sortedIndexesOffset, splatRenderCount);
        if (!sortedIndexesOut || sortedIndexesOut.length < splatRenderCount) {
            sortedIndexesOut = new Uint32Array(splatRenderCount);
        }
        sortedIndexesOut.set(sortedIndexes);
        sortMessage.sortedIndexes = sortedIndexesOut;
        const sortEndTime = performance.now();

        sortMessage.sortTime = sortEndTime - sortStartTime;

        self.postMessage(sortMessage);
    }

    self.onmessage = (e) => {
        if (e.data.centers) {
            centers = e.data.centers;
            sceneIndexes = e.data.sceneIndexes;
            if (integerBasedSort) {
                new Int32Array(wasmMemory, centersOffset + e.data.range.from * Constants.BytesPerInt * 4,
                               e.data.range.count * 4).set(new Int32Array(centers));
            } else {
                new Float32Array(wasmMemory, centersOffset + e.data.range.from * Constants.BytesPerFloat * 4,
                                 e.data.range.count * 4).set(new Float32Array(centers));
            }
            if (dynamicMode) {
                new Uint32Array(wasmMemory, sceneIndexesOffset + e.data.range.from * 4,
                                e.data.range.count).set(new Uint32Array(sceneIndexes));
            }
            uploadedSplatCount = e.data.range.from + e.data.range.count;
        } else if (e.data.sort) {
            const usePrecomputedDistances = e.data.sort.usePrecomputedDistances;
            // When distances are supplied by the GPU, the worker does not
            // need center data at all. Do not clamp the render range to the
            // CPU-center upload watermark (which is intentionally zero on
            // the GPU path).
            const availableSplatCount = usePrecomputedDistances ? splatCount : uploadedSplatCount;
            const renderCount = Math.min(e.data.sort.splatRenderCount || 0, availableSplatCount);
            const sortCount = Math.min(e.data.sort.splatSortCount || 0, availableSplatCount);

            const copyIndexesToSort = e.data.sort.indexesToSort;
            const copyTransforms = e.data.sort.transforms;
            const copyPrecomputedDistances = usePrecomputedDistances ? e.data.sort.precomputedDistances : null;
            sort(sortCount, renderCount, e.data.sort.modelViewProj, usePrecomputedDistances,
                 copyIndexesToSort, copyPrecomputedDistances, copyTransforms);
        } else if (e.data.init) {
            // Yep, this is super hacky and gross :(
            Constants = e.data.init.Constants;

            splatCount = e.data.init.splatCount;
            integerBasedSort = e.data.init.integerBasedSort;
            dynamicMode = e.data.init.dynamicMode;
            const fastPositions = e.data.init.compactGaussianQuantizedPositions || null;
            compactGaussianQuantizedMode = !!fastPositions;
            compactGaussianGroupSize = fastPositions?.groupSize || 0;
            distanceMapRange = e.data.init.distanceMapRange;
            uploadedSplatCount = 0;

            const CENTERS_BYTES_PER_ENTRY = integerBasedSort ? (Constants.BytesPerInt * 4) : (Constants.BytesPerFloat * 4);

            const sorterWasmBytes = new Uint8Array(e.data.init.sorterWasmBytes);

            const matrixSize = 16 * Constants.BytesPerFloat;
            const memoryRequiredForIndexesToSort = splatCount * Constants.BytesPerInt;
            const memoryRequiredForCenters = compactGaussianQuantizedMode ? 0 : splatCount * CENTERS_BYTES_PER_ENTRY;
            const memoryRequiredForFastPositionPlane = compactGaussianQuantizedMode ? splatCount * 3 : 0;
            const memoryRequiredForFastPositionGroups = compactGaussianQuantizedMode ? fastPositions.groupMin.byteLength : 0;
            const memoryRequiredForModelViewProjectionMatrix = matrixSize;
            const memoryRequiredForPrecomputedDistances = integerBasedSort ?
                                                          (splatCount * Constants.BytesPerInt) : (splatCount * Constants.BytesPerFloat);
            const memoryRequiredForMappedDistances = splatCount * Constants.BytesPerInt;
            const memoryRequiredForSortedIndexes = splatCount * Constants.BytesPerInt;
            const memoryRequiredForIntermediateSortBuffers = integerBasedSort ? (distanceMapRange * Constants.BytesPerInt * 2) :
                                                                                (distanceMapRange * Constants.BytesPerFloat * 2);
            const memoryRequiredforTransformIndexes = dynamicMode ? (splatCount * Constants.BytesPerInt) : 0;
            const memoryRequiredforTransforms = dynamicMode ? (Constants.MaxScenes * matrixSize) : 0;
            const extraMemory = Constants.MemoryPageSize * 32;

            const totalRequiredMemory = memoryRequiredForIndexesToSort +
                                        memoryRequiredForCenters +
                                        memoryRequiredForModelViewProjectionMatrix +
                                        memoryRequiredForPrecomputedDistances +
                                        memoryRequiredForMappedDistances +
                                        memoryRequiredForIntermediateSortBuffers +
                                        memoryRequiredForSortedIndexes +
                                        memoryRequiredforTransformIndexes +
                                        memoryRequiredforTransforms +
                                        memoryRequiredForFastPositionPlane * 2 +
                                        memoryRequiredForFastPositionGroups * 2 +
                                        extraMemory;
            const totalPagesRequired = Math.floor(totalRequiredMemory / Constants.MemoryPageSize ) + 1;
            const sorterWasmImport = {
                module: {},
                env: {
                    memory: new WebAssembly.Memory({
                        initial: totalPagesRequired,
                        maximum: totalPagesRequired
                    }),
                }
            };
            WebAssembly.compile(sorterWasmBytes)
            .then((wasmModule) => {
                return WebAssembly.instantiate(wasmModule, sorterWasmImport);
            })
            .then((instance) => {
                wasmInstance = instance;
                indexesToSortOffset = 0;
                centersOffset = indexesToSortOffset + memoryRequiredForIndexesToSort;
                modelViewProjOffset = centersOffset + memoryRequiredForCenters;
                precomputedDistancesOffset = modelViewProjOffset + memoryRequiredForModelViewProjectionMatrix;
                mappedDistancesOffset = precomputedDistancesOffset + memoryRequiredForPrecomputedDistances;
                frequenciesOffset = mappedDistancesOffset + memoryRequiredForMappedDistances;
                sortedIndexesOffset = frequenciesOffset + memoryRequiredForIntermediateSortBuffers;
                sceneIndexesOffset = sortedIndexesOffset + memoryRequiredForSortedIndexes;
                transformsOffset = sceneIndexesOffset + memoryRequiredforTransformIndexes;
                positionLowOffset = transformsOffset + memoryRequiredforTransforms;
                positionHighOffset = positionLowOffset + memoryRequiredForFastPositionPlane;
                positionGroupMinOffset = (positionHighOffset + memoryRequiredForFastPositionPlane + 3) & ~3;
                positionGroupStepOffset = positionGroupMinOffset + memoryRequiredForFastPositionGroups;
                wasmMemory = sorterWasmImport.env.memory.buffer;
                if (compactGaussianQuantizedMode) {
                    new Uint8Array(wasmMemory, positionLowOffset, memoryRequiredForFastPositionPlane)
                        .set(new Uint8Array(fastPositions.positionLow));
                    new Uint8Array(wasmMemory, positionHighOffset, memoryRequiredForFastPositionPlane)
                        .set(new Uint8Array(fastPositions.positionHigh));
                    new Float32Array(wasmMemory, positionGroupMinOffset, fastPositions.groupMin.byteLength / 4)
                        .set(new Float32Array(fastPositions.groupMin));
                    new Float32Array(wasmMemory, positionGroupStepOffset, fastPositions.groupStep.byteLength / 4)
                        .set(new Float32Array(fastPositions.groupStep));
                    uploadedSplatCount = splatCount;
                }
                self.postMessage({
                    'sortSetupPhase1Complete': true
                });
            });
        }
    };
}

export function createSortWorker(splatCount, enableSIMDInSort, integerBasedSort, dynamicMode,
                                 splatSortDistanceMapPrecision = Constants.DefaultSplatSortDistanceMapPrecision,
                                 compactGaussianQuantizedPositions = null) {
    const worker = new Worker(
        URL.createObjectURL(
            new Blob(['(', sortWorker.toString(), ')(self)'], {
                type: 'application/javascript',
            }),
        ),
    );

    const sourceWasm = enableSIMDInSort ? SorterWasm : SorterWasmNoSIMD;

    const sorterWasmBinaryString = atob(sourceWasm);
    const sorterWasmBytes = new Uint8Array(sorterWasmBinaryString.length);
    for (let i = 0; i < sorterWasmBinaryString.length; i++) {
        sorterWasmBytes[i] = sorterWasmBinaryString.charCodeAt(i);
    }

    let compactGaussianInit = null;
    const transferList = [sorterWasmBytes.buffer];
    if (compactGaussianQuantizedPositions) {
        const positionLow = compactGaussianQuantizedPositions.positionLow.slice().buffer;
        const positionHigh = compactGaussianQuantizedPositions.positionHigh.slice().buffer;
        const groupMin = compactGaussianQuantizedPositions.groupMin.slice().buffer;
        const groupStep = compactGaussianQuantizedPositions.groupStep.slice().buffer;
        compactGaussianInit = { positionLow, positionHigh, groupMin, groupStep,
                       groupSize: compactGaussianQuantizedPositions.groupSize };
        transferList.push(positionLow, positionHigh, groupMin, groupStep);
    }
    worker.postMessage({
        'init': {
            'sorterWasmBytes': sorterWasmBytes.buffer,
            'splatCount': splatCount,
            'integerBasedSort': integerBasedSort,
            'dynamicMode': dynamicMode,
            'distanceMapRange': 1 << splatSortDistanceMapPrecision,
            'compactGaussianQuantizedPositions': compactGaussianInit,
            // Super hacky
            'Constants': {
                'BytesPerFloat': Constants.BytesPerFloat,
                'BytesPerInt': Constants.BytesPerInt,
                'MemoryPageSize': Constants.MemoryPageSize,
                'MaxScenes': Constants.MaxScenes
            }
        }
    }, transferList);
    return worker;
}
