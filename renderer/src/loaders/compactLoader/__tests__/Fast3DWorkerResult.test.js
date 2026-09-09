import assert from 'node:assert/strict';
import test from 'node:test';

globalThis.self = { __COMPACTGS_SPLAT_DECODER_BOOTSTRAP__: true };
const {
    buildDecodeTransferList,
    copyCompressedTextureUvs,
    copyCompactGSPreparedResult,
    extractCompressedPayload
} = await import('../SplatDecoder.worker.js');

function fixture(width = 4, height = 4) {
    const capacity = width * height;
    const heap = new ArrayBuffer(capacity * 66 + 4096);
    const HEAPU8 = new Uint8Array(heap);
    const HEAPF32 = new Float32Array(heap);
    let pointer = 64;
    const putBytes = (length, fill) => {
        const offset = pointer;
        HEAPU8.fill(fill, offset, offset + length);
        pointer += length;
        return offset;
    };
    const putFloats = (values) => {
        pointer = (pointer + 3) & ~3;
        const offset = pointer;
        HEAPF32.set(values, offset >>> 2);
        pointer += values.length * 4;
        return offset;
    };
    const prepared = {
        profileIdc: 2,
        layoutVersion: 2,
        pointCount: 3,
        shDegree: 3,
        pointMapWidth: width,
        pointMapHeight: height,
        positionGroupSize: 256,
        positionGroupFloatCount: 3,
        positionGroupMinPtr: putFloats([1, 2, 3]),
        positionGroupStepPtr: putFloats([0.1, 0.2, 0.3]),
        shapeComponentCount: 7,
        shapeMinPtr: putFloats([1, 2, 3, 4, 5, 6, 7]),
        shapeStepPtr: putFloats([.1, .2, .3, .4, .5, .6, .7]),
        importanceMin: 0.1,
        importanceStep: 0.9 / 255,
        fastShComponentCount: 45,
        fastShMinPtr: putFloats(Array.from({ length: 45 }, (_, index) => -1 + index * 0.01)),
        fastShStepPtr: putFloats(new Array(45).fill(2 / 255)),
        positionLowSize: width * height * 3,
        positionHighSize: width * height * 3,
        colorRgbSize: width * height * 3,
        shapePlaneSize: width * 4 * height * 3,
        decodeFeaturesRest: true,
        shPlaneSize: width * 4 * height * 4 * 3,
        encodedMinimumAlpha: 0,
        parseMs: 1,
        decodeNonVideoSubstreamsMs: 2,
        astcTextureDecodeMs: 0,
        bcTextureEncodeMs: 0,
        textureInputBytes: 0,
        textureOutputBytes: 0,
        decodeVideoFallbackMs: 0,
        decodeSubstreamsMs: 3,
        totalMs: 4,
        wallTimings: {}
    };
    prepared.positionLowPtr = putBytes(prepared.positionLowSize, 1);
    prepared.positionHighPtr = putBytes(prepared.positionHighSize, 2);
    prepared.colorRgbPtr = putBytes(prepared.colorRgbSize, 3);
    prepared.shapePlanePtr = putBytes(prepared.shapePlaneSize, 4);
    prepared.shPlanePtr = putBytes(prepared.shPlaneSize, 5);
    return { Module: { HEAPU8, HEAPF32 }, prepared };
}

test('copies exact COMPACTGS planes out of the WASM heap without reconstruction fields', () => {
    const { Module, prepared } = fixture();
    const result = copyCompactGSPreparedResult(Module, prepared, null, { videoDecoderPath: 'raw', streams: [] });
    assert.equal(result.variant, 'compactgs');
    assert.equal(result.descriptor.pointCapacity, 16);
    assert.deepEqual([...result.descriptor.positionGroupMin], [1, 2, 3]);
    assert.equal(result.planes.positionLow.byteLength, 48);
    assert.equal(result.planes.colorRgb.byteLength, 48);
    assert.equal(result.planes.shape.byteLength, 192);
    assert.equal(result.planes.shAtlas.byteLength, 768);
    assert.equal(result.timings.wall.reconstructionWallMs, null);
    assert.equal(result.timings.wall.reconstructionApplicable, false);
    assert.equal('positions' in result, false);
    Module.HEAPU8.fill(9);
    assert.equal(result.planes.positionLow[0], 1);
});

test('rejects a corrupted COMPACTGS plane length', () => {
    const { Module, prepared } = fixture();
    prepared.colorRgbSize--;
    assert.throws(
        () => copyCompactGSPreparedResult(Module, prepared, null, { videoDecoderPath: 'raw', streams: [] }),
        /colorRgbSize/
    );
});

test('accepts COMPACTGS atlases above the former 4096 limit and rejects dimensions above 8192', () => {
    const accepted = fixture(1028, 4);
    const result = copyCompactGSPreparedResult(
        accepted.Module, accepted.prepared, null, { videoDecoderPath: 'raw', streams: [] }
    );
    assert.equal(result.descriptor.pointMapWidth, 1028);

    const rejected = fixture(2052, 4);
    assert.throws(
        () => copyCompactGSPreparedResult(
            rejected.Module, rejected.prepared, null, { videoDecoderPath: 'raw', streams: [] }
        ),
        /invalid COMPACTGS_V1 descriptor/
    );
});

test('COMPACTGS compressed textures neither copy nor transfer legacy per-point UVs', () => {
    const Module = {};
    Object.defineProperty(Module, 'HEAPU32', {
        get() { throw new Error('COMPACTGS must not read the WASM UV heap'); }
    });
    const prepared = { profileIdc: 2, pointCount: 1_048_408, astcUvCount: 0, astcUvPtr: 0 };
    assert.equal(copyCompressedTextureUvs(Module, prepared, 'astc'), null);
    assert.throws(
        () => copyCompressedTextureUvs(Module, { ...prepared, astcUvCount: 2 }, 'astc'),
        /must not export per-point astc UVs/
    );

    const { Module: fixtureModule, prepared: fastPrepared } = fixture();
    const data = copyCompactGSPreparedResult(fixtureModule, fastPrepared, {
        format: 'astc',
        raw: new Uint8Array(16),
        uvs: null,
        metas: new Uint32Array(6)
    }, { videoDecoderPath: 'raw', streams: [] });
    const transferList = buildDecodeTransferList(data);
    assert.equal(data.compressedTextureData.uvs, null);
    assert.equal(transferList.includes(data.compressedTextureData.raw.buffer), true);
    assert.equal(transferList.includes(data.compressedTextureData.metas.buffer), true);
    assert.equal(transferList.every((buffer) => buffer instanceof ArrayBuffer), true);
});

test('profile 1 compressed textures keep strict pointCount*2 UV copy and transfer behavior', () => {
    const heap = new ArrayBuffer(64);
    const HEAPU32 = new Uint32Array(heap);
    HEAPU32.set([3, 4, 5, 6], 2);
    const prepared = { profileIdc: 1, pointCount: 2, astcUvCount: 4, astcUvPtr: 8 };
    const uvs = copyCompressedTextureUvs({ HEAPU32 }, prepared, 'astc');
    assert.deepEqual([...uvs], [3, 4, 5, 6]);
    HEAPU32.fill(9);
    assert.deepEqual([...uvs], [3, 4, 5, 6]);
    assert.throws(
        () => copyCompressedTextureUvs({ HEAPU32 }, { ...prepared, astcUvCount: 2 }, 'astc'),
        /invalid astc UV count/
    );

    const data = {
        variant: 'legacy',
        positions: new Float32Array(3),
        scales: new Float32Array(3),
        rotations: new Float32Array(4),
        colors: new Uint8Array(4),
        compressedTextureData: {
            raw: new Uint8Array(16),
            uvs,
            metas: new Uint32Array(6)
        }
    };
    assert.equal(buildDecodeTransferList(data).includes(uvs.buffer), true);
});

test('extracts the exact GSBS bytes from a glTF GaussianSlim bufferView', () => {
    const payload = new Uint8Array([0, 0, 4, 195, 1, 2, 3]);
    const document = {
        asset: { version: '2.0' },
        extensionsUsed: ['COMPACTGS_primitive_3DGS_compression'],
        extensions: { COMPACTGS_primitive_3DGS_compression: { bufferView: 0 } },
        buffers: [{ byteLength: payload.length }],
        bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: payload.length }]
    };
    const encoder = new TextEncoder();
    const jsonRaw = encoder.encode(JSON.stringify(document));
    const jsonLength = Math.ceil(jsonRaw.length / 4) * 4;
    const binLength = Math.ceil(payload.length / 4) * 4;
    const glb = new ArrayBuffer(12 + 8 + jsonLength + 8 + binLength);
    const view = new DataView(glb);
    view.setUint32(0, 0x46546c67, true);
    view.setUint32(4, 2, true);
    view.setUint32(8, glb.byteLength, true);
    view.setUint32(12, jsonLength, true);
    view.setUint32(16, 0x4e4f534a, true);
    new Uint8Array(glb, 20, jsonLength).fill(0x20);
    new Uint8Array(glb, 20, jsonRaw.length).set(jsonRaw);
    const binHeader = 20 + jsonLength;
    view.setUint32(binHeader, binLength, true);
    view.setUint32(binHeader + 4, 0x004e4942, true);
    new Uint8Array(glb, binHeader + 8, payload.length).set(payload);

    const extracted = extractCompressedPayload(glb);
    assert.equal(extracted.compressedPayload, true);
    assert.deepEqual([...extracted.bytes], [...payload]);
});
