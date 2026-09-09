import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';

import { SplatMesh } from '../../../splatmesh/SplatMesh.js';

function makeFastData(pointCount = 2) {
    const width = 4;
    const height = 4;
    const capacity = width * height;
    return {
        isCompactGSSceneData: true,
        shDegree: 3,
        minimumAlpha: 7,
        compressedTextureData: null,
        sceneCenter: new THREE.Vector3(0.5, 0.5, 0.5),
        descriptor: {
            layoutVersion: 2,
            pointMapWidth: width,
            pointMapHeight: height,
            positionGroupSize: 256,
            positionGroupMin: new Float32Array([0, 0, 0]),
            positionGroupStep: new Float32Array([1 / 32767, 1 / 32767, 1 / 32767]),
            shapeMin: new Float32Array(7),
            shapeStep: new Float32Array(7).fill(1 / 255),
            importanceMin: 0.1,
            importanceStep: 0.9 / 255,
            shMin: new Float32Array(45).fill(-1),
            shStep: new Float32Array(45).fill(2 / 255)
        },
        planes: {
            positionLow: new Uint8Array(capacity * 3),
            positionHigh: new Uint8Array(capacity * 3),
            colorRgb: new Uint8Array(capacity * 3).fill(255),
            shape: new Uint8Array(width * height * 12),
            shAtlas: new Uint8Array(width * height * 48)
        },
        covariances: new Float32Array(pointCount * 6),
        getSplatCount() { return pointCount; },
        getMaxSplatCount() { return pointCount; },
        getMinSphericalHarmonicsDegree() { return 3; }
    };
}

test('builds a static COMPACTGS mesh directly from typed planes', () => {
    const mesh = new SplatMesh(undefined, false, false, false, 1, false, false,
                               false, 324, 0, 3);
    const data = makeFastData();
    const result = mesh.build([data], [{ splatAlphaRemovalThreshold: 13 }], true, true);

    assert.equal(mesh.compactGaussianMode, true);
    assert.equal(mesh.getSplatCount(), 2);
    assert.equal(mesh.splatDataTextures.baseData, undefined);
    assert.equal(mesh.material.uniforms.fastMinimumAlpha.value, 13);
    assert.match(mesh.material.vertexShader, /sortIndexesQuantized|fastPositionLowTexture|readShape/);
    assert.match(mesh.material.vertexShader, /fastShMin\[45\]/);
    assert.match(mesh.material.vertexShader, /readShapeQ\(pointCoord, fastCovarianceMode/);
    assert.match(mesh.material.vertexShader, /fastCovarianceTexture/);
    assert.match(mesh.material.vertexShader, /readDirectCovariance/);
    assert.equal((mesh.material.vertexShader.match(/1\.0 \/ importance/g) || []).length, 1);
    assert.equal(result.compactGaussianQuantizedPositions.positionLow.byteLength, 6);
    assert.equal(result.centers.byteLength, 0);
    assert.equal(mesh.splatDataTextures.fastPositionLow.texture.unpackAlignment, 1);
    mesh.dispose();
});

test('keeps a valid ASTC SH atlas compressed through the COMPACTGS mesh build', () => {
    const mesh = new SplatMesh(undefined, false, false, false, 1, false, false,
                               false, 324, 0, 3);
    const data = makeFastData();
    data.planes.shAtlas = null;
    data.compressedTextureData = {
        format: 'astc',
        raw: new Uint8Array(256),
        metas: new Uint32Array([4, 16, 16, 4, 16, 256]),
        width: 16,
        height: 16,
        layers: 1,
        blockWidth: 4,
        blockHeight: 4,
        singleWidth: 4,
        singleHeight: 4,
        regionPixelCount: 16,
        textureNum: 1,
        shDegree: 3,
        shnMin: -1,
        shnMax: 1
    };

    mesh.build([data], [{}], true, true);
    assert.equal(mesh.useASTC, true);
    assert.equal(mesh.splatDataTextures.fastSH.data, data.compressedTextureData.raw);
    assert.equal(mesh.material.uniforms.fastShCompressedTexture.value.isCompressedArrayTexture, true);
    assert.equal(mesh.splatDataTextures.baseData, undefined);
    mesh.dispose();
});

test('accepts the BC3 SH fallback without creating legacy base textures', () => {
    const mesh = new SplatMesh(undefined, false, false, false, 1, false, false,
                               false, 324, 0, 3);
    const data = makeFastData();
    data.planes.shAtlas = null;
    data.compressedTextureData = {
        format: 'bc3', raw: new Uint8Array(256),
        metas: new Uint32Array([4, 16, 16, 4, 16, 256]),
        width: 16, height: 16, layers: 1, blockWidth: 4, blockHeight: 4,
        singleWidth: 4, singleHeight: 4, regionPixelCount: 16, textureNum: 1,
        shDegree: 3, shnMin: -1, shnMax: 1
    };
    mesh.build([data], [{}], true, true);
    assert.equal(mesh.useCompressedTexture, true);
    assert.equal(mesh.useASTC, false);
    assert.equal(mesh.splatDataTextures.baseData, undefined);
    mesh.dispose();
});

test('rejects transforms and multi-scene COMPACTGS builds', () => {
    const data = makeFastData(1);
    const transformed = new SplatMesh(undefined, false, false, false, 1, false, false,
                                      false, 324, 0, 3);
    assert.throws(() => transformed.build([data], [{ position: [1, 0, 0] }]), /identity scene transform/);
    const multiple = new SplatMesh(undefined, false, false, false, 1, false, false,
                                   false, 324, 0, 3);
    assert.throws(() => multiple.build([data, data], [{}, {}]), /exactly one scene/);
});
