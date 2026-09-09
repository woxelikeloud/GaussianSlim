import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

async function runQuantizedSort(wasmName) {
    const bytes = await fs.readFile(new URL(`../../../worker/${wasmName}`, import.meta.url));
    const memory = new WebAssembly.Memory({ initial: 2, maximum: 2 });
    const { instance } = await WebAssembly.instantiate(bytes, { env: { memory }, module: {} });
    const indexesOffset = 0;
    const lowOffset = 64;
    const highOffset = 96;
    const groupMinOffset = 128;
    const groupStepOffset = 144;
    const mappedOffset = 160;
    const frequenciesOffset = 192;
    const mvpOffset = 512;
    const outputOffset = 576;
    new Uint32Array(memory.buffer, indexesOffset, 3).set([0, 1, 2]);
    new Uint8Array(memory.buffer, lowOffset, 9).set([0, 0, 0, 10, 0, 0, 5, 0, 0]);
    new Float32Array(memory.buffer, groupMinOffset, 3).set([0, 0, 0]);
    new Float32Array(memory.buffer, groupStepOffset, 3).set([1, 1, 1]);
    const mvp = new Float32Array(memory.buffer, mvpOffset, 16);
    mvp[2] = 1;
    instance.exports.sortIndexesQuantized(indexesOffset, lowOffset, highOffset, groupMinOffset, groupStepOffset,
                                           256, mappedOffset, frequenciesOffset, mvpOffset, outputOffset,
                                           64, 3, 3, 3);
    return [...new Uint32Array(memory.buffer, outputOffset, 3)];
}

for (const wasmName of ['sorter.wasm', 'sorter_no_simd.wasm']) {
    test(`${wasmName} sorts directly from COMPACTGS q15 positions`, async () => {
        assert.deepEqual(await runQuantizedSort(wasmName), [1, 2, 0]);
    });
}

