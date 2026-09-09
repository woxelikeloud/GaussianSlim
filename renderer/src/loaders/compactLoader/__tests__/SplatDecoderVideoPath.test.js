import assert from 'node:assert/strict';
import test from 'node:test';

globalThis.self = { __COMPACTGS_SPLAT_DECODER_BOOTSTRAP__: true };
const { composeVideoDecoderPath } = await import('../SplatDecoder.worker.js');

test('labels no-video and raw-only staged prepares', () => {
    assert.equal(composeVideoDecoderPath(0, 0), 'none');
    assert.equal(composeVideoDecoderPath(2, 0), 'raw');
});

test('combines raw adoption with encoded video decoder paths', () => {
    assert.equal(composeVideoDecoderPath(1, 2, 'webcodecs'), 'raw+webcodecs');
    assert.equal(composeVideoDecoderPath(1, 2, 'ffmpeg'), 'raw+ffmpeg');
    assert.equal(composeVideoDecoderPath(1, 2, 'mixed'), 'raw+mixed');
    assert.equal(composeVideoDecoderPath(0, 2, 'mixed'), 'mixed');
});
