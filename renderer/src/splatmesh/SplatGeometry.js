import * as THREE from 'three';

export class SplatGeometry {

    /**
     * Build the Three.js geometry that will be used to render the splats. The geometry is instanced and is made up of
     * vertices for a single quad as well as an attribute buffer for the splat indexes.
     * @param {number} maxSplatCount The maximum number of splats that the geometry will need to accomodate
     * @return {THREE.InstancedBufferGeometry}
     */
    static build(maxSplatCount, batched = false, batchSize = 128) {

        const baseGeometry = new THREE.BufferGeometry();
        const quadCount = batched ? batchSize : 1;
        const indexArray = new Uint32Array(quadCount * 6);
        for (let quad = 0; quad < quadCount; quad++) {
            const base = quad * 4;
            const index = quad * 6;
            indexArray[index] = base;
            indexArray[index + 1] = base + 1;
            indexArray[index + 2] = base + 2;
            indexArray[index + 3] = base;
            indexArray[index + 4] = base + 2;
            indexArray[index + 5] = base + 3;
        }
        baseGeometry.setIndex(Array.from(indexArray));

        // Vertices for the instanced quad
        const positionsArray = new Float32Array(quadCount * 4 * 3);
        const positions = new THREE.BufferAttribute(positionsArray, 3);
        baseGeometry.setAttribute('position', positions);
        for (let quad = 0; quad < quadCount; quad++) {
            const base = quad * 4;
            positions.setXYZ(base, -1.0, -1.0, 0.0);
            positions.setXYZ(base + 1, -1.0, 1.0, 0.0);
            positions.setXYZ(base + 2, 1.0, 1.0, 0.0);
            positions.setXYZ(base + 3, 1.0, -1.0, 0.0);
        }
        positions.needsUpdate = true;

        const geometry = new THREE.InstancedBufferGeometry().copy(baseGeometry);

        // Splat index buffer
        const instanceCount = batched ? Math.ceil(maxSplatCount / batchSize) : maxSplatCount;
        const splatIndexArray = new Uint32Array(instanceCount);
        const splatIndexes = new THREE.InstancedBufferAttribute(splatIndexArray, 1, false);
        splatIndexes.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('splatIndex', splatIndexes);

        if (batched) {
            const localOffsets = new Uint32Array(quadCount * 4);
            for (let quad = 0; quad < quadCount; quad++) {
                localOffsets[quad * 4] = quad;
                localOffsets[quad * 4 + 1] = quad;
                localOffsets[quad * 4 + 2] = quad;
                localOffsets[quad * 4 + 3] = quad;
            }
            geometry.setAttribute('splatLocalIndex', new THREE.BufferAttribute(localOffsets, 1, false));
            geometry.userData.fastBatchSize = batchSize;
            geometry.userData.fastBatched = true;
        }

        geometry.instanceCount = 0;

        return geometry;
    }
}
