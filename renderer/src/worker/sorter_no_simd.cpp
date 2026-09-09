#include <emscripten/emscripten.h>
#include <iostream>

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

#define computeMatMul4x4ThirdRow(a, b, out) \
    out[0] = a[2] * b[0] +  a[6] * b[1] + a[10] * b[2] + a[14] * b[3]; \
    out[1] = a[2] * b[4] +  a[6] * b[5] + a[10] * b[6] + a[14] * b[7]; \
    out[2] = a[2] * b[8] +  a[6] * b[9] + a[10] * b[10] + a[14] * b[11]; \
    out[3] = a[2] * b[12] +  a[6] * b[13] + a[10] * b[14] + a[14] * b[15];

static void countingSortMapped(unsigned int* indexes, int* mappedDistances, unsigned int* frequencies,
                               unsigned int* indexesOut, unsigned int distanceMapRange,
                               unsigned int sortStart, unsigned int renderCount,
                               int minDistance, int maxDistance) {
    if (renderCount == 0) return;
    const float distanceRange = (float)maxDistance - (float)minDistance;
    const float rangeMap = distanceRange > 0.0f ? (float)(distanceMapRange - 1) / distanceRange : 0.0f;
    for (unsigned int i = sortStart; i < renderCount; i++) {
        unsigned int bucket = distanceRange > 0.0f ?
            (unsigned int)((float)(mappedDistances[i] - minDistance) * rangeMap) : 0;
        if (bucket >= distanceMapRange) bucket = distanceMapRange - 1;
        mappedDistances[i] = (int)bucket;
        frequencies[bucket]++;
    }
    unsigned int cumulative = frequencies[0];
    for (unsigned int i = 1; i < distanceMapRange; i++) {
        cumulative += frequencies[i];
        frequencies[i] = cumulative;
    }
    for (int i = (int)sortStart - 1; i >= 0; i--) indexesOut[i] = indexes[i];
    for (int i = (int)renderCount - 1; i >= (int)sortStart; i--) {
        const unsigned int bucket = (unsigned int)mappedDistances[i];
        const unsigned int frequency = frequencies[bucket];
        indexesOut[renderCount - frequency] = indexes[i];
        frequencies[bucket] = frequency - 1;
    }
}

EXTERN EMSCRIPTEN_KEEPALIVE void sortIndexes(unsigned int* indexes, void* centers, void* precomputedDistances, 
                                             int* mappedDistances, unsigned int * frequencies, float* modelViewProj,
                                             unsigned int* indexesOut,  unsigned int* sceneIndexes, float* transforms,
                                             unsigned int distanceMapRange, unsigned int sortCount, unsigned int renderCount,
                                             unsigned int splatCount, bool usePrecomputedDistances, bool useIntegerSort,
                                             bool dynamicMode) {

    int maxDistance = -2147483640;
    int minDistance = 2147483640;

    float fMVPTRow3[4];
    int iMVPTRow3[4];
    unsigned int sortStart = renderCount - sortCount;
    if (useIntegerSort) {
        int* intCenters = (int*)centers;
        if (usePrecomputedDistances) {
            int* intPrecomputedDistances = (int*)precomputedDistances;
            for (unsigned int i = sortStart; i < renderCount; i++) {
                int distance = intPrecomputedDistances[indexes[i]];
                mappedDistances[i] = distance;
                if (distance > maxDistance) maxDistance = distance;
                if (distance < minDistance) minDistance = distance;
            }
        } else {
            if (dynamicMode) {
                int lastTransformIndex = -1;
                for (unsigned int i = sortStart; i < renderCount; i++) {
                    unsigned int realIndex = indexes[i];
                    unsigned int indexOffset = 4 * realIndex;
                    unsigned int sceneIndex = sceneIndexes[realIndex];
                    if ((int)sceneIndex != lastTransformIndex) {
                        float* transform = &transforms[sceneIndex * 16];
                        computeMatMul4x4ThirdRow(modelViewProj, transform, fMVPTRow3);
                        iMVPTRow3[0] = (int)(fMVPTRow3[0] * 1000.0);
                        iMVPTRow3[1] = (int)(fMVPTRow3[1] * 1000.0);
                        iMVPTRow3[2] = (int)(fMVPTRow3[2] * 1000.0);
                        iMVPTRow3[3] = (int)(fMVPTRow3[3] * 1000.0);
                        lastTransformIndex = (int)sceneIndex;
                    }
                    int distance =
                        (int)((iMVPTRow3[0] * intCenters[indexOffset] +
                               iMVPTRow3[1] * intCenters[indexOffset + 1] +
                               iMVPTRow3[2] * intCenters[indexOffset + 2] +
                               iMVPTRow3[3] * intCenters[indexOffset + 3]));
                    mappedDistances[i] = distance;
                    if (distance > maxDistance) maxDistance = distance;
                    if (distance < minDistance) minDistance = distance;
                }
            } else {
                iMVPTRow3[0] = (int)(modelViewProj[2] * 1000.0);
                iMVPTRow3[1] = (int)(modelViewProj[6] * 1000.0);
                iMVPTRow3[2] = (int)(modelViewProj[10] * 1000.0);
                iMVPTRow3[3] = 1;
                for (unsigned int i = sortStart; i < renderCount; i++) {
                    unsigned int indexOffset = 4 * (unsigned int)indexes[i];
                    int distance =
                        (int)((iMVPTRow3[0] * intCenters[indexOffset] +
                               iMVPTRow3[1] * intCenters[indexOffset + 1] +
                               iMVPTRow3[2] * intCenters[indexOffset + 2]));
                    mappedDistances[i] = distance;
                    if (distance > maxDistance) maxDistance = distance;
                    if (distance < minDistance) minDistance = distance;
                }
            }
        }
    } else {
        float* floatCenters = (float*)centers;
        if (usePrecomputedDistances) {
            float* floatPrecomputedDistances = (float*)precomputedDistances;
            for (unsigned int i = sortStart; i < renderCount; i++) {
                int distance = (int)(floatPrecomputedDistances[indexes[i]] * 4096.0);
                mappedDistances[i] = distance;
                if (distance > maxDistance) maxDistance = distance;
                if (distance < minDistance) minDistance = distance;
            }
        } else {
            float* fMVP = (float*)modelViewProj;
            float* floatTransforms = (float *)transforms;

            if (dynamicMode) {
                int lastTransformIndex = -1;
                for (unsigned int i = sortStart; i < renderCount; i++) {
                    unsigned int realIndex = indexes[i];
                    unsigned int indexOffset = 4 * realIndex;
                    unsigned int sceneIndex = sceneIndexes[realIndex];
                    if ((int)sceneIndex != lastTransformIndex) {
                        float* transform = &transforms[sceneIndex * 16];
                        computeMatMul4x4ThirdRow(modelViewProj, transform, fMVPTRow3);
                        lastTransformIndex = (int)sceneIndex;
                    }
                    int distance =
                        (int)((fMVPTRow3[0] * floatCenters[indexOffset] +
                               fMVPTRow3[1] * floatCenters[indexOffset + 1] +
                               fMVPTRow3[2] * floatCenters[indexOffset + 2] +
                               fMVPTRow3[3] * floatCenters[indexOffset + 3]) * 4096.0);
                    mappedDistances[i] = distance;
                    if (distance > maxDistance) maxDistance = distance;
                    if (distance < minDistance) minDistance = distance;
                }
            } else {
                for (unsigned int i = sortStart; i < renderCount; i++) {
                    unsigned int indexOffset = 4 * (unsigned int)indexes[i];
                    int distance =
                        (int)((fMVP[2] * floatCenters[indexOffset] +
                               fMVP[6] * floatCenters[indexOffset + 1] +
                               fMVP[10] * floatCenters[indexOffset + 2]) * 4096.0);
                    mappedDistances[i] = distance;
                    if (distance > maxDistance) maxDistance = distance;
                    if (distance < minDistance) minDistance = distance;
                }
            }
        }
    }

    countingSortMapped(indexes, mappedDistances, frequencies, indexesOut, distanceMapRange,
                       sortStart, renderCount, minDistance, maxDistance);
}

EXTERN EMSCRIPTEN_KEEPALIVE void sortIndexesQuantized(
    unsigned int* indexes, const unsigned char* positionLow, const unsigned char* positionHigh,
    const float* groupMin, const float* groupStep, unsigned int groupSize,
    int* mappedDistances, unsigned int* frequencies, const float* modelViewProj,
    unsigned int* indexesOut, unsigned int distanceMapRange, unsigned int sortCount,
    unsigned int renderCount, unsigned int splatCount) {
    if (renderCount == 0 || groupSize == 0) return;
    const unsigned int sortStart = renderCount - sortCount;
    int maxDistance = -2147483640;
    int minDistance = 2147483640;
    for (unsigned int i = sortStart; i < renderCount; i++) {
        const unsigned int pointIndex = indexes[i];
        if (pointIndex >= splatCount) continue;
        const unsigned int pointBase = pointIndex * 3;
        const unsigned int groupBase = (pointIndex / groupSize) * 3;
        const float x = groupMin[groupBase] + groupStep[groupBase] *
            (float)(((unsigned int)positionHigh[pointBase] << 7) | positionLow[pointBase]);
        const float y = groupMin[groupBase + 1] + groupStep[groupBase + 1] *
            (float)(((unsigned int)positionHigh[pointBase + 1] << 7) | positionLow[pointBase + 1]);
        const float z = groupMin[groupBase + 2] + groupStep[groupBase + 2] *
            (float)(((unsigned int)positionHigh[pointBase + 2] << 7) | positionLow[pointBase + 2]);
        const int distance = (int)((modelViewProj[2] * x + modelViewProj[6] * y + modelViewProj[10] * z) * 4096.0f);
        mappedDistances[i] = distance;
        if (distance > maxDistance) maxDistance = distance;
        if (distance < minDistance) minDistance = distance;
    }
    countingSortMapped(indexes, mappedDistances, frequencies, indexesOut, distanceMapRange,
                       sortStart, renderCount, minDistance, maxDistance);
}
