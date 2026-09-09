#pragma once
#ifndef DECODER_H
#define DECODER_H
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "nlohmann/json.hpp"
#include "cgltf.h"
#include "compactgs_gsdecoder/gaussian_model/gs_data.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

typedef struct {
    float *posBitstream;
    float *opacityBitstream;
    float *scaleBitstream;
    float *rotationBitstream;
    float *colorBitstream;
    float *SHBitstream;
    uint8_t *astcRawStream;
	uint32_t *astcUVStream;
} GSAttributeBitstream;

struct AstcMeta {
    uint32_t astcBlockSize;
    uint32_t astcWidth;
    uint32_t astcHeight;
    uint32_t singleWidth;
    uint32_t numPoints;
    uint32_t streamSize;
};

struct ShnAstcMeta {
    std::vector<AstcMeta> astcMetas;
    uint32_t textureNum;
    float shnMin;
    float shnMax;
};

// ����ṹ��
struct GSData {
    int GSpointnum;
    int SHdegree;
    size_t posBitstreamSize;
    size_t opacityBitstreamSize;
    size_t scaleBitstreamSize;
    size_t rotationBitstreamSize;
    size_t colorBitstreamSize;
    size_t SHBitstreamSize;
    size_t astcBitstreamSize;
	size_t astcUVstreamSize;
	ShnAstcMeta shnAstcMeta;
    GSAttributeBitstream gsAttributeBitstream;
};

struct ViewInfo {
    float yfov;
    glm::vec3 initialPositionLookAt;
    glm::vec3 target;
    glm::quat initialRotationLookAt;
    glm::vec2 longitudeRange;
    glm::vec2 latitudeRange;
    glm::vec2 distanceRange;
    glm::vec3 boundingBoxMin;
    glm::vec3 boundingBoxMax;
};

struct CameraMotionParams {
    int id;
    double yfov;
    glm::vec3 position;
    glm::quat rotation;
};

typedef void *DecodeHandle;

struct DecodedDataHandle {
    GSData gsData;
    ViewInfo viewInfo;
    std::vector<CameraMotionParams> motionParams;
    std::unique_ptr<SplatData> ownedSplatData;
};

// ���뺯�������뻺���������ذ����ṹ��Ľ��
int decodeUI(DecodeHandle *decoderHandle, const uint8_t *buffer, size_t bufferSize, bool astcCpuDecode);
int decodeCompressed3DGS(DecodeHandle *decoderHandle, const uint8_t *buffer, size_t bufferSize, bool astcCpuDecode);
void parseCameraAnimation(const nlohmann::json &animation, const nlohmann::json &gltfJson,
                          const std::vector<uint8_t> &binData, DecodedDataHandle *handle);
GSData *getStructGSData(DecodeHandle handle);
ViewInfo *getStructViewInfo(DecodeHandle handle);
std::vector<CameraMotionParams> *getMotionParams(DecodeHandle handle);

// ���ٺ������ͷŽ�������ռ�õ��ڴ�
int destroyDecode(DecodeHandle decoderHandle);

#endif  // DECODER_H
