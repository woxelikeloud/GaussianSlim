#include <emscripten/bind.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "compactgs_gsdecoder/gaussian_model/gs_decoder.h"

using namespace emscripten;

namespace {

bool textureOutputModeFromAbi(uint32_t value, TextureOutputMode& mode) {
    switch (value) {
        case 0: mode = TextureOutputMode::ASTC; return true;
        case 1: mode = TextureOutputMode::CPU; return true;
        case 2: mode = TextureOutputMode::BC7; return true;
        case 3: mode = TextureOutputMode::BC3; return true;
        default: return false;
    }
}

double substreamTimingOrZero(const TimingStats& timing, uint32_t streamIndex) {
    const auto found = timing.substreamTimings.find("stream_" + std::to_string(streamIndex));
    return found == timing.substreamTimings.end() ? 0.0 : found->second;
}

struct StagedPrepareResult {
    bool success = false;
    uint32_t protocolVersion = compactgs::kShardProtocolVersion;
    uint32_t buildVersion = compactgs::kShardBuildVersion;
    uint32_t pendingVideoCount = 0;
    uint32_t rawVideoStreamCount = 0;
    double rawVideoInputBytes = 0.0;
    double rawVideoAdoptMs = 0.0;
    double parseMs = 0.0;
    double decodeNonVideoSubstreamsMs = 0.0;
    double astcTextureDecodeMs = 0.0;
    double bcTextureEncodeMs = 0.0;
    double textureInputBytes = 0.0;
    double textureOutputBytes = 0.0;
};

struct PendingVideoResult {
    bool success = false;
    uint32_t streamIndex = 0;
    uint32_t codecId = 0;
    uint32_t frameWidth = 0;
    uint32_t frameHeight = 0;
    uint32_t frameCount = 0;
    uintptr_t encodedPtr = 0;
    uint32_t encodedSize = 0;
};

struct PreparedDecodeResult {
    bool success = false;
    uint32_t protocolVersion = compactgs::kShardProtocolVersion;
    uint32_t buildVersion = compactgs::kShardBuildVersion;
    uint32_t pointCount = 0;
    uint32_t shDegree = 0;
    uint32_t blockSide = 0;
    uint32_t pointsPerBlock = 0;
    uint32_t blockCount = 0;
    uint32_t textureOutputMode = static_cast<uint32_t>(TextureOutputMode::CPU);
    bool decodeFeaturesRest = false;
    uintptr_t metadataPtr = 0;
    uint32_t metadataSize = 0;
    uintptr_t astcRawPtr = 0;
    uint32_t astcRawSize = 0;
    uintptr_t astcUvPtr = 0;
    uint32_t astcUvCount = 0;
    uintptr_t astcMetasPtr = 0;
    uint32_t astcMetasCount = 0;
    float shnMin = 0.0f;
    float shnMax = 0.0f;
    uint32_t textureNum = 0;
    uint32_t profileIdc = 1;
    uint32_t layoutVersion = 0;
    uint32_t pointMapWidth = 0;
    uint32_t pointMapHeight = 0;
    uint32_t positionGroupSize = 0;
    uintptr_t positionGroupMinPtr = 0;
    uintptr_t positionGroupStepPtr = 0;
    uint32_t positionGroupFloatCount = 0;
    uintptr_t shapeMinPtr = 0;
    uintptr_t shapeStepPtr = 0;
    uint32_t shapeComponentCount = 0;
    uintptr_t covarianceGroupMinPtr = 0;
    uintptr_t covarianceGroupStepPtr = 0;
    uint32_t covarianceGroupFloatCount = 0;
    double encodedMinimumAlpha = 0.0;
    double importanceMin = 0.0;
    double importanceStep = 0.0;
    uintptr_t fastShMinPtr = 0;
    uintptr_t fastShStepPtr = 0;
    uint32_t fastShComponentCount = 0;
    uintptr_t positionLowPtr = 0;
    uint32_t positionLowSize = 0;
    uintptr_t positionHighPtr = 0;
    uint32_t positionHighSize = 0;
    uintptr_t colorRgbPtr = 0;
    uint32_t colorRgbSize = 0;
    uintptr_t shapePlanePtr = 0;
    uint32_t shapePlaneSize = 0;
    uintptr_t shPlanePtr = 0;
    uint32_t shPlaneSize = 0;
    double parseMs = 0.0;
    double decodeNonVideoSubstreamsMs = 0.0;
    double astcTextureDecodeMs = 0.0;
    double bcTextureEncodeMs = 0.0;
    double textureInputBytes = 0.0;
    double textureOutputBytes = 0.0;
    double substream0Ms = 0.0;
    double substream1Ms = 0.0;
    double substream2Ms = 0.0;
    double substream3Ms = 0.0;
    double substream4Ms = 0.0;
    double decodeVideoFallbackMs = 0.0;
    double decodeSubstreamsMs = 0.0;
    double totalMs = 0.0;
};

struct PackedShardResult {
    bool success = false;
    uint32_t protocolVersion = compactgs::kShardProtocolVersion;
    uint32_t buildVersion = compactgs::kShardBuildVersion;
    uint32_t startBlock = 0;
    uint32_t blockCount = 0;
    uint32_t startPoint = 0;
    uint32_t pointCount = 0;
    uintptr_t packetPtr = 0;
    uint32_t packetSize = 0;
    double packMs = 0.0;
    double unpackMs = 0.0;
};

class GaussianSlimLoaderCoordinatorWasm {
public:
    uint32_t getTextureEncoderMask() const {
#ifdef USE_BC_TEXTURE_ENCODERS
        // BC7 remains compiled for an inexpensive future re-enable, but BC3 is
        // the only runtime-advertised PC path while startup latency is prioritized.
        return 2;
#else
        return 0;
#endif
    }

    PreparedDecodeResult prepare(uintptr_t bufferPtr,
                                 size_t bufferSize,
                                 uint32_t textureOutputMode,
                                 bool compressedPayload) {
        const StagedPrepareResult staged = beginPrepare(
            bufferPtr, bufferSize, textureOutputMode, compressedPayload);
        if (!staged.success) return PreparedDecodeResult();
        if (!decodePendingVideosWithFallback()) {
            decoder_.releasePreparedDecode();
            return PreparedDecodeResult();
        }
        PreparedDecodeResult result = finishPrepare();
        if (!result.success) decoder_.releasePreparedDecode();
        return result;
    }

    StagedPrepareResult beginPrepare(uintptr_t bufferPtr,
                                     size_t bufferSize,
                                     uint32_t textureOutputModeValue,
                                     bool compressedPayload) {
        (void)compressedPayload;
        release();
        StagedPrepareResult result;
        if (bufferPtr == 0 || bufferSize == 0) {
            lastError_ = "Coordinator received an empty input buffer";
            return result;
        }
        TextureOutputMode textureOutputMode;
        if (!textureOutputModeFromAbi(textureOutputModeValue, textureOutputMode)) {
            lastError_ = "Coordinator received an invalid texture output mode";
            return result;
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(bufferPtr);
        if (!decoder_.beginPrepareFromMemory(bytes, bufferSize, textureOutputMode)) {
            lastError_ = decoder_.getLastError();
            return result;
        }

        const TimingStats& timing = getTimingStats();
        result.success = true;
        result.pendingVideoCount = static_cast<uint32_t>(decoder_.getPendingVideoStreams().size());
        result.rawVideoStreamCount = timing.rawVideoStreamCount;
        result.rawVideoInputBytes = static_cast<double>(timing.rawVideoInputBytes);
        result.rawVideoAdoptMs = timing.rawVideoAdoptMs;
        result.parseMs = timing.parseStreamMs;
        result.decodeNonVideoSubstreamsMs = timing.decodeNonVideoSubstreamsMs;
        result.astcTextureDecodeMs = timing.astcTextureDecodeMs;
        result.bcTextureEncodeMs = timing.bcTextureEncodeMs;
        result.textureInputBytes = static_cast<double>(timing.textureInputBytes);
        result.textureOutputBytes = static_cast<double>(timing.textureOutputBytes);
        return result;
    }

    PendingVideoResult getPendingVideo(uint32_t pendingIndex) {
        PendingVideoResult result;
        const auto& pendingVideos = decoder_.getPendingVideoStreams();
        if (pendingIndex >= pendingVideos.size()) {
            lastError_ = "Pending video index is outside the staged descriptor table";
            return result;
        }
        const auto& pending = pendingVideos[pendingIndex];
        const auto* encoded = decoder_.getPendingVideoEncodedData(pending.streamIndex);
        if (!encoded || encoded->size() > std::numeric_limits<uint32_t>::max()) {
            lastError_ = "Pending video buffers exceed the WebAssembly bridge limits";
            return result;
        }

        result.success = true;
        result.streamIndex = pending.streamIndex;
        result.codecId = pending.codecId;
        result.frameWidth = pending.frameWidth;
        result.frameHeight = pending.frameHeight;
        result.frameCount = pending.frameCount;
        result.encodedPtr = reinterpret_cast<uintptr_t>(encoded->data());
        result.encodedSize = static_cast<uint32_t>(encoded->size());
        return result;
    }

    bool injectDecodedVideo(uint32_t streamIndex,
                            uintptr_t decodedPtr,
                            size_t decodedSize,
                            uint32_t pixelFormatValue) {
        VideoPixelFormat pixelFormat;
        if (!videoPixelFormatFromAbi(pixelFormatValue, pixelFormat)) {
            lastError_ = "Decoded video injection used an unsupported pixel format";
            return false;
        }
        const auto* decoded = reinterpret_cast<const uint8_t*>(decodedPtr);
        if (!decoder_.injectDecodedVideo(streamIndex, decoded, decodedSize, pixelFormat)) {
            lastError_ = decoder_.getLastError();
            return false;
        }
        lastError_.clear();
        return true;
    }

    bool decodePendingVideosWithFallback() {
        if (!decoder_.decodePendingVideosWithFallback()) {
            lastError_ = decoder_.getLastError();
            return false;
        }
        lastError_.clear();
        return true;
    }

    PreparedDecodeResult finishPrepare() {
        PreparedDecodeResult result;
        if (!decoder_.finishPreparedDecode()) {
            lastError_ = decoder_.getLastError();
            return result;
        }

        const DecodeDescriptor& descriptor = decoder_.getDecodeDescriptor();
        const auto& metadata = decoder_.getReconstructionMetadataPacket();
        const auto& auxiliary = decoder_.getPreparedAuxiliaryData();
        flatAstcMetas_.resize(auxiliary.astcMetas.size() * 6);
        for (size_t i = 0; i < auxiliary.astcMetas.size(); ++i) {
            const auto& source = auxiliary.astcMetas[i];
            flatAstcMetas_[i * 6] = source.astcBlockSize;
            flatAstcMetas_[i * 6 + 1] = source.astcWidth;
            flatAstcMetas_[i * 6 + 2] = source.astcHeight;
            flatAstcMetas_[i * 6 + 3] = source.singleWidth;
            flatAstcMetas_[i * 6 + 4] = source.numPoints;
            flatAstcMetas_[i * 6 + 5] = source.streamSize;
        }

        const TimingStats& timing = getTimingStats();
        result.success = true;
        result.pointCount = descriptor.pointCount;
        result.shDegree = descriptor.shDegree;
        result.blockSide = descriptor.blockSide;
        result.pointsPerBlock = descriptor.pointsPerBlock;
        result.blockCount = descriptor.blockCount;
        result.textureOutputMode = static_cast<uint32_t>(descriptor.textureOutputMode);
        result.decodeFeaturesRest = descriptor.decodeFeaturesRest;
        result.metadataPtr = reinterpret_cast<uintptr_t>(metadata.data());
        result.metadataSize = static_cast<uint32_t>(metadata.size());
        result.astcRawPtr = reinterpret_cast<uintptr_t>(auxiliary.astcRawStream.data());
        result.astcRawSize = static_cast<uint32_t>(auxiliary.astcRawStream.size());
        result.astcUvPtr = auxiliary.astcUV.empty() ? 0 :
            reinterpret_cast<uintptr_t>(auxiliary.astcUV.data());
        result.astcUvCount = static_cast<uint32_t>(auxiliary.astcUV.size());
        result.astcMetasPtr = reinterpret_cast<uintptr_t>(flatAstcMetas_.data());
        result.astcMetasCount = static_cast<uint32_t>(flatAstcMetas_.size());
        result.shnMin = auxiliary.shnMin;
        result.shnMax = auxiliary.shnMax;
        result.textureNum = auxiliary.astcTextureNum;
        result.profileIdc = descriptor.profileIdc;
        if (const auto* fast = decoder_.getCompactGSDescriptor()) {
            result.layoutVersion = fast->layoutVersion;
            result.pointMapWidth = fast->pointMapWidth;
            result.pointMapHeight = fast->pointMapHeight;
            result.positionGroupSize = fast->positionGroupSize;
            result.positionGroupMinPtr = reinterpret_cast<uintptr_t>(fast->positionGroupMin.data());
            result.positionGroupStepPtr = reinterpret_cast<uintptr_t>(fast->positionGroupStep.data());
            result.positionGroupFloatCount = static_cast<uint32_t>(fast->positionGroupMin.size());
            result.shapeMinPtr = reinterpret_cast<uintptr_t>(fast->shapeMin);
            result.shapeStepPtr = reinterpret_cast<uintptr_t>(fast->shapeStep);
            result.shapeComponentCount = fast->layoutVersion >= 3 ? 6 : 7;
            if (fast->layoutVersion >= 3) {
                result.covarianceGroupMinPtr = reinterpret_cast<uintptr_t>(fast->covarianceGroupMin.data());
                result.covarianceGroupStepPtr = reinterpret_cast<uintptr_t>(fast->covarianceGroupStep.data());
                result.covarianceGroupFloatCount = static_cast<uint32_t>(fast->covarianceGroupMin.size());
            }
            result.encodedMinimumAlpha = fast->encodedMinimumAlpha;
            result.importanceMin = fast->importanceMin;
            result.importanceStep = fast->importanceStep;
            result.fastShMinPtr = reinterpret_cast<uintptr_t>(fast->shMin);
            result.fastShStepPtr = reinterpret_cast<uintptr_t>(fast->shStep);
            result.fastShComponentCount = 45;
            auto setPlane = [&](CompactGSStreamRole role, uintptr_t& pointer, uint32_t& size) {
                const auto* plane = decoder_.getCompactGSPlane(role);
                if (!plane || plane->size() > std::numeric_limits<uint32_t>::max()) return;
                pointer = reinterpret_cast<uintptr_t>(plane->data());
                size = static_cast<uint32_t>(plane->size());
            };
            setPlane(CompactGSStreamRole::POSITION_LOW, result.positionLowPtr, result.positionLowSize);
            setPlane(CompactGSStreamRole::POSITION_HIGH, result.positionHighPtr, result.positionHighSize);
            setPlane(CompactGSStreamRole::COLOR_RGB, result.colorRgbPtr, result.colorRgbSize);
            setPlane(CompactGSStreamRole::SHAPE_VIDEO, result.shapePlanePtr, result.shapePlaneSize);
            setPlane(CompactGSStreamRole::SH_ASTC, result.shPlanePtr, result.shPlaneSize);
        }
        result.astcTextureDecodeMs = timing.astcTextureDecodeMs;
        result.bcTextureEncodeMs = timing.bcTextureEncodeMs;
        result.textureInputBytes = static_cast<double>(timing.textureInputBytes);
        result.textureOutputBytes = static_cast<double>(timing.textureOutputBytes);
        result.substream0Ms = substreamTimingOrZero(timing, 0);
        result.substream1Ms = substreamTimingOrZero(timing, 1);
        result.substream2Ms = substreamTimingOrZero(timing, 2);
        result.substream3Ms = substreamTimingOrZero(timing, 3);
        result.substream4Ms = substreamTimingOrZero(timing, 4);
        result.parseMs = timing.parseStreamMs;
        result.decodeNonVideoSubstreamsMs = timing.decodeNonVideoSubstreamsMs;
        result.decodeVideoFallbackMs = timing.decodeVideoFallbackMs;
        result.decodeSubstreamsMs = timing.decodeSubstreamsMs;
        result.totalMs = timing.totalMs;
        lastError_.clear();
        return result;
    }

    PackedShardResult packShard(uint32_t startBlock, uint32_t blockCount) {
        PackedShardResult result;
        const auto before = getTimingStats().packShardMs;
        const auto unpackBefore = getTimingStats().unpackMs;
        if (!decoder_.packShard(startBlock, blockCount, packet_)) {
            lastError_ = decoder_.getLastError();
            return result;
        }
        const auto& descriptor = decoder_.getDecodeDescriptor();
        result.success = true;
        result.startBlock = startBlock;
        result.blockCount = blockCount;
        result.startPoint = startBlock * descriptor.pointsPerBlock;
        result.pointCount = std::min(descriptor.pointCount - result.startPoint,
                                     blockCount * descriptor.pointsPerBlock);
        result.packetPtr = reinterpret_cast<uintptr_t>(packet_.data());
        result.packetSize = static_cast<uint32_t>(packet_.size());
        result.packMs = getTimingStats().packShardMs - before;
        result.unpackMs = getTimingStats().unpackMs - unpackBefore;
        return result;
    }

    void releasePackedShard() {
        packet_.clear();
    }

    void release() {
        decoder_.releasePreparedDecode();
        packet_.clear();
        flatAstcMetas_.clear();
        lastError_.clear();
    }

    std::string getLastError() const {
        return lastError_.empty() ? decoder_.getLastError() : lastError_;
    }

private:
    GSDecoder decoder_;
    std::vector<uint8_t> packet_;
    std::vector<uint32_t> flatAstcMetas_;
    std::string lastError_;
};

} // namespace

EMSCRIPTEN_BINDINGS(gaussianslim_coordinator_module) {
    value_object<StagedPrepareResult>("StagedPrepareResult")
        .field("success", &StagedPrepareResult::success)
        .field("protocolVersion", &StagedPrepareResult::protocolVersion)
        .field("buildVersion", &StagedPrepareResult::buildVersion)
        .field("pendingVideoCount", &StagedPrepareResult::pendingVideoCount)
        .field("rawVideoStreamCount", &StagedPrepareResult::rawVideoStreamCount)
        .field("rawVideoInputBytes", &StagedPrepareResult::rawVideoInputBytes)
        .field("rawVideoAdoptMs", &StagedPrepareResult::rawVideoAdoptMs)
        .field("parseMs", &StagedPrepareResult::parseMs)
        .field("decodeNonVideoSubstreamsMs", &StagedPrepareResult::decodeNonVideoSubstreamsMs)
        .field("astcTextureDecodeMs", &StagedPrepareResult::astcTextureDecodeMs)
        .field("bcTextureEncodeMs", &StagedPrepareResult::bcTextureEncodeMs)
        .field("textureInputBytes", &StagedPrepareResult::textureInputBytes)
        .field("textureOutputBytes", &StagedPrepareResult::textureOutputBytes);

    value_object<PendingVideoResult>("PendingVideoResult")
        .field("success", &PendingVideoResult::success)
        .field("streamIndex", &PendingVideoResult::streamIndex)
        .field("codecId", &PendingVideoResult::codecId)
        .field("frameWidth", &PendingVideoResult::frameWidth)
        .field("frameHeight", &PendingVideoResult::frameHeight)
        .field("frameCount", &PendingVideoResult::frameCount)
        .field("encodedPtr", &PendingVideoResult::encodedPtr)
        .field("encodedSize", &PendingVideoResult::encodedSize);

    value_object<PreparedDecodeResult>("PreparedDecodeResult")
        .field("success", &PreparedDecodeResult::success)
        .field("protocolVersion", &PreparedDecodeResult::protocolVersion)
        .field("buildVersion", &PreparedDecodeResult::buildVersion)
        .field("pointCount", &PreparedDecodeResult::pointCount)
        .field("shDegree", &PreparedDecodeResult::shDegree)
        .field("blockSide", &PreparedDecodeResult::blockSide)
        .field("pointsPerBlock", &PreparedDecodeResult::pointsPerBlock)
        .field("blockCount", &PreparedDecodeResult::blockCount)
        .field("textureOutputMode", &PreparedDecodeResult::textureOutputMode)
        .field("decodeFeaturesRest", &PreparedDecodeResult::decodeFeaturesRest)
        .field("metadataPtr", &PreparedDecodeResult::metadataPtr)
        .field("metadataSize", &PreparedDecodeResult::metadataSize)
        .field("astcRawPtr", &PreparedDecodeResult::astcRawPtr)
        .field("astcRawSize", &PreparedDecodeResult::astcRawSize)
        .field("astcUvPtr", &PreparedDecodeResult::astcUvPtr)
        .field("astcUvCount", &PreparedDecodeResult::astcUvCount)
        .field("astcMetasPtr", &PreparedDecodeResult::astcMetasPtr)
        .field("astcMetasCount", &PreparedDecodeResult::astcMetasCount)
        .field("shnMin", &PreparedDecodeResult::shnMin)
        .field("shnMax", &PreparedDecodeResult::shnMax)
        .field("textureNum", &PreparedDecodeResult::textureNum)
        .field("profileIdc", &PreparedDecodeResult::profileIdc)
        .field("layoutVersion", &PreparedDecodeResult::layoutVersion)
        .field("pointMapWidth", &PreparedDecodeResult::pointMapWidth)
        .field("pointMapHeight", &PreparedDecodeResult::pointMapHeight)
        .field("positionGroupSize", &PreparedDecodeResult::positionGroupSize)
        .field("positionGroupMinPtr", &PreparedDecodeResult::positionGroupMinPtr)
        .field("positionGroupStepPtr", &PreparedDecodeResult::positionGroupStepPtr)
        .field("positionGroupFloatCount", &PreparedDecodeResult::positionGroupFloatCount)
        .field("shapeMinPtr", &PreparedDecodeResult::shapeMinPtr)
        .field("shapeStepPtr", &PreparedDecodeResult::shapeStepPtr)
        .field("shapeComponentCount", &PreparedDecodeResult::shapeComponentCount)
        .field("covarianceGroupMinPtr", &PreparedDecodeResult::covarianceGroupMinPtr)
        .field("covarianceGroupStepPtr", &PreparedDecodeResult::covarianceGroupStepPtr)
        .field("covarianceGroupFloatCount", &PreparedDecodeResult::covarianceGroupFloatCount)
        .field("encodedMinimumAlpha", &PreparedDecodeResult::encodedMinimumAlpha)
        .field("importanceMin", &PreparedDecodeResult::importanceMin)
        .field("importanceStep", &PreparedDecodeResult::importanceStep)
        .field("fastShMinPtr", &PreparedDecodeResult::fastShMinPtr)
        .field("fastShStepPtr", &PreparedDecodeResult::fastShStepPtr)
        .field("fastShComponentCount", &PreparedDecodeResult::fastShComponentCount)
        .field("positionLowPtr", &PreparedDecodeResult::positionLowPtr)
        .field("positionLowSize", &PreparedDecodeResult::positionLowSize)
        .field("positionHighPtr", &PreparedDecodeResult::positionHighPtr)
        .field("positionHighSize", &PreparedDecodeResult::positionHighSize)
        .field("colorRgbPtr", &PreparedDecodeResult::colorRgbPtr)
        .field("colorRgbSize", &PreparedDecodeResult::colorRgbSize)
        .field("shapePlanePtr", &PreparedDecodeResult::shapePlanePtr)
        .field("shapePlaneSize", &PreparedDecodeResult::shapePlaneSize)
        .field("shPlanePtr", &PreparedDecodeResult::shPlanePtr)
        .field("shPlaneSize", &PreparedDecodeResult::shPlaneSize)
        .field("parseMs", &PreparedDecodeResult::parseMs)
        .field("decodeNonVideoSubstreamsMs", &PreparedDecodeResult::decodeNonVideoSubstreamsMs)
        .field("astcTextureDecodeMs", &PreparedDecodeResult::astcTextureDecodeMs)
        .field("bcTextureEncodeMs", &PreparedDecodeResult::bcTextureEncodeMs)
        .field("textureInputBytes", &PreparedDecodeResult::textureInputBytes)
        .field("textureOutputBytes", &PreparedDecodeResult::textureOutputBytes)
        .field("substream0Ms", &PreparedDecodeResult::substream0Ms)
        .field("substream1Ms", &PreparedDecodeResult::substream1Ms)
        .field("substream2Ms", &PreparedDecodeResult::substream2Ms)
        .field("substream3Ms", &PreparedDecodeResult::substream3Ms)
        .field("substream4Ms", &PreparedDecodeResult::substream4Ms)
        .field("decodeVideoFallbackMs", &PreparedDecodeResult::decodeVideoFallbackMs)
        .field("decodeSubstreamsMs", &PreparedDecodeResult::decodeSubstreamsMs)
        .field("totalMs", &PreparedDecodeResult::totalMs);

    value_object<PackedShardResult>("PackedShardResult")
        .field("success", &PackedShardResult::success)
        .field("protocolVersion", &PackedShardResult::protocolVersion)
        .field("buildVersion", &PackedShardResult::buildVersion)
        .field("startBlock", &PackedShardResult::startBlock)
        .field("blockCount", &PackedShardResult::blockCount)
        .field("startPoint", &PackedShardResult::startPoint)
        .field("pointCount", &PackedShardResult::pointCount)
        .field("packetPtr", &PackedShardResult::packetPtr)
        .field("packetSize", &PackedShardResult::packetSize)
        .field("packMs", &PackedShardResult::packMs)
        .field("unpackMs", &PackedShardResult::unpackMs);

    class_<GaussianSlimLoaderCoordinatorWasm>("GaussianSlimLoaderCoordinatorWasm")
        .constructor<>()
        .function("getTextureEncoderMask", &GaussianSlimLoaderCoordinatorWasm::getTextureEncoderMask)
        .function("prepare", &GaussianSlimLoaderCoordinatorWasm::prepare)
        .function("beginPrepare", &GaussianSlimLoaderCoordinatorWasm::beginPrepare)
        .function("getPendingVideo", &GaussianSlimLoaderCoordinatorWasm::getPendingVideo)
        .function("injectDecodedVideo", &GaussianSlimLoaderCoordinatorWasm::injectDecodedVideo)
        .function("decodePendingVideosWithFallback", &GaussianSlimLoaderCoordinatorWasm::decodePendingVideosWithFallback)
        .function("finishPrepare", &GaussianSlimLoaderCoordinatorWasm::finishPrepare)
        .function("packShard", &GaussianSlimLoaderCoordinatorWasm::packShard)
        .function("releasePackedShard", &GaussianSlimLoaderCoordinatorWasm::releasePackedShard)
        .function("release", &GaussianSlimLoaderCoordinatorWasm::release)
        .function("getLastError", &GaussianSlimLoaderCoordinatorWasm::getLastError);
}
