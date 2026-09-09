#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gaussian_model/gs_decoder.h"
#include "processor/platform_video_decoder.h"
#include "processor/stream.h"
#include "processor/unpacker.h"

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << std::endl;
    return condition;
}

void appendUint32(std::vector<uint8_t>& output, uint32_t value) {
    output.push_back(static_cast<uint8_t>(value >> 24));
    output.push_back(static_cast<uint8_t>(value >> 16));
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value));
}

std::vector<uint8_t> makeRawPayload(uint32_t width, uint32_t height, uint32_t frames) {
    std::vector<uint8_t> payload(static_cast<size_t>(width) * height * frames * 3);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        for (uint32_t channel = 0; channel < 3; ++channel) {
            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    const size_t offset = (((static_cast<size_t>(frame) * 3 + channel) * height + y) * width + x);
                    payload[offset] = static_cast<uint8_t>(frame * 100 + channel * 20 + y * width + x);
                }
            }
        }
    }
    return payload;
}

std::shared_ptr<VideoMeta> makeVideoMeta(uint32_t width, uint32_t height, uint32_t frames) {
    auto video = std::make_shared<VideoMeta>();
    video->videoDecodeInformation.packingMapVideoCodecId = 0;
    auto& packing = video->videoPackingInformation;
    packing.packingMapWidth = static_cast<uint16_t>(width);
    packing.packingMapHeight = static_cast<uint16_t>(height);
    packing.regionWidth = static_cast<uint16_t>(width);
    packing.regionHeight = static_cast<uint16_t>(height);
    packing.packingMapFrameNumMinus1 = static_cast<uint16_t>(frames - 1);
    packing.packingScaningType = 0;
    packing.packingRegionCountMinus1 = 0;
    packing.initialize();
    packing.regionFrameIndex[0] = 1;
    packing.regionTopLeftX[0] = 0;
    packing.regionTopLeftY[0] = 0;
    packing.attributeType[0] = 0;
    packing.attributeChannelOffset[0] = 0;
    packing.attributeChannelNum[0] = 3;
    packing.byteshift[0] = 0;
    return video;
}

std::vector<uint8_t> makeRawGsbs(const std::vector<uint8_t>& payload,
                                 uint32_t width, uint32_t height, uint32_t frames) {
    Unit metadataUnit(0);
    auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    metadata.initialize2(width * height, 1, 0, 1);
    metadata.subGsPointsNum[0] = width * height;
    metadata.subBitstreamSize[0] = static_cast<uint32_t>(payload.size());
    metadata.gsSubsetId[0] = 0;
    metadata.subBitstreamDecodeType[0] = 2;
    metadata.subBitstreamMeta.push_back(makeVideoMeta(width, height, frames));
    metadata.subBitstreamMetaType.push_back(SubBitstreamMetaType::VIDEO);
    metadata.reconstructionInformation.resize(1);
    metadata.reconstructionCount[0] = 0;
    if (!metadataUnit.write()) return {};

    Unit substreamUnit(1);
    substreamUnit.unitPayload.gsbsSubBitstreams->initialize(1);
    substreamUnit.unitPayload.gsbsSubBitstreams->gstcSubBitstreamData[0] = payload;
    if (!substreamUnit.write()) return {};

    const auto metadataBytes = metadataUnit.writer.getBytes();
    const auto substreamBytes = substreamUnit.writer.getBytes();
    std::vector<uint8_t> output;
    output.reserve(8 + metadataBytes.size() + substreamBytes.size());
    appendUint32(output, static_cast<uint32_t>(metadataBytes.size()));
    output.insert(output.end(), metadataBytes.begin(), metadataBytes.end());
    appendUint32(output, static_cast<uint32_t>(substreamBytes.size()));
    output.insert(output.end(), substreamBytes.begin(), substreamBytes.end());
    return output;
}

bool testPixelFormatAndPlanarUnpack() {
    constexpr uint32_t width = 3;
    constexpr uint32_t height = 2;
    constexpr uint32_t frames = 2;
    const auto payload = makeRawPayload(width, height, frames);
    const VideoFrameLayout layout{VideoPixelFormat::YUV444P, width, height, frames};
    VideoPixelFormat abiFormat = VideoPixelFormat::YUV444_INTERLEAVED;
    if (!expect(videoPixelFormatFromAbi(4, abiFormat) && abiFormat == VideoPixelFormat::YUV444P,
                "stable pixel-format ABI value 4 must resolve only to YUV444P")) {
        return false;
    }
    if (!expect(videoPixelFormatMatchesCodec(CODEC_ID_RAW, VideoPixelFormat::YUV444P) &&
                !videoPixelFormatMatchesCodec(CODEC_ID_RAW, VideoPixelFormat::YUV444_INTERLEAVED),
                "codec0 validation must reject non-planar YUV444 layouts")) {
        return false;
    }
    size_t byteLength = 0;
    if (!expect(videoFrameByteLength(layout, byteLength), "YUV444P byte length must be valid") ||
        !expect(byteLength == payload.size(), "YUV444P byte length must include every frame and plane")) {
        return false;
    }

    auto video = makeVideoMeta(width, height, frames);
    Unpacker unpacker;
    std::map<std::string, std::vector<int32_t>> attributes;
    if (!expect(unpacker.unpackVideo(payload, attributes, *video, layout, 0, -1, width * height),
                "planar multi-frame video must unpack with per-frame height")) {
        return false;
    }
    const auto found = attributes.find("means");
    if (!expect(found != attributes.end(), "unpack must produce means")) return false;
    const auto& values = found->second;
    if (!expect(values.size() == width * height * 3, "unpacked means size must be pixel-major RGB")) return false;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t pixel = static_cast<size_t>(y) * width + x;
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const int32_t expected = static_cast<int32_t>(100 + channel * 20 + pixel);
                if (!expect(values[pixel * 3 + channel] == expected,
                            "unpacker must sample frame/channel/y/x planar order")) return false;
            }
        }
    }
    return true;
}

bool testRawOnlyStagedLifecycle() {
    constexpr uint32_t width = 3;
    constexpr uint32_t height = 2;
    constexpr uint32_t frames = 2;
    const auto payload = makeRawPayload(width, height, frames);
    const auto stream = makeRawGsbs(payload, width, height, frames);
    GSDecoder decoder;
    if (!expect(decoder.beginPrepareFromMemory(stream.data(), stream.size(), TextureOutputMode::CPU),
                "raw-only beginPrepareFromMemory must succeed")) {
        std::cerr << decoder.getLastError() << std::endl;
        return false;
    }
    if (!expect(decoder.getPendingVideoStreams().empty(), "codec0 must never enter the pending queue") ||
        !expect(getTimingStats().rawVideoStreamCount == 1, "raw stream count must be recorded") ||
        !expect(getTimingStats().rawVideoInputBytes == payload.size(), "raw input bytes must be recorded") ||
        !expect(decoder.finishPreparedDecode(), "raw-only finishPreparedDecode must succeed")) {
        std::cerr << decoder.getLastError() << std::endl;
        return false;
    }
    return true;
}

bool testMalformedLengthsFail() {
    constexpr uint32_t width = 3;
    constexpr uint32_t height = 2;
    constexpr uint32_t frames = 2;
    const auto valid = makeRawPayload(width, height, frames);
    for (int delta : {-1, 1}) {
        auto malformed = valid;
        if (delta < 0) malformed.pop_back();
        else malformed.push_back(0);
        const auto stream = makeRawGsbs(malformed, width, height, frames);
        GSDecoder decoder;
        if (!expect(!decoder.beginPrepareFromMemory(stream.data(), stream.size(), TextureOutputMode::CPU),
                    delta < 0 ? "N-1 raw payload must fail" : "N+1 raw payload must fail")) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    if (!testPixelFormatAndPlanarUnpack() ||
        !testRawOnlyStagedLifecycle() ||
        !testMalformedLengthsFail()) {
        return 1;
    }
    std::cout << "codec0 tests passed" << std::endl;
    return 0;
}
