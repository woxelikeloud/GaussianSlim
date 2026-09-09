#include "gs_decoder.h"
#include "../processor/platform_video_decoder.h"
#include "../processor/texture_transcoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <set>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "GSDecoder"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOG_SUMMARY(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#ifndef NDEBUG
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) ((void)0)
#endif
#else
#define LOGE(fmt, ...) fprintf(stderr, "[GSDecoder] ERROR: " fmt "\n", ##__VA_ARGS__)
#define LOG_SUMMARY(fmt, ...) printf("[GSDecoder] " fmt "\n", ##__VA_ARGS__)
#ifndef NDEBUG
#define LOGI(fmt, ...) printf("[GSDecoder] " fmt "\n", ##__VA_ARGS__)
#else
#define LOGI(fmt, ...) ((void)0)
#endif
#endif

namespace {

using Clock = std::chrono::high_resolution_clock;

double elapsedMs(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void copyAuxiliaryData(const SplatData& source, SplatData& destination) {
    destination.astcRawStream = source.astcRawStream;
    destination.astcUV = source.astcUV;
    destination.astcMetas = source.astcMetas;
    destination.astcTextureNum = source.astcTextureNum;
    destination.shnMin = source.shnMin;
    destination.shnMax = source.shnMax;
}

bool checkedPointProduct(uint32_t side, uint32_t& points) {
    if (side == 0 || side > std::numeric_limits<uint32_t>::max() / side) return false;
    points = side * side;
    return true;
}

bool checkedSizeProduct(size_t left, size_t right, size_t& product) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) return false;
    product = left * right;
    return true;
}

bool canonicalChannelsForAttributeType(int attributeType, uint32_t& channels) {
    compactgs::ShardAttribute attribute;
    if (!compactgs::attributeFromName(attributeTypeToName(attributeType), attribute)) return false;
    channels = compactgs::attributeChannels(attribute);
    return channels != 0;
}

bool expectedDecodedBytes(const GsbsMetadata& metadata,
                          int streamIndex,
                          size_t& byteCount,
                          bool& requireExactSize,
                          std::string& error) {
    byteCount = 0;
    requireExactSize = false;
    const auto metaType = metadata.subBitstreamMetaType[streamIndex];
    if (metaType == SubBitstreamMetaType::ENTROPY) {
        const auto entropy = std::static_pointer_cast<EntropyMeta>(metadata.subBitstreamMeta[streamIndex]);
        uint32_t channels = 0;
        if (!entropy || !canonicalChannelsForAttributeType(entropy->attributeType, channels) ||
            (entropy->bitdepth != 8 && entropy->bitdepth != 16)) {
            error = "Invalid entropy layout";
            return false;
        }
        size_t valueCount = 0;
        if (!checkedSizeProduct(metadata.gsPointsNum, channels, valueCount) ||
            !checkedSizeProduct(valueCount, entropy->bitdepth / 8, byteCount)) {
            error = "Entropy layout is too large";
            return false;
        }
        return true;
    }

    if (metaType == SubBitstreamMetaType::TEXTURE) {
        const auto texture = std::static_pointer_cast<TextureMeta>(metadata.subBitstreamMeta[streamIndex]);
        if (!texture) {
            error = "Missing texture metadata";
            return false;
        }
        const auto& packing = texture->texturePackingInformation;
        size_t pixelCount = 0;
        if (!checkedSizeProduct(packing.packingMapWidth, packing.packingMapHeight, pixelCount) ||
            !checkedSizeProduct(pixelCount, packing.textureChannelNum, byteCount)) {
            error = "Texture map is too large";
            return false;
        }
        requireExactSize = true;
        return true;
    }

    error = "Video buffers require an explicit pixel layout";
    return false;
}

bool normalizeUnpackedAttribute(std::vector<int32_t>& values,
                                uint32_t channels,
                                uint32_t pointCount,
                                uint32_t globalBlock,
                                uint32_t globalPoint,
                                const DecodeDescriptor& descriptor,
                                bool allowTrailingPadding,
                                std::string& error) {
    size_t expectedValues = 0;
    size_t maximumValues = 0;
    if (channels == 0 ||
        !checkedSizeProduct(pointCount, channels, expectedValues) ||
        !checkedSizeProduct(descriptor.pointsPerBlock, channels, maximumValues)) {
        error = "Unpacked attribute layout is too large";
        return false;
    }
    if (values.size() == expectedValues) return true;
    if (values.size() < expectedValues) {
        error = "Unpacked attribute is shorter than the declared block";
        return false;
    }
    if (!allowTrailingPadding || values.size() % channels != 0 ||
        values.size() > maximumValues || pointCount >= descriptor.pointsPerBlock ||
        globalBlock + 1 != descriptor.blockCount ||
        static_cast<uint64_t>(globalPoint) + pointCount != descriptor.pointCount) {
        error = "Unpacked attribute has non-trailing or invalid padding";
        return false;
    }

    values.resize(expectedValues);
    return true;
}

bool checkedCompressedTextureBytes(uint32_t width,
                                   uint32_t height,
                                   uint32_t blockWidth,
                                   uint32_t blockHeight,
                                   size_t& byteCount) {
    if (width == 0 || height == 0 || blockWidth == 0 || blockHeight == 0) return false;
    const uint64_t blocksX = (static_cast<uint64_t>(width) + blockWidth - 1) / blockWidth;
    const uint64_t blocksY = (static_cast<uint64_t>(height) + blockHeight - 1) / blockHeight;
    const uint64_t bytes = blocksX * blocksY * 16;
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max()) return false;
    byteCount = static_cast<size_t>(bytes);
    return true;
}

bool isSupportedSquareAstcBlock(uint32_t blockSide) {
    return blockSide == 4 || blockSide == 5 || blockSide == 6 ||
           blockSide == 8 || blockSide == 10 || blockSide == 12;
}

bool buildCompressedFeatureLayout(const TextureMeta& textureMeta,
                                  uint32_t pointCount,
                                  uint32_t textureWidth,
                                  uint32_t textureHeight,
                                  uint32_t gpuBlockSide,
                                  size_t streamSize,
                                  bool generateUvs,
                                  SplatData::AstcMeta& outputMeta,
                                  std::vector<uint32_t>& outputUvs,
                                  std::string& error) {
    const auto& packing = textureMeta.texturePackingInformation;
    const uint32_t regionWidth = packing.regionWidth;
    const uint32_t regionHeight = packing.regionHeight;
    const size_t regionCount = static_cast<size_t>(packing.packingRegionCountMinus1) + 1;
    const uint64_t regionCapacity = static_cast<uint64_t>(regionWidth) * regionHeight;
    if (textureWidth != packing.packingMapWidth || textureHeight != packing.packingMapHeight) {
        error = "Compressed features_rest dimensions do not match the texture packing map";
        return false;
    }
    if (textureWidth > 16384 || textureHeight > 16384 || gpuBlockSide == 0 ||
        regionWidth == 0 || regionHeight == 0 || pointCount > regionCapacity ||
        streamSize > std::numeric_limits<uint32_t>::max() ||
        packing.regionTopLeftX.size() != regionCount || packing.regionTopLeftY.size() != regionCount) {
        error = "Compressed features_rest layout exceeds the GPU protocol limits";
        return false;
    }

    const int32_t baseXValue = packing.regionTopLeftX[0];
    const int32_t baseYValue = packing.regionTopLeftY[0];
    if (baseXValue < 0 || baseYValue < 0) {
        error = "Compressed features_rest has a negative base region offset";
        return false;
    }
    const uint32_t baseX = static_cast<uint32_t>(baseXValue);
    const uint32_t baseY = static_cast<uint32_t>(baseYValue);

    // The shader advances regions in row-major order. Reject layouts it cannot address
    // so the worker can retry through the CPU fallback instead of silently sampling bad SH.
    for (size_t region = 0; region < regionCount; ++region) {
        const uint64_t linearX = static_cast<uint64_t>(baseX) + region * regionWidth;
        const uint32_t expectedX = static_cast<uint32_t>(linearX % textureWidth);
        const uint64_t expectedY64 = static_cast<uint64_t>(baseY) +
            (linearX / textureWidth) * regionHeight;
        if (expectedY64 > std::numeric_limits<uint32_t>::max() ||
            expectedX + regionWidth > textureWidth || expectedY64 + regionHeight > textureHeight ||
            packing.regionTopLeftX[region] != static_cast<int32_t>(expectedX) ||
            packing.regionTopLeftY[region] != static_cast<int32_t>(expectedY64)) {
            error = "Compressed features_rest regions are not in the supported row-major layout";
            return false;
        }
    }

    uint32_t scanBlockSide = 1;
    if (packing.packingScaningType == 1) {
        scanBlockSide = packing.packingScaningBlockSize;
        if (scanBlockSide == 0 || regionWidth % scanBlockSide != 0 || regionHeight % scanBlockSide != 0) {
            error = "Compressed features_rest has invalid block-scan geometry";
            return false;
        }
    } else if (packing.packingScaningType != 0) {
        error = "Compressed features_rest uses an unsupported packing scan";
        return false;
    }

    outputUvs.clear();
    if (generateUvs) {
        size_t uvCount = 0;
        constexpr size_t kMaxUvBytes = 256ull * 1024 * 1024;
        if (!checkedSizeProduct(pointCount, 2, uvCount) ||
            uvCount > kMaxUvBytes / sizeof(uint32_t)) {
            error = "Compressed features_rest UV table is too large";
            return false;
        }
        try {
            outputUvs.resize(uvCount);
        } catch (const std::bad_alloc&) {
            error = "Compressed features_rest UV allocation failed";
            return false;
        }
        if (packing.packingScaningType == 1) {
            const uint32_t blockArea = scanBlockSide * scanBlockSide;
            const uint32_t blocksPerRow = regionWidth / scanBlockSide;
            for (uint32_t point = 0; point < pointCount; ++point) {
                const uint32_t block = point / blockArea;
                const uint32_t inBlock = point % blockArea;
                outputUvs[static_cast<size_t>(point) * 2] = baseX +
                    (block % blocksPerRow) * scanBlockSide + inBlock % scanBlockSide;
                outputUvs[static_cast<size_t>(point) * 2 + 1] = baseY +
                    (block / blocksPerRow) * scanBlockSide + inBlock / scanBlockSide;
            }
        } else {
            for (uint32_t point = 0; point < pointCount; ++point) {
                outputUvs[static_cast<size_t>(point) * 2] = baseX + point % regionWidth;
                outputUvs[static_cast<size_t>(point) * 2 + 1] = baseY + point / regionWidth;
            }
        }
    }

    outputMeta.astcBlockSize = gpuBlockSide;
    outputMeta.astcWidth = textureWidth;
    outputMeta.astcHeight = textureHeight;
    outputMeta.singleWidth = regionWidth;
    outputMeta.numPoints = static_cast<uint32_t>(regionCapacity);
    outputMeta.streamSize = static_cast<uint32_t>(streamSize);
    return true;
}

} // namespace

static TimingStats g_timingStats;

TimingStats& getTimingStats() {
    return g_timingStats;
}

void TimingStats::print() const {
    std::cout << "\n========== Timing Statistics ==========\n" << std::fixed << std::setprecision(2)
              << "  Read File:        " << std::setw(10) << readFileMs << " ms\n"
              << "  Parse Stream:     " << std::setw(10) << parseStreamMs << " ms\n"
              << "  Decode Non-Video: " << std::setw(10) << decodeNonVideoSubstreamsMs << " ms\n"
              << "  ASTC Texture:     " << std::setw(10) << astcTextureDecodeMs << " ms\n"
              << "  BC Encode:        " << std::setw(10) << bcTextureEncodeMs << " ms\n"
              << "  Raw Video Adopt:  " << std::setw(10) << rawVideoAdoptMs << " ms ("
              << rawVideoStreamCount << " streams, " << rawVideoInputBytes << " bytes)\n"
              << "  Video Fallback:   " << std::setw(10) << decodeVideoFallbackMs << " ms\n"
              << "  Decode Substreams:" << std::setw(10) << decodeSubstreamsMs << " ms\n"
              << "  Unpack:           " << std::setw(10) << unpackMs << " ms\n"
              << "  Pack Shards:      " << std::setw(10) << packShardMs << " ms\n"
              << "  Prediction:       " << std::setw(10) << predictionMs << " ms\n"
              << "  Dequantize:       " << std::setw(10) << dequantizeMs << " ms\n"
              << "  Transform:        " << std::setw(10) << transformMs << " ms\n"
              << "  Nonlinear:        " << std::setw(10) << nonlinearMs << " ms\n"
              << "  Output Copy:      " << std::setw(10) << outputCopyMs << " ms\n"
              << "  Prune:            " << std::setw(10) << pruneMs << " ms\n"
              << "  Decode Total:     " << std::setw(10) << totalMs << " ms\n"
              << "  ========================================\n";
}

GSDecoder::GSDecoder()
    : metadataUnit(0), substreamUnit(1) {
}

GSDecoder::~GSDecoder() {
    releasePreparedDecode();
}

void GSDecoder::setError(const std::string& error) {
    lastError_ = error;
    LOGE("%s", error.c_str());
}

bool GSDecoder::readFile(const std::string& filename, std::vector<uint8_t>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize <= 0) return false;
    file.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(fileSize));
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    return file.good();
}

bool GSDecoder::decode(const std::string& binFile, SplatData& gsData) {
    g_timingStats = TimingStats();
    std::vector<uint8_t> fileData;
    const auto readStart = Clock::now();
    if (!readFile(binFile, fileData)) {
        setError("Failed to read stream file: " + binFile);
        return false;
    }
    g_timingStats.readFileMs = elapsedMs(readStart);
    return decodeFromMemory(fileData.data(), fileData.size(), gsData);
}

bool GSDecoder::prepareFromMemory(const uint8_t* buffer,
                                  size_t bufferSize,
                                  TextureOutputMode textureOutputMode) {
    if (!beginPrepareFromMemory(buffer, bufferSize, textureOutputMode)) return false;
    if (!decodePendingVideosWithFallback() || !finishPreparedDecode()) {
        releasePreparedDecode();
        return false;
    }
    return true;
}

bool GSDecoder::beginPrepareFromMemory(const uint8_t* buffer,
                                       size_t bufferSize,
                                       TextureOutputMode textureOutputMode) {
    releasePreparedDecode();
    lastError_.clear();
    g_timingStats = TimingStats();
    prepareStartedAt_ = Clock::now();

    if (!buffer || bufferSize == 0) {
        setError("Cannot prepare an empty GSBS buffer");
        return false;
    }
    if (textureOutputMode != TextureOutputMode::ASTC && textureOutputMode != TextureOutputMode::CPU &&
        textureOutputMode != TextureOutputMode::BC7 && textureOutputMode != TextureOutputMode::BC3) {
        setError("Invalid texture output mode");
        return false;
    }
#ifndef USE_BC_TEXTURE_ENCODERS
    if (textureOutputMode == TextureOutputMode::BC7 || textureOutputMode == TextureOutputMode::BC3) {
        setError("BC texture output requested, but this build has no BC encoder");
        return false;
    }
#endif
    const bool decodeFeaturesRest = textureOutputMode == TextureOutputMode::CPU;

    std::vector<uint8_t> stream;
    if (!decompressZstdIfNeeded(buffer, bufferSize, stream)) {
        setError("Failed to decompress the GSBS input payload");
        return false;
    }

    const auto parseStart = Clock::now();
    if (!parseGsbsStream(stream, metadataUnit, substreamUnit)) {
        setError("Failed to parse GSBS metadata or substreams");
        return false;
    }
    g_timingStats.parseStreamMs = elapsedMs(parseStart);
    descriptor_.textureOutputMode = textureOutputMode;
    descriptor_.decodeFeaturesRest = decodeFeaturesRest;

    if (!validatePreparedMetadata()) {
        releasePreparedDecode();
        return false;
    }

    const auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    auto& encodedStreams = substreamUnit.unitPayload.gsbsSubBitstreams->gstcSubBitstreamData;
    uint32_t videoStreamCount = 0;
    const auto rawAdoptStart = Clock::now();
    for (int i = 0; i < metadata.subBitstreamNum; ++i) {
        if (metadata.subBitstreamMetaType[i] != SubBitstreamMetaType::VIDEO) continue;
        ++videoStreamCount;
        const auto video = std::static_pointer_cast<VideoMeta>(metadata.subBitstreamMeta[i]);
        if (!video) {
            setError("Missing video metadata for substream " + std::to_string(i));
            releasePreparedDecode();
            return false;
        }
        const auto& packing = video->videoPackingInformation;
        const uint32_t frameCount = static_cast<uint32_t>(packing.packingMapFrameNumMinus1) + 1;
        const uint32_t codecId = video->videoDecodeInformation.packingMapVideoCodecId;
        if (codecId == CODEC_ID_RAW) {
            const VideoFrameLayout layout = {
                VideoPixelFormat::YUV444P,
                packing.packingMapWidth,
                packing.packingMapHeight,
                frameCount
            };
            size_t expectedBytes = 0;
            if (static_cast<size_t>(i) >= encodedStreams.size() ||
                !videoFrameByteLength(layout, expectedBytes) ||
                encodedStreams[i].size() != expectedBytes ||
                expectedBytes > std::numeric_limits<uint64_t>::max() - g_timingStats.rawVideoInputBytes) {
                setError("Raw video substream " + std::to_string(i) +
                         " does not match its exact YUV444P byte layout");
                releasePreparedDecode();
                return false;
            }
            const std::string streamName = "stream_" + std::to_string(i);
            decodedStreams[streamName].swap(encodedStreams[i]);
            decodedVideoLayouts_[streamName] = layout;
            g_timingStats.rawVideoInputBytes += expectedBytes;
            ++g_timingStats.rawVideoStreamCount;
            g_timingStats.substreamTimings[streamName] = 0.0;
            continue;
        }
        pendingVideoStreams_.push_back({
            static_cast<uint32_t>(i),
            codecId,
            packing.packingMapWidth,
            packing.packingMapHeight,
            frameCount
        });
    }
    if (g_timingStats.rawVideoStreamCount > 0) {
        g_timingStats.rawVideoAdoptMs = elapsedMs(rawAdoptStart);
    }
    if (g_timingStats.rawVideoStreamCount + pendingVideoStreams_.size() != videoStreamCount) {
        setError("Internal video routing count mismatch");
        releasePreparedDecode();
        return false;
    }

    if (!setupDecoders()) {
        if (lastError_.empty()) setError("Failed to initialize one or more GSBS substream decoders");
        releasePreparedDecode();
        return false;
    }

    const auto decodeStart = Clock::now();
    preparedAuxiliaryData_.numPoints = static_cast<int>(descriptor_.pointCount);
    preparedAuxiliaryData_.shDegree = static_cast<int>(descriptor_.shDegree);
    if (!decodeNonVideoSubstreams(preparedAuxiliaryData_, textureOutputMode)) {
        if (lastError_.empty()) setError("Failed to decode a GSBS substream");
        releasePreparedDecode();
        return false;
    }
    g_timingStats.decodeNonVideoSubstreamsMs = elapsedMs(decodeStart);
    g_timingStats.decodeSubstreamsMs =
        g_timingStats.rawVideoAdoptMs + g_timingStats.decodeNonVideoSubstreamsMs;
    preparing_ = true;
    return true;
}

const std::vector<uint8_t>* GSDecoder::getPendingVideoEncodedData(uint32_t streamIndex) const {
    if (!preparing_ || !substreamUnit.unitPayload.gsbsSubBitstreams) return nullptr;
    const auto pending = std::find_if(pendingVideoStreams_.begin(), pendingVideoStreams_.end(),
        [streamIndex](const PendingVideoStreamDescriptor& descriptor) {
            return descriptor.streamIndex == streamIndex;
        });
    if (pending == pendingVideoStreams_.end()) return nullptr;
    const auto& encodedStreams = substreamUnit.unitPayload.gsbsSubBitstreams->gstcSubBitstreamData;
    if (streamIndex >= encodedStreams.size()) return nullptr;
    return &encodedStreams[streamIndex];
}

bool GSDecoder::injectDecodedVideo(uint32_t streamIndex,
                                   const uint8_t* decoded,
                                   size_t decodedSize,
                                   VideoPixelFormat pixelFormat) {
    if (!preparing_) {
        setError("Decoded video injection requires an active staged prepare");
        return false;
    }
    const auto pending = std::find_if(pendingVideoStreams_.begin(), pendingVideoStreams_.end(),
        [streamIndex](const PendingVideoStreamDescriptor& descriptor) {
            return descriptor.streamIndex == streamIndex;
        });
    if (pending == pendingVideoStreams_.end()) {
        setError("Decoded video injection referenced a non-video substream");
        return false;
    }
    const VideoFrameLayout layout = {
        pixelFormat,
        pending->frameWidth,
        pending->frameHeight,
        pending->frameCount
    };
    size_t expectedBytes = 0;
    if (!decoded || !videoFrameByteLength(layout, expectedBytes) || decodedSize != expectedBytes) {
        setError("Externally decoded video substream " + std::to_string(streamIndex) +
                 " has an invalid byte layout");
        return false;
    }

    const std::string streamName = "stream_" + std::to_string(streamIndex);
    decodedStreams[streamName].assign(decoded, decoded + decodedSize);
    decodedVideoLayouts_[streamName] = layout;
    g_timingStats.substreamTimings[streamName] = 0.0;
    lastError_.clear();
    return true;
}

bool GSDecoder::finishPreparedDecode() {
    if (!preparing_) {
        setError("finishPreparedDecode called without an active staged prepare");
        return false;
    }
    const auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    for (int streamIndex = 0; streamIndex < metadata.subBitstreamNum; ++streamIndex) {
        if (metadata.subBitstreamMetaType[streamIndex] != SubBitstreamMetaType::VIDEO) continue;
        const std::string streamName = "stream_" + std::to_string(streamIndex);
        const auto decoded = decodedStreams.find(streamName);
        const auto layout = decodedVideoLayouts_.find(streamName);
        if (decoded == decodedStreams.end() || layout == decodedVideoLayouts_.end() ||
            !validateDecodedSubstream(streamIndex, decoded->second)) {
            if (lastError_.empty()) {
                setError("Video substream " + std::to_string(streamIndex) + " is still pending");
            }
            return false;
        }
    }

    if (metadata.profileIdc == 2) {
        if (!buildCompactGSPlanes()) return false;
        reconstructionMetadataPacket_.clear();
    } else if (!buildReconstructionMetadata()) {
        return false;
    }

    // Reconstruction only needs decoded streams and parsed metadata. Drop the
    // compressed copies and codec contexts before shard packets are produced.
    decoders.clear();
    if (substreamUnit.unitPayload.gsbsSubBitstreams) {
        for (auto& encoded : substreamUnit.unitPayload.gsbsSubBitstreams->gstcSubBitstreamData) {
            std::vector<uint8_t>().swap(encoded);
        }
    }

    preparing_ = false;
    prepared_ = true;
    g_timingStats.totalMs = elapsedMs(prepareStartedAt_);
    LOG_SUMMARY("Prepared GSBS profile=%u: points=%u blocks=%u blockSide=%u metadata=%zu bytes",
                descriptor_.profileIdc,
                descriptor_.pointCount, descriptor_.blockCount, descriptor_.blockSide,
                reconstructionMetadataPacket_.size());
    return true;
}

bool GSDecoder::decodeFromMemory(const uint8_t* buffer,
                                 size_t bufferSize,
                                 SplatData& gsData,
                                 TextureOutputMode textureOutputMode) {
    const auto totalStart = Clock::now();
    if (!prepareFromMemory(buffer, bufferSize, textureOutputMode)) return false;
    const bool decodeFeaturesRest = textureOutputMode == TextureOutputMode::CPU;

    gsData.clear();
    gsData.numPoints = static_cast<int>(descriptor_.pointCount);
    gsData.shDegree = static_cast<int>(descriptor_.shDegree);
    gsData.means.resize(static_cast<size_t>(descriptor_.pointCount) * 3);
    gsData.opacity.resize(descriptor_.pointCount);
    gsData.scaling.resize(static_cast<size_t>(descriptor_.pointCount) * 3);
    gsData.rotation.resize(static_cast<size_t>(descriptor_.pointCount) * 4);
    gsData.features_dc.resize(static_cast<size_t>(descriptor_.pointCount) * 3);
    if (decodeFeaturesRest) {
        gsData.features_rest.resize(static_cast<size_t>(descriptor_.pointCount) * 45);
    }
    copyAuxiliaryData(preparedAuxiliaryData_, gsData);

    constexpr uint32_t kSerialBlocksPerShard = 64;
    for (uint32_t startBlock = 0; startBlock < descriptor_.blockCount; startBlock += kSerialBlocksPerShard) {
        const uint32_t blockCount = std::min(kSerialBlocksPerShard, descriptor_.blockCount - startBlock);
        std::vector<uint8_t> packet;
        if (!packShard(startBlock, blockCount, packet)) {
            releasePreparedDecode();
            return false;
        }

        compactgs::ReconstructionResult result;
        std::string error;
        if (!compactgs::reconstructPackedShard(reconstructionMetadata_, packet.data(), packet.size(), result, error)) {
            setError("Serial shard reconstruction failed: " + error);
            releasePreparedDecode();
            return false;
        }

        const size_t pointOffset = result.startPoint;
        std::copy(result.positions.begin(), result.positions.end(), gsData.means.begin() + pointOffset * 3);
        std::copy(result.opacity.begin(), result.opacity.end(), gsData.opacity.begin() + pointOffset);
        std::copy(result.scales.begin(), result.scales.end(), gsData.scaling.begin() + pointOffset * 3);
        std::copy(result.rotations.begin(), result.rotations.end(), gsData.rotation.begin() + pointOffset * 4);
        std::copy(result.featuresDc.begin(), result.featuresDc.end(), gsData.features_dc.begin() + pointOffset * 3);
        if (decodeFeaturesRest) {
            std::copy(result.featuresRest.begin(), result.featuresRest.end(), gsData.features_rest.begin() + pointOffset * 45);
        }
        g_timingStats.predictionMs += result.timings.predictionMs;
        g_timingStats.dequantizeMs += result.timings.dequantizeMs;
        g_timingStats.transformMs += result.timings.transformMs;
        g_timingStats.nonlinearMs += result.timings.nonlinearMs;
        g_timingStats.outputCopyMs += result.timings.outputCopyMs;
    }

    const auto pruneStart = Clock::now();
    if (!pruneInvalidGaussians(gsData)) {
        releasePreparedDecode();
        return false;
    }
    g_timingStats.pruneMs = elapsedMs(pruneStart);
    g_timingStats.totalMs = elapsedMs(totalStart);
    releasePreparedDecode();
    return true;
}

bool GSDecoder::validatePreparedMetadata() {
    if (!metadataUnit.unitPayload.gsbsMetadata || !substreamUnit.unitPayload.gsbsSubBitstreams) {
        setError("GSBS stream is missing metadata or substream units");
        return false;
    }
    const auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    const auto& substreams = *substreamUnit.unitPayload.gsbsSubBitstreams;

    if (metadata.gsSubsetNum != 1 || metadata.subGsPointsNum.size() != 1) {
        setError("Unsupported GSBS layout: worker sharding currently requires exactly one subset");
        return false;
    }
    if (metadata.gsPointsNum == 0 || metadata.subGsPointsNum[0] != metadata.gsPointsNum) {
        setError("Invalid GSBS subset point counts: the single subset must cover the full model");
        return false;
    }
    if (metadata.subBitstreamNum <= 0 || substreams.subBitstreamNum != metadata.subBitstreamNum ||
        metadata.subBitstreamMeta.size() != static_cast<size_t>(metadata.subBitstreamNum) ||
        metadata.subBitstreamMetaType.size() != static_cast<size_t>(metadata.subBitstreamNum) ||
        metadata.subBitstreamDecodeType.size() != static_cast<size_t>(metadata.subBitstreamNum) ||
        metadata.gsSubsetId.size() != static_cast<size_t>(metadata.subBitstreamNum)) {
        setError("Inconsistent GSBS substream metadata counts");
        return false;
    }
    for (uint8_t subsetId : metadata.gsSubsetId) {
        if (subsetId != 0) {
            setError("Invalid GSBS subset id for a single-subset stream");
            return false;
        }
    }

    if (metadata.profileIdc == 2) {
        if (!metadata.compactgs || metadata.subBitstreamNum != 5 ||
            metadata.reconstructionCount.size() != 1 || metadata.reconstructionCount[0] != 0) {
            setError("COMPACTGS_V1 requires one five-stream subset and no reconstruction metadata");
            return false;
        }
        const auto& fast = *metadata.compactgs;
        const CompactGSStreamRole expectedRoles[] = {
            CompactGSStreamRole::POSITION_LOW, CompactGSStreamRole::POSITION_HIGH,
            CompactGSStreamRole::COLOR_RGB, CompactGSStreamRole::SHAPE_VIDEO,
            CompactGSStreamRole::SH_ASTC
        };
        if ((fast.layoutVersion != 2 && fast.layoutVersion != 3) || fast.pointOrder != 1 || fast.pointMapWidth == 0 ||
            fast.pointMapHeight == 0 || fast.pointMapWidth % 4 != 0 || fast.pointMapHeight % 4 != 0 ||
            static_cast<uint32_t>(fast.pointMapWidth) * 4 > CompactGSDescriptor::kMaximumAtlasDimension ||
            static_cast<uint32_t>(fast.pointMapHeight) * 4 > CompactGSDescriptor::kMaximumAtlasDimension ||
            fast.pointBlockSize == 0 || fast.positionBits != 15 || fast.positionLowBits != 7 ||
            fast.positionGroupSize != CompactGSDescriptor::kRequiredPositionGroupSize ||
            ((fast.layoutVersion == 2 && fast.shapeEncoding != 1) || (fast.layoutVersion >= 3 && fast.shapeEncoding != 2)) ||
            fast.streamRoles.size() != 5 || fast.planes.size() != 5) {
            setError("Invalid COMPACTGS_V1 fixed descriptor");
            return false;
        }
        const uint64_t capacity = static_cast<uint64_t>(fast.pointMapWidth) * fast.pointMapHeight;
        const uint32_t groupCount = (metadata.gsPointsNum + fast.positionGroupSize - 1) / fast.positionGroupSize;
        if (capacity < metadata.gsPointsNum || fast.positionGroupMin.size() != static_cast<size_t>(groupCount) * 3 ||
            fast.positionGroupStep.size() != static_cast<size_t>(groupCount) * 3) {
            setError("COMPACTGS_V1 point capacity or group table is inconsistent");
            return false;
        }
        const uint16_t expectedWidths[] = {fast.pointMapWidth, fast.pointMapWidth, fast.pointMapWidth,
                                           static_cast<uint16_t>(fast.pointMapWidth * 4),
                                           static_cast<uint16_t>(fast.pointMapWidth * 4)};
        const uint16_t expectedHeights[] = {fast.pointMapHeight, fast.pointMapHeight, fast.pointMapHeight,
                                            static_cast<uint16_t>(fast.pointMapHeight * (fast.layoutVersion >= 3 ? 4 : 3)),
                                            static_cast<uint16_t>(fast.pointMapHeight * 4)};
        const uint8_t expectedChannels[] = {3, 3, 3, 1, 3};
        const uint8_t expectedRows[] = {1, 1, 1, static_cast<uint8_t>(fast.layoutVersion >= 3 ? 4 : 3), 4};
        const uint8_t expectedCols[] = {1, 1, 1, 4, 4};
        const uint8_t expectedDecodeTypes[] = {0, 0, 0, 2, 1};
        for (size_t i = 0; i < 5; ++i) {
            const auto& plane = fast.planes[i];
            if (fast.streamRoles[i] != expectedRoles[i] || plane.role != expectedRoles[i] ||
                plane.width != expectedWidths[i] || plane.height != expectedHeights[i] ||
                plane.channels != expectedChannels[i] || plane.componentType != CompactGSComponentType::UINT8 ||
                plane.tileRows != expectedRows[i] || plane.tileCols != expectedCols[i] ||
                metadata.subBitstreamDecodeType[i] != expectedDecodeTypes[i]) {
                setError("COMPACTGS_V1 stream role, plane layout, or codec role mismatch at stream " + std::to_string(i));
                return false;
            }
            if (i < 3) {
                const auto entropy = std::static_pointer_cast<EntropyMeta>(metadata.subBitstreamMeta[i]);
                if (metadata.subBitstreamMetaType[i] != SubBitstreamMetaType::ENTROPY ||
                    !entropy || entropy->bitdepth != 8) {
                    setError("COMPACTGS_V1 entropy role has invalid codec metadata"); return false;
                }
            } else if (i == 3) {
                const auto video = std::static_pointer_cast<VideoMeta>(metadata.subBitstreamMeta[i]);
                if (metadata.subBitstreamMetaType[i] != SubBitstreamMetaType::VIDEO || !video ||
                    video->videoPackingInformation.packingMapWidth != plane.width ||
                    video->videoPackingInformation.packingMapHeight != plane.height ||
                    video->videoPackingInformation.packingMapFrameNumMinus1 != 0) {
                    setError("COMPACTGS_V1 shape role has invalid video metadata"); return false;
                }
            } else {
                const auto texture = std::static_pointer_cast<TextureMeta>(metadata.subBitstreamMeta[i]);
                if (metadata.subBitstreamMetaType[i] != SubBitstreamMetaType::TEXTURE || !texture ||
                    texture->texturePackingInformation.packingMapWidth != plane.width ||
                    texture->texturePackingInformation.packingMapHeight != plane.height ||
                    texture->texturePackingInformation.textureChannelNum != plane.channels) {
                    setError("COMPACTGS_V1 SH role has invalid texture metadata"); return false;
                }
            }
        }
        for (float value : fast.positionGroupMin) if (!std::isfinite(value)) {
            setError("COMPACTGS_V1 position group table contains a non-finite minimum"); return false;
        }
        for (float value : fast.positionGroupStep) if (!std::isfinite(value) || value < 0.0f) {
            setError("COMPACTGS_V1 position group table contains an invalid step"); return false;
        }
        for (size_t component = 0; component < 7; ++component) {
            if (!std::isfinite(fast.shapeMin[component]) || !std::isfinite(fast.shapeStep[component]) ||
                fast.shapeStep[component] < 0.0f) {
                setError("COMPACTGS_V1 shape quantization table is invalid"); return false;
            }
        }
        if (fast.layoutVersion >= 3) {
            if (fast.covarianceGroupMin.size() != static_cast<size_t>(groupCount) * 6 ||
                fast.covarianceGroupStep.size() != static_cast<size_t>(groupCount) * 6) {
                setError("COMPACTGS covariance quantization table is inconsistent"); return false;
            }
            for (float value : fast.covarianceGroupMin) if (!std::isfinite(value)) {
                setError("COMPACTGS covariance minimum table contains a non-finite value"); return false;
            }
            for (float value : fast.covarianceGroupStep) if (!std::isfinite(value) || value < 0.0f) {
                setError("COMPACTGS covariance step table contains an invalid value"); return false;
            }
        }
        if (!std::isfinite(fast.encodedMinimumAlpha) || fast.encodedMinimumAlpha < 0.0f ||
            fast.encodedMinimumAlpha > 1.0f || !std::isfinite(fast.importanceMin) ||
            !std::isfinite(fast.importanceStep) || fast.importanceStep < 0.0f) {
            setError("COMPACTGS_V1 scalar metadata is invalid");
            return false;
        }
        for (size_t component = 0; component < 45; ++component) {
            if (!std::isfinite(fast.shMin[component]) || !std::isfinite(fast.shStep[component]) ||
                fast.shStep[component] < 0.0f) {
                setError("COMPACTGS_V1 SH quantization table is invalid"); return false;
            }
        }
        descriptor_.profileIdc = 2;
        descriptor_.pointCount = metadata.gsPointsNum;
        descriptor_.shDegree = static_cast<uint32_t>(metadata.shDegree);
        descriptor_.blockSide = fast.pointBlockSize;
        if (!checkedPointProduct(fast.pointBlockSize, descriptor_.pointsPerBlock)) {
            setError("COMPACTGS_V1 point block size overflows"); return false;
        }
        descriptor_.blockCount = (descriptor_.pointCount + descriptor_.pointsPerBlock - 1) /
                                 descriptor_.pointsPerBlock;
        return true;
    }

    auto [predictionMetadata, blockSide] = metadata.getPredictionMeta();
    if (blockSide <= 0) blockSide = 16;
    uint32_t pointsPerBlock = 0;
    if (!checkedPointProduct(static_cast<uint32_t>(blockSide), pointsPerBlock)) {
        setError("Invalid reconstruction superblock side");
        return false;
    }
    for (const auto& [name, prediction] : predictionMetadata) {
        if (prediction.predictionType != 0 && prediction.predictionType != 1) {
            setError("Unsupported prediction type " + std::to_string(prediction.predictionType) + " for " + name);
            return false;
        }
        if (prediction.predictionType == 1 &&
            (prediction.blocksize <= 0 || blockSide % prediction.blocksize != 0)) {
            setError("Prediction block for " + name + " does not divide the reconstruction superblock");
            return false;
        }
    }

    auto validScanGeometry = [&](uint8_t scanType, uint32_t packingBlockSide,
                                 uint32_t regionWidth, uint32_t regionHeight,
                                 int streamIndex) {
        if (scanType > 1) {
            setError("Unsupported packing scan type for substream " + std::to_string(streamIndex));
            return false;
        }
        if (scanType == 0) return true;
        if (packingBlockSide == 0 || regionWidth % packingBlockSide != 0 ||
            regionHeight % packingBlockSide != 0 ||
            static_cast<uint32_t>(blockSide) % packingBlockSide != 0) {
            setError("Invalid block-scan geometry for substream " + std::to_string(streamIndex));
            return false;
        }
        return true;
    };

    for (int i = 0; i < metadata.subBitstreamNum; ++i) {
        if (metadata.subBitstreamMetaType[i] == SubBitstreamMetaType::ENTROPY) continue;

        if (metadata.subBitstreamMetaType[i] == SubBitstreamMetaType::TEXTURE) {
            const auto texture = std::static_pointer_cast<TextureMeta>(metadata.subBitstreamMeta[i]);
            if (!texture) {
                setError("Missing texture metadata for substream " + std::to_string(i));
                return false;
            }
            const auto& packing = texture->texturePackingInformation;
            const size_t regionCount = static_cast<size_t>(packing.packingRegionCountMinus1) + 1;
            const uint64_t regionCapacity = static_cast<uint64_t>(packing.regionWidth) * packing.regionHeight;
            uint32_t channels = 0;
            if (packing.packingMapWidth == 0 || packing.packingMapHeight == 0 ||
                packing.regionWidth == 0 || packing.regionHeight == 0 ||
                regionCapacity < metadata.gsPointsNum || packing.textureChannelNum == 0 ||
                packing.regionTopLeftX.size() != regionCount ||
                packing.regionTopLeftY.size() != regionCount ||
                !canonicalChannelsForAttributeType(texture->attributeType, channels) ||
                regionCount * packing.textureChannelNum != channels ||
                !validScanGeometry(packing.packingScaningType, packing.packingScaningBlockSize,
                                   packing.regionWidth, packing.regionHeight, i)) {
                if (lastError_.empty()) setError("Invalid texture packing metadata for substream " + std::to_string(i));
                return false;
            }
            for (size_t region = 0; region < regionCount; ++region) {
                const int32_t x = packing.regionTopLeftX[region];
                const int32_t y = packing.regionTopLeftY[region];
                if (x < 0 || y < 0 || static_cast<uint32_t>(x) + packing.regionWidth > packing.packingMapWidth ||
                    static_cast<uint32_t>(y) + packing.regionHeight > packing.packingMapHeight) {
                    setError("Texture region is outside its packing map for substream " + std::to_string(i));
                    return false;
                }
            }
            continue;
        }

        const auto video = std::static_pointer_cast<VideoMeta>(metadata.subBitstreamMeta[i]);
        if (!video) {
            setError("Missing video metadata for substream " + std::to_string(i));
            return false;
        }
        const uint8_t rawCodecId = video->videoDecodeInformation.packingMapVideoCodecId;
        if (rawCodecId != 0 && rawCodecId != CODEC_ID_H264 && rawCodecId != CODEC_ID_H265) {
            setError("Unsupported video codec id for substream " + std::to_string(i));
            return false;
        }
        if (metadata.subBitstreamDecodeType[i] != 2) {
            setError("Video substream " + std::to_string(i) + " does not select the video decoder");
            return false;
        }
        const auto& packing = video->videoPackingInformation;
        const size_t regionCount = static_cast<size_t>(packing.packingRegionCountMinus1) + 1;
        const uint32_t frameCount = static_cast<uint32_t>(packing.packingMapFrameNumMinus1) + 1;
        const uint64_t regionCapacity = static_cast<uint64_t>(packing.regionWidth) * packing.regionHeight;
        if (packing.packingMapWidth == 0 || packing.packingMapHeight == 0 ||
            packing.regionWidth == 0 || packing.regionHeight == 0 ||
            regionCapacity < metadata.gsPointsNum ||
            packing.regionFrameIndex.size() != regionCount ||
            packing.regionTopLeftX.size() != regionCount ||
            packing.regionTopLeftY.size() != regionCount ||
            packing.attributeType.size() != regionCount ||
            packing.attributeChannelOffset.size() != regionCount ||
            packing.attributeChannelNum.size() != regionCount ||
            packing.byteshift.size() != regionCount ||
            !validScanGeometry(packing.packingScaningType, packing.packingScaningBlockSize,
                               packing.regionWidth, packing.regionHeight, i)) {
            if (lastError_.empty()) setError("Invalid video packing metadata for substream " + std::to_string(i));
            return false;
        }
        for (size_t region = 0; region < regionCount; ++region) {
            uint32_t channels = 0;
            const int32_t frame = packing.regionFrameIndex[region];
            const int32_t x = packing.regionTopLeftX[region];
            const int32_t y = packing.regionTopLeftY[region];
            const int32_t channelOffset = packing.attributeChannelOffset[region];
            const int32_t channelCount = packing.attributeChannelNum[region];
            if (!canonicalChannelsForAttributeType(packing.attributeType[region], channels) ||
                frame < 0 || static_cast<uint32_t>(frame) >= frameCount || x < 0 || y < 0 ||
                static_cast<uint32_t>(x) + packing.regionWidth > packing.packingMapWidth ||
                static_cast<uint32_t>(y) + packing.regionHeight > packing.packingMapHeight ||
                channelOffset < 0 || channelCount <= 0 || channelCount > 3 ||
                static_cast<uint32_t>(channelOffset + channelCount) > channels) {
                setError("Invalid video region layout for substream " + std::to_string(i));
                return false;
            }
        }
    }

    descriptor_.pointCount = metadata.gsPointsNum;
    descriptor_.shDegree = static_cast<uint32_t>(metadata.shDegree);
    descriptor_.blockSide = static_cast<uint32_t>(blockSide);
    descriptor_.pointsPerBlock = pointsPerBlock;
    descriptor_.blockCount = (descriptor_.pointCount + pointsPerBlock - 1) / pointsPerBlock;
    descriptor_.profileIdc = 1;
    return true;
}

bool GSDecoder::setupDecoders() {
    const auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    decoders.clear();
    for (int i = 0; i < metadata.subBitstreamNum; ++i) {
        if (metadata.subBitstreamMetaType[i] == SubBitstreamMetaType::VIDEO) continue;
        const int decodeType = metadata.subBitstreamDecodeType[i];
        auto decoder = DecoderFactory::createDecoder(decodeType);
        if (!decoder) return false;
        if (decodeType == 1) {
            auto textureMeta = metadata.getTextureMeta(i);
            auto* textureDecoder = dynamic_cast<TextureDecoder*>(decoder.get());
            if (textureMeta && textureDecoder) {
                textureDecoder->entropyDecodeType = textureMeta->textureDecodeInformation.entropyDecodeType;
            }
        }
        decoders["stream_" + std::to_string(i)] = std::move(decoder);
    }
    return true;
}

bool GSDecoder::captureAstcFeatureRestStream(const std::vector<uint8_t>& encoded,
                                             const TextureMeta& textureMeta,
                                             const QuantMeta& quantMeta,
                                             SplatData& gsData) {
    if (!gsData.astcRawStream.empty() || !gsData.astcMetas.empty()) {
        setError("Multiple ASTC features_rest streams are not supported by the current single-subset protocol");
        return false;
    }

    std::vector<uint8_t> astcData;
    if (!decompressZstdIfNeeded(encoded, astcData) || astcData.size() < 16) {
        setError("Failed to prepare the ASTC features_rest stream");
        return false;
    }
    const uint8_t* header = astcData.data();
    const uint32_t magic = header[0] | (header[1] << 8) | (header[2] << 16) | (header[3] << 24);
    if (magic != 0x5ca1ab13) {
        setError("Invalid ASTC features_rest header");
        return false;
    }

    const uint32_t astcBlockSide = header[4];
    const uint32_t astcWidth = header[7] | (header[8] << 8) | (header[9] << 16);
    const uint32_t astcHeight = header[10] | (header[11] << 8) | (header[12] << 16);
    const uint32_t astcDepth = header[13] | (header[14] << 8) | (header[15] << 16);
    if (!isSupportedSquareAstcBlock(astcBlockSide) || header[5] != header[4] ||
        header[6] != 1 || astcDepth != 1 || astcWidth == 0 || astcHeight == 0) {
        setError("GPU ASTC path requires a square-block 2D features_rest texture");
        return false;
    }

    size_t expectedPayloadBytes = 0;
    if (!checkedCompressedTextureBytes(
            astcWidth, astcHeight, astcBlockSide, astcBlockSide, expectedPayloadBytes) ||
        astcData.size() - 16 != expectedPayloadBytes) {
        setError("ASTC features_rest byte count does not match its header");
        return false;
    }

    SplatData::AstcMeta astcMeta;
    std::vector<uint32_t> astcUvs;
    std::string layoutError;
    const bool generateUvs = descriptor_.profileIdc != 2;
    if (!buildCompressedFeatureLayout(textureMeta, descriptor_.pointCount, astcWidth, astcHeight,
                                      astcBlockSide, expectedPayloadBytes, generateUvs,
                                      astcMeta, astcUvs, layoutError)) {
        setError(layoutError);
        return false;
    }

    astcData.erase(astcData.begin(), astcData.begin() + 16);
    gsData.astcRawStream = std::move(astcData);
    gsData.astcUV = std::move(astcUvs);
    gsData.astcMetas.push_back(astcMeta);
    gsData.astcTextureNum = 1;
    gsData.shnMin = quantMeta.minVals.empty() ? 0.0f : quantMeta.minVals[0];
    gsData.shnMax = quantMeta.maxVals.empty() ? 0.0f : quantMeta.maxVals[0];
    g_timingStats.textureInputBytes += expectedPayloadBytes;
    g_timingStats.textureOutputBytes += expectedPayloadBytes;
    return true;
}

bool GSDecoder::transcodeAstcFeatureRestStream(const std::vector<uint8_t>& encoded,
                                                const TextureMeta& textureMeta,
                                                const QuantMeta& quantMeta,
                                                TextureOutputMode textureOutputMode,
                                                SplatData& gsData) {
#ifdef USE_BC_TEXTURE_ENCODERS
    if (!gsData.astcRawStream.empty() || !gsData.astcMetas.empty()) {
        setError("Multiple compressed features_rest streams are not supported");
        return false;
    }

    std::vector<uint8_t> astcData;
    if (!decompressZstdIfNeeded(encoded, astcData)) {
        setError("Failed to decompress the ASTC features_rest stream before BC transcoding");
        return false;
    }

    BcTranscodeResult transcoded;
    std::string transcodeError;
    const BcTextureFormat format = textureOutputMode == TextureOutputMode::BC7 ?
        BcTextureFormat::BC7 : BcTextureFormat::BC3;
    if (!transcodeAstcToBc(astcData, format, transcoded, transcodeError)) {
        setError("ASTC to BC transcode failed: " + transcodeError);
        return false;
    }

    SplatData::AstcMeta outputMeta;
    std::vector<uint32_t> outputUvs;
    std::string layoutError;
    const bool generateUvs = descriptor_.profileIdc != 2;
    if (!buildCompressedFeatureLayout(
            textureMeta, descriptor_.pointCount, transcoded.width, transcoded.height,
            4, transcoded.data.size(), generateUvs, outputMeta, outputUvs, layoutError)) {
        setError(layoutError);
        return false;
    }

    gsData.astcRawStream = std::move(transcoded.data);
    gsData.astcUV = std::move(outputUvs);
    gsData.astcMetas.push_back(outputMeta);
    gsData.astcTextureNum = 1;
    gsData.shnMin = quantMeta.minVals.empty() ? 0.0f : quantMeta.minVals[0];
    gsData.shnMax = quantMeta.maxVals.empty() ? 0.0f : quantMeta.maxVals[0];

    g_timingStats.astcTextureDecodeMs += transcoded.astcDecodeMs;
    g_timingStats.bcTextureEncodeMs += transcoded.bcEncodeMs;
    g_timingStats.textureInputBytes += astcData.size() - 16;
    g_timingStats.textureOutputBytes += gsData.astcRawStream.size();
    return true;
#else
    (void)encoded;
    (void)textureMeta;
    (void)quantMeta;
    (void)textureOutputMode;
    (void)gsData;
    setError("BC texture output requested, but this build has no BC encoder");
    return false;
#endif
}

bool GSDecoder::decodeNonVideoSubstreams(SplatData& gsData, TextureOutputMode textureOutputMode) {
    const auto& substreams = *substreamUnit.unitPayload.gsbsSubBitstreams;
    const auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    const auto quantMetadata = metadata.getQuantMeta();
    for (int i = 0; i < substreams.subBitstreamNum; ++i) {
        if (metadata.subBitstreamMetaType[i] == SubBitstreamMetaType::VIDEO) continue;

        const std::string streamName = "stream_" + std::to_string(i);
        const auto decoderIt = decoders.find(streamName);
        if (decoderIt == decoders.end()) {
            setError("Missing decoder for substream " + std::to_string(i));
            return false;
        }

        if (textureOutputMode != TextureOutputMode::CPU &&
            metadata.subBitstreamMetaType[i] == SubBitstreamMetaType::TEXTURE) {
            auto textureMeta = std::static_pointer_cast<TextureMeta>(metadata.subBitstreamMeta[i]);
            if (textureMeta && attributeTypeToName(textureMeta->attributeType) == "features_rest") {
                const auto quantIt = quantMetadata.find("features_rest");
                QuantMeta quant = quantIt == quantMetadata.end() ? QuantMeta() : quantIt->second;
                if (metadata.profileIdc == 2 && metadata.compactgs) {
                    quant.minVals.assign(std::begin(metadata.compactgs->shMin), std::end(metadata.compactgs->shMin));
                    quant.maxVals.resize(45);
                    for (size_t component = 0; component < 45; ++component) {
                        quant.maxVals[component] = metadata.compactgs->shMin[component] +
                                                   255.0f * metadata.compactgs->shStep[component];
                    }
                }
                const auto start = Clock::now();
                const bool prepared = textureOutputMode == TextureOutputMode::ASTC ?
                    captureAstcFeatureRestStream(
                        substreams.gstcSubBitstreamData[i], *textureMeta, quant, gsData) :
                    transcodeAstcFeatureRestStream(
                        substreams.gstcSubBitstreamData[i], *textureMeta, quant, textureOutputMode, gsData);
                if (!prepared) {
                    return false;
                }
                g_timingStats.substreamTimings[streamName] = elapsedMs(start);
                LOG_SUMMARY("Decoded substream %d: %zu -> %zu bytes in %.2f ms", i,
                            substreams.gstcSubBitstreamData[i].size(), gsData.astcRawStream.size(),
                            g_timingStats.substreamTimings[streamName]);
                continue;
            }
        }

        std::vector<int> shape;
        int bitDepth = 8;
        if (metadata.subBitstreamMetaType[i] == SubBitstreamMetaType::ENTROPY) {
            const auto entropyMeta = std::static_pointer_cast<EntropyMeta>(metadata.subBitstreamMeta[i]);
            if (entropyMeta) bitDepth = entropyMeta->bitdepth;
        }

        const auto start = Clock::now();
        auto& decoded = decodedStreams[streamName];
        if (!decoderIt->second->decode(substreams.gstcSubBitstreamData[i], decoded, shape, bitDepth)) {
            decodedStreams.erase(streamName);
            setError("Failed to decode substream " + std::to_string(i));
            return false;
        }
        if (!validateDecodedSubstream(i, decoded)) {
            decodedStreams.erase(streamName);
            return false;
        }
        g_timingStats.substreamTimings[streamName] = elapsedMs(start);
        LOG_SUMMARY("Decoded substream %d: %zu -> %zu bytes in %.2f ms", i,
                    substreams.gstcSubBitstreamData[i].size(), decoded.size(),
                    g_timingStats.substreamTimings[streamName]);
    }
    return true;
}

bool GSDecoder::validateDecodedSubstream(int streamIndex, const std::vector<uint8_t>& decoded) {
    if (!metadataUnit.unitPayload.gsbsMetadata) {
        setError("Cannot validate a decoded substream without GSBS metadata");
        return false;
    }
    const auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    if (streamIndex < 0 || streamIndex >= metadata.subBitstreamNum) {
        setError("Decoded substream index is outside the metadata table");
        return false;
    }
    if (metadata.profileIdc == 2 && metadata.compactgs &&
        metadata.subBitstreamMetaType[streamIndex] != SubBitstreamMetaType::VIDEO) {
        const auto& plane = metadata.compactgs->planes[streamIndex];
        size_t pixels = 0;
        size_t expected = 0;
        if (!checkedSizeProduct(plane.width, plane.height, pixels) ||
            !checkedSizeProduct(pixels, plane.channels, expected) || decoded.size() != expected) {
            setError("COMPACTGS_V1 decoded plane length mismatch at stream " + std::to_string(streamIndex));
            return false;
        }
        return true;
    }
    if (metadata.subBitstreamMetaType[streamIndex] == SubBitstreamMetaType::VIDEO) {
        const std::string streamName = "stream_" + std::to_string(streamIndex);
        const auto layoutIt = decodedVideoLayouts_.find(streamName);
        const auto video = std::static_pointer_cast<VideoMeta>(metadata.subBitstreamMeta[streamIndex]);
        if (layoutIt == decodedVideoLayouts_.end() || !video) {
            setError("Decoded video substream " + std::to_string(streamIndex) +
                     " is missing its explicit pixel layout");
            return false;
        }
        const auto& packing = video->videoPackingInformation;
        const uint32_t frameCount = static_cast<uint32_t>(packing.packingMapFrameNumMinus1) + 1;
        const uint32_t codecId = video->videoDecodeInformation.packingMapVideoCodecId;
        const auto& layout = layoutIt->second;
        size_t requiredBytes = 0;
        if (!videoPixelFormatMatchesCodec(codecId, layout.pixelFormat) ||
            layout.width != packing.packingMapWidth || layout.height != packing.packingMapHeight ||
            layout.frameCount != frameCount || !videoFrameByteLength(layout, requiredBytes) ||
            decoded.size() != requiredBytes) {
            setError("Decoded video substream " + std::to_string(streamIndex) +
                     " has an invalid explicit pixel layout");
            return false;
        }
        return true;
    }
    size_t requiredBytes = 0;
    bool requireExactSize = false;
    std::string layoutError;
    if (!expectedDecodedBytes(metadata, streamIndex, requiredBytes, requireExactSize, layoutError) ||
        decoded.size() < requiredBytes || (requireExactSize && decoded.size() != requiredBytes)) {
        setError("Decoded substream " + std::to_string(streamIndex) + " has an invalid byte layout" +
                 (layoutError.empty() ? std::string() : ": " + layoutError));
        return false;
    }
    return true;
}

bool GSDecoder::decodeVideoSubstreamWithFallback(const PendingVideoStreamDescriptor& pending) {
    if (!substreamUnit.unitPayload.gsbsSubBitstreams || !metadataUnit.unitPayload.gsbsMetadata) {
        setError("Cannot run video fallback without staged GSBS data");
        return false;
    }
    const auto& encodedStreams = substreamUnit.unitPayload.gsbsSubBitstreams->gstcSubBitstreamData;
    if (pending.streamIndex >= encodedStreams.size()) {
        setError("Video fallback substream index is outside the payload table");
        return false;
    }

    const std::string streamName = "stream_" + std::to_string(pending.streamIndex);
    auto decoderIt = decoders.find(streamName);
    if (decoderIt == decoders.end()) {
        const int decodeType = metadataUnit.unitPayload.gsbsMetadata->subBitstreamDecodeType[pending.streamIndex];
        auto decoder = DecoderFactory::createDecoder(decodeType);
        auto* videoDecoder = dynamic_cast<VideoDecoder*>(decoder.get());
        if (!videoDecoder) {
            setError("Cannot initialize fallback decoder for video substream " +
                     std::to_string(pending.streamIndex));
            return false;
        }
        videoDecoder->setCodecId(static_cast<int>(pending.codecId));
        decoderIt = decoders.emplace(streamName, std::move(decoder)).first;
    }
    const auto video = std::static_pointer_cast<VideoMeta>(
        metadataUnit.unitPayload.gsbsMetadata->subBitstreamMeta[pending.streamIndex]);
    if (!video) {
        setError("Missing video metadata for fallback substream " + std::to_string(pending.streamIndex));
        return false;
    }

    const auto start = Clock::now();
    auto& decoded = decodedStreams[streamName];
    auto* videoDecoder = dynamic_cast<VideoDecoder*>(decoderIt->second.get());
    if (!videoDecoder) {
        setError("Fallback decoder lost its video interface for substream " +
                 std::to_string(pending.streamIndex));
        return false;
    }
    const std::vector<int> shape = {
        static_cast<int>(pending.frameHeight),
        static_cast<int>(pending.frameWidth),
        static_cast<int>(pending.frameCount)
    };
    VideoFrameLayout layout;
    if (!videoDecoder->decodePlanar(encodedStreams[pending.streamIndex], decoded, shape, 8, layout)) {
        decodedStreams.erase(streamName);
        decodedVideoLayouts_.erase(streamName);
        setError("Failed to decode video substream " + std::to_string(pending.streamIndex) +
                 " with the platform fallback");
        return false;
    }
    decodedVideoLayouts_[streamName] = layout;
    if (!validateDecodedSubstream(static_cast<int>(pending.streamIndex), decoded)) {
        decodedVideoLayouts_.erase(streamName);
        decodedStreams.erase(streamName);
        return false;
    }

    const double durationMs = elapsedMs(start);
    g_timingStats.substreamTimings[streamName] = durationMs;
    LOG_SUMMARY("Decoded video substream %u with fallback (%s): %zu -> %zu bytes in %.2f ms",
                pending.streamIndex, videoPixelFormatName(layout.pixelFormat),
                encodedStreams[pending.streamIndex].size(), decoded.size(), durationMs);
    return true;
}

bool GSDecoder::decodePendingVideosWithFallback() {
    if (!preparing_) {
        setError("Video fallback requires an active staged prepare");
        return false;
    }
    bool usedFallback = false;
    auto fallbackStart = Clock::now();
    for (const auto& pending : pendingVideoStreams_) {
        const std::string streamName = "stream_" + std::to_string(pending.streamIndex);
        const auto decoded = decodedStreams.find(streamName);
        if (decoded != decodedStreams.end() && decodedVideoLayouts_.find(streamName) != decodedVideoLayouts_.end() &&
            validateDecodedSubstream(static_cast<int>(pending.streamIndex), decoded->second)) {
            continue;
        }
        lastError_.clear();
        if (decoded != decodedStreams.end()) decodedStreams.erase(decoded);
        decodedVideoLayouts_.erase(streamName);
        if (!usedFallback) {
            usedFallback = true;
            fallbackStart = Clock::now();
        }
        if (!decodeVideoSubstreamWithFallback(pending)) {
            g_timingStats.decodeVideoFallbackMs += elapsedMs(fallbackStart);
            g_timingStats.decodeSubstreamsMs =
                g_timingStats.rawVideoAdoptMs + g_timingStats.decodeNonVideoSubstreamsMs +
                g_timingStats.decodeVideoFallbackMs;
            return false;
        }
    }
    if (usedFallback) g_timingStats.decodeVideoFallbackMs += elapsedMs(fallbackStart);
    g_timingStats.decodeSubstreamsMs =
        g_timingStats.rawVideoAdoptMs + g_timingStats.decodeNonVideoSubstreamsMs +
        g_timingStats.decodeVideoFallbackMs;
    lastError_.clear();
    return true;
}

bool GSDecoder::buildReconstructionMetadata() {
    const auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    auto [predictionMetadata, blockSide] = metadata.getPredictionMeta();
    const auto quantizationMetadata = metadata.getQuantMeta();
    const auto transformMetadata = metadata.getTransformMeta();

    reconstructionMetadata_ = compactgs::ReconstructionMetadata();
    reconstructionMetadata_.totalPoints = descriptor_.pointCount;
    reconstructionMetadata_.blockSide = descriptor_.blockSide;
    reconstructionMetadata_.pointsPerBlock = descriptor_.pointsPerBlock;
    reconstructionMetadata_.shDegree = descriptor_.shDegree;
    reconstructionMetadata_.decodeFeaturesRest = descriptor_.decodeFeaturesRest;

    std::set<std::string> encodedAttributes = {"means", "opacity", "scaling", "rotation", "features_dc"};
    for (const auto& reconInfo : metadata.reconstructionInformation[0]) {
        const std::string name = attributeTypeToName(reconInfo.attributeType);
        compactgs::ShardAttribute ignored;
        if (!compactgs::attributeFromName(name, ignored)) {
            setError("Unsupported reconstructed GSBS attribute: " + name);
            return false;
        }
        encodedAttributes.insert(name);
    }
    if (!descriptor_.decodeFeaturesRest) encodedAttributes.erase("features_rest");

    const char* orderedNames[] = {
        "means", "opacity", "scaling", "rotation", "features_dc", "features_rest", "importance"
    };
    for (const char* nameValue : orderedNames) {
        const std::string name(nameValue);
        if (encodedAttributes.find(name) == encodedAttributes.end()) continue;
        compactgs::ShardAttribute attributeId;
        if (!compactgs::attributeFromName(name, attributeId)) continue;

        compactgs::ReconstructionAttributeMetadata attribute;
        attribute.attribute = attributeId;
        attribute.name = name;
        attribute.channels = compactgs::attributeChannels(attributeId);
        const auto predictionIt = predictionMetadata.find(name);
        attribute.prediction = predictionIt == predictionMetadata.end() ? PredictionMeta() : predictionIt->second;
        if (attribute.prediction.blocksize <= 0) attribute.prediction.blocksize = static_cast<int>(descriptor_.blockSide);
        const auto quantizationIt = quantizationMetadata.find(name);
        attribute.quantization = quantizationIt == quantizationMetadata.end() ? QuantMeta() : quantizationIt->second;
        const auto transformIt = transformMetadata.transformMap.find(name);
        attribute.transformType = transformIt == transformMetadata.transformMap.end() ? 0 : transformIt->second;
        reconstructionMetadata_.attributes.push_back(std::move(attribute));
    }

    std::string error;
    if (!compactgs::serializeReconstructionMetadata(reconstructionMetadata_, reconstructionMetadataPacket_, error)) {
        setError("Failed to serialize reconstruction metadata: " + error);
        return false;
    }
    return true;
}

const CompactGSDescriptor* GSDecoder::getCompactGSDescriptor() const {
    if (!metadataUnit.unitPayload.gsbsMetadata ||
        metadataUnit.unitPayload.gsbsMetadata->profileIdc != 2) return nullptr;
    return metadataUnit.unitPayload.gsbsMetadata->compactgs.get();
}

const std::vector<uint8_t>* GSDecoder::getCompactGSPlane(CompactGSStreamRole role) const {
    const auto found = compactgsPlanes_.find(role);
    return found == compactgsPlanes_.end() ? nullptr : &found->second;
}

bool GSDecoder::buildCompactGSPlanes() {
    const auto* fast = getCompactGSDescriptor();
    if (!fast) {
        setError("COMPACTGS_V1 plane export requires a parsed descriptor");
        return false;
    }
    compactgsPlanes_.clear();
    for (size_t streamIndex = 0; streamIndex < fast->streamRoles.size(); ++streamIndex) {
        const auto role = fast->streamRoles[streamIndex];
        const std::string streamName = "stream_" + std::to_string(streamIndex);
        const auto decoded = decodedStreams.find(streamName);
        if (role == CompactGSStreamRole::SH_ASTC &&
            descriptor_.textureOutputMode != TextureOutputMode::CPU) {
            if (preparedAuxiliaryData_.astcRawStream.empty()) {
                setError("COMPACTGS_V1 compressed SH output is missing");
                return false;
            }
            continue;
        }
        if (decoded == decodedStreams.end()) {
            setError("COMPACTGS_V1 decoded plane is missing at stream " + std::to_string(streamIndex));
            return false;
        }
        const auto& planeDescriptor = fast->planes[streamIndex];
        size_t pixels = 0;
        size_t expectedBytes = 0;
        if (!checkedSizeProduct(planeDescriptor.width, planeDescriptor.height, pixels) ||
            !checkedSizeProduct(pixels, planeDescriptor.channels, expectedBytes)) {
            setError("COMPACTGS_V1 plane size overflows at stream " + std::to_string(streamIndex));
            return false;
        }
        if (role == CompactGSStreamRole::SHAPE_VIDEO) {
            const auto layout = decodedVideoLayouts_.find(streamName);
            if (layout == decodedVideoLayouts_.end() || layout->second.frameCount != 1 ||
                decoded->second.size() < pixels) {
                setError("COMPACTGS_V1 shape video does not expose a complete luma plane");
                return false;
            }
            // The FFmpeg fallback exposes YUV444 as the legacy interleaved
            // ABI layout (Y,U,V per pixel), while I400/I420/NV12/YUV444P
            // expose a planar luma prefix.  COMPACTGS shape values are stored in
            // the luma channel only; taking the first `pixels` bytes from an
            // interleaved frame would mix U/V into every third shape sample.
            if (layout->second.pixelFormat == VideoPixelFormat::YUV444_INTERLEAVED) {
                const size_t expectedInterleaved = pixels * 3;
                if (decoded->second.size() < expectedInterleaved) {
                    setError("COMPACTGS_V1 interleaved YUV444 shape video is truncated");
                    return false;
                }
                auto& shapePlane = compactgsPlanes_[role];
                shapePlane.resize(pixels);
                for (size_t pixel = 0; pixel < pixels; ++pixel) {
                    shapePlane[pixel] = decoded->second[pixel * 3];
                }
            } else {
                compactgsPlanes_[role].assign(decoded->second.begin(), decoded->second.begin() + pixels);
            }
        } else {
            if (decoded->second.size() != expectedBytes) {
                setError("COMPACTGS_V1 decoded plane has an unexpected byte length");
                return false;
            }
            compactgsPlanes_[role] = decoded->second;
        }
    }

    const uint64_t capacity = static_cast<uint64_t>(fast->pointMapWidth) * fast->pointMapHeight;
    for (auto role : {CompactGSStreamRole::POSITION_LOW, CompactGSStreamRole::POSITION_HIGH,
                      CompactGSStreamRole::COLOR_RGB}) {
        const auto& plane = compactgsPlanes_[role];
        const size_t channels = 3;
        const size_t paddingStart = static_cast<size_t>(descriptor_.pointCount) * channels;
        const size_t expected = static_cast<size_t>(capacity) * channels;
        if (plane.size() != expected ||
            std::any_of(plane.begin() + paddingStart, plane.end(), [](uint8_t value) { return value != 0; })) {
            setError("COMPACTGS_V1 entropy plane has non-zero or malformed tail padding");
            return false;
        }
    }
    return true;
}

bool GSDecoder::packShard(uint32_t startBlock, uint32_t blockCount, std::vector<uint8_t>& packet) {
    const auto packStart = Clock::now();
    packet.clear();
    if (!prepared_) {
        setError("packShard called without a prepared decode");
        return false;
    }
    if (descriptor_.profileIdc == 2) {
        setError("packShard is not available for COMPACTGS_V1");
        return false;
    }
    if (blockCount == 0 || startBlock >= descriptor_.blockCount ||
        blockCount > descriptor_.blockCount - startBlock) {
        setError("packShard received an invalid block range");
        return false;
    }

    const auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    compactgs::PackedShard shard;
    shard.startBlock = startBlock;
    shard.blockCount = blockCount;
    shard.startPoint = startBlock * descriptor_.pointsPerBlock;
    shard.pointCount = std::min(descriptor_.pointCount - shard.startPoint,
                                blockCount * descriptor_.pointsPerBlock);
    shard.blocks.reserve(blockCount);

    for (uint32_t localBlock = 0; localBlock < blockCount; ++localBlock) {
        const uint32_t globalBlock = startBlock + localBlock;
        const uint32_t globalPoint = globalBlock * descriptor_.pointsPerBlock;
        const uint32_t pointCount = std::min(descriptor_.pointsPerBlock, descriptor_.pointCount - globalPoint);
        std::map<std::string, std::vector<int32_t>> quantized;

        for (int streamIndex = 0; streamIndex < metadata.subBitstreamNum; ++streamIndex) {
            const std::string streamName = "stream_" + std::to_string(streamIndex);
            const auto decodedIt = decodedStreams.find(streamName);
            if (decodedIt == decodedStreams.end()) continue;

            std::map<std::string, std::vector<int32_t>> streamQuantized;
            bool unpacked = false;
            const auto unpackStart = Clock::now();
            if (metadata.subBitstreamMetaType[streamIndex] == SubBitstreamMetaType::TEXTURE) {
                const auto texture = std::static_pointer_cast<TextureMeta>(metadata.subBitstreamMeta[streamIndex]);
                unpacked = unpacker.unpackTexture(decodedIt->second, streamQuantized, *texture, streamIndex,
                                                  static_cast<int>(globalBlock),
                                                  static_cast<int>(descriptor_.pointsPerBlock));
            } else if (metadata.subBitstreamMetaType[streamIndex] == SubBitstreamMetaType::VIDEO) {
                const auto video = std::static_pointer_cast<VideoMeta>(metadata.subBitstreamMeta[streamIndex]);
                const auto layout = decodedVideoLayouts_.find(streamName);
                unpacked = layout != decodedVideoLayouts_.end() &&
                    unpacker.unpackVideo(decodedIt->second, streamQuantized, *video, layout->second,
                                         streamIndex, static_cast<int>(globalBlock),
                                         static_cast<int>(descriptor_.pointsPerBlock));
            } else if (metadata.subBitstreamMetaType[streamIndex] == SubBitstreamMetaType::ENTROPY) {
                const auto entropy = std::static_pointer_cast<EntropyMeta>(metadata.subBitstreamMeta[streamIndex]);
                unpacked = unpacker.unpackEntropy(decodedIt->second, streamQuantized, *entropy, streamIndex,
                                                  globalPoint, globalPoint + pointCount);
            }
            g_timingStats.unpackMs += elapsedMs(unpackStart);
            if (!unpacked) {
                setError("Failed to unpack substream " + std::to_string(streamIndex) +
                         " for block " + std::to_string(globalBlock));
                return false;
            }

            const bool allowTrailingPadding =
                metadata.subBitstreamMetaType[streamIndex] != SubBitstreamMetaType::ENTROPY;
            for (auto& [name, values] : streamQuantized) {
                if (!descriptor_.decodeFeaturesRest && name == "features_rest") continue;
                compactgs::ShardAttribute attribute;
                if (!compactgs::attributeFromName(name, attribute)) {
                    setError("Cannot pack unsupported attribute: " + name);
                    return false;
                }
                std::string layoutError;
                if (!normalizeUnpackedAttribute(values, compactgs::attributeChannels(attribute), pointCount,
                                                globalBlock, globalPoint, descriptor_,
                                                allowTrailingPadding, layoutError)) {
                    setError("Invalid " + name + " layout from substream " +
                             std::to_string(streamIndex) + " in block " +
                             std::to_string(globalBlock) + ": " + layoutError);
                    return false;
                }

                auto destination = quantized.find(name);
                if (destination == quantized.end()) {
                    quantized.emplace(name, std::move(values));
                    continue;
                }
                if (destination->second.size() != values.size()) {
                    setError("Substream attribute length mismatch for " + name + " in block " +
                             std::to_string(globalBlock));
                    return false;
                }
                for (size_t valueIndex = 0; valueIndex < values.size(); ++valueIndex) {
                    destination->second[valueIndex] += values[valueIndex];
                }
            }
        }
        if (!descriptor_.decodeFeaturesRest) quantized.erase("features_rest");

        compactgs::QuantizedBlock block;
        block.globalBlock = globalBlock;
        block.pointCount = pointCount;
        for (auto& [name, values] : quantized) {
            compactgs::ShardAttribute attribute;
            if (!compactgs::attributeFromName(name, attribute)) {
                setError("Cannot pack unsupported attribute: " + name);
                return false;
            }
            const uint32_t channels = compactgs::attributeChannels(attribute);
            size_t expectedValues = 0;
            if (!checkedSizeProduct(pointCount, channels, expectedValues) || values.size() != expectedValues) {
                setError("Unpacked attribute length mismatch for " + name + " in block " +
                         std::to_string(globalBlock));
                return false;
            }
            block.attributes.push_back({attribute, std::move(values)});
        }
        shard.blocks.push_back(std::move(block));
    }

    std::string error;
    if (!compactgs::serializePackedShard(shard, packet, error)) {
        setError("Failed to serialize shard: " + error);
        return false;
    }
    g_timingStats.packShardMs += elapsedMs(packStart);
    return true;
}

bool GSDecoder::pruneInvalidGaussians(SplatData& gsData) {
    if (gsData.opacity.empty()) return true;
    std::vector<size_t> validIndices;
    validIndices.reserve(gsData.opacity.size());
    for (size_t point = 0; point < gsData.opacity.size(); ++point) {
        if (gsData.opacity[point] > -9.0f) validIndices.push_back(point);
    }
    if (validIndices.size() == gsData.opacity.size() || validIndices.empty()) return true;

    const size_t oldPointCount = gsData.opacity.size();
    const size_t featuresRestChannels = gsData.features_rest.empty() ? 0 : gsData.features_rest.size() / oldPointCount;
    auto compactFloat = [&](std::vector<float>& values, size_t channels) {
        std::vector<float> compacted(validIndices.size() * channels);
        for (size_t destination = 0; destination < validIndices.size(); ++destination) {
            const size_t source = validIndices[destination];
            std::copy_n(values.begin() + source * channels, channels, compacted.begin() + destination * channels);
        }
        values = std::move(compacted);
    };
    compactFloat(gsData.means, 3);
    compactFloat(gsData.opacity, 1);
    compactFloat(gsData.scaling, 3);
    compactFloat(gsData.rotation, 4);
    compactFloat(gsData.features_dc, 3);
    if (featuresRestChannels > 0) compactFloat(gsData.features_rest, featuresRestChannels);
    if (!gsData.astcUV.empty()) {
        std::vector<uint32_t> compacted(validIndices.size() * 2);
        for (size_t destination = 0; destination < validIndices.size(); ++destination) {
            const size_t source = validIndices[destination];
            compacted[destination * 2] = gsData.astcUV[source * 2];
            compacted[destination * 2 + 1] = gsData.astcUV[source * 2 + 1];
        }
        gsData.astcUV = std::move(compacted);
    }
    gsData.numPoints = static_cast<int>(validIndices.size());
    return true;
}

void GSDecoder::releasePreparedDecode() {
    preparing_ = false;
    prepared_ = false;
    decoders.clear();
    decodedStreams.clear();
    decodedVideoLayouts_.clear();
    reconstructionMetadataPacket_.clear();
    reconstructionMetadata_ = compactgs::ReconstructionMetadata();
    descriptor_ = DecodeDescriptor();
    preparedAuxiliaryData_.clear();
    pendingVideoStreams_.clear();
    compactgsPlanes_.clear();
    metadataUnit = Unit(0);
    substreamUnit = Unit(1);
}

bool GSDecoder::savePLY(const std::string& plyFile, const SplatData& gsData) {
    return gsData.savePLY(plyFile);
}
