#include "unpacker.h"
#include <iostream>
#include <cmath>

// Default channel counts.
constexpr int CHANNELS_MEANS = 3;       // means: x, y, z
constexpr int CHANNELS_OPACITY = 1;     // Opacity: one channel.
constexpr int CHANNELS_SCALING = 3;     // scaling: x, y, z
constexpr int CHANNELS_ROTATION = 4;    // rotation: w, x, y, z
constexpr int CHANNELS_FEATURES_DC = 3; // features_dc: r, g, b
constexpr int CHANNELS_IMPORTANCE = 1;  // Importance: one channel.

Unpacker::Unpacker() {
}

Unpacker::~Unpacker() {
}

std::string Unpacker::attributeTypeToName(int attrType) {
    switch (attrType) {
        case 0: return "means";
        case 1: return "opacity";
        case 2: return "scaling";
        case 3: return "rotation";
        case 4: return "features_dc";
        case 20: return "features_rest";
        case 21: return "importance";
        default:
            if (attrType >= 5 && attrType <= 19) {
                return "features_rest";
            }
            return "unknown";
    }
}

void Unpacker::extractBlockScan(
    const std::vector<uint8_t>& srcData,
    int32_t* dstData,
    int regionX, int regionY, int regionW, int regionH,
    int mapWidth, int channelNum, int blockSize,
    int totalChannels, int channelOffset, int srcChannelNum,
    int byteshift,
    int blockIdx, int chunkblocksize) {

    int blocksPerRow = regionW / blockSize;
    int pointsPerBlock = blockSize * blockSize;

    // A negative blockIdx processes all blocks for compatibility with the legacy interface.
    bool processAll = (blockIdx < 0);

    // Precompute values used for byte shifting.
    int shift = (byteshift < 0) ? 0 : byteshift;

    // Determine the block range to process.
    int startBlock = processAll ? 0 : blockIdx * (chunkblocksize / pointsPerBlock);
    int endBlock = processAll ? (blocksPerRow * (regionH / blockSize)) : std::min((startBlock + (chunkblocksize / pointsPerBlock)), (blocksPerRow * (regionH / blockSize)));

    for (int b = startBlock; b < endBlock; b++) {
        int blockY = b / blocksPerRow;
        int blockX = b % blocksPerRow;

        // Iterate over every pixel in the block.
        for (int by = 0; by < blockSize; by++) {
            for (int bx = 0; bx < blockSize; bx++) {
                int y = blockY * blockSize + by;
                int x = blockX * blockSize + bx;

                // Calculate the row-major source position in the packing map.
                int srcY = regionY + y;
                int srcX = regionX + x;
                int srcIdx = (srcY * mapWidth + srcX) * srcChannelNum;

                // Calculate the contiguous zero-based destination position.
                int dstIdx = (b - startBlock) * pointsPerBlock + (by * blockSize + bx);

                // Apply byte shifting and type conversion while writing directly to the destination.
                for (int c = 0; c < channelNum; c++) {
                    int32_t value = static_cast<int32_t>(srcData[srcIdx + c]);
                    // Apply the byte shift.
                    value = value << shift;
                    // Accumulate into the destination element.
                    dstData[dstIdx * totalChannels + channelOffset + c] += value;
                }
            }
        }
    }
}

void Unpacker::extractRowFirstScan(
    const std::vector<uint8_t>& srcData,
    int32_t* dstData,
    int regionX, int regionY, int regionW, int regionH,
    int mapWidth, int channelNum,
    int totalChannels, int channelOffset, int srcChannelNum,
    int byteshift,
    int idx, int chunkblocksize) {

    // A negative row index processes all rows for compatibility with the legacy interface.
    bool processAll = (idx < 0);

    // Precompute values used for byte shifting.
    int shiftLimit = (byteshift < 0) ? ((1 << (-byteshift)) - 1) : 0;

    // Determine the range to process.
    int startIdx = processAll ? 0 : idx * chunkblocksize;
    int endIdx = processAll ? regionH * regionW : std::min(regionH * regionW, idx * chunkblocksize + chunkblocksize);

    for (int i = startIdx; i < endIdx; i++) {
        int y = i / regionW;
        int x = i % regionW;

        int srcIdx = ((regionY + y) * mapWidth + (regionX + x)) * srcChannelNum;
        // Calculate the contiguous zero-based destination position.
        int dstIdx = i - startIdx;

        // Apply byte shifting and type conversion while writing directly to the destination.
        for (int c = 0; c < channelNum; c++) {
            if (srcIdx + c < static_cast<int>(srcData.size())) {
                int32_t value = static_cast<int32_t>(srcData[srcIdx + c]);

                // Apply the byte shift.
                if (byteshift >= 0) {
                    value = value << byteshift;
                } else {
                    value = std::min(value, shiftLimit);
                }

                // Accumulate into the destination element.
                dstData[dstIdx * totalChannels + channelOffset + c] += value;
            }
        }
    }
}

bool Unpacker::unpackTexture(
    const std::vector<uint8_t>& decodedData,
    std::map<std::string, std::vector<int32_t>>& unpackedAttrs,
    const TextureMeta& textureMeta,
    int streamIndex,
    int blockIdx, int chunkblocksize) {

    const auto& packingInfo = textureMeta.texturePackingInformation;

    // Read the texture stream attribute type.
    int attrType = textureMeta.attributeType;
    std::string attrName = attributeTypeToName(attrType);
    int byteshift = packingInfo.byteshift;

    int channelNum = packingInfo.textureChannelNum;

    // Extract data from each region.
    int regionCount = packingInfo.packingRegionCountMinus1 + 1;

    // Precompute the total channel count.
    int totalChannels = regionCount * channelNum;

    // Calculate the block size.
    int blockSize = packingInfo.packingScaningBlockSize;
    int pointsPerBlock = blockSize * blockSize;

    // Size the buffer according to blockIdx.
    int numPixels;
    if (blockIdx >= 0) {
        // Process a single block.
        numPixels = std::min(packingInfo.regionWidth * packingInfo.regionHeight - blockIdx * chunkblocksize, chunkblocksize);
    } else {
        // In full-model mode, process every pixel.
        numPixels = packingInfo.regionWidth * packingInfo.regionHeight;
    }

    if (unpackedAttrs.find(attrName) == unpackedAttrs.end()) {
        unpackedAttrs[attrName] = std::vector<int32_t>(numPixels * totalChannels, 0);
    }

    for (int r = 0; r < regionCount; r++) {
        int regionX = packingInfo.regionTopLeftX[r];
        int regionY = packingInfo.regionTopLeftY[r];
        int regionW = packingInfo.regionWidth;
        int regionH = packingInfo.regionHeight;

        // Calculate channelOffset; for texture packing it is regionIndex * textureChannelNum.
        int channelOffset = r * channelNum;

        // Read the source channel count from the metadata.
        int srcChannelNum = textureMeta.texturePackingInformation.textureChannelNum;

        // Write directly to the destination buffer without a temporary buffer.
        int32_t* dstData = unpackedAttrs[attrName].data();

        // Select the extraction routine for the configured scan type.
        if (packingInfo.packingScaningType == 1) {
            // Use block-order scanning.
            extractBlockScan(decodedData, dstData, regionX, regionY, regionW, regionH,
                           packingInfo.packingMapWidth, channelNum, packingInfo.packingScaningBlockSize,
                           totalChannels, channelOffset, srcChannelNum, byteshift, blockIdx, chunkblocksize);
        }
        else {
            // Handle other scan types, such as row-first or Morton order.
            extractRowFirstScan(decodedData, dstData, regionX, regionY, regionW, regionH,
                              packingInfo.packingMapWidth, channelNum,
                              totalChannels, channelOffset, srcChannelNum, byteshift, blockIdx, chunkblocksize);
        }
    }

    return true;
}

bool Unpacker::unpackVideo(
    const std::vector<uint8_t>& decodedData,
    std::map<std::string, std::vector<int32_t>>& unpackedAttrs,
    const VideoMeta& videoMeta,
    const VideoFrameLayout& layout,
    int streamIndex,
    int blockIdx, int chunkblocksize) {

    (void)streamIndex;
    const auto& packingInfo = videoMeta.videoPackingInformation;
    int frameCount = packingInfo.packingMapFrameNumMinus1 + 1;
    int regionCount = packingInfo.packingRegionCountMinus1 + 1;
    int frameHeight = packingInfo.packingMapHeight;
    int blockSize = packingInfo.packingScaningBlockSize;

    size_t expectedBytes = 0;
    if (layout.width != packingInfo.packingMapWidth ||
        layout.height != static_cast<uint32_t>(frameHeight) ||
        layout.frameCount != static_cast<uint32_t>(frameCount) ||
        !videoFrameByteLength(layout, expectedBytes) || decodedData.size() != expectedBytes) {
        std::cerr << "      Error: Invalid explicit video layout" << std::endl;
        return false;
    }

    const size_t width = layout.width;
    const size_t height = layout.height;
    const size_t lumaBytes = width * height;
    const size_t chromaBytes = lumaBytes / 4;
    const size_t frameStride = expectedBytes / layout.frameCount;
    auto sampleChannel = [&](int frame, int x, int y, int channel) -> uint8_t {
        const size_t frameBase = static_cast<size_t>(frame) * frameStride;
        const size_t pixel = static_cast<size_t>(y) * width + x;
        if (layout.pixelFormat == VideoPixelFormat::YUV444_INTERLEAVED) {
            return decodedData[frameBase + pixel * 3 + channel];
        }
        if (layout.pixelFormat == VideoPixelFormat::YUV444P) {
            return decodedData[frameBase + static_cast<size_t>(channel) * lumaBytes + pixel];
        }
        if (channel == 0) return decodedData[frameBase + pixel];
        if (layout.pixelFormat == VideoPixelFormat::I400) return 128;

        const size_t chroma = (static_cast<size_t>(y) / 2) * (width / 2) +
            static_cast<size_t>(x) / 2;
        if (layout.pixelFormat == VideoPixelFormat::I420) {
            return decodedData[frameBase + lumaBytes + (channel - 1) * chromaBytes + chroma];
        }
        return decodedData[frameBase + lumaBytes + chroma * 2 + channel - 1];
    };

    for (int r = 0; r < regionCount; r++) {
        int frameIdx = packingInfo.regionFrameIndex[r];
        int regionX = packingInfo.regionTopLeftX[r];
        int regionY = packingInfo.regionTopLeftY[r];
        int regionW = packingInfo.regionWidth;
        int regionH = packingInfo.regionHeight;
        int attrType = packingInfo.attributeType[r];
        int channelOffset = packingInfo.attributeChannelOffset[r];
        int channelNum = packingInfo.attributeChannelNum[r];
        int byteshift = packingInfo.byteshift[r];

        std::string attrName = attributeTypeToName(attrType);

        if (frameIdx < 0 || frameIdx >= frameCount || regionX < 0 || regionY < 0 ||
            regionW <= 0 || regionH <= 0 || channelNum <= 0 || channelNum > 3 ||
            regionX + regionW > static_cast<int>(layout.width) ||
            regionY + regionH > static_cast<int>(layout.height)) {
            std::cerr << "      Error: Video region is outside its explicit frame layout" << std::endl;
            return false;
        }

        // Determine the pixel count from blockIdx.
        int numPixels;
        if (blockIdx >= 0) {
            numPixels = std::min(regionW * regionH - blockIdx * chunkblocksize, chunkblocksize);
        } else {
            numPixels = regionW * regionH;
        }

        // Determine the attribute's total channel count.
        int totalChannels = 0;

        if (attrName == "means") {
            totalChannels = CHANNELS_MEANS;
        }
        else if (attrName == "opacity") {
            totalChannels = CHANNELS_OPACITY;
        }
        else if (attrName == "scaling") {
            totalChannels = CHANNELS_SCALING;
        }
        else if (attrName == "rotation") {
            totalChannels = CHANNELS_ROTATION;
        }
        else if (attrName == "features_dc") {
            totalChannels = CHANNELS_FEATURES_DC;
        }
        else if (attrName == "features_rest") {
            totalChannels = channelOffset + channelNum;
        }
        else if (attrName == "importance") {
            totalChannels = CHANNELS_IMPORTANCE;
        }
        else {
            totalChannels = channelOffset + channelNum;
        }

        if (numPixels <= 0 || totalChannels <= 0 || channelOffset < 0 ||
            channelOffset + channelNum > totalChannels) {
            return false;
        }
        const size_t requiredValues = static_cast<size_t>(numPixels) * totalChannels;

        if (unpackedAttrs.find(attrName) == unpackedAttrs.end()) {
            unpackedAttrs[attrName] = std::vector<int32_t>(requiredValues, 0);
        }
        else {
            if (unpackedAttrs[attrName].size() < requiredValues) {
                unpackedAttrs[attrName].resize(requiredValues, 0);
            }
        }

        int32_t* dstData = unpackedAttrs[attrName].data();

        if (packingInfo.packingScaningType == 1) {
            if (blockSize <= 0 || regionW % blockSize != 0 || regionH % blockSize != 0) return false;
            const int blocksPerRow = regionW / blockSize;
            const int pointsPerBlock = blockSize * blockSize;
            const int totalBlocks = blocksPerRow * (regionH / blockSize);
            const bool processAll = blockIdx < 0;
            const int startBlock = processAll ? 0 : blockIdx * (chunkblocksize / pointsPerBlock);
            const int endBlock = processAll ? totalBlocks :
                std::min(startBlock + chunkblocksize / pointsPerBlock, totalBlocks);
            const int shift = byteshift < 0 ? 0 : byteshift;
            for (int block = startBlock; block < endBlock; ++block) {
                const int blockY = block / blocksPerRow;
                const int blockX = block % blocksPerRow;
                for (int y = 0; y < blockSize; ++y) {
                    for (int x = 0; x < blockSize; ++x) {
                        const int destinationPixel = (block - startBlock) * pointsPerBlock + y * blockSize + x;
                        for (int channel = 0; channel < channelNum; ++channel) {
                            const int32_t value = static_cast<int32_t>(sampleChannel(
                                frameIdx, regionX + blockX * blockSize + x,
                                regionY + blockY * blockSize + y, channel));
                            dstData[destinationPixel * totalChannels + channelOffset + channel] += value << shift;
                        }
                    }
                }
            }
        } else {
            const bool processAll = blockIdx < 0;
            const int startPixel = processAll ? 0 : blockIdx * chunkblocksize;
            const int endPixel = processAll ? regionW * regionH :
                std::min(regionW * regionH, startPixel + chunkblocksize);
            const int shiftLimit = byteshift < 0 ? (1 << (-byteshift)) - 1 : 0;
            for (int pixel = startPixel; pixel < endPixel; ++pixel) {
                const int y = pixel / regionW;
                const int x = pixel % regionW;
                const int destinationPixel = pixel - startPixel;
                for (int channel = 0; channel < channelNum; ++channel) {
                    int32_t value = static_cast<int32_t>(sampleChannel(
                        frameIdx, regionX + x, regionY + y, channel));
                    value = byteshift >= 0 ? value << byteshift : std::min(value, shiftLimit);
                    dstData[destinationPixel * totalChannels + channelOffset + channel] += value;
                }
            }
        }
    }

    return true;
}

bool Unpacker::unpackEntropy(
    const std::vector<uint8_t>& decodedData,
    std::map<std::string, std::vector<int32_t>>& unpackedAttrs,
    const EntropyMeta& entropyMeta,
    int streamIndex,
    size_t startIdx,
    size_t endIdx) {

    // Read the entropy stream attribute type.
    int attrType = entropyMeta.attributeType;
    std::string attrName = attributeTypeToName(attrType);
    int bitdepth = entropyMeta.bitdepth;
    int byteshift = entropyMeta.byteshift;

    // Determine the channel count.
    int numChannels = 1;
    if (attrName == "means") {
        numChannels = 3;
    } else if (attrName == "scaling") {
        numChannels = 3;
    } else if (attrName == "rotation") {
        numChannels = 4;
    } else if (attrName == "features_dc") {
        numChannels = 3;
    } else if (attrName == "features_rest") {
        numChannels = 45;  // 15 * 3
    } else if (attrName == "opacity") {
        numChannels = 1;
    } else if (attrName == "importance") {
        numChannels = 1;
    }

    // Calculate the starting and ending value indices.
    size_t startValueIdx = startIdx * numChannels;
    size_t endValueIdx = endIdx * numChannels;

    // Process only the requested range when one is specified.
    if (endIdx == 0) {
        endValueIdx = (bitdepth == 16) ? decodedData.size() / 2 : decodedData.size();
    }

    size_t numValuesToProcess = endValueIdx - startValueIdx;

    // Initialize the attribute buffer.
    if (unpackedAttrs.find(attrName) == unpackedAttrs.end()) {
        unpackedAttrs[attrName] = std::vector<int32_t>(numValuesToProcess, 0);
    }

    // Process the requested range directly to avoid a temporary buffer.
    if (bitdepth == 16) {
        // 16-bit data: combine each pair of bytes as a big-endian value.
        for (size_t i = 0; i < numValuesToProcess; i++) {
            size_t srcIdx = (startValueIdx + i) * 2;
            if (srcIdx + 1 < decodedData.size()) {
                int32_t value = static_cast<int32_t>(decodedData[srcIdx] << 8) |
                               (static_cast<int32_t>(decodedData[srcIdx + 1]));

                if (byteshift >= 0) {
                    value <<= byteshift;
                } else {
                    int limit = (1 << (-byteshift)) - 1;
                    value = std::min(value, limit);
                }

                unpackedAttrs[attrName][i] += value;
            }
        }
    }
    else if (bitdepth == 8) {
        // 8-bit data: use each byte directly.
        for (size_t i = 0; i < numValuesToProcess; i++) {
            size_t srcIdx = startValueIdx + i;
            if (srcIdx < decodedData.size()) {
                int32_t value = static_cast<int32_t>(decodedData[srcIdx]);

                if (byteshift >= 0) {
                    value <<= byteshift;
                } else {
                    int limit = (1 << (-byteshift)) - 1;
                    value = std::min(value, limit);
                }

                unpackedAttrs[attrName][i] += value;
            }
        }
    }
    else {
        std::cerr << "      Error: Unsupported bitdepth " << bitdepth << std::endl;
        return false;
    }

    return true;
}
