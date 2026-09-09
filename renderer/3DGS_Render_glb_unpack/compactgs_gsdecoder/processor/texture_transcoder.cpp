#include "texture_transcoder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <mutex>
#include <new>

#include "astcenc.h"
#include "bc7enc.h"
#include "rgbcx.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kMaxTextureDimension = 16384;
constexpr uint64_t kMaxRgbaBytes = 512ull * 1024 * 1024;

double elapsedMs(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

uint32_t readU24(const uint8_t* value) {
    return static_cast<uint32_t>(value[0]) |
           (static_cast<uint32_t>(value[1]) << 8) |
           (static_cast<uint32_t>(value[2]) << 16);
}

bool checkedPixelBytes(uint32_t width, uint32_t height, size_t& byteCount) {
    const uint64_t bytes = static_cast<uint64_t>(width) * height * 4;
    if (bytes == 0 || bytes > kMaxRgbaBytes || bytes > std::numeric_limits<size_t>::max()) return false;
    byteCount = static_cast<size_t>(bytes);
    return true;
}

bool checkedBlockBytes(uint32_t width,
                       uint32_t height,
                       uint32_t blockWidth,
                       uint32_t blockHeight,
                       size_t& byteCount) {
    if (blockWidth == 0 || blockHeight == 0) return false;
    const uint64_t blocksX = (static_cast<uint64_t>(width) + blockWidth - 1) / blockWidth;
    const uint64_t blocksY = (static_cast<uint64_t>(height) + blockHeight - 1) / blockHeight;
    const uint64_t bytes = blocksX * blocksY * 16;
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max()) return false;
    byteCount = static_cast<size_t>(bytes);
    return true;
}

void initializeBcEncoders() {
    static std::once_flag once;
    std::call_once(once, []() {
        bc7enc_compress_block_init();
        rgbcx::init();
    });
}

void gatherBlock(const std::vector<uint8_t>& rgba,
                 uint32_t width,
                 uint32_t height,
                 uint32_t blockX,
                 uint32_t blockY,
                 std::array<uint8_t, 64>& block) {
    for (uint32_t y = 0; y < 4; ++y) {
        const uint32_t sourceY = std::min(blockY * 4 + y, height - 1);
        for (uint32_t x = 0; x < 4; ++x) {
            const uint32_t sourceX = std::min(blockX * 4 + x, width - 1);
            const size_t source = (static_cast<size_t>(sourceY) * width + sourceX) * 4;
            const size_t destination = (y * 4 + x) * 4;
            std::copy_n(rgba.data() + source, 4, block.data() + destination);
        }
    }
}

} // namespace

bool transcodeAstcToBc(const std::vector<uint8_t>& astcFile,
                       BcTextureFormat format,
                       BcTranscodeResult& result,
                       std::string& error) {
    result = BcTranscodeResult();
    error.clear();
    if (format != BcTextureFormat::BC7 && format != BcTextureFormat::BC3) {
        error = "Unsupported BC texture format";
        return false;
    }
    if (astcFile.size() < 16) {
        error = "ASTC texture is smaller than its header";
        return false;
    }

    const uint8_t* header = astcFile.data();
    const uint32_t magic = static_cast<uint32_t>(header[0]) |
                           (static_cast<uint32_t>(header[1]) << 8) |
                           (static_cast<uint32_t>(header[2]) << 16) |
                           (static_cast<uint32_t>(header[3]) << 24);
    if (magic != 0x5ca1ab13) {
        error = "ASTC texture has an invalid magic number";
        return false;
    }

    const uint32_t blockWidth = header[4];
    const uint32_t blockHeight = header[5];
    const uint32_t blockDepth = header[6];
    const uint32_t width = readU24(header + 7);
    const uint32_t height = readU24(header + 10);
    const uint32_t depth = readU24(header + 13);
    if (blockWidth == 0 || blockHeight == 0 || blockDepth != 1 || width == 0 || height == 0 || depth != 1) {
        error = "Only non-empty 2D ASTC textures can be transcoded to BC";
        return false;
    }
    if (width > kMaxTextureDimension || height > kMaxTextureDimension) {
        error = "ASTC texture dimensions exceed the WebGL texture limit";
        return false;
    }

    size_t astcPayloadBytes = 0;
    if (!checkedBlockBytes(width, height, blockWidth, blockHeight, astcPayloadBytes) ||
        astcPayloadBytes > std::numeric_limits<size_t>::max() - 16 ||
        astcFile.size() != 16 + astcPayloadBytes) {
        error = "ASTC texture byte count does not match its header";
        return false;
    }

    const auto decodeStart = Clock::now();
    astcenc_config config;
    astcenc_error status = astcenc_config_init(
        ASTCENC_PRF_LDR, blockWidth, blockHeight, blockDepth,
        ASTCENC_PRE_FASTEST, ASTCENC_FLG_DECOMPRESS_ONLY, &config);
    if (status != ASTCENC_SUCCESS) {
        error = "ASTC decoder rejected the texture block dimensions";
        return false;
    }

    size_t rgbaBytes = 0;
    if (!checkedPixelBytes(width, height, rgbaBytes)) {
        error = "ASTC texture exceeds the CPU transcode memory budget";
        return false;
    }
    std::vector<uint8_t> rgba;
    try {
        rgba.resize(rgbaBytes);
    } catch (const std::bad_alloc&) {
        error = "ASTC RGBA decode buffer allocation failed";
        return false;
    }

    astcenc_context* context = nullptr;
    status = astcenc_context_alloc(&config, 1, &context);
    if (status != ASTCENC_SUCCESS || !context) {
        error = "ASTC decoder context allocation failed";
        return false;
    }

    void* slices[] = {rgba.data()};
    astcenc_image image{width, height, 1, ASTCENC_TYPE_U8, slices};
    const astcenc_swizzle swizzle = {
        ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A
    };
    status = astcenc_decompress_image(
        context, astcFile.data() + 16, astcFile.size() - 16, &image, &swizzle, 0);
    astcenc_context_free(context);
    if (status != ASTCENC_SUCCESS) {
        error = "ASTC texture decode failed";
        return false;
    }
    result.astcDecodeMs = elapsedMs(decodeStart);

    const uint32_t blocksX = (width + 3) / 4;
    const uint32_t blocksY = (height + 3) / 4;
    size_t outputBytes = 0;
    if (!checkedBlockBytes(width, height, 4, 4, outputBytes)) {
        error = "BC texture dimensions exceed addressable memory";
        return false;
    }
    try {
        result.data.resize(outputBytes);
    } catch (const std::bad_alloc&) {
        error = "BC output buffer allocation failed";
        return false;
    }

    initializeBcEncoders();
    bc7enc_compress_block_params bc7Params;
    bc7enc_compress_block_params_init(&bc7Params);
    bc7enc_compress_block_params_init_linear_weights(&bc7Params);
    bc7Params.m_mode_mask = 1u << 6;
    bc7Params.m_max_partitions = 0;
    bc7Params.m_uber_level = 0;
    bc7Params.m_force_alpha = false;

    const auto encodeStart = Clock::now();
    std::array<uint8_t, 64> block{};
    for (uint32_t blockY = 0; blockY < blocksY; ++blockY) {
        for (uint32_t blockX = 0; blockX < blocksX; ++blockX) {
            gatherBlock(rgba, width, height, blockX, blockY, block);
            uint8_t* destination = result.data.data() +
                (static_cast<size_t>(blockY) * blocksX + blockX) * 16;
            if (format == BcTextureFormat::BC7) {
                bc7enc_compress_block(destination, block.data(), &bc7Params);
            } else {
                rgbcx::encode_bc3(0, destination, block.data());
            }
        }
    }
    result.bcEncodeMs = elapsedMs(encodeStart);
    result.width = width;
    result.height = height;
    result.sourceBlockWidth = blockWidth;
    result.sourceBlockHeight = blockHeight;
    return true;
}
