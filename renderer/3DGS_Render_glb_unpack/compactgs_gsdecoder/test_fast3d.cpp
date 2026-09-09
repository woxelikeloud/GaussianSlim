#include "processor/stream.h"
#include "gaussian_model/gs_decoder.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include <zstd.h>

namespace {

[[noreturn]] void fail(const char* message) {
    std::cerr << "COMPACTGS_V1 test failure: " << message << std::endl;
    std::exit(EXIT_FAILURE);
}

void require(bool condition, const char* message) {
    if (!condition) {
        fail(message);
    }
}

std::vector<uint8_t> compressZstd(const std::vector<uint8_t>& input) {
    std::vector<uint8_t> output(ZSTD_compressBound(input.size()));
    const size_t encodedSize = ZSTD_compress(
        output.data(), output.size(), input.data(), input.size(), 1);
    require(!ZSTD_isError(encodedSize), "fixture Zstd compression failed");
    output.resize(encodedSize);
    return output;
}

std::vector<uint8_t> makeAstcFixture() {
    constexpr uint32_t width = 32;
    constexpr uint32_t height = 16;
    constexpr uint32_t blockSide = 4;
    const size_t payloadSize = (width / blockSide) * (height / blockSide) * 16;
    std::vector<uint8_t> astc(16 + payloadSize, 0);
    astc[0] = 0x13;
    astc[1] = 0xab;
    astc[2] = 0xa1;
    astc[3] = 0x5c;
    astc[4] = blockSide;
    astc[5] = blockSide;
    astc[6] = 1;
    astc[7] = static_cast<uint8_t>(width);
    astc[10] = static_cast<uint8_t>(height);
    astc[13] = 1;
    return astc;
}

std::vector<uint8_t> makePaddedPointPlane(uint8_t value) {
    std::vector<uint8_t> plane(8 * 4 * 3, 0);
    std::fill(plane.begin(), plane.begin() + 17 * 3, value);
    return plane;
}

CompactGSDescriptor makeGroupTableDescriptor(size_t groupCount) {
    CompactGSDescriptor descriptor;
    // This fixture exercises only the legacy descriptor table limits; keep it
    // on layout 2 so no covariance tables are required.
    descriptor.layoutVersion = 2;
    descriptor.shapeEncoding = 1;
    descriptor.pointMapWidth = 2048;
    descriptor.pointMapHeight = 2048;
    descriptor.positionGroupMin.resize(groupCount * 3, -1.0f);
    descriptor.positionGroupStep.resize(groupCount * 3, 0.01f);
    descriptor.streamRoles = {CompactGSStreamRole::POSITION_LOW};
    descriptor.planes = {{CompactGSStreamRole::POSITION_LOW, 2048, 2048, 3,
                          CompactGSComponentType::UINT8, 1, 1}};
    return descriptor;
}

std::vector<uint8_t> makeCompactGSStream(uint8_t profile = 2, uint8_t layout = 2) {
    Unit metadataUnit(0);
    auto& metadata = *metadataUnit.unitPayload.gsbsMetadata;
    const bool metadataInitialized = metadata.initialize2(17, 5, 3, 1);
    require(metadataInitialized, "metadata initialization failed");
    metadata.profileIdc = profile;
    metadata.subGsPointsNum[0] = 17;
    metadata.reconstructionInformation.resize(1);
    const std::vector<std::vector<uint8_t>> payloads = {
        compressZstd(makePaddedPointPlane(1)),
        compressZstd(makePaddedPointPlane(2)),
        compressZstd(makePaddedPointPlane(3)),
        std::vector<uint8_t>(32 * 12 * 3, 4),
        makeAstcFixture()
    };
    metadata.subBitstreamSize.clear();
    for (const auto& payload : payloads) {
        metadata.subBitstreamSize.push_back(static_cast<uint32_t>(payload.size()));
    }
    metadata.gsSubsetId.assign(5, 0);
    metadata.subBitstreamDecodeType = {0, 0, 0, 2, 1};
    metadata.reconstructionCount = {0};

    for (int stream = 0; stream < 3; ++stream) {
        auto entropy = std::make_shared<EntropyMeta>();
        entropy->entropyDecodeType = 0;
        entropy->attributeType = stream == 2 ? 4 : 0;
        entropy->bitdepth = 8;
        metadata.subBitstreamMeta.push_back(entropy);
        metadata.subBitstreamMetaType.push_back(SubBitstreamMetaType::ENTROPY);
    }
    auto video = std::make_shared<VideoMeta>();
    video->videoDecodeInformation.packingMapVideoCodecId = 0;
    video->videoPackingInformation.packingMapWidth = 32;
    video->videoPackingInformation.packingMapHeight = 12;
    video->videoPackingInformation.regionWidth = 8;
    video->videoPackingInformation.regionHeight = 4;
    video->videoPackingInformation.packingRegionCountMinus1 = 7;
    video->videoPackingInformation.initialize();
    metadata.subBitstreamMeta.push_back(video);
    metadata.subBitstreamMetaType.push_back(SubBitstreamMetaType::VIDEO);

    auto texture = std::make_shared<TextureMeta>();
    texture->attributeType = 20;
    texture->textureDecodeInformation.packingMapTextureCodecId = 1;
    texture->texturePackingInformation.packingMapWidth = 32;
    texture->texturePackingInformation.packingMapHeight = 16;
    texture->texturePackingInformation.regionWidth = 8;
    texture->texturePackingInformation.regionHeight = 4;
    texture->texturePackingInformation.packingRegionCountMinus1 = 14;
    texture->texturePackingInformation.textureChannelNum = 3;
    texture->texturePackingInformation.initialize();
    metadata.subBitstreamMeta.push_back(texture);
    metadata.subBitstreamMetaType.push_back(SubBitstreamMetaType::TEXTURE);

    if (profile == 2) {
        metadata.compactgs = std::make_shared<CompactGSDescriptor>();
        auto& fast = *metadata.compactgs;
        fast.layoutVersion = layout;
        fast.shapeEncoding = layout >= 3 ? 2 : 1;
        fast.pointMapWidth = 8;
        fast.pointMapHeight = 4;
        fast.positionGroupMin = {-1.0f, -2.0f, -3.0f};
        fast.positionGroupStep = {0.01f, 0.02f, 0.03f};
        fast.shapeMin[0] = 0.5f;
        fast.shapeStep[0] = 0.001f;
        fast.importanceMin = 0.1f;
        fast.importanceStep = 0.9f / 255.0f;
        for (size_t component = 0; component < 45; ++component) {
            fast.shMin[component] = -1.0f + static_cast<float>(component) * 0.001f;
            fast.shStep[component] = 2.0f / 255.0f;
        }
        fast.streamRoles = {
            CompactGSStreamRole::POSITION_LOW, CompactGSStreamRole::POSITION_HIGH,
            CompactGSStreamRole::COLOR_RGB, CompactGSStreamRole::SHAPE_VIDEO,
            CompactGSStreamRole::SH_ASTC
        };
        fast.planes = {
            {CompactGSStreamRole::POSITION_LOW, 8, 4, 3, CompactGSComponentType::UINT8, 1, 1},
            {CompactGSStreamRole::POSITION_HIGH, 8, 4, 3, CompactGSComponentType::UINT8, 1, 1},
            {CompactGSStreamRole::COLOR_RGB, 8, 4, 3, CompactGSComponentType::UINT8, 1, 1},
            {CompactGSStreamRole::SHAPE_VIDEO, 32, 12, 1,
             CompactGSComponentType::UINT8, 3, 4},
            {CompactGSStreamRole::SH_ASTC, 32, 16, 3, CompactGSComponentType::UINT8, 4, 4},
        };
    }
    const bool metadataWritten = metadataUnit.write();
    if (!metadataWritten) {
        require(layout != 2, "layout 2 metadata serialization failed");
        return {};
    }

    Unit payloadUnit(1);
    const bool payloadInitialized = payloadUnit.unitPayload.gsbsSubBitstreams->initialize(5);
    require(payloadInitialized, "payload initialization failed");
    payloadUnit.unitPayload.gsbsSubBitstreams->gstcSubBitstreamData = payloads;
    const bool payloadWritten = payloadUnit.write();
    require(payloadWritten, "payload serialization failed");
    const auto metadataBytes = metadataUnit.writer.getBytes();
    const auto payloadBytes = payloadUnit.writer.getBytes();
    std::vector<uint8_t> stream;
    auto append32 = [&](uint32_t value) {
        stream.push_back(static_cast<uint8_t>(value >> 24));
        stream.push_back(static_cast<uint8_t>(value >> 16));
        stream.push_back(static_cast<uint8_t>(value >> 8));
        stream.push_back(static_cast<uint8_t>(value));
    };
    append32(static_cast<uint32_t>(metadataBytes.size()));
    stream.insert(stream.end(), metadataBytes.begin(), metadataBytes.end());
    append32(static_cast<uint32_t>(payloadBytes.size()));
    stream.insert(stream.end(), payloadBytes.begin(), payloadBytes.end());
    return stream;
}

} // namespace

int main(int argc, char** argv) {
    {
        constexpr size_t kGardenGroupCount = 5237;
        const auto descriptor = makeGroupTableDescriptor(kGardenGroupCount);
        BitStreamWriter writer;
        require(descriptor.write(writer), "descriptor with more than 4096 groups did not serialize");
        const auto bytes = writer.getBytes();
        BitStreamReader reader;
        reader.setData(bytes);
        CompactGSDescriptor parsed;
        require(parsed.read(reader), "descriptor with more than 4096 groups did not parse");
        require(parsed.positionGroupMin.size() == kGardenGroupCount * 3,
                "large position group table changed size during round trip");

        const auto oversized = makeGroupTableDescriptor(CompactGSDescriptor::kMaximumPositionGroups + 1);
        BitStreamWriter oversizedWriter;
        require(!oversized.write(oversizedWriter), "descriptor above the maximum group count was accepted");

        const auto maximum = makeGroupTableDescriptor(CompactGSDescriptor::kMaximumPositionGroups);
        BitStreamWriter maximumWriter;
        require(maximum.write(maximumWriter), "descriptor at the maximum group count did not serialize");
        auto oversizedBytes = maximumWriter.getBytes();
        constexpr size_t kGroupCountOffset = 16;
        const uint32_t oversizedGroupCount = CompactGSDescriptor::kMaximumPositionGroups + 1;
        oversizedBytes[kGroupCountOffset] = static_cast<uint8_t>(oversizedGroupCount >> 24);
        oversizedBytes[kGroupCountOffset + 1] = static_cast<uint8_t>(oversizedGroupCount >> 16);
        oversizedBytes[kGroupCountOffset + 2] = static_cast<uint8_t>(oversizedGroupCount >> 8);
        oversizedBytes[kGroupCountOffset + 3] = static_cast<uint8_t>(oversizedGroupCount);
        const size_t groupTableEnd = kGroupCountOffset + sizeof(uint32_t) +
                                     CompactGSDescriptor::kMaximumPositionGroups * 6 * sizeof(float);
        oversizedBytes.insert(oversizedBytes.begin() + groupTableEnd, 6 * sizeof(float), 0);
        BitStreamReader oversizedReader;
        oversizedReader.setData(oversizedBytes);
        CompactGSDescriptor oversizedParsed;
        require(!oversizedParsed.read(oversizedReader),
                "parser accepted a descriptor above the maximum group count");
    }
    {
        const auto stream = makeCompactGSStream();
        Unit metadata(0);
        Unit payload(1);
        const bool parsedSuccessfully = parseGsbsStream(stream, metadata, payload);
        require(parsedSuccessfully, "valid COMPACTGS stream did not parse");
        const auto& parsed = *metadata.unitPayload.gsbsMetadata;
        require(parsed.profileIdc == 2, "parsed profile is not COMPACTGS");
        require(static_cast<bool>(parsed.compactgs), "COMPACTGS descriptor is missing");
        require(parsed.compactgs->layoutVersion == 2, "unexpected COMPACTGS layout version");
        require(parsed.compactgs->pointMapWidth == 8, "unexpected point map width");
        require(parsed.compactgs->streamRoles.size() == 5, "unexpected stream-role count");
        require(parsed.compactgs->streamRoles[4] == CompactGSStreamRole::SH_ASTC,
                "SH stream role was not preserved");
        require(parsed.compactgs->planes.size() > 3, "shape plane descriptor is missing");
        require(parsed.compactgs->planes[3].width == 32, "unexpected shape plane width");
        require(parsed.compactgs->positionGroupMin.size() == 3,
                "position group metadata is missing");

        GSDecoder decoder;
        const bool preparedSuccessfully = decoder.prepareFromMemory(
            stream.data(), stream.size(), TextureOutputMode::ASTC);
        require(preparedSuccessfully, "self-contained COMPACTGS ASTC fixture did not decode");
        require(decoder.getPreparedAuxiliaryData().astcUV.empty(),
                "COMPACTGS ASTC decode allocated legacy per-point UVs");
        require(!decoder.getPreparedAuxiliaryData().astcRawStream.empty(),
                "COMPACTGS ASTC decode did not retain the compressed atlas");
    }
    if (argc >= 2) {
        std::ifstream input(argv[1], std::ios::binary);
        require(input.good(), "fixture could not be opened");
        std::vector<uint8_t> stream((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
        GSDecoder decoder;
        const bool compressedSh = argc >= 3 && std::string(argv[2]) == "astc";
        const bool preparedSuccessfully = decoder.prepareFromMemory(
            stream.data(), stream.size(),
            compressedSh ? TextureOutputMode::ASTC : TextureOutputMode::CPU);
        require(preparedSuccessfully, "decoder could not prepare fixture");
        const auto& descriptor = decoder.getDecodeDescriptor();
        std::cout << "fixture profile=" << descriptor.profileIdc << " points=" << descriptor.pointCount << std::endl;
        require(descriptor.profileIdc == 2, "fixture profile is not COMPACTGS");
        require(descriptor.pointCount > 0, "fixture has no points");
        require(decoder.getReconstructionMetadataPacket().empty(),
                "COMPACTGS fixture unexpectedly produced reconstruction metadata");
        const auto* fast = decoder.getCompactGSDescriptor();
        require(fast != nullptr, "decoded COMPACTGS descriptor is missing");
        const size_t capacity = static_cast<size_t>(fast->pointMapWidth) * fast->pointMapHeight;
        require(capacity >= descriptor.pointCount, "point atlas is too small");
        const auto* positionLow = decoder.getCompactGSPlane(CompactGSStreamRole::POSITION_LOW);
        const auto* colorRgb = decoder.getCompactGSPlane(CompactGSStreamRole::COLOR_RGB);
        const auto* shapeVideo = decoder.getCompactGSPlane(CompactGSStreamRole::SHAPE_VIDEO);
        require(positionLow != nullptr && positionLow->size() == capacity * 3,
                "position-low plane has an invalid length");
        require(colorRgb != nullptr && colorRgb->size() == capacity * 3,
                "color plane has an invalid length");
        require(shapeVideo != nullptr && shapeVideo->size() == capacity * 12,
                "shape plane has an invalid length");
        if (compressedSh) {
            require(decoder.getCompactGSPlane(CompactGSStreamRole::SH_ASTC) == nullptr,
                    "compressed SH unexpectedly produced a CPU plane");
            require(!decoder.getPreparedAuxiliaryData().astcRawStream.empty(),
                    "compressed SH payload is missing");
        } else {
            const auto* sh = decoder.getCompactGSPlane(CompactGSStreamRole::SH_ASTC);
            require(sh != nullptr && sh->size() == capacity * 48,
                    "CPU SH plane has an invalid length");
        }
        std::vector<uint8_t> shard;
        const bool shardPacked = decoder.packShard(0, 1, shard);
        require(!shardPacked, "COMPACTGS unexpectedly entered the reconstruction shard path");
    }
    {
        const auto stream = makeCompactGSStream(9);
        Unit metadata(0);
        Unit payload(1);
        const bool unknownProfileParsed = parseGsbsStream(stream, metadata, payload);
        require(!unknownProfileParsed, "unknown profile was accepted");
    }
    {
        // Layout 1 used a different color/SH metadata schema and must not be
        // reinterpreted as layout 2.
        const auto stream = makeCompactGSStream(2, 1);
        require(stream.empty(), "obsolete COMPACTGS layout was serialized");
    }
    {
        auto stream = makeCompactGSStream();
        stream.resize(stream.size() - 1);
        Unit metadata(0);
        Unit payload(1);
        const bool truncatedStreamParsed = parseGsbsStream(stream, metadata, payload);
        require(!truncatedStreamParsed, "truncated COMPACTGS stream was accepted");
    }
    std::cout << "COMPACTGS_V1 stream tests passed" << std::endl;
    return 0;
}
