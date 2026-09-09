import * as THREE from 'three';
import { SplatMaterial } from './SplatMaterial.js';

export class SplatMaterial3D {

    static buildCompactGS(maxScreenSpaceSplatSize = 2048, splatScale = 1.0,
                       pointCloudModeEnabled = false, maxSphericalHarmonicsDegree = 3,
                       kernel2DSize = 0.3, compressedSH = false) {
        const uniforms = SplatMaterial.getUniforms(false, false, maxSphericalHarmonicsDegree,
                                                   splatScale, pointCloudModeEnabled, false);
        Object.assign(uniforms, {
            // Positions are decoded once while the COMPACTGS mesh is built.  The
            // shader intentionally consumes a regular float texture here so
            // quantisation arithmetic is not repeated for every vertex/frame.
            fastPositionTexture: { type: 't', value: null },
            fastPositionColorTexture: { type: 't', value: null },
            // Kept as compatibility handles for tooling/sorter diagnostics;
            // the vertex shader no longer samples these quantized planes.
            fastPositionLowTexture: { type: 't', value: null },
            fastPositionHighTexture: { type: 't', value: null },
            fastColorTexture: { type: 't', value: null },
            fastShapeTexture: { type: 't', value: null },
            fastPositionGroupMinTexture: { type: 't', value: null },
            fastPositionGroupStepTexture: { type: 't', value: null },
            fastCovarianceGroupMinTexture: { type: 't', value: null },
            fastCovarianceGroupStepTexture: { type: 't', value: null },
            fastCovarianceTexture: { type: 't', value: null },
            fastShTexture: { type: 't', value: null },
            fastShPackedTexture: { type: 't', value: null },
            fastShCompressedTexture: { type: 't', value: null },
            fastOrderTexture: { type: 't', value: null },
            fastPointMapSize: { type: 'v2', value: new THREE.Vector2() },
            fastOrderTextureSize: { type: 'v2', value: new THREE.Vector2() },
            fastCovarianceTextureSize: { type: 'v2', value: new THREE.Vector2() },
            fastShapeMin: { type: 'fv1', value: new Float32Array(7) },
            fastShapeStep: { type: 'fv1', value: new Float32Array(7) },
            fastImportanceMin: { type: 'f', value: 0.0 },
            fastImportanceStep: { type: 'f', value: 0.0 },
            fastShMin: { type: 'fv1', value: new Float32Array(45) },
            fastShStep: { type: 'fv1', value: new Float32Array(45) },
            fastMinimumAlpha: { type: 'f', value: 0.0 },
            fastCovarianceMode: { type: 'i', value: 0 },
            fastBatched: { type: 'i', value: 0 },
            fastBatchSize: { type: 'i', value: 128 },
            fastRenderSplatCount: { type: 'i', value: 0 },
            fastUsePositionColorTexture: { type: 'i', value: 0 },
            fastUsePackedSH: { type: 'i', value: 0 }
        });

        const vertexShader = SplatMaterial3D.buildCompactGSVertexShader(
            maxScreenSpaceSplatSize, kernel2DSize, maxSphericalHarmonicsDegree, compressedSH
        );
        return new THREE.ShaderMaterial({
            uniforms,
            vertexShader,
            fragmentShader: SplatMaterial3D.buildFragmentShader(),
            transparent: true,
            alphaTest: 1.0,
            blending: THREE.NormalBlending,
            depthTest: true,
            depthWrite: false,
            side: THREE.DoubleSide
        });
    }

    static buildCompactGSVertexShader(maxScreenSpaceSplatSize, kernel2DSize, maxSHDegree, compressedSH) {
        const shSampler = compressedSH ?
            'texelFetch(fastShCompressedTexture, ivec3(atlasCoord, 0), 0).rgb' :
            'texelFetch(fastShTexture, atlasCoord, 0).rgb';
        return `
            precision highp float;
            precision highp int;
            precision highp usampler2D;
            precision highp sampler2DArray;
            #include <common>

            attribute uint splatIndex;
            attribute uint splatLocalIndex;
            uniform highp sampler2D fastPositionTexture;
            uniform highp sampler2D fastPositionColorTexture;
            uniform highp usampler2D fastPositionLowTexture;
            uniform highp usampler2D fastPositionHighTexture;
            uniform highp sampler2D fastColorTexture;
            uniform highp sampler2D fastShapeTexture;
            uniform highp sampler2D fastPositionGroupMinTexture;
            uniform highp sampler2D fastPositionGroupStepTexture;
            uniform highp sampler2D fastCovarianceGroupMinTexture;
            uniform highp sampler2D fastCovarianceGroupStepTexture;
            uniform highp sampler2D fastCovarianceTexture;
            uniform highp sampler2D fastShTexture;
            uniform highp sampler2D fastShPackedTexture;
            uniform highp sampler2DArray fastShCompressedTexture;
            uniform highp usampler2D fastOrderTexture;
            uniform vec2 fastPointMapSize;
            uniform vec2 fastOrderTextureSize;
            uniform vec2 fastCovarianceTextureSize;
            uniform float fastShapeMin[7];
            uniform float fastShapeStep[7];
            uniform float fastImportanceMin;
            uniform float fastImportanceStep;
            uniform float fastShMin[45];
            uniform float fastShStep[45];
            uniform float fastMinimumAlpha;
            uniform int fastCovarianceMode;
            uniform int fastBatched;
            uniform int fastBatchSize;
            uniform int fastRenderSplatCount;
            uniform int fastUsePositionColorTexture;
            uniform int fastUsePackedSH;
            uniform vec2 focal;
            uniform float orthoZoom;
            uniform int orthographicMode;
            uniform int pointCloudModeEnabled;
            uniform float inverseFocalAdjustment;
            uniform vec2 viewport;
            uniform vec2 basisViewport;
            uniform int sphericalHarmonicsDegree;
            uniform float visibleRegionRadius;
            uniform float visibleRegionFadeStartRadius;
            uniform float firstRenderTime;
            uniform float currentTime;
            uniform int fadeInComplete;
            uniform vec3 sceneCenter;
            uniform float splatScale;

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;

            const float sqrt8 = sqrt(8.0);
            const float SH_C1 = 0.4886025119029199;
            const float SH_C2[5] = float[](1.0925484, -1.0925484, 0.3153916, -1.0925484, 0.5462742);
            const float SH_C3[7] = float[](-0.59004359, 2.89061144, -0.45704580, 0.37317633,
                                           -0.45704580, 1.44530572, -0.59004359);

            vec3 readFastSH(ivec2 pointCoord, int coefficientIndex) {
                int tileX = coefficientIndex & 3;
                int tileY = coefficientIndex >> 2;
                ivec2 atlasCoord = pointCoord + ivec2(tileX * int(fastPointMapSize.x),
                                                      tileY * int(fastPointMapSize.y));
                vec3 q = ${shSampler} * 255.0;
                int base = coefficientIndex * 3;
                return q * vec3(fastShStep[base], fastShStep[base + 1], fastShStep[base + 2]) +
                    vec3(fastShMin[base], fastShMin[base + 1], fastShMin[base + 2]);
            }

            vec3 decodeFastSH(vec3 q, int coefficientIndex) {
                q *= 255.0;
                int base = coefficientIndex * 3;
                return q * vec3(fastShStep[base], fastShStep[base + 1], fastShStep[base + 2]) +
                    vec3(fastShMin[base], fastShMin[base + 1], fastShMin[base + 2]);
            }

            void readPackedSHDegree2(ivec2 pointCoord,
                                     out vec3 sh0, out vec3 sh1, out vec3 sh2, out vec3 sh3,
                                     out vec3 sh4, out vec3 sh5, out vec3 sh6, out vec3 sh7) {
                int atlasWidth = int(fastPointMapSize.x);
                vec4 t0 = texelFetch(fastShPackedTexture, pointCoord + ivec2(0 * atlasWidth, 0), 0);
                vec4 t1 = texelFetch(fastShPackedTexture, pointCoord + ivec2(1 * atlasWidth, 0), 0);
                vec4 t2 = texelFetch(fastShPackedTexture, pointCoord + ivec2(2 * atlasWidth, 0), 0);
                vec4 t3 = texelFetch(fastShPackedTexture, pointCoord + ivec2(3 * atlasWidth, 0), 0);
                vec4 t4 = texelFetch(fastShPackedTexture, pointCoord + ivec2(4 * atlasWidth, 0), 0);
                vec4 t5 = texelFetch(fastShPackedTexture, pointCoord + ivec2(5 * atlasWidth, 0), 0);
                sh0 = decodeFastSH(t0.rgb, 0);
                sh1 = decodeFastSH(vec3(t0.a, t1.rg), 1);
                sh2 = decodeFastSH(vec3(t1.ba, t2.r), 2);
                sh3 = decodeFastSH(t2.gba, 3);
                sh4 = decodeFastSH(t3.rgb, 4);
                sh5 = decodeFastSH(vec3(t3.a, t4.rg), 5);
                sh6 = decodeFastSH(vec3(t4.ba, t5.r), 6);
                sh7 = decodeFastSH(t5.gba, 7);
            }

            float readShapeQ(ivec2 pointCoord, int component) {
                ivec2 atlasCoord = pointCoord + ivec2((component & 3) * int(fastPointMapSize.x),
                                                      (component >> 2) * int(fastPointMapSize.y));
                return texelFetch(fastShapeTexture, atlasCoord, 0).r * 255.0;
            }

            float readShape(ivec2 pointCoord, int component) {
                return readShapeQ(pointCoord, component) * fastShapeStep[component] + fastShapeMin[component];
            }

            void readCovariance(ivec2 pointCoord, int groupIndex, out vec3 covA, out vec3 covB) {
                vec4 mn = texelFetch(fastCovarianceGroupMinTexture, ivec2(groupIndex * 2, 0), 0);
                vec4 st = texelFetch(fastCovarianceGroupStepTexture, ivec2(groupIndex * 2, 0), 0);
                vec4 mn2 = texelFetch(fastCovarianceGroupMinTexture, ivec2(groupIndex * 2 + 1, 0), 0);
                vec4 st2 = texelFetch(fastCovarianceGroupStepTexture, ivec2(groupIndex * 2 + 1, 0), 0);
                int vals[6];
                for (int c = 0; c < 6; ++c) {
                    int loTile = (c * 2) & 3;
                    int loRow = (c * 2) >> 2;
                    int hiTile = (c * 2 + 1) & 3;
                    int hiRow = (c * 2 + 1) >> 2;
                    ivec2 loCoord = pointCoord + ivec2(loTile * int(fastPointMapSize.x), loRow * int(fastPointMapSize.y));
                    ivec2 hiCoord = pointCoord + ivec2(hiTile * int(fastPointMapSize.x), hiRow * int(fastPointMapSize.y));
                    vals[c] = int(texelFetch(fastShapeTexture, loCoord, 0).r * 255.0 + 0.5) |
                              (int(texelFetch(fastShapeTexture, hiCoord, 0).r * 255.0 + 0.5) << 8);
                }
                covA = vec3(mn.x + st.x * float(vals[0]), mn.y + st.y * float(vals[1]), mn.z + st.z * float(vals[2]));
                covB = vec3(mn.w + st.w * float(vals[3]), mn2.x + st2.x * float(vals[4]), mn2.y + st2.y * float(vals[5]));
            }

            void readDirectCovariance(int splat, out vec3 covA, out vec3 covB) {
                int mapWidth = int(fastCovarianceTextureSize.x);
                ivec2 uv = ivec2((splat % mapWidth) * 2, splat / mapWidth);
                vec4 a = texelFetch(fastCovarianceTexture, uv, 0);
                vec4 b = texelFetch(fastCovarianceTexture, uv + ivec2(1, 0), 0);
                covA = a.rgb;
                covB = vec3(a.a, b.rg);
            }

            void main() {
                uint resolvedSplatIndex = splatIndex;
                if (fastBatched == 1) {
                    uint renderOffset = uint(gl_InstanceID) * uint(fastBatchSize) + splatLocalIndex;
                    if (renderOffset >= uint(fastRenderSplatCount)) {
                        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                        return;
                    }
                    ivec2 orderCoord = ivec2(int(renderOffset) % int(fastOrderTextureSize.x),
                                             int(renderOffset) / int(fastOrderTextureSize.x));
                    resolvedSplatIndex = texelFetch(fastOrderTexture, orderCoord, 0).r;
                }
                int mapWidth = int(fastPointMapSize.x);
                ivec2 pointCoord = ivec2(int(resolvedSplatIndex) % mapWidth, int(resolvedSplatIndex) / mapWidth);
                int groupIndex = int(resolvedSplatIndex) >> 8;
                // COMPACTGS positions are decoded at build time.  Keep this
                // lookup as a single float fetch in the per-frame shader.
                vec4 positionColor;
                if (fastUsePositionColorTexture == 1) positionColor = texelFetch(fastPositionColorTexture, pointCoord, 0);
                else positionColor = texelFetch(fastPositionTexture, pointCoord, 0);
                vec3 splatCenter = positionColor.xyz;
                if (fastUsePositionColorTexture == 1) {
                    // The alpha lane stores an exact 24-bit RGB integer as a
                    // regular float. Do not use uintBitsToFloat here: values
                    // with an 0xff high byte are NaNs, whose payload may be
                    // canonicalized by Apple's Metal/WebGL implementation.
                    uint packedColor = uint(positionColor.a + 0.5);
                    vColor = vec4(vec3(float(packedColor & 0xffu), float((packedColor >> 8u) & 0xffu),
                                       float((packedColor >> 16u) & 0xffu)) / 255.0, 1.0);
                } else {
                    vColor = texelFetch(fastColorTexture, pointCoord, 0);
                }
                vColor.a = readShapeQ(pointCoord, fastCovarianceMode == 1 ? 12 : 7) / 255.0;
                if (vColor.a * 255.0 < fastMinimumAlpha) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }

                vec4 viewCenter = modelViewMatrix * vec4(splatCenter, 1.0);
                vec4 clipCenter = projectionMatrix * viewCenter;
                float clip = 1.2 * clipCenter.w;
                if (clipCenter.z < -clip || clipCenter.x < -clip || clipCenter.x > clip ||
                    clipCenter.y < -clip || clipCenter.y > clip) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }
                vec3 ndcCenter = clipCenter.xyz / clipCenter.w;
                vPosition = position.xy;

                if (sphericalHarmonicsDegree >= 1) {
                    float importance = clamp(fastImportanceMin + readShapeQ(pointCoord, fastCovarianceMode == 1 ? 13 : 8) *
                                             fastImportanceStep, 1e-6, 1.0);
                    float inverseImportance = 1.0 / importance;
                    vec3 d = normalize(splatCenter - cameraPosition);
                    float x = d.x, y = d.y, z = d.z;
                    vec3 sh1;
                    vec3 sh2;
                    vec3 sh3;
                    vec3 sh4;
                    vec3 sh5;
                    vec3 sh6;
                    vec3 sh7;
                    vec3 sh8;
                    if (fastUsePackedSH == 1 && sphericalHarmonicsDegree == 2) {
                        readPackedSHDegree2(pointCoord, sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8);
                    } else {
                        sh1 = readFastSH(pointCoord, 0);
                        sh2 = readFastSH(pointCoord, 1);
                        sh3 = readFastSH(pointCoord, 2);
                    }
                    vColor.rgb += inverseImportance * SH_C1 * (-sh1 * y + sh2 * z - sh3 * x);
                    float xx = x*x, yy = y*y, zz = z*z;
                    float xy = x*y, yz = y*z, xz = x*z;
                    if (sphericalHarmonicsDegree >= 2) {
                        vColor.rgb += inverseImportance * (
                            SH_C2[0] * xy * ((fastUsePackedSH == 1) ? sh4 : readFastSH(pointCoord, 3)) +
                            SH_C2[1] * yz * ((fastUsePackedSH == 1) ? sh5 : readFastSH(pointCoord, 4)) +
                            SH_C2[2] * (2.0*zz-xx-yy) * ((fastUsePackedSH == 1) ? sh6 : readFastSH(pointCoord, 5)) +
                            SH_C2[3] * xz * ((fastUsePackedSH == 1) ? sh7 : readFastSH(pointCoord, 6)) +
                            SH_C2[4] * (xx-yy) * ((fastUsePackedSH == 1) ? sh8 : readFastSH(pointCoord, 7)));
                    }
                    if (sphericalHarmonicsDegree >= 3) {
                        vColor.rgb += inverseImportance * (
                            SH_C3[0] * y * (3.0*xx-yy) * readFastSH(pointCoord, 8) +
                            SH_C3[1] * xy * z * readFastSH(pointCoord, 9) +
                            SH_C3[2] * y * (4.0*zz-xx-yy) * readFastSH(pointCoord, 10) +
                            SH_C3[3] * z * (2.0*zz-3.0*xx-3.0*yy) * readFastSH(pointCoord, 11) +
                            SH_C3[4] * x * (4.0*zz-xx-yy) * readFastSH(pointCoord, 12) +
                            SH_C3[5] * z * (xx-yy) * readFastSH(pointCoord, 13) +
                            SH_C3[6] * x * (xx-3.0*yy) * readFastSH(pointCoord, 14));
                    }
                    vColor.rgb = clamp(vColor.rgb, vec3(0.0), vec3(1.0));
                }

                mat3 Vrk;
                if (fastCovarianceMode == 1 || fastCovarianceMode == 2) {
                    // Layout 3 carries the six symmetric covariance terms
                    // directly; avoid exp/normalize/quaternion rotation work.
                    vec3 covA; vec3 covB;
                    if (fastCovarianceMode == 2) readDirectCovariance(int(resolvedSplatIndex), covA, covB);
                    else readCovariance(pointCoord, groupIndex, covA, covB);
                    Vrk = mat3(covA.x, covA.y, covA.z,
                               covA.y, covB.x, covB.y,
                               covA.z, covB.y, covB.z);
                } else {
                    vec3 scale = exp(vec3(readShape(pointCoord, 0), readShape(pointCoord, 1), readShape(pointCoord, 2)));
                    vec4 rq = normalize(vec4(readShape(pointCoord, 3), readShape(pointCoord, 4),
                                             readShape(pointCoord, 5), readShape(pointCoord, 6)));
                    float wq = rq.x, xq = rq.y, yq = rq.z, zq = rq.w;
                    mat3 R = mat3(
                        1.0 - 2.0*(yq*yq + zq*zq), 2.0*(xq*yq + zq*wq),       2.0*(xq*zq - yq*wq),
                        2.0*(xq*yq - zq*wq),       1.0 - 2.0*(xq*xq + zq*zq), 2.0*(yq*zq + xq*wq),
                        2.0*(xq*zq + yq*wq),       2.0*(yq*zq - xq*wq),       1.0 - 2.0*(xq*xq + yq*yq));
                    mat3 M = R * mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
                    Vrk = M * transpose(M);
                }
                mat3 J;
                if (orthographicMode == 1) {
                    J = transpose(mat3(orthoZoom,0.0,0.0, 0.0,orthoZoom,0.0, 0.0,0.0,0.0));
                } else {
                    float s = 1.0 / (viewCenter.z * viewCenter.z);
                    J = mat3(focal.x/viewCenter.z,0.0,-focal.x*viewCenter.x*s,
                             0.0,focal.y/viewCenter.z,-focal.y*viewCenter.y*s, 0.0,0.0,0.0);
                }
                mat3 T = transpose(mat3(modelViewMatrix)) * J;
                mat3 cov2Dm = transpose(T) * Vrk * T;
                cov2Dm[0][0] += ${kernel2DSize};
                cov2Dm[1][1] += ${kernel2DSize};
                float a = cov2Dm[0][0], b = cov2Dm[0][1], dd = cov2Dm[1][1];
                float traceOver2 = 0.5 * (a + dd);
                float term2 = sqrt(max(0.1, traceOver2*traceOver2 - (a*dd-b*b)));
                float eigenValue1 = traceOver2 + term2;
                float eigenValue2 = traceOver2 - term2;
                if (pointCloudModeEnabled == 1) eigenValue1 = eigenValue2 = 0.2;
                if (eigenValue2 <= 0.0) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }
                vec2 eigenVector1 = abs(b) > 1e-8 ? normalize(vec2(b, eigenValue1-a)) : vec2(1.0, 0.0);
                vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);
                vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8*sqrt(eigenValue1), ${parseInt(maxScreenSpaceSplatSize)}.0);
                vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8*sqrt(eigenValue2), ${parseInt(maxScreenSpaceSplatSize)}.0);
                vec2 ndcOffset = vec2(vPosition.x*basisVector1 + vPosition.y*basisVector2) *
                                 basisViewport * 2.0 * inverseFocalAdjustment;
                gl_Position = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
                vPosition *= sqrt8;
            }
        `;
    }

    /**
     * Build the Three.js material that is used to render the splats.
     * @param {number} dynamicMode If true, it means the scene geometry represented by this splat mesh is not stationary or
     *                             that the splat count might change
     * @param {boolean} enableOptionalEffects When true, allows for usage of extra properties and attributes in the shader for effects
     *                                        such as opacity adjustment. Default is false for performance reasons.
     * @param {boolean} antialiased If true, calculate compensation factor to deal with gaussians being rendered at a significantly
     *                              different resolution than that of their training
     * @param {number} maxScreenSpaceSplatSize The maximum clip space splat size
     * @param {number} splatScale Value by which all splats are scaled in screen-space (default is 1.0)
     * @param {number} pointCloudModeEnabled Render all splats as screen-space circles
     * @param {number} maxSphericalHarmonicsDegree Degree of spherical harmonics to utilize in rendering splats
     * @return {THREE.ShaderMaterial}
     */
    static build(dynamicMode = false, enableOptionalEffects = false, antialiased = false, maxScreenSpaceSplatSize = 2048,
                 splatScale = 1.0, pointCloudModeEnabled = false, maxSphericalHarmonicsDegree = 0, kernel2DSize = 0.3, useASTC = false) {

        const customVertexVars = `
            uniform vec2 covariancesTextureSize;
            uniform highp sampler2D covariancesTexture;
            uniform highp usampler2D covariancesTextureHalfFloat;
            uniform int covariancesAreHalfFloat;

            void fromCovarianceHalfFloatV4(uvec4 val, out vec4 first, out vec4 second) {
                vec2 r = unpackHalf2x16(val.r);
                vec2 g = unpackHalf2x16(val.g);
                vec2 b = unpackHalf2x16(val.b);

                first = vec4(r.x, r.y, g.x, g.y);
                second = vec4(b.x, b.y, 0.0, 0.0);
            }
        `;

        let vertexShaderSource = SplatMaterial.buildVertexShaderBase(dynamicMode, enableOptionalEffects,
                                                                     maxSphericalHarmonicsDegree, useASTC, customVertexVars);
        vertexShaderSource += SplatMaterial3D.buildVertexShaderProjection(antialiased, enableOptionalEffects,
                                                                          maxScreenSpaceSplatSize, kernel2DSize);
        const fragmentShaderSource = SplatMaterial3D.buildFragmentShader();

        const uniforms = SplatMaterial.getUniforms(dynamicMode, enableOptionalEffects,
                                                   maxSphericalHarmonicsDegree, splatScale, pointCloudModeEnabled, useASTC);

        uniforms['covariancesTextureSize'] = {
            'type': 'v2',
            'value': new THREE.Vector2(1024, 1024)
        };
        uniforms['covariancesTexture'] = {
            'type': 't',
            'value': null
        };
        uniforms['covariancesTextureHalfFloat'] = {
            'type': 't',
            'value': null
        };
        uniforms['covariancesAreHalfFloat'] = {
            'type': 'i',
            'value': 0
        };

        const material = new THREE.ShaderMaterial({
            uniforms: uniforms,
            vertexShader: vertexShaderSource,
            fragmentShader: fragmentShaderSource,
            transparent: true,
            alphaTest: 1.0,
            blending: THREE.NormalBlending,
            depthTest: true,
            depthWrite: false,
            side: THREE.DoubleSide
        });

        return material;
    }

    static buildVertexShaderProjection(antialiased, enableOptionalEffects, maxScreenSpaceSplatSize, kernel2DSize) {
        let vertexShaderSource = `

            vec4 sampledCovarianceA;
            vec4 sampledCovarianceB;
            vec3 cov3D_M11_M12_M13;
            vec3 cov3D_M22_M23_M33;
            if (covariancesAreHalfFloat == 0) {
                sampledCovarianceA = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset,
                                                                            covariancesTextureSize));
                sampledCovarianceB = texture(covariancesTexture, getDataUVF(nearestEvenIndex, 1.5, oddOffset + uint(1),
                                                                            covariancesTextureSize));

                cov3D_M11_M12_M13 = vec3(sampledCovarianceA.rgb) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceA.ba, sampledCovarianceB.r) * fOddOffset;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg) * (1.0 - fOddOffset) +
                                    vec3(sampledCovarianceB.gba) * fOddOffset;
            } else {
                uvec4 sampledCovarianceU = texture(covariancesTextureHalfFloat, getDataUV(1, 0, covariancesTextureSize));
                fromCovarianceHalfFloatV4(sampledCovarianceU, sampledCovarianceA, sampledCovarianceB);
                cov3D_M11_M12_M13 = sampledCovarianceA.rgb;
                cov3D_M22_M23_M33 = vec3(sampledCovarianceA.a, sampledCovarianceB.rg);
            }
        
            // Construct the 3D covariance matrix
            mat3 Vrk = mat3(
                cov3D_M11_M12_M13.x, cov3D_M11_M12_M13.y, cov3D_M11_M12_M13.z,
                cov3D_M11_M12_M13.y, cov3D_M22_M23_M33.x, cov3D_M22_M23_M33.y,
                cov3D_M11_M12_M13.z, cov3D_M22_M23_M33.y, cov3D_M22_M23_M33.z
            );

            mat3 J;
            if (orthographicMode == 1) {
                // Since the projection is linear, we don't need an approximation
                J = transpose(mat3(orthoZoom, 0.0, 0.0,
                                0.0, orthoZoom, 0.0,
                                0.0, 0.0, 0.0));
            } else {
                // Construct the Jacobian of the affine approximation of the projection matrix. It will be used to transform the
                // 3D covariance matrix instead of using the actual projection matrix because that transformation would
                // require a non-linear component (perspective division) which would yield a non-gaussian result.
                float s = 1.0 / (viewCenter.z * viewCenter.z);
                J = mat3(
                    focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * s,
                    0., focal.y / viewCenter.z, -(focal.y * viewCenter.y) * s,
                    0., 0., 0.
                );
            }

            // Concatenate the projection approximation with the model-view transformation
            mat3 W = transpose(mat3(transformModelViewMatrix));
            mat3 T = W * J;

            // Transform the 3D covariance matrix (Vrk) to compute the 2D covariance matrix
            mat3 cov2Dm = transpose(T) * Vrk * T;
            `;

        if (antialiased) {
            vertexShaderSource += `
                float detOrig = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
                cov2Dm[0][0] += ${kernel2DSize};
                cov2Dm[1][1] += ${kernel2DSize};
                float detBlur = cov2Dm[0][0] * cov2Dm[1][1] - cov2Dm[0][1] * cov2Dm[0][1];
                vColor.a *= sqrt(max(detOrig / detBlur, 0.0));
                if (vColor.a < minAlpha) return;
            `;
        } else {
            vertexShaderSource += `
                cov2Dm[0][0] += ${kernel2DSize};
                cov2Dm[1][1] += ${kernel2DSize};
            `;
        }

        vertexShaderSource += `

            // We are interested in the upper-left 2x2 portion of the projected 3D covariance matrix because
            // we only care about the X and Y values. We want the X-diagonal, cov2Dm[0][0],
            // the Y-diagonal, cov2Dm[1][1], and the correlation between the two cov2Dm[0][1]. We don't
            // need cov2Dm[1][0] because it is a symetric matrix.
            vec3 cov2Dv = vec3(cov2Dm[0][0], cov2Dm[0][1], cov2Dm[1][1]);

            // We now need to solve for the eigen-values and eigen vectors of the 2D covariance matrix
            // so that we can determine the 2D basis for the splat. This is done using the method described
            // here: https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
            // After calculating the eigen-values and eigen-vectors, we calculate the basis for rendering the splat
            // by normalizing the eigen-vectors and then multiplying them by (sqrt(8) * sqrt(eigen-value)), which is
            // equal to scaling them by sqrt(8) standard deviations.
            //
            // This is a different approach than in the original work at INRIA. In that work they compute the
            // max extents of the projected splat in screen space to form a screen-space aligned bounding rectangle
            // which forms the geometry that is actually rasterized. The dimensions of that bounding box are 3.0
            // times the square root of the maximum eigen-value, or 3 standard deviations. They then use the inverse
            // 2D covariance matrix (called 'conic') in the CUDA rendering thread to determine fragment opacity by
            // calculating the full gaussian: exp(-0.5 * (X - mean) * conic * (X - mean)) * splat opacity
            float a = cov2Dv.x;
            float d = cov2Dv.z;
            float b = cov2Dv.y;
            float D = a * d - b * b;
            float trace = a + d;
            float traceOver2 = 0.5 * trace;
            float term2 = sqrt(max(0.1f, traceOver2 * traceOver2 - D));
            float eigenValue1 = traceOver2 + term2;
            float eigenValue2 = traceOver2 - term2;

            if (pointCloudModeEnabled == 1) {
                eigenValue1 = eigenValue2 = 0.2;
            }

            if (eigenValue2 <= 0.0) return;

            vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
            // since the eigen vectors are orthogonal, we derive the second one from the first
            vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

            // We use sqrt(8) standard deviations instead of 3 to eliminate more of the splat with a very low opacity.
            vec2 basisVector1 = eigenVector1 * splatScale * min(sqrt8 * sqrt(eigenValue1), ${parseInt(maxScreenSpaceSplatSize)}.0);
            vec2 basisVector2 = eigenVector2 * splatScale * min(sqrt8 * sqrt(eigenValue2), ${parseInt(maxScreenSpaceSplatSize)}.0);
            `;

        if (enableOptionalEffects) {
            vertexShaderSource += `
                vColor.a *= splatOpacityFromScene;
            `;
        }

        vertexShaderSource += `
            vec2 ndcOffset = vec2(vPosition.x * basisVector1 + vPosition.y * basisVector2) *
                             basisViewport * 2.0 * inverseFocalAdjustment;

            vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
            gl_Position = quadPos;

            // Scale the position data we send to the fragment shader
            vPosition *= sqrt8;
        `;

        vertexShaderSource += SplatMaterial.getVertexShaderFadeIn();
        vertexShaderSource += `}`;

        return vertexShaderSource;
    }

    static buildFragmentShader() {
        let fragmentShaderSource = `
            precision highp float;
            #include <common>
 
            uniform vec3 debugColor;

            varying vec4 vColor;
            varying vec2 vUv;
            varying vec2 vPosition;
        `;

        fragmentShaderSource += `
            void main () {
                // Compute the positional squared distance from the center of the splat to the current fragment.
                float A = dot(vPosition, vPosition);
                // Since the positional data in vPosition has been scaled by sqrt(8), the squared result will be
                // scaled by a factor of 8. If the squared result is larger than 8, it means it is outside the ellipse
                // defined by the rectangle formed by vPosition. It also means it's farther
                // away than sqrt(8) standard deviations from the mean.
                if (A > 8.0) discard;
                vec3 color = vColor.rgb;

                // Since the rendered splat is scaled by sqrt(8), the inverse covariance matrix that is part of
                // the gaussian formula becomes the identity matrix. We're then left with (X - mean) * (X - mean),
                // and since 'mean' is zero, we have X * X, which is the same as A:
                float opacity = exp(-0.5 * A) * vColor.a;

                gl_FragColor = vec4(color.rgb, opacity);
            }
        `;

        return fragmentShaderSource;
    }

}
