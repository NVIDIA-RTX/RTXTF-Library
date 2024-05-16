/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#ifndef __STF_DEFINITIONS_H__
#define __STF_DEFINITIONS_H__

#define STF_PI 3.14159265359

// Filter types
#define STF_FILTER_TYPE_POINT               0
#define STF_FILTER_TYPE_LINEAR              1
#define STF_FILTER_TYPE_CUBIC               2
#define STF_FILTER_TYPE_GAUSSIAN            3

// Mip-Level overrides and bias
#define STF_MIP_VALUE_MODE_NONE             0
#define STF_MIP_VALUE_MODE_MIP_LEVEL        1
#define STF_MIP_VALUE_MODE_MIP_BIAS         2

// Magnification strategies
#define STF_MAGNIFICATION_METHOD_NONE                        0
#define STF_MAGNIFICATION_METHOD_2x2_QUAD                    1
#define STF_MAGNIFICATION_METHOD_2x2_FINE                    2
#define STF_MAGNIFICATION_METHOD_2x2_FINE_TEMPORAL           3 // Require FrameIndex to be set on the STFSamplerState
#define STF_MAGNIFICATION_METHOD_3x3_FINE_ALU                4
#define STF_MAGNIFICATION_METHOD_3x3_FINE_LUT                5
#define STF_MAGNIFICATION_METHOD_4x4_FINE                    6

#define STF_WAVE_READ_SAMPLES_PER_PIXEL     8

// No anisotropy
#define STF_ANISO_LOD_METHOD_NONE           0
// The method described in the STF paper
#define STF_ANISO_LOD_METHOD_DEFAULT        1

// Texture addressing modes - the values match D3D12
#define STF_ADDRESS_MODE_WRAP               1
#define STF_ADDRESS_MODE_MIRROR             2
#define STF_ADDRESS_MODE_CLAMP              3
#define STF_ADDRESS_MODE_BORDER             4
#define STF_ADDRESS_MODE_MIRROR_ONCE        5

// Shader stages
#define STF_SHADER_STAGE_PIXEL              1
#define STF_SHADER_STAGE_VERTEX	            2
#define STF_SHADER_STAGE_GEOMETRY           3
#define STF_SHADER_STAGE_HULL               4
#define STF_SHADER_STAGE_DOMAIN             5
#define STF_SHADER_STAGE_COMPUTE            6
#define STF_SHADER_STAGE_AMPLIFICATION      7
#define STF_SHADER_STAGE_MESH               8
#define STF_SHADER_STAGE_LIBRARY            9

#endif // #ifndef __STF_DEFINITIONS_H__
