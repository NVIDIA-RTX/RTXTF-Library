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

#include "STFDefinitions.h"

#ifndef __STF_MACROS_HLSLI__
#define __STF_MACROS_HLSLI__

// Slang
#if __SLANG_COMPILER__
    #define STF_MUTATING [mutating]
#else
    #define STF_MUTATING
#endif // #if __SLANG_COMPILER__

#ifndef STF_SHADER_STAGE
    #if defined(__SHADER_TARGET_STAGE) // I.e dxc - we can derive STF_SHADER_STAGE automatically
        #if __SHADER_TARGET_STAGE == __SHADER_STAGE_PIXEL
            #define STF_SHADER_STAGE STF_SHADER_STAGE_PIXEL
        #elif __SHADER_TARGET_STAGE == __SHADER_STAGE_VERTEX
            #define STF_SHADER_STAGE STF_SHADER_STAGE_VERTEX
        #elif __SHADER_TARGET_STAGE == __SHADER_STAGE_GEOMETRY
            #define STF_SHADER_STAGE STF_SHADER_STAGE_GEOMETRY
        #elif __SHADER_TARGET_STAGE == __SHADER_STAGE_HULL
            #define STF_SHADER_STAGE STF_SHADER_STAGE_HULL
        #elif __SHADER_TARGET_STAGE == __SHADER_STAGE_DOMAIN
            #define STF_SHADER_STAGE STF_SHADER_STAGE_DOMAIN
        #elif __SHADER_TARGET_STAGE == __SHADER_STAGE_COMPUTE
            #define STF_SHADER_STAGE STF_SHADER_STAGE_COMPUTE
        #elif __SHADER_TARGET_STAGE == __SHADER_STAGE_AMPLIFICATION
            #define STF_SHADER_STAGE STF_SHADER_STAGE_AMPLIFICATION
        #elif __SHADER_TARGET_STAGE == __SHADER_STAGE_MESH
            #define STF_SHADER_STAGE STF_SHADER_STAGE_MESH
        #elif  __SHADER_TARGET_STAGE == __SHADER_STAGE_LIBRARY
            #define STF_SHADER_STAGE STF_SHADER_STAGE_LIBRARY
        #else
            #error "unknown value of __SHADER_TARGET_STAGE"
        #endif
    #endif
#endif // #ifndef STF_SHADER_STAGE

#if !defined(STF_SHADER_STAGE)
    #error "STF_SHADER_STAGE must be defined"
#elif (STF_SHADER_STAGE < 0 || STF_SHADER_STAGE > STF_SHADER_STAGE_LIBRARY)
    #error "Invalid value of STF_SHADER_STAGE"
#endif

#ifndef STF_SHADER_MODEL_MAJOR
    #if defined(__SHADER_TARGET_MAJOR) // I.e dxc - we can derive STF_SHADER_MODEL_MAJOR automatically
        #define STF_SHADER_MODEL_MAJOR __SHADER_TARGET_MAJOR
    #else
        #error "STF_SHADER_MODEL_MAJOR must be defined"
    #endif
#endif // #ifndef STF_SHADER_MODEL_MAJOR

#ifndef STF_SHADER_MODEL_MINOR
    #if defined(__SHADER_TARGET_MINOR) // I.e dxc - we can derive STF_SHADER_MODEL_MINOR automatically
        #define STF_SHADER_MODEL_MINOR __SHADER_TARGET_MINOR
    #else
        #error "STF_SHADER_MODEL_MINOR must be defined"
    #endif
#endif // #ifndef STF_SHADER_MODEL_MINOR

// STF_ALLOW_WAVE_READ is needed for all magnicifaction methods (except STF_MAGNIFICATION_METHOD_NONE) to work properly.
#ifndef STF_ALLOW_WAVE_READ
    #define STF_ALLOW_WAVE_READ (STF_SHADER_MODEL_MAJOR >= 6)
#endif // #ifndef STF_ALLOW_WAVE_READ

#endif // #ifndef __STF_MACROS_HLSLI__