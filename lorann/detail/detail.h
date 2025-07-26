#pragma once

#include <simsimd/simsimd.h>

#if SIMSIMD_NATIVE_BF16
#include "bfloat.h"
#endif
#include "binary.h"
#include "float.h"
#if SIMSIMD_NATIVE_F16
#include "half.h"
#endif
#include "traits.h"
#include "uint8.h"