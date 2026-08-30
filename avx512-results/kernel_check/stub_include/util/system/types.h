#pragma once

// Stand-in for the arcadia util header, for the standalone kernel check only:
// the real one needs the arcadia libc++ setup. Provides just the fixed-width
// aliases the kernels use.

#include <cstddef>
#include <cstdint>

using i8 = std::int8_t;
using ui8 = std::uint8_t;
using i16 = std::int16_t;
using ui16 = std::uint16_t;
using i32 = std::int32_t;
using ui32 = std::uint32_t;
using i64 = std::int64_t;
using ui64 = std::uint64_t;
