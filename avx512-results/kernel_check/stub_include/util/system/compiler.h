#pragma once

// Stand-in for the arcadia util header, for the standalone kernel check only.

#ifndef Y_FORCE_INLINE
    #define Y_FORCE_INLINE inline __attribute__((always_inline))
#endif

#ifndef Y_PURE_FUNCTION
    #define Y_PURE_FUNCTION
#endif

#ifndef Y_PREFETCH_READ
    #define Y_PREFETCH_READ(ptr, locality) __builtin_prefetch((const void*)(ptr), 0, locality)
#endif

#ifndef Y_PREFETCH_WRITE
    #define Y_PREFETCH_WRITE(ptr, locality) __builtin_prefetch((const void*)(ptr), 1, locality)
#endif
