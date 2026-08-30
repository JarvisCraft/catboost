#pragma once

// Stand-in for the arcadia util header, for the standalone kernel check only.

#include <cassert>

#ifndef Y_ASSERT
    #define Y_ASSERT(condition) assert(condition)
#endif
