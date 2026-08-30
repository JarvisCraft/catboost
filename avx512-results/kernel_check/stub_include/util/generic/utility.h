#pragma once

// Stand-in for the arcadia util header, for the standalone kernel check only.

template <class T>
constexpr const T& Min(const T& lhs, const T& rhs) {
    return lhs < rhs ? lhs : rhs;
}

template <class T>
constexpr const T& Max(const T& lhs, const T& rhs) {
    return lhs > rhs ? lhs : rhs;
}
