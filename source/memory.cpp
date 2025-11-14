#include <cstdlib>

#include "memory.h"
#include "misc.h"

#if __has_include("features.h")
    #include <features.h>
#endif

#if defined(__linux__) && !defined(__ANDROID__)
    #include <sys/mman.h>
#endif

#if defined(__APPLE__) || defined(__ANDROID__) || defined(__OpenBSD__) \
  || (defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC) && !defined(_WIN32)) \
  || defined(__e2k__)
    #define POSIXALIGNEDALLOC
    #include <stdlib.h>
#endif

#ifdef _WIN32
    #if _WIN32_WINNT < 0x0601
        #undef _WIN32_WINNT
        #define _WIN32_WINNT 0x0601  // Force to include needed API prototypes
    #endif

    #ifndef NOMINMAX
        #define NOMINMAX
    #endif

    #include <ios>       // std::hex, std::dec
    #include <iostream>  // std::cerr
    #include <ostream>   // std::endl
    #include <windows.h>

// The needed Windows API for processor groups could be missed from old Windows
// versions, so instead of calling them directly (forcing the linker to resolve
// the calls at compile time), try to load them at runtime. To do this we need
// first to define the corresponding function pointers.

#endif

namespace YaneuraOu {

// Wrappers for systems where the c++17 implementation does not guarantee the
// availability of aligned_alloc(). Memory allocated with std_aligned_alloc()
// must be freed with std_aligned_free().

void* std_aligned_alloc(size_t alignment, size_t size) {
#if defined(_ISOC11_SOURCE)
    return aligned_alloc(alignment, size);
#elif defined(POSIXALIGNEDALLOC)
    void* mem = nullptr;
    posix_memalign(&mem, alignment, size);
    return mem;
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
    return _mm_malloc(size, alignment);
#elif defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void std_aligned_free(void* ptr) {

#if defined(POSIXALIGNEDALLOC)
    free(ptr);
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
    _mm_free(ptr);
#elif defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// aligned_large_pages_alloc() will return suitably aligned memory,
// if possible using large pages.

#if defined(_WIN32)

static void* aligned_large_pages_alloc_windows([[maybe_unused]] size_t allocSize) {

    return windows_try_with_large_page_priviliges(
      [&](size_t largePageSize) {
          // Round up size to full pages and allocate
          allocSize = (allocSize + largePageSize - 1) & ~size_t(largePageSize - 1);
          return VirtualAlloc(nullptr, allocSize, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                              PAGE_READWRITE);
      },
      []() { return (void*) nullptr; });
}

#if !STOCKFISH
// 🌈 やねうら王独自
// 初回のlarge pageのallocationに対してメッセージを出力するためのフラグ。
bool first_large_pages_allocation = true;
#endif

void* aligned_large_pages_alloc(size_t allocSize) {

	// ※　ここでは4KB単位でalignされたメモリが返ることは保証されているので
	//     引数でalignを指定できる必要はない。(それを超えた大きなalignを行いたいケースがない)

    // Try to allocate large pages
    void* mem = aligned_large_pages_alloc_windows(allocSize);

    // Fall back to regular, page-aligned, allocation if necessary
	// 必要に応じて、通常のpage-alignedなメモリ割り当てにfall backする。

	// ⇨ LargePage非対応の環境であれば、std::aligned_alloc()を用いて確保しておく。
	//   最低でも4KBでalignされたメモリが返るので、引数でalign sizeを指定できるようにする必要はない。
    if (!mem)
        mem = VirtualAlloc(nullptr, allocSize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

#if !STOCKFISH
	else
	{
		// 🌈 やねうら王独自
		// large pageのallocationに成功したのでメッセージを出力。
		
		if (first_large_pages_allocation)
		{
			// Large Pagesを確保した旨を出力。
			sync_cout << "info string Large Pages are used." << sync_endl;
			first_large_pages_allocation = false;
		}
	}
#endif

    return mem;
}

#else

void* aligned_large_pages_alloc(size_t allocSize) {

    #if defined(__linux__)
    constexpr size_t alignment = 2 * 1024 * 1024;  // 2MB page size assumed
    #else
    constexpr size_t alignment = 4096;  // small page size assumed
    #endif

    // Round up to multiples of alignment
    size_t size = ((allocSize + alignment - 1) / alignment) * alignment;
    void*  mem  = std_aligned_alloc(alignment, size);
    #if defined(MADV_HUGEPAGE)
    madvise(mem, size, MADV_HUGEPAGE);
    #endif
    return mem;
}

#endif

bool has_large_pages() {

#if defined(_WIN32)

    constexpr size_t page_size = 2 * 1024 * 1024;  // 2MB page size assumed
    void*            mem       = aligned_large_pages_alloc_windows(page_size);
    if (mem == nullptr)
    {
        return false;
    }
    else
    {
        aligned_large_pages_free(mem);
        return true;
    }

#elif defined(__linux__)

    #if defined(MADV_HUGEPAGE)
    return true;
    #else
    return false;
    #endif

#else

    return false;

#endif
}


// aligned_large_pages_free() will free the previously memory allocated
// by aligned_large_pages_alloc(). The effect is a nop if mem == nullptr.

#if defined(_WIN32)

void aligned_large_pages_free(void* mem) {

    if (mem && !VirtualFree(mem, 0, MEM_RELEASE))
    {
        DWORD err = GetLastError();
        std::cerr << "Failed to free large page memory. Error code: 0x" << std::hex << err
                  << std::dec << std::endl;
        exit(EXIT_FAILURE);
    }
}

#else

void aligned_large_pages_free(void* mem) { std_aligned_free(mem); }

#endif

} // namespace YaneuraOu
