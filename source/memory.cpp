#include "memory.h"

#include <cstdlib>
#include <iostream>  // std::cerr

#if __has_include("features.h")
    #include <features.h>
#endif

#if defined(__linux__) && !defined(__ANDROID__)
    #include <errno.h>
    #include <sys/mman.h>
    // IWYU pragma: no_include <bits/mman-map-flags-generic.h>
    #include <cstring>
    #include <mutex>
    #include <map>
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
        // 必要なAPIのプロトタイプ宣言が読み込まれるようにする。
    #endif

    #ifndef NOMINMAX
        #define NOMINMAX
    #endif

    #include <ios>  // std::hex, std::dec
    #include <windows.h>

// The needed Windows API for processor groups could be missed from old Windows
// versions, so instead of calling them directly (forcing the linker to resolve
// the calls at compile time), try to load them at runtime. To do this we need
// first to define the corresponding function pointers.
// 古いWindowsで不足するAPIを実行時に読み込めるようにする。
// 対応する関数ポインターの型はmemory.hで定義する。

#endif


namespace YaneuraOu {

// Wrappers for systems where the c++17 implementation does not guarantee the
// availability of aligned_alloc(). Memory allocated with std_aligned_alloc()
// must be freed with std_aligned_free().
// C++17のaligned_alloc()が利用できない環境にも対応するためのラッパー。
// 確保したメモリはstd_aligned_free()で解放する。

void* std_aligned_alloc(usize alignment, usize size) {
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
// 必要なアラインメントを満たすメモリを、可能ならLarge Pagesで確保する。

#if defined(_WIN32)

static void* aligned_large_pages_alloc_windows([[maybe_unused]] usize allocSize) {

    return windows_try_with_large_page_privileges(
      [&](usize largePageSize) {
          // Round up size to full pages and allocate
          // ページサイズの倍数に切り上げて確保する。
          allocSize = (allocSize + largePageSize - 1) & ~usize(largePageSize - 1);
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

void* aligned_large_pages_alloc_with_hint(usize allocSize, bool) {

	// ※　ここでは4KB単位でalignされたメモリが返ることは保証されているので
	//     引数でalignを指定できる必要はない。(それを超えた大きなalignを行いたいケースがない)

    // Try to allocate large pages
    void* mem = aligned_large_pages_alloc_windows(allocSize);

    // Fall back to regular, page-aligned, allocation if necessary
	// 必要に応じて、通常のpage-alignedなメモリ割り当てにfall backする。

	// ⇨ LargePageを確保できなければ、通常のVirtualAlloc()で確保しておく。
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

    #if defined(__linux__) && !defined(__ANDROID__)
// mmapで確保した領域のサイズを記録し、解放時にmunmapへ渡す。
// mallocのarenaを再利用せず、スレッドのNUMA配置に合わせて新たに確保する。
static std::map<void*, usize> large_page_sizes;
static std::mutex             large_page_sizes_mtx;
    #endif

    #if defined(__linux__) && defined(MAP_HUGE_SHIFT) && defined(__x86_64__)
        #define HAS_HUGE_PAGES

static void* try_huge_pages_alloc(usize allocSize) {
    usize size = ((allocSize + HugePageSize - 1) / HugePageSize) * HugePageSize;
    void* mem  = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (30 << MAP_HUGE_SHIFT), -1, 0);

    if (mem == MAP_FAILED)
        return nullptr;

    std::lock_guard lg(large_page_sizes_mtx);
    large_page_sizes[mem] = size;
    return mem;
}
    #endif  // defined(__linux__) && defined(MAP_HUGE_SHIFT) && defined(__x86_64__)

void* aligned_large_pages_alloc_with_hint(usize allocSize, [[maybe_unused]] bool hugePageHint) {
    #ifdef HAS_HUGE_PAGES
    if (hugePageHint && allocSize >= HugePageSize)
    {
        void* mem = try_huge_pages_alloc(allocSize);
        if (mem)
            return mem;
    }
    #endif

    #if defined(__linux__) && !defined(__ANDROID__)
    constexpr usize alignment = 2 * 1024 * 1024;  // 2MB page size assumed
    // 2MiBページを想定する。

    // Round up to multiples of alignment
    // alignmentの倍数に切り上げる。
    usize size = ((allocSize + alignment - 1) / alignment) * alignment;
    void* mem  = mmap_huge_aligned(size, MAP_PRIVATE | MAP_ANONYMOUS);
    if (mem != MAP_FAILED)
    {
        #if defined(MADV_HUGEPAGE)
        madvise(mem, size, MADV_HUGEPAGE);
        #endif
        std::lock_guard lg(large_page_sizes_mtx);
        large_page_sizes[mem] = size;
    }
    else
        mem = nullptr;
    return mem;
    #else
    constexpr usize alignment = 4096;  // small page size assumed
    // 通常のページサイズを想定する。
    usize           size      = ((allocSize + alignment - 1) / alignment) * alignment;
    void*           mem       = std_aligned_alloc(alignment, size);
        #if defined(MADV_HUGEPAGE)
    if (mem)
        madvise(mem, size, MADV_HUGEPAGE);
        #endif
    return mem;
    #endif
}

#endif

void* aligned_large_pages_alloc(usize size) {
    return aligned_large_pages_alloc_with_hint(size, false);
}

bool has_large_pages() {

#if defined(_WIN32)

    constexpr usize page_size = 2 * 1024 * 1024;  // 2MB page size assumed
    void*           mem       = aligned_large_pages_alloc_windows(page_size);
    if (mem == nullptr)
        return false;

    aligned_large_pages_free(mem);
    return true;

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
// aligned_large_pages_alloc()またはヒント付き版で確保したメモリを解放する。
// memがnullptrなら何もしない。

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

void aligned_large_pages_free(void* mem) {
    if (!mem)
        return;

    #if defined(__linux__) && !defined(__ANDROID__)
    {
        std::lock_guard lg(large_page_sizes_mtx);
        if (auto it = large_page_sizes.find(mem); it != large_page_sizes.end())
        {
            if (munmap(mem, it->second) != 0)
            {
                std::cerr << "munmap failed: " << strerror(errno) << std::endl;
                exit(EXIT_FAILURE);
            }
            large_page_sizes.erase(it);
            return;
        }
    }
    #endif

    std_aligned_free(mem);
}

#endif
} // namespace YaneuraOu
