#ifndef MEMORY_H_INCLUDED
#define MEMORY_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "types.h"
#include "misc.h"

#if defined(_WIN64)

    #if _WIN32_WINNT < 0x0601
        #undef _WIN32_WINNT
        #define _WIN32_WINNT 0x0601  // Force to include needed API prototypes
    #endif

    #if !defined(NOMINMAX)
        #define NOMINMAX
    #endif
    #include <windows.h>

    // Some Windows headers (RPC/old headers) define short macros such
    // as 'small' expanding to 'char', which breaks identifiers in the code.
    // Undefine those macros immediately after including <windows.h>.

	// いくつかの Windows ヘッダー（RPC／古いヘッダーなど）は、
	// 'small' のような短いマクロを 'char' に展開するなどして、
	// コード内の識別子を壊してしまうことがあります。
	// <windows.h> をインクルードした直後に、そうしたマクロを
	// 即座に undef してください。

    #ifdef small
        #undef small
    #endif

    #include <psapi.h>

extern "C" {
using OpenProcessToken_t      = bool (*)(HANDLE, DWORD, PHANDLE);
using LookupPrivilegeValueA_t = bool (*)(LPCSTR, LPCSTR, PLUID);
using AdjustTokenPrivileges_t =
  bool (*)(HANDLE, BOOL, PTOKEN_PRIVILEGES, DWORD, PTOKEN_PRIVILEGES, PDWORD);
}
#endif

#if defined(__linux__) && !defined(__ANDROID__)
    #include <sys/mman.h>
    #include <unistd.h>
#endif

namespace YaneuraOu {

// 通常のメモリ確保。alignmentされたsize[byte]のメモリを確保する。
// 💡 AlignedPtrから内部的に呼び出して使うために用意されている。この関数は直接呼び出されない。
void* std_aligned_alloc(usize alignment, usize size);
void  std_aligned_free(void* ptr);

// Memory aligned by page size, min alignment: 4096 bytes
// 大きなメモリの確保用のalloc/free。4096 bytes単位でalignmentされていることは保証されている。
// ⇨ この関数は置換表のメモリ確保のために使われる。
// 💡 Windowsのlarge pages allocの機能が使えるなら、それを用いる。
void* aligned_large_pages_alloc(usize size);
void  aligned_large_pages_free(void* mem);

// 現環境でlarge pagesが使えるか判定して返す。
bool has_large_pages();

// Frees memory which was placed there with placement new.
// Works for both single objects and arrays of unknown bound.

// placement new によって確保されたメモリを解放します。
// 単一オブジェクトにも、サイズ不明の配列にも対応します。

template<typename T, typename FREE_FUNC>
void memory_deleter(T* ptr, FREE_FUNC free_func) {
    if (!ptr)
        return;

    // Explicitly needed to call the destructor
    if constexpr (!std::is_trivially_destructible_v<T>)
        ptr->~T();

    free_func(ptr);
}

// Frees memory which was placed there with placement new.
// Works for both single objects and arrays of unknown bound.

// placement new を使って確保したメモリを解放します。
// 単一のオブジェクトにも、不定長の配列にも対応します。

template<typename T, typename FREE_FUNC>
void memory_deleter_array(T* ptr, FREE_FUNC free_func) {
    if (!ptr)
        return;


    // Move back on the pointer to where the size is allocated
	// 配列サイズが格納されている位置までポインタを戻します

	const usize array_offset = std::max(sizeof(usize), alignof(T));
    char*        raw_memory   = reinterpret_cast<char*>(ptr) - array_offset;

    if constexpr (!std::is_trivially_destructible_v<T>)
    {
        const usize size = *reinterpret_cast<usize*>(raw_memory);

        // Explicitly call the destructor for each element in reverse order
        for (usize i = size; i-- > 0;)
            ptr[i].~T();
    }

    free_func(raw_memory);
}

// Allocates memory for a single object and places it there with placement new
// 単一のオブジェクト用にメモリを確保し、placement new でその場所に配置します

template<typename T, typename ALLOC_FUNC, typename... Args>
inline std::enable_if_t<!std::is_array_v<T>, T*> memory_allocator(ALLOC_FUNC alloc_func,
                                                                  Args&&... args) {
    void* raw_memory = alloc_func(sizeof(T));
    ASSERT_ALIGNED(raw_memory, alignof(T));
    return new (raw_memory) T(std::forward<Args>(args)...);
}

// Allocates memory for an array of unknown bound and places it there with placement new
// サイズが不明な配列用にメモリを確保し、placement new でその場所に配置します

template<typename T, typename ALLOC_FUNC>
inline std::enable_if_t<std::is_array_v<T>, std::remove_extent_t<T>*>
memory_allocator(ALLOC_FUNC alloc_func, usize num) {
    using ElementType = std::remove_extent_t<T>;

    const usize array_offset = std::max(sizeof(usize), alignof(ElementType));

    // Save the array size in the memory location
    char* raw_memory =
      reinterpret_cast<char*>(alloc_func(array_offset + num * sizeof(ElementType)));
    ASSERT_ALIGNED(raw_memory, alignof(T));

    new (raw_memory) usize(num);

    for (usize i = 0; i < num; ++i)
        new (raw_memory + array_offset + i * sizeof(ElementType)) ElementType();

    // Need to return the pointer at the start of the array so that
    // the indexing in unique_ptr<T[]> works.

	// unique_ptr<T[]> での添字アクセスが正しく動作するように、
	// 配列の先頭のポインタを返す必要があります。

    return reinterpret_cast<ElementType*>(raw_memory + array_offset);
}

//
//
// aligned large page unique ptr
//
//

template<typename T>
struct LargePageDeleter {
    void operator()(T* ptr) const { return memory_deleter<T>(ptr, aligned_large_pages_free); }
};

template<typename T>
struct LargePageArrayDeleter {
    void operator()(T* ptr) const { return memory_deleter_array<T>(ptr, aligned_large_pages_free); }
};

template<typename T>
using LargePagePtr =
  std::conditional_t<std::is_array_v<T>,
                     std::unique_ptr<T, LargePageArrayDeleter<std::remove_extent_t<T>>>,
                     std::unique_ptr<T, LargePageDeleter<T>>>;

// make_unique_large_page for single objects
// 単一オブジェクト用の make_unique_large_page

template<typename T, typename... Args>
std::enable_if_t<!std::is_array_v<T>, LargePagePtr<T>> make_unique_large_page(Args&&... args) {
    static_assert(alignof(T) <= 4096,
                  "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");

    T* obj = memory_allocator<T>(aligned_large_pages_alloc, std::forward<Args>(args)...);

    return LargePagePtr<T>(obj);
}

// make_unique_large_page for arrays of unknown bound
// サイズが不明な配列用の make_unique_large_page

template<typename T>
std::enable_if_t<std::is_array_v<T>, LargePagePtr<T>> make_unique_large_page(usize num) {
    using ElementType = std::remove_extent_t<T>;

    static_assert(alignof(ElementType) <= 4096,
                  "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");

    ElementType* memory = memory_allocator<T>(aligned_large_pages_alloc, num);

    return LargePagePtr<T>(memory);
}

//
//
// aligned unique ptr
//
//

template<typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const { return memory_deleter<T>(ptr, std_aligned_free); }
};

template<typename T>
struct AlignedArrayDeleter {
    void operator()(T* ptr) const { return memory_deleter_array<T>(ptr, std_aligned_free); }
};

// alignされたunique_ptr<T>
template<typename T>
using AlignedPtr =
  std::conditional_t<std::is_array_v<T>,
                     std::unique_ptr<T, AlignedArrayDeleter<std::remove_extent_t<T>>>,
                     std::unique_ptr<T, AlignedDeleter<T>>>;

// make_unique_aligned for single objects
// 単一オブジェクト用の make_unique_aligned

template<typename T, typename... Args>
std::enable_if_t<!std::is_array_v<T>, AlignedPtr<T>> make_unique_aligned(Args&&... args) {
    const auto func = [](usize size) { return std_aligned_alloc(alignof(T), size); };
    T*         obj  = memory_allocator<T>(func, std::forward<Args>(args)...);

    return AlignedPtr<T>(obj);
}

// make_unique_aligned for arrays of unknown bound
// サイズが不明な配列用の make_unique_aligned

template<typename T>
std::enable_if_t<std::is_array_v<T>, AlignedPtr<T>> make_unique_aligned(usize num) {
    using ElementType = std::remove_extent_t<T>;

    const auto   func   = [](usize size) { return std_aligned_alloc(alignof(ElementType), size); };
    ElementType* memory = memory_allocator<T>(func, num);

    return AlignedPtr<T>(memory);
}


// Get the first aligned element of an array.
// ptr must point to an array of size at least `sizeof(T) * N + alignment` bytes,
// where N is the number of elements in the array.

// 配列の最初のアラインされた要素を取得します。
// ptr は、配列の要素数を N とした場合、少なくとも `sizeof(T) * N + alignment` バイト
// のサイズを持つ配列を指している必要があります。

template<uintptr_t Alignment, typename T>
T* align_ptr_up(T* ptr) {
    static_assert(alignof(T) < Alignment);

    const uintptr_t ptrint = reinterpret_cast<uintptr_t>(reinterpret_cast<char*>(ptr));
    return reinterpret_cast<T*>(
      reinterpret_cast<char*>((ptrint + (Alignment - 1)) / Alignment * Alignment));
}

#if defined(__linux__) && !defined(__ANDROID__)

// Allocate size bytes aligned to a 2 MB boundary using mmap.
// On success the returned pointer can be freed with munmap(ptr, size).
// mmapを使って2MiB境界に整列した領域を確保する。
// 成功時のポインターはmunmap(ptr, size)で解放する。
inline void* mmap_huge_aligned(usize size, int flags, int fd = -1, off_t offset = 0) {
    constexpr usize Alignment = 2 * 1024 * 1024;
    const long      pageSize  = sysconf(_SC_PAGESIZE);

    if (size >= Alignment && pageSize > 0)
    {
        const usize mappingSize =
          ((size + static_cast<usize>(pageSize) - 1) / static_cast<usize>(pageSize))
          * static_cast<usize>(pageSize);
        const usize reservationSize = mappingSize + Alignment;
        void*       reservation =
          mmap(nullptr, reservationSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (reservation != MAP_FAILED)
        {
            char* const base        = static_cast<char*>(reservation);
            char* const alignedBase = align_ptr_up<Alignment>(base);

            void* mapped =
              mmap(alignedBase, size, PROT_READ | PROT_WRITE, flags | MAP_FIXED, fd, offset);

            if (mapped != MAP_FAILED)
            {
                const usize prefixSize = static_cast<usize>(alignedBase - base);
                const usize suffixSize = reservationSize - prefixSize - mappingSize;
                if (prefixSize)
                    munmap(reservation, prefixSize);
                if (suffixSize)
                    munmap(alignedBase + mappingSize, suffixSize);
                return mapped;
            }

            munmap(reservation, reservationSize);
        }
    }

    return mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, fd, offset);
}

#endif

#if defined(_WIN32)

template<typename FuncYesT, typename FuncNoT>
auto windows_try_with_large_page_privileges([[maybe_unused]] FuncYesT&& fyes, FuncNoT&& fno) {

    #if !defined(_WIN64)
    return fno();
    #else

    HANDLE hProcessToken{};
    LUID   luid{};

    const usize largePageSize = GetLargePageMinimum();
    if (!largePageSize)
        return fno();

    // Dynamically link OpenProcessToken, LookupPrivilegeValue and AdjustTokenPrivileges

    HMODULE hAdvapi32 = GetModuleHandle(TEXT("advapi32.dll"));

    if (!hAdvapi32)
        hAdvapi32 = LoadLibrary(TEXT("advapi32.dll"));

    auto OpenProcessToken_f =
      OpenProcessToken_t((void (*)()) GetProcAddress(hAdvapi32, "OpenProcessToken"));
    if (!OpenProcessToken_f)
        return fno();
    auto LookupPrivilegeValueA_f =
      LookupPrivilegeValueA_t((void (*)()) GetProcAddress(hAdvapi32, "LookupPrivilegeValueA"));
    if (!LookupPrivilegeValueA_f)
        return fno();
    auto AdjustTokenPrivileges_f =
      AdjustTokenPrivileges_t((void (*)()) GetProcAddress(hAdvapi32, "AdjustTokenPrivileges"));
    if (!AdjustTokenPrivileges_f)
        return fno();

    // We need SeLockMemoryPrivilege, so try to enable it for the process

    if (!OpenProcessToken_f(  // OpenProcessToken()
          GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hProcessToken))
        return fno();

    if (!LookupPrivilegeValueA_f(nullptr, "SeLockMemoryPrivilege", &luid))
        return fno();

    TOKEN_PRIVILEGES tp{};
    TOKEN_PRIVILEGES prevTp{};
    DWORD            prevTpLen = 0;

    tp.PrivilegeCount           = 1;
    tp.Privileges[0].Luid       = luid;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    // Try to enable SeLockMemoryPrivilege. Note that even if AdjustTokenPrivileges()
    // succeeds, we still need to query GetLastError() to ensure that the privileges
    // were actually obtained.

    if (!AdjustTokenPrivileges_f(hProcessToken, FALSE, &tp, sizeof(TOKEN_PRIVILEGES), &prevTp,
                                 &prevTpLen)
        || GetLastError() != ERROR_SUCCESS)
        return fno();

    auto&& ret = fyes(largePageSize);

    // Privilege no longer needed, restore previous state
    AdjustTokenPrivileges_f(hProcessToken, FALSE, &prevTp, 0, nullptr, nullptr);

    CloseHandle(hProcessToken);

    return std::forward<decltype(ret)>(ret);

    #endif
}

#endif

} // namespace YaneuraOu

#endif  // #ifndef MEMORY_H_INCLUDED
