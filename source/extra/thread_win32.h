#ifndef THREAD_WIN32_H_INCLUDED
#define THREAD_WIN32_H_INCLUDED

// mingwやgccによるSTLのthread libraryは、Windows用にクロスコンパイルされるときに
// libwinpthreadに依存する。目下のところ、libwinpthreadはmutexをWindows セマフォの
// トップに直接実装してある。 セマフォはカーネルオブジェクトであり、lockやunlockのために
// kernel modeへの移行を必要とするので、interlocked operationに比較して非常に遅い。
// (bench testで30%程度遅い) この問題を回避するために、我々は低レベルのWin32 callの
// wrapperを定義した。我々は、Windows XPやそれより古いバージョンをサポートするために
// critical sectionを用いる。不運にも、cond_wait()はunlock()とWaitForSingleObject()で
// race状態になるが、それでもSRW lockと同様のスピードである。

// 注記)
// Stockfish、新しくなってから探索部はlazy SMPで、Mutexで同期を取る必要がなく(USIへの出力の時ぐらい？)
// この部分のパフォーマンスが性能に及ぼす影響はほとんどない。

#include <condition_variable>
#include <mutex>

#if defined(_WIN32) && !defined(_MSC_VER)

#ifndef NOMINMAX
#  define NOMINMAX // Disable macros min() and max()
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#undef NOMINMAX

// MutexとConditionVariable構造体は、低レベルなロック機構のwrapperであり、
// C++11の同名のクラスと同じように使えるようにデザインされている。

struct Mutex {
  Mutex() { InitializeCriticalSection(&cs); }
 ~Mutex() { DeleteCriticalSection(&cs); }
  void lock() { EnterCriticalSection(&cs); }
  void unlock() { LeaveCriticalSection(&cs); }

private:
  CRITICAL_SECTION cs;
};

typedef std::condition_variable_any ConditionVariable;

#else // Default case: use STL classes

typedef std::mutex Mutex;
typedef std::condition_variable ConditionVariable;

#endif

#endif // #ifndef THREAD_WIN32_H_INCLUDED
