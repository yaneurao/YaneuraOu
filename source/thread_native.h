#ifndef THREAD_NATIVE_H_INCLUDED
#define THREAD_NATIVE_H_INCLUDED

// ----------------------------
//     thread wrapper
//     スレッド生成のラッパー
// ----------------------------

#ifdef _MSC_VER
    #include <thread>
#else
    #include <pthread.h>
    #include <cstdlib>
    #include <cstring>
    #include <functional>
    #include <iostream>
    #include <tuple>
    #include <utility>

    #include "misc.h"
#endif

namespace YaneuraOu {

struct NativeThreadOptions {
    bool largeStack{};

    NativeThreadOptions& setLargeStack(bool value) {
        largeStack = value;
        return *this;
    }
};

#ifdef _MSC_VER

// MSVC-compatible toolchains use std::thread because they do not provide
// pthreads by default. On all other platforms, pthreads is required and used.
// MSVC互換のツールチェーンは標準ではpthreadsを提供しないため、std::threadを使う。
// それ以外のプラットフォームではpthreadsを必須として使用する。

using NativeThread = std::thread;

template<class Function, class... Args>
NativeThread create_native_thread(NativeThreadOptions options, Function&& fun, Args&&... args) {
    // TODO: implement fallible thread creation for MSVC
    // TODO: MSVCでもスレッド生成の失敗を呼び出し側で検出できるようにする。
    (void) options;
    return NativeThread(std::forward<Function>(fun), std::forward<Args>(args)...);
}

#else

// type erase F
// 関数オブジェクトFの型を消去し、共通のインターフェースで呼び出せるようにする。
struct ThreadCallableBase {
    virtual ~ThreadCallableBase() = default;
    virtual void run()            = 0;
};

template<typename F, typename... Args>
struct ThreadCallable final: ThreadCallableBase {
    F                   func_;
    std::tuple<Args...> args_;
    ThreadCallable(F&& f, Args&&... args) :
        func_(std::forward<F>(f)),
        args_(std::make_tuple(std::forward<Args>(args)...)) {}
    void run() override { std::apply(func_, args_); }
};

// On OSX threads other than the main thread are created with a reduced stack
// size of 512KB by default, this is too low for deep searches, which require
// somewhat more than 1MB stack, so adjust it to TH_STACK_SIZE.
// The implementation calls pthread_create() with the stack size parameter
// equal to the Linux 8MB default, on platforms that support it.
// OSXではメインスレッド以外のスタックサイズが標準で512KBと小さく、
// 1MBを超えるスタックを必要とする深い探索には足りないため、TH_STACK_SIZEに変更する。
// 対応するプラットフォームでは、Linuxの標準値と同じ8MBのスタックサイズを指定して
// pthread_create()を呼び出す。

// スレッドのデフォルトスタックサイズが小さい環境で、
// 少なくとも8MB確保するためのthread生成のためのwrapper

class NativeThread {
    pthread_t thread;
    bool      running_ = false;

	// 少なくともこのサイズのスタック領域を確保したい
    static constexpr usize TH_STACK_SIZE = 8 * 1024 * 1024;

    void start(NativeThreadOptions options, ThreadCallableBase* func) {
        pthread_attr_t attr_storage, *attr = &attr_storage;
        pthread_attr_init(attr);
        if (options.largeStack)
            pthread_attr_setstacksize(attr, TH_STACK_SIZE);

        auto start_routine = [](void* ptr) -> void* {
            auto f = reinterpret_cast<ThreadCallableBase*>(ptr);
            // Call the function
            // 関数を呼び出す。
            f->run();
            delete f;
            return nullptr;
        };

        const int rc = pthread_create(&thread, attr, start_routine, func);
        pthread_attr_destroy(attr);

        if (rc != 0)
            delete func;
        else
            running_ = true;
    }

   public:
    NativeThread()                               = default;
    NativeThread(const NativeThread&)            = delete;
    NativeThread& operator=(const NativeThread&) = delete;

    NativeThread(NativeThread&& other) noexcept {
        thread         = other.thread;
        running_       = other.running_;
        other.running_ = false;
    }

    NativeThread& operator=(NativeThread&& other) noexcept {
        if (&other != this)
        {
            assert(!running_ && "Thread was not joined");
            thread         = other.thread;
            running_       = other.running_;
            other.running_ = false;
        }
        return *this;
    }

    ~NativeThread() { assert(!running_ && "Thread was not joined"); }

    bool joinable() const { return running_; }
    void join() {
        if (running_)
        {
            pthread_join(thread, nullptr);
            running_ = false;
        }
    }

    template<class Function, class... Args>
    friend NativeThread create_native_thread(NativeThreadOptions, Function&&, Args&&...);
};

template<class Function, class... Args>
inline NativeThread
create_native_thread(NativeThreadOptions options, Function&& fun, Args&&... args) {
    NativeThread thread{};
    using Callable = ThreadCallable<std::decay_t<Function>, std::decay_t<Args>...>;
    if (auto func =
          new (std::nothrow) Callable(std::forward<Function>(fun), std::forward<Args>(args)...))
        thread.start(options, func);
    return thread;
}

#endif  // _MSC_VER

}  // namespace YaneuraOu

#endif  // #ifndef THREAD_NATIVE_H_INCLUDED
