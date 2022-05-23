#ifndef ENGINE_NEGOTIATOR_H_INCLUDED
#define ENGINE_NEGOTIATOR_H_INCLUDED

#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include "ClusterCommon.h"
#include "ProcessNegotiator.h"

namespace YaneuraouTheCluster
{
	// EngineNegotiatorは、
	// エンジンとやりとりするためのクラス
	//
	// 状態遷移を指示するメソッドだけをpublicにしてあるので、
	// 外部からはこれを呼び出すことで状態管理を簡単化する考え。

	// EngineNegotiatorのinterface
	class IEngineNegotiator
	{
	public:
		// [main thread]
		// エンジンを起動する。
		// このメソッドは起動直後に、最初に一度だけmain threadから呼び出す。
		// (これとdisconnect以外のメソッドは observer が生成したスレッドから呼び出される)
		// path      : エンジンの実行path。 ("engines/" 相対で指定する)
		// engine_id : このengineのID。これでエンジンを識別する。親クラスがuniqueになるように付与する。 
		virtual void connect(const std::string& path, size_t engine_id_) = 0;

		virtual ~IEngineNegotiator(){}

		// -------------------------------------------------------
		//    Property
		// -------------------------------------------------------

		// [main thread][receive thread]
		// プロセスの終了判定
		virtual bool is_terminated() const = 0;

		// [receive thread]
		// エンジンが対局中であるか。
		// "usinewgame"～"gameover"の間であるか。
		virtual bool is_in_game() const = 0;

		// エンジンIDを取得する。
		// これは、connect()の時に親クラスから渡されたもの。
		virtual size_t get_engine_id() const = 0;

		// Messageを解釈してエンジンに送信する。
		// 結果はすぐに返る。親クラス(ClusterObserver)の送受信用スレッドから呼び出す。
		// send()とreceive()とは同時に呼び出されない。
		// (親クラスの送受信用のスレッドは、送受信のために1つしかスレッドが走っていないため。)
		virtual void send(Message message) = 0;

		// エンジンからメッセージを受信して、dispatchする。
		// このメソッドは親クラス(ClusterObserver)の送受信用スレッドから定期的に呼び出される。(呼び出さなければならない)
		// メッセージを一つでも受信したならtrueを返す。
		virtual bool receive() = 0;

		// 現在、"go","go ponder"によって探索中の局面。
		// ただし、"go"に対してエンジンが"bestmove"を返したあとも
		// その探索していた局面のsfenを、このメソッドで取得できる。
		virtual std::string get_searching_sfen() const = 0;

		// 直前のコマンドはponderhitであるか？
		// (現在is_state_go()==true (探索中)であるとして、
		// 送られてきたコマンドが"go"なのか"ponderhit"なのかを区別するのに使う)
		virtual bool is_ponderhit() const = 0;

		// エンジン側から受け取った"bestmove XX ponder YY"を返す。
		// 一度このメソッドを呼び出すと、次以降は(エンジン側からさらに"bestmove XX ponder YY"を受信するまで)空の文字列が返る。
		// つまりこれは、size = 1 の PC-queueとみなしている。
		virtual std::string get_bestmove() = 0;

		// エンジンの状態を取得する。
		// エンジンの状態は、send() , receive()でしか変化しないから、これで取得中に変化することはない。
		virtual EngineState get_state() const = 0;
	};

	// EngineNegotiatorの入れ物。
	// std::move()できて欲しいので、Pimpl型にする。
	class EngineNegotiator : public IEngineNegotiator
	{
	public:
		virtual void        connect(const std::string& path, size_t engine_id_) {        ptr->connect(path, engine_id_); }
		virtual bool        is_terminated() const                               { return ptr->is_terminated();           }
		virtual bool        is_in_game()    const                               { return ptr->is_in_game();              }
		virtual size_t      get_engine_id() const                               { return ptr->get_engine_id();           }
		virtual void        send(Message message)                               {        ptr->send(message);             }
		virtual bool        receive()                                           { return ptr->receive();                 }
		virtual std::string get_searching_sfen() const                          { return ptr->get_searching_sfen();      }
		virtual bool        is_ponderhit() const                                { return ptr->is_ponderhit();            }
		virtual std::string get_bestmove()                                      { return ptr->get_bestmove();            }
		virtual EngineState get_state() const                                   { return ptr->get_state();               }

		EngineNegotiator();
		EngineNegotiator(EngineNegotiator&&) = default; // default move constructor
		virtual ~EngineNegotiator(){}

		// --- エンジンの状態を判定するhelper property

		// 現在のstateが"isready"の送信待ちの状態か？
		bool is_waiting_isready() const { return get_state() == EngineState::WAIT_ISREADY; }

		// エンジンが対局中で休憩モード(bestmoveを返したあとなど)に入っているのか？
		bool is_idle_in_game()    const { return get_state() == EngineState::IDLE_IN_GAME; }

		// GO_PONDER状態なのか？
		bool is_state_go_ponder() const { return get_state() == EngineState::GO_PONDER; }

		// GO状態なのか？
		bool is_state_go() const { return get_state() == EngineState::GO; }

	protected:
		std::unique_ptr<IEngineNegotiator> ptr;
	};
}

#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
#endif
