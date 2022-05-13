#ifndef PROCESS_NEGOTIATOR_H_INCLUDED
#define PROCESS_NEGOTIATOR_H_INCLUDED

#include "../../config.h"

#if defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))

#include <string>
#include <memory>

// ---------------------------------------
//          ProcessNegotiator
// ---------------------------------------

// 子プロセスを実行して、子プロセスの標準入出力をリダイレクトするのをお手伝いするクラス。
// 1つの子プロセスのつき、1つのProcessNegotiatorの instance が必要。
// 
// 親プロセス(このプログラム)の終了時に、子プロセスを自動的に終了させたいが、それは簡単ではない。
// アプリケーションが終了するときに、子プロセスを自動的に終了させる方法 : https://qiita.com/kenichiuda/items/3079ab93dae564dd5d17
// 親プロセスは必ず quit コマンドか何かで正常に終了させるものとする。
//
// また、このclass自体は、worker threadを持たない。
// send()/receive()を定期的に親classから呼び出すものとする。
//
// また、std::mutex()やstd::atomicなどを持っているとstd::move()が出来ないので、
// std::unique_ptrで管理する。
// 
struct IProcessNegotiator
{
	// 子プロセスの実行
	// workingDirectory : エンジンを実行する時の作業ディレクトリ 
	// app_path         : エンジンの実行ファイルのpath (.batファイルでも可) 絶対pathで。
	virtual void connect(const std::string& workingDirectory , const std::string& app_path) = 0;

	// 子プロセスへの接続を切断する。
	virtual void disconnect() = 0;

	// 接続されている子プロセスから1行読み込む。
	// 子プロセスと切断されていることがわかったら、以降is_terminated()==trueを返すようになる。
	// (この関数のなかで切断されているかをチェックしている。)
	virtual std::string receive() = 0;

	// 接続されている子プロセス(の標準入力)に1行送る。改行は自動的に付与される。
	// sendでは、terminatedの判定はしていない。
	virtual bool send(const std::string& message) = 0;

	// プロセスの終了判定
	virtual bool is_terminated() const = 0;

	// エンジンの実行path
	// これはconnectの直後に設定され、そのあとは変更されない。connect以降でしか
	// このプロパティにはアクセスしないので同期は問題とならない。
	virtual std::string get_engine_path() const = 0;

	virtual ~IProcessNegotiator(){}
};

struct ProcessNegotiator : public IProcessNegotiator
{
	// 子プロセスの実行
	// workingDirectory : エンジンを実行する時の作業ディレクトリ("engines/"相対で指定)
	// app_path         : エンジンの実行ファイルのpath (.batファイルでも可) workingDirectory相対で指定。
	virtual void connect(const std::string& workingDirectory , const std::string& app_path)
	{ ptr->connect(workingDirectory, app_path);}

	// 子プロセスへの接続を切断する。
	virtual void disconnect() { ptr->disconnect(); }

	// 接続されている子プロセスから1行読み込む。
	// 子プロセスと切断されていることがわかったら、以降is_terminated()==trueを返すようになる。
	// (この関数のなかで切断されているかをチェックしている。)
	virtual std::string receive() { return ptr->receive(); }

	// 接続されている子プロセス(の標準入力)に1行送る。改行は自動的に付与される。
	// sendでは、terminatedの判定はしていない。
	virtual bool send(const std::string& message) { return ptr->send(message); }

	// プロセスの終了判定
	virtual bool is_terminated() const { return ptr->is_terminated(); }

	// エンジンの実行path
	// これはconnectの直後に設定され、そのあとは変更されない。connect以降でしか
	// このプロパティにはアクセスしないので同期は問題とならない。
	virtual std::string get_engine_path() const { return ptr->get_engine_path(); }

	ProcessNegotiator();

protected:
	std::unique_ptr<IProcessNegotiator> ptr;
};

#endif // defined(USE_YO_CLUSTER) && (defined(YANEURAOU_ENGINE_DEEP) || defined(YANEURAOU_ENGINE_NNUE))
#endif // ndef PROCESS_NEGOTIATOR_H_INCLUDED
