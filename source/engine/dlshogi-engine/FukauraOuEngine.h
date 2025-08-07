#ifndef __FUKAURAOU_ENGINE_H_INCLUDED__
#define __FUKAURAOU_ENGINE_H_INCLUDED__
#include "../../config.h"

#if defined(YANEURAOU_ENGINE_DEEP)

#include "../../engine.h"
#include "../../book/book.h"

#include "dlshogi_types.h"
#include "FukauraOuSearch.h"

namespace dlshogi {

/*
	この探索部は、dlshogiのソースコードを参考にさせていただいています。🙇
	DeepLearningShogi GitHub : https://github.com/TadaoYamaoka/DeepLearningShogi
*/

// 残り時間チェックを行ったり、main threadからのみアクセスされる探索manager
// 💡 Stockfishの同名のclassとほぼ同じ設計。Stockfishのsearch.hにあるSearchManagerも参考にすること。
// 🤔 YaneuraOuEngineの1インスタンスに対して、SearchManagerが1インスタンスあれば良いので、
//     やねうら王では、YaneuraOuEngineのメンバーとして持たせることにする。
namespace Search {
class SearchManager {


    // 探索オプション
    SearchOptions search_options;
};
}

class FukauraOuEngine : public YaneuraOu::Engine
{
	// エンジンoptionを生やす。
    virtual void add_options() override;

	// "isready"コマンド応答。
	virtual void isready() override;

	// エンジン作者名の変更。
	virtual std::string get_engine_author() const override;

    // 定跡の指し手を選択するモジュール。
    YaneuraOu::Book::BookMoveSelector book;

	// 探索マネージャー。
	Search::SearchManager manager;

    // Stockfish風のコードが書けるように同名のメソッドを定義しておく。
    Search::SearchManager* main_manager() { return &manager; }

}; // class FukauraOuEngine

} // namespace dlshogi

#endif // defined(YANEURAOU_ENGINE_DEEP)

#endif // ndef __FUKAURAOU_ENGINE_H_INCLUDED__
