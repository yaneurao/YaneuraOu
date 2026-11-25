// Positionの単体テスト。局面処理本体との依存を分離する。
#include "../position.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "../attacks.h"
#include "../book/book.h"
#include "../evaluate.h"
#include "../misc.h"
#include "../movegen.h"
#include "../usi.h"
#include "unit_test.h"

namespace YaneuraOu {

using std::string;
using namespace Attacks;

// ----------------------------------
//			UnitTest
// ----------------------------------

namespace {
	// performance test
	// ある局面から、全合法手を生成して depth深さまで辿り、局面数がいくらあったかを返す。
	u64 perft(Position& pos, Depth depth)
	{

		u64 cnt, nodes = 0;

		auto ml = MoveList<LEGAL_ALL>(pos);
		if (depth == 1)
			return ml.size();

		const bool leaf = (depth == 2);
		for (const auto& m : ml)
		{
			pos.do_move(m);
			cnt = leaf ? MoveList<LEGAL_ALL>(pos).size() : perft(pos, depth - 1);
			nodes += cnt;
			pos.undo_move(m);
		}
		return nodes;
	};
}

void Position::UnitTest(Test::UnitTester& tester, IEngine& engine) {
    auto section1 = tester.section("Position");

    {
        auto section = tester.section("StateInfo ring");
        auto root = std::make_unique<Position>();
        auto copy = std::make_unique<Position>();
        const int oldWindow = max_repetition_ply;
        tester.test("reject negative history window", !root->set_max_repetition_ply(-1)
                                                     && max_repetition_ply == oldWindow);
#if !defined(YANEURAOU_MATE_ENGINE) && !defined(TANUKI_MATE_ENGINE)
        tester.test("reject oversized history window",
                    !root->set_max_repetition_ply(MAX_HIST) && max_repetition_ply == oldWindow);
#endif
        tester.test("accept history window", root->set_max_repetition_ply(oldWindow));
        root->set_hirate();
        const char* cycle[] = {"5i6h", "5a6b", "6h5i", "6b5a"};
        for (int i = 0; i < int(MAX_HIST) * 2 + 12; ++i)
            root->do_move(USIEngine::to_move(*root, cycle[i % 4]));
        const auto rootSfen = root->sfen();
        const auto rootKey = root->key();
        const auto rootRep = root->is_repetition(16);
        copy->copy_from(*root);
        tester.test("private root state", copy->st != root->st);
        tester.test("wrapped repetition", rootRep == REPETITION_DRAW
                                         && copy->is_repetition(16) == rootRep);
        bool ok = true;
        for (int run = 0; run < 4; ++run) {
            copy->reset_to_root(*root);
            std::vector<Move> moves;
            for (int i = 0; i < MAX_PLY; ++i) {
                Move m = USIEngine::to_move(*copy, cycle[i % 4]);
                copy->do_move(m);
                moves.push_back(m);
                ok &= copy->pos_is_ok();
            }
            while (!moves.empty()) {
                copy->undo_move(moves.back());
                moves.pop_back();
            }
            ok &= copy->sfen() == rootSfen && copy->key() == rootKey
               && copy->is_repetition(16) == rootRep;
            copy->do_null_move();
            copy->undo_null_move();
            ok &= copy->key() == rootKey;
            // Leave a rollout advanced; reset_to_root must recover without undo.
            // プレイアウトを戻さず進めたままにし、次のresetだけで復元できるか確認する。
            copy->do_move(USIEngine::to_move(*copy, cycle[0]));
        }
        copy->reset_to_root(*root);
        tester.test("rollout reset and do/undo", ok && copy->sfen() == rootSfen);
        tester.test("source root unchanged", root->key() == rootKey && root->sfen() == rootSfen);
    }


    Position  pos;


    // 任意局面での初期化。
    auto pos_init = [&](const std::string& sfen_) { pos.set(sfen_); };

    // 平手初期化
    auto hirate_init = [&] { pos.set_hirate(); };
    // 2枚落ち初期化
    auto handi2_sfen = "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
    auto handi2_init = [&] { pos.set(handi2_sfen); };

    // 4枚落ち初期化
    auto handi4_sfen = "1nsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
    auto handi4_init = [&] { pos.set(handi4_sfen); };

    // 指し手生成祭りの局面
    auto matsuri_sfen = "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1";
    auto matsuri_init = [&] { pos.set(matsuri_sfen); };

    Move16 m16;
    Move   m;

    // to_move() のテスト
    {
        auto section2 = tester.section("to_move()");

        // 平手初期化
        hirate_init();

        // is_ok(m) == falseな指し手に対して、to_move()がその指し手をそのまま返すことを保証する。
        tester.test("MOVE_NONE", pos.to_move(Move16::none()) == Move::none());
        tester.test("MOVE_WIN", pos.to_move(Move16::win()) == Move::win());
        tester.test("MOVE_NULL", pos.to_move(Move16::null()) == Move::null());

        // 88の角を22に不成で移動。(非合法手) 移動後の駒は先手の角。
        m16 = make_move16(SQ_88, SQ_22);
        tester.test("make_move16(SQ_88, SQ_22)",
                    pos.to_move(m16) == (Move) ((u32) m16.to_u16() + (u32) (B_BISHOP << 16)));

        // 88の角を22に成る移動。(非合法手) 移動後の駒は先手の馬。
        m16 = make_move_promote16(SQ_88, SQ_22);
        tester.test("make_move_promote16(SQ_88, SQ_22)",
                    pos.to_move(m16) == (Move) ((u32) m16.to_u16() + (u32) (B_HORSE << 16)));

        // 22の角を88に不成で移動。(非合法手) 移動後の駒は後手の角。
        m16 = make_move16(SQ_22, SQ_88);
        tester.test("make_move16(SQ_22, SQ_88)", pos.to_move(m16) == Move::none());

        // 22の角を88に成る移動。(非合法手) 移動後の駒は後手の馬。
        m16 = make_move_promote16(SQ_22, SQ_88);
        tester.test("make_move_promote16(SQ_22, SQ_88)", pos.to_move(m16) == Move::none());

        matsuri_init();
        m16 = make_move_drop16(GOLD, SQ_55);
        tester.test("make_move_drop(SQ_55,GOLD)",
                    pos.to_move(m16) == (Move) ((u32) m16.to_u16() + (u32) (W_GOLD << 16)));
    }

    // pseudo_legal() , legal() のテスト
    {
        auto section2 = tester.section("legality");

        // 平手初期化
        hirate_init();

        // 77の歩を76に移動。(合法手)
        // これはpseudo_legalではある。
        m16 = make_move16(SQ_77, SQ_76);
        m   = pos.to_move(m16);
        tester.test("make_move(SQ_77, SQ_76) is pseudo_legal == true",
                    pos.pseudo_legal(m, true) == true);



        // 88の先手の角を22に移動。これは途中に駒があって移動できないのでpseudo_legalではない。
        // (pseudo_legalは、その駒が移動できる(移動先の升にその駒の利きがある)ことを保証する)
        m16 = make_move16(SQ_88, SQ_22);
        m   = pos.to_move(m16);
        tester.test("make_move(SQ_88, SQ_22) is pseudo_legal == false",
                    pos.pseudo_legal(m, true) == false);
    }

    // attacks_bb() のテスト
    {
        auto section2 = tester.section("attacks_bb");
        hirate_init();
        tester.test("attacks_by<BLACK,PAWN>", pos.attacks_by<BLACK, PAWN>() == BB_Table::Rank6BB);
        tester.test("attacks_by<WHITE,PAWN>", pos.attacks_by<WHITE, PAWN>() == BB_Table::Rank4BB);
        tester.test("attacks_by<BLACK,KNIGHT>",
                    pos.attacks_by<BLACK, KNIGHT>()
                      == (square_bb(SQ_97) | square_bb(SQ_77) | square_bb(SQ_37) | square_bb(SQ_17)));
        tester.test("attacks_by<BLACK,GOLDS>",
                    pos.attacks_by<BLACK, GOLDS>()
                      == (goldEffect<BLACK>(SQ_69) | goldEffect<BLACK>(SQ_49)));
        tester.test("attacks_by<WHITE,GOLDS>",
                    pos.attacks_by<WHITE, GOLDS>()
                      == (goldEffect<WHITE>(SQ_61) | goldEffect<WHITE>(SQ_41)));
    }

#if defined(ENABLE_QUICK_DRAW)
	// ENABLE_QUICK_DRAWを定義していない時は、4回出現しないと千日手として扱わない。

    // 千日手検出のテスト
    {
        auto section2 = tester.section("is_repetition");


        // 4手前の局面に戻っているパターン
        BookTools::feed_position_string(pos, "startpos moves 5i5h 5a5b 5h5i 5b5a");

        int  found_ply;
        auto rep = pos.is_repetition(16, found_ply);

        tester.test("REPETITION_DRAW", rep == REPETITION_DRAW && found_ply == 4);

        // 初期局面から先手の飛車が46,後手玉が54に移動している局面。
        // ここから56飛(46)→44玉(54)→46飛(56)→54玉(44)で先手の反則負け
        pos_init("lnsg1gsnl/1r5b1/ppppppppp/4k4/9/5R3/PPPPPPPPP/1B7/LNSGKGSNL b - 1");

        m = pos.to_move(make_move16(SQ_46, SQ_56));
        pos.do_move(m);
        m = pos.to_move(make_move16(SQ_54, SQ_44));
        pos.do_move(m);
        m = pos.to_move(make_move16(SQ_56, SQ_46));
        pos.do_move(m);
        m = pos.to_move(make_move16(SQ_44, SQ_54));
        pos.do_move(m);

        // いま先手番であり、先手の反則負けが確定しているはず。
        auto draw_value = pos.is_repetition();
        tester.test("REPETITION_LOSE", draw_value == REPETITION_LOSE);

        // 初期局面から先手の飛車が56,後手玉が54に移動している局面。(王手がかかっていて後手番)
        // ここから44玉(54)→46飛(56)→54玉(44)→56飛(46)で(後手番において)先手の反則負け
        pos_init("lnsg1gsnl/1r5b1/ppppppppp/4k4/9/4R4/PPPPPPPPP/1B7/LNSGKGSNL w - 1");

        m = pos.to_move(make_move16(SQ_54, SQ_44));
        pos.do_move(m);
        m = pos.to_move(make_move16(SQ_56, SQ_46));
        pos.do_move(m);
        m = pos.to_move(make_move16(SQ_44, SQ_54));
        pos.do_move(m);
        m = pos.to_move(make_move16(SQ_46, SQ_56));
        pos.do_move(m);

        draw_value = pos.is_repetition();
        tester.test("REPETITION_WIN", draw_value == REPETITION_WIN);
    }
#endif

    // 入玉のテスト
    {
        auto section2 = tester.section("EnteringKing");

        {
            // 27点法の入玉可能点数 平手 : 先手=28,後手=27
            auto section3 = tester.section("EKR_27_POINT");

            hirate_init();
            pos.set_ekr(EKR_27_POINT);

            tester.test("hirate",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 27);

            // 2枚落ち初期化 , 駒落ち対応でないなら、この時も 先手=28,後手=27
            handi2_init();
            pos.set_ekr(EKR_27_POINT);

            tester.test("handi2",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 27);
        }

        {
            // 24点法の入玉可能点数 平手 : 先手=31,後手=31
            auto section3 = tester.section("EKR_24_POINT");

            hirate_init();
            pos.set_ekr(EKR_24_POINT);

            tester.test("hirate",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 31);

            // 2枚落ち初期化 , 駒落ち対応でないなら、この時も 先手=31,後手=31
            handi2_init();
            pos.set_ekr(EKR_24_POINT);

			tester.test("handi2",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 31);
        }

        {
            // 27点法の入玉可能点数 平手 : 先手=28,後手=27
            auto section3 = tester.section("EKR_27_POINT_H");

            hirate_init();
            pos.set_ekr(EKR_27_POINT_H);

            tester.test("hirate",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 27);

            // 2枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=17,下手(BLACK)=28
            handi2_init();
            pos.set_ekr(EKR_27_POINT_H);
            tester.test("handi2",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 17);

            // 4枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=15,下手(BLACK)=28
            handi4_init();
            pos.set_ekr(EKR_27_POINT_H);
            tester.test("handi4",
                        pos.enteringKingPoint[BLACK] == 28 && pos.enteringKingPoint[WHITE] == 15);
        }

        {
            // 24点法の入玉可能点数 平手 : 先手=31,後手=31
            auto section3 = tester.section("EKR_24_POINT_H");

            hirate_init();
            pos.set_ekr(EKR_24_POINT_H);

            tester.test("hirate",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 31);

            // 2枚落ち初期化 , 駒落ち対応なのでこの時 上手(WHITE)=21,下手(BLACK)=31
            handi2_init();
            pos.set_ekr(EKR_24_POINT_H);
            tester.test("handi2",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 21);

            // 4枚落ち初期化 , 駒落ち対応なので この時 上手(WHITE)=19,下手(BLACK)=31
            handi4_init();
            pos.set_ekr(EKR_24_POINT_H);
            tester.test("handi4",
                        pos.enteringKingPoint[BLACK] == 31 && pos.enteringKingPoint[WHITE] == 19);
        }
    }

    {
        // 指し手生成のテスト
        auto section2 = tester.section("GenMove");

        {
            // 23歩不成ができ、かつ、23歩不成では駒の捕獲にはならない局面。
            pos_init("lnsgk1snl/1r4g2/p1ppppb1p/6pP1/7R1/2P6/P2PPPP1P/1SG6/LN2KGSNL b BP2p 21");
            Move move1 = make_move(SQ_24, SQ_23, B_PAWN);
            Move move2 = make_move_promote(SQ_24, SQ_23, B_PAWN);

            Move move_buf[MAX_MOVES], *move_last;
            // move_bufからmove_lastのなかにmoveがあるかを探す。あればtrueを返す。
            auto find_move = [&](Move m) {
                for (Move* em = &move_buf[0]; em != move_last; ++em)
                    if (*em == m)
                        return true;
                return false;
            };

            bool all = true;

            move_last = generate<QUIETS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<CAPTURES>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<NON_EVASIONS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<NON_EVASIONS_ALL>(pos, move_buf);
            all &= find_move(move1);
            all &= find_move(move2);

            move_last = generate<CAPTURES>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS_ALL>(pos, move_buf);
            all &= find_move(
              move1);  // 歩の不成はこちらに含めることになった。(movegenの実装の修正が難しいので)
            all &= find_move(move2);

            move_last = generate<QUIETS_PRO_MINUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<QUIETS_PRO_MINUS_ALL>(pos, move_buf);
            all &= !find_move(move1);  // 歩の不成はこちらには含まれていないので注意。
            all &= !find_move(move2);

            tester.test("pawn's unpromoted move", all);

            // 23角不成で5手詰め
            // https://github.com/yaneurao/YaneuraOu/issues/257
            pos_init("5B1n1/8k/6Rpp/9/9/9/1+p7/9/K8 b rb4g4s3n4l15p 1");
            // 23角不成(41)と23角成(41)
            move1 = make_move(SQ_41, SQ_23, B_BISHOP);
            move2 = make_move_promote(SQ_41, SQ_23, B_BISHOP);
            all   = true;

            move_last = generate<LEGAL_ALL>(pos, move_buf);
            all &= find_move(move1);
            all &= find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<QUIETS_PRO_MINUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS>(pos, move_buf);
            all &= !find_move(move1);
            all &= find_move(move2);

            move_last = generate<QUIETS_PRO_MINUS_ALL>(pos, move_buf);
            all &= !find_move(move1);
            all &= !find_move(move2);

            move_last = generate<CAPTURES_PRO_PLUS_ALL>(pos, move_buf);
            all &= find_move(move1);
            all &= find_move(move2);

            tester.test("bishop's unpromoted move", all);
        }
    }
    {
        auto section = tester.section("LegalMoveGeneration");

        // ピンされた金は玉と飛車を結ぶ線上には動けるが、横には動けない。
        pos_init("4k4/9/9/9/4r4/9/4G4/9/4K4 b - 1");
        auto quiets = MoveList<LEGAL_ALL>(pos);
        tester.test("pinned gold",
                    quiets.contains(make_move(SQ_57, SQ_56, B_GOLD))
                      && !quiets.contains(make_move(SQ_57, SQ_47, B_GOLD)));

        // 王手している飛車の利きへ玉を動かす手は生成せず、合駒は生成する。
        pos_init("4k4/9/9/9/4r4/9/9/9/4K4 b G 1");
        auto evasions = MoveList<LEGAL_ALL>(pos);
        tester.test("king safety and interposition",
                    !evasions.contains(make_move(SQ_59, SQ_58, B_KING))
                      && evasions.contains(make_move_drop(GOLD, SQ_58, BLACK)));

        // 両王手は玉の移動だけで回避する。手駒があっても合駒は生成しない。
        pos_init("4k4/9/9/9/4r4/9/6b2/9/4K4 b G 1");
        auto doubleCheck = MoveList<LEGAL_ALL>(pos);
        tester.test("double check",
                    pos.checkers().more_than_one() && doubleCheck.size() != 0
                      && std::all_of(doubleCheck.begin(), doubleCheck.end(), [&](Move move) {
                             return !move.is_drop() && type_of(pos.moved_piece(move)) == KING
                                 && pos.pseudo_legal_s<true>(move) && pos.legal(move);
                         }));

        // 王手となる歩打ちでも、打ち歩詰めならquiet/王手生成の両方から除く。
        pos_init("4k4/9/3LGL3/9/9/9/9/9/4K4 b P 1");
        const Move pawnMate = make_move_drop(PAWN, SQ_52, BLACK);
        tester.test("illegal pawn-drop mate",
                    !MoveList<QUIETS_ALL>(pos).contains(pawnMate)
                      && !MoveList<CHECKS_ALL>(pos).contains(pawnMate)
                      && MoveList<QUIETS_ALL>(pos).contains(make_move_drop(PAWN, SQ_42, BLACK)));
    }

    {
        // see_ge()のテスト
        auto      section = tester.section("see_ge");


        // 平手初期化
        hirate_init();

        // see_geのしきい値がv以下の時だけtrueが返ってくるかをテストする。
        // つまりはsee値がvであるかをテストする関数。
        auto see_ge_th = [&](int v) {
            Value th     = Value(v);
            bool  all_ok = true;
            all_ok &= pos.see_ge(m, th);       // see_ge(m, th) == true
            all_ok &= !pos.see_ge(m, th + 1);  // see_ge(m,  1) == false
            all_ok &= pos.see_ge(m, th - 1);   // see_ge(m, -1) == true
            return all_ok;
        };

        // 76歩、34歩の局面を作る。
        m = pos.to_move(make_move16(SQ_77, SQ_76));
        pos.do_move(m);
        m = pos.to_move(make_move16(SQ_33, SQ_34));
        pos.do_move(m);
        // 22角成りの指し手について
        m = pos.to_move(make_move_promote16(SQ_88, SQ_22));
        // 角を取るが、see値は、同銀と取り返されて、駒の損得なし。

        tester.test("pos1move", see_ge_th(0));

        pos.do_move(m);
        // 馬を取り返さずにあえて84歩
        m = pos.to_move(make_move16(SQ_83, SQ_24));
        pos.do_move(m);

        // この局面で31馬は、同金とされて、(see値は)馬、銀の交換 = 馬を損して銀を得する
        m = pos.to_move(make_move16(SQ_22, SQ_31));
        tester.test("pos2move", see_ge_th(-Eval::HorseValue + Eval::SilverValue));

        // この局面で33角打ちは、同桂で同馬。(see値は)角損 + 桂得。
        m = pos.to_move(make_move_drop16(BISHOP, SQ_33));
        tester.test("pos2drop", see_ge_th(-Eval::BishopValue + Eval::KnightValue));

        // この局面で33馬は、同桂でタダ。(see値は)馬損。
        m = pos.to_move(make_move16(SQ_22, SQ_33));
        tester.test("pos2move", see_ge_th(-Eval::HorseValue));
    }

    {
        // null moveのテスト
        auto section = tester.section("nullmove");
        matsuri_init();


        // null moveして、局面情報がおかしくならないかのテスト。
        pos.do_null_move();
        tester.test("pos_is_ok()", pos.pos_is_ok());
    }

    {
        // packed sfenのテスト
        auto section = tester.section("PackedSfen");

        std::vector<string> test_sfens = {
          "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
          "lns1kgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
          "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGK4 w -",
          "lnsgk4/9/ppppppppp/9/9/9/PPPPPPPPP/9/LNSGK4 w GBRgbr",
        };

        // packed by cshogi
        /*
			board = cshogi.Board()
			psfen = np.zeros(32, dtype=np.uint8)
			board.set_sfen(sfen)
			board.to_psfen(psfen)
			print(np.array2string(psfen, separator=', '))
		*/
        std::vector<PackedSfen> packed_sfens = {
          {89,  164, 81, 34, 12,  171, 68, 252, 44,  167, 68, 56, 94,  137, 240, 72,
           132, 87,  34, 60, 167, 68,  56, 86,  137, 248, 88, 70, 137, 48,  188, 126},
          {89, 164, 81,  34, 12, 171, 68,  252, 44,  167, 68, 56, 94,  137, 240, 72,
           4,  18,  225, 57, 37, 194, 177, 74,  196, 199, 50, 74, 132, 97,  191, 126},
          {89,  164, 81, 34,  88, 37,  226, 199, 41,  17,  188, 18, 129, 68,  120, 37,
           194, 115, 74, 132, 99, 149, 136, 143, 101, 148, 8,   67, 106, 107, 191, 126},
          {89,  36,  18, 1,  137, 128, 68,  64,  34, 144, 8,   175, 68,  120, 78,  137,
           112, 172, 18, 97, 25,  37,  194, 112, 30, 159, 251, 252, 166, 212, 218, 90}};

        bool success = true;
        for (size_t i = 0; i < test_sfens.size(); ++i)
        {
            auto  sfen        = test_sfens[i];
            auto& packed_sfen = packed_sfens[i];

            pos.set(sfen);

            PackedSfen ps;
            pos.sfen_pack(ps);

            // バイナリ列として一致するか。
            success &= ps == packed_sfen;

            // decodeで元のsfenになることは、このあとのランダムプレイヤーのテストで散々やっているから
            // ここでやる必要なし。
        }
        tester.test("handicapped sfen", success);
    }

#if defined(USE_PARTIAL_KEY)
    {
        // 部分hashkeyのテスト
        auto section = tester.section("PartialKey");
        {
            PRNG     my_rand(114514);
            Position pos2;

            bool ok = true;

            for (int i = 0; i < 1000; ++i)
            {

                pos.set_hirate();
                for (int j = 1; j < MAX_PLY; ++j)
                {
                    MoveList<LEGAL_ALL> ml(pos);

                    // 指し手がない == 負け == 終了
                    if (ml.size() == 0)
                        break;

                    Move m = Move(ml.at(size_t(my_rand.rand(ml.size()))));
                    pos.do_move(m);

					// 📓 sfen経由でset()を呼び出す。この時、set()によってpartial keyが初期化される。
					//     差分更新したpartial keyと一致するかをテストする。

					auto      sfen = pos.sfen();

                    pos2.set(sfen);

                    // clang-format off

                    // 部分keyが一致するか。
                    ok &= pos.st->board_key     == pos2.st->board_key;
                    ok &= pos.st->hand_key      == pos2.st->hand_key;
                    ok &= pos.st->pawnKey       == pos2.st->pawnKey;
                    ok &= pos.st->nonPawnKey[0] == pos2.st->nonPawnKey[0];
                    ok &= pos.st->nonPawnKey[1] == pos2.st->nonPawnKey[1];
                    ok &= pos.st->materialKey   == pos2.st->materialKey;
                    ok &= pos.st->minorPieceKey == pos2.st->minorPieceKey;

                    // clang-format on

                    //if (!ok)
                    //    tester.test("game " + std::to_string(i) + " ply = " + std::to_string(j), ok);
                }
            }
            tester.test("partial hash key", ok);
        }
    }
#endif

    {
        // それ以外のテスト
        auto section = tester.section("misc");
        {
            // 盤面の反転

            // 23歩不成ができ、かつ、23歩不成では駒の捕獲にはならない局面。
            pos_init("lnsgk1snl/1r4g2/p1ppppb1p/6pP1/7R1/2P6/P2PPPP1P/1SG6/LN2KGSNL b BP2p 21");
            auto flipped = pos.flipped_sfen();
            tester.test(
              "flip sfen",
              flipped == "lnsgk2nl/6gs1/p1pppp2p/6p2/1r7/1pP6/P1BPPPP1P/2G4R1/LNS1KGSNL w 2Pbp 21");
        }
    }

    {
        // 深いdepthのperftのテストが通っていれば、利きの計算、指し手生成はおおよそ間違っていないと言える。

        auto section2 = tester.section("Perft");

        {
            auto section3 = tester.section("hirate");
            hirate_init();
            const s64 p_nodes[] = {0, 30, 900, 25470, 719731, 19861490, 547581517};

            for (Depth d = 1; d <= 6; ++d)
            {
                u64 nodes = perft(pos, d);
                u64 pn    = p_nodes[d];
                tester.test("depth " + std::to_string(d) + " = " + std::to_string(pn),
                            nodes == pn && pos.pos_is_ok());
            }
        }

        {
            auto section3 = tester.section("matsuri");
            matsuri_init();

            const s64 p_nodes[] = {0, 207, 28684, 4809015, 516925165};

            for (Depth d = 1; d <= 4; ++d)
            {
                u64 nodes = perft(pos, d);
                u64 pn    = p_nodes[d];
                tester.test("depth " + std::to_string(d) + " = " + std::to_string(nodes),
                            nodes == pn && pos.pos_is_ok());
            }
        }
    }

    // ランダムプレイヤーでの対局によるテスト

    // packed sfenのtest
    auto extra_test1 = [&](Position& pos) {
        PackedSfen ps;

        string     sfen     = pos.sfen();
        int        game_ply = pos.game_ply();
        pos.sfen_pack(ps);

        Position pos2;
        pos2.set_from_packed_sfen(ps);
        string sfen2 = pos2.sfen(game_ply);

        return sfen == sfen2;
    };

    // 駒落ちのpacked sfenのテスト
    auto extra_test2 = [&](Position& pos) {
        PackedSfen ps;

        string     sfen     = pos.sfen();
        int        game_ply = pos.game_ply();
        pos.sfen_pack(ps);

        Position pos2;
        pos2.set_from_packed_sfen(ps);
        // ここから駒を5枚ほど落とす。
        int count = 0;
        for (auto sq : SQ)
        {
            auto pc = pos2.piece_on(sq);
            if (pc != NO_PIECE && type_of(pc) != KING)
            {
                pos2.board[sq] = NO_PIECE;  // 自分のclass内なので直接書き換えてしまう。
                if (++count >= 5)
                    break;
            }
        }
        string sfen2 = pos2.sfen(game_ply);
        pos2.sfen_pack(ps);  // 駒落ちのpacked sfenができた。

        Position pos3;
        pos3.set_from_packed_sfen(ps);

        string sfen3 = pos3.sfen(game_ply);

        return sfen2 == sfen3;
    };

    {
        // 対局回数→0ならskip
        s64 random_player_loop = tester.options["random_player_loop"];
        if (random_player_loop)
        {
            auto section2 = tester.section("GamesOfRandomPlayer");

            // seed固定乱数(再現性ある乱数)
            PRNG      my_rand(114514);


            for (s64 i = 0; i < random_player_loop; ++i)
            {
                // 平手初期化
                hirate_init();
                bool fail = false;

                // 512手目まで
                for (int ply = 0; ply < 512; ++ply)
                {
                    MoveList<LEGAL_ALL> ml(pos);

                    // 指し手がない == 負け == 終了
                    if (ml.size() == 0)
                        break;

                    Move m = Move(ml.at(size_t(my_rand.rand(ml.size()))));

                    pos.do_move(m);

                    if (!pos.pos_is_ok() || !extra_test1(pos) || !extra_test2(pos))
                        fail = true;
                }

                // 今回のゲームのなかでおかしいものがなかったか
                tester.test(string("game ") + std::to_string(i + 1), !fail);
            }
        }
    }
}


} // namespace YaneuraOu
