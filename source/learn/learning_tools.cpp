#include "learning_tools.h"

#if defined (EVAL_LEARN)

namespace EvalLearningTools
{

	// --- static variables

#if defined (ADA_GRAD_UPDATE)
	double Weight::eta;
#elif defined (SGD_UPDATE)
	PRNG Weight::prng;
#endif

	// --- tables

	// あるBonaPieceを相手側から見たときの値
	// BONA_PIECE_INITが-1なので符号型で持つ必要がある。
	// KPPTを拡張しても当面、BonaPieceが2^15を超えることはないのでs16で良しとする。
	s16 inv_piece_[Eval::fe_end];

	// 盤面上のあるBonaPieceをミラーした位置にあるものを返す。
	s16 mir_piece_[Eval::fe_end];

	std::vector<bool> min_index_flag;


	// --- methods

	// あるBonaPieceを相手側から見たときの値を返す
	Eval::BonaPiece inv_piece(Eval::BonaPiece p) { return (Eval::BonaPiece)inv_piece_[p]; }

	// 盤面上のあるBonaPieceをミラーした位置にあるものを返す。
	Eval::BonaPiece mir_piece(Eval::BonaPiece p) { return (Eval::BonaPiece)mir_piece_[p]; }


	// --- 個別のテーブルごとの初期化

	void init_min_index_flag()
	{
		// 次元下げ用フラグ配列の初期化
		u64 size = KPP::max_index();
		min_index_flag.resize(size);
#pragma omp parallel for schedule(guided)
		for (u64 index = 0; index < size; ++index)
		{
			if (KK::is_ok(index))
			{
				min_index_flag[index] = true;
				// indexからの変換と逆変換によって元のindexに戻ることを確認しておく。
				// 起動時に1回しか実行しない処理なのでASSERT_LV1で書いておく。
				ASSERT_LV1(KK::fromIndex(index).toIndex() == index);
				// 次元下げの1つ目の要素が元のindexと同一であることを確認しておく。
				KK a[1];
				KK::fromIndex(index).toLowerDimensions(a);
				ASSERT_LV1(a[0].toIndex() == index);
			}
			else if (KKP::is_ok(index))
			{
				KKP x = KKP::fromIndex(index);
				KKP a[2];
				x.toLowerDimensions(a);
				u64 id[2] = { a[0].toIndex(),a[1].toIndex() };
				min_index_flag[index] = (std::min({ id[0],id[1] }) == index);
				ASSERT_LV1(id[0] == index);
			}
			else if (KPP::is_ok(index))
			{
				KPP x = KPP::fromIndex(index);

#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
				// 普通の正方配列のとき、次元下げは4つ。
				KPP a[4];
				x.toLowerDimensions(a);
				u64 id[4] = { a[0].toIndex() , a[1].toIndex(), a[2].toIndex() , a[3].toIndex()};
				min_index_flag[index] = (std::min({ id[0],id[1],id[2],id[3] }) == index);
#else
				// 3角配列を用いるなら、次元下げは2つ。
				KPP a[2];
				x.toLowerDimensions(a);
				u64 id[2] = { a[0].toIndex() , a[1].toIndex()};
				min_index_flag[index] = (std::min({ id[0],id[1] }) == index);
#endif
				ASSERT_LV1(KPP::fromIndex(index).toIndex() == index);
				ASSERT_LV1(id[0] == index);
			}
			else
			{
				ASSERT_LV3(false);
			}
		}
	}

	using namespace Eval;
	void init_mir_inv_tables()
	{
		// mirrorとinverseのテーブルの初期化。

			// fとeとの交換
		int t[] = {
			f_hand_pawn - 1    , e_hand_pawn - 1   ,
			f_hand_lance - 1   , e_hand_lance - 1  ,
			f_hand_knight - 1  , e_hand_knight - 1 ,
			f_hand_silver - 1  , e_hand_silver - 1 ,
			f_hand_gold - 1    , e_hand_gold - 1   ,
			f_hand_bishop - 1  , e_hand_bishop - 1 ,
			f_hand_rook - 1    , e_hand_rook - 1   ,
			f_pawn             , e_pawn            ,
			f_lance            , e_lance           ,
			f_knight           , e_knight          ,
			f_silver           , e_silver          ,
			f_gold             , e_gold            ,
			f_bishop           , e_bishop          ,
			f_horse            , e_horse           ,
			f_rook             , e_rook            ,
			f_dragon           , e_dragon          ,
		};

		// 未初期化の値を突っ込んでおく。
		for (BonaPiece p = BONA_PIECE_ZERO; p < fe_end; ++p)
		{
			inv_piece_[p] = BONA_PIECE_NOT_INIT;

			// mirrorは手駒に対しては機能しない。元の値を返すだけ。
			mir_piece_[p] = (p < f_pawn) ? p : BONA_PIECE_NOT_INIT;
		}

		for (BonaPiece p = BONA_PIECE_ZERO; p < fe_end; ++p)
		{
			for (int i = 0; i < 32 /* t.size() */; i += 2)
			{
				if (t[i] <= p && p < t[i + 1])
				{
					Square sq = (Square)(p - t[i]);

					// 見つかった!!
					BonaPiece q = (p < fe_hand_end) ? BonaPiece(sq + t[i + 1]) : (BonaPiece)(Inv(sq) + t[i + 1]);
					inv_piece_[p] = q;
					inv_piece_[q] = p;

					/*
					ちょっとトリッキーだが、pに関して盤上の駒は
					p >= fe_hand_end
					のとき。

					このpに対して、nを整数として(上のコードのiは偶数しかとらない)、
					a)  t[2n + 0] <= p < t[2n + 1] のときは先手の駒
					b)  t[2n + 1] <= p < t[2n + 2] のときは後手の駒
					　である。

					 ゆえに、a)の範囲にあるpをq = Inv(p-t[2n+0]) + t[2n+1] とすると180度回転させた升にある後手の駒となる。
					 そこでpとqをswapさせてinv_piece[ ]を初期化してある。
					 */

					 // 手駒に関してはmirrorなど存在しない。
					if (p < fe_hand_end)
						continue;

					BonaPiece r1 = (BonaPiece)(Mir(sq) + t[i]);
					mir_piece_[p] = r1;
					mir_piece_[r1] = p;

					BonaPiece p2 = (BonaPiece)(sq + t[i + 1]);
					BonaPiece r2 = (BonaPiece)(Mir(sq) + t[i + 1]);
					mir_piece_[p2] = r2;
					mir_piece_[r2] = p2;

					break;
				}
			}
		}

		for (BonaPiece p = BONA_PIECE_ZERO; p < fe_end; ++p)
			if (inv_piece_[p] == BONA_PIECE_NOT_INIT
				|| mir_piece_[p] == BONA_PIECE_NOT_INIT
				)
			{
				// 未初期化のままになっている。上のテーブルの初期化コードがおかしい。
				ASSERT(false);
			}

#if 0
		// 評価関数のミラーをしても大丈夫であるかの事前検証
		// 値を書き込んだときにassertionがあるので、ミラーしてダメである場合、
		// そのassertに引っかかるはず。

		// AperyのWCSC26の評価関数、kppのp1==0とかp1==20(後手の0枚目の歩)とかの
		// ところにゴミが入っていて、これを回避しないとassertに引っかかる。

		std::unordered_set<BonaPiece> s;
		vector<int> a = {
			f_hand_pawn - 1,e_hand_pawn - 1,
			f_hand_lance - 1, e_hand_lance - 1,
			f_hand_knight - 1, e_hand_knight - 1,
			f_hand_silver - 1, e_hand_silver - 1,
			f_hand_gold - 1, e_hand_gold - 1,
			f_hand_bishop - 1, e_hand_bishop - 1,
			f_hand_rook - 1, e_hand_rook - 1,
		};
		for (auto b : a)
			s.insert((BonaPiece)b);

		// さらに出現しない升の盤上の歩、香、桂も除外(Aperyはここにもゴミが入っている)
		for (Rank r = RANK_1; r <= RANK_2; ++r)
			for (File f = FILE_1; f <= FILE_9; ++f)
			{
				if (r == RANK_1)
				{
					// 1段目の歩
					BonaPiece b1 = BonaPiece(f_pawn + (f | r));
					s.insert(b1);
					s.insert(inv_piece[b1]);

					// 1段目の香
					BonaPiece b2 = BonaPiece(f_lance + (f | r));
					s.insert(b2);
					s.insert(inv_piece[b2]);
				}

				// 1,2段目の桂
				BonaPiece b = BonaPiece(f_knight + (f | r));
				s.insert(b);
				s.insert(inv_piece[b]);
			}

		cout << "\nchecking kpp_write()..";
		for (auto sq : SQ)
		{
			cout << sq << ' ';
			for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
				for (BonaPiece p2 = BONA_PIECE_ZERO; p2 < fe_end; ++p2)
					if (!s.count(p1) && !s.count(p2))
						kpp_write(sq, p1, p2, kpp[sq][p1][p2]);
		}
		cout << "\nchecking kkp_write()..";

		for (auto sq1 : SQ)
		{
			cout << sq1 << ' ';
			for (auto sq2 : SQ)
				for (BonaPiece p1 = BONA_PIECE_ZERO; p1 < fe_end; ++p1)
					if (!s.count(p1))
						kkp_write(sq1, sq2, p1, kkp[sq1][sq2][p1]);
		}
		cout << "..done!" << endl;
#endif
	}


	// このEvalLearningTools全体の初期化
	void init()
	{
		//std::cout << "EvalLearningTools init..";

		init_min_index_flag();
		init_mir_inv_tables();

		//std::cout << "done." << std::endl;
	}
}

#endif
