#include "mate.h"
#if defined (USE_MATE_1PLY)
#include "../position.h"

namespace Mate
{
	// 1手詰めテーブルの初期化関数
	// ※　これは、mate1ply_without_effect.cppか、mate1ply_with_effect.cppのいずれかで定義されている。
	void init_mate_1ply();

	// ---------------------
	//     Mate::init()
	// ---------------------

	// Mate関連で使うテーブルの初期化
	void init()
	{
		init_mate_1ply();
	}
		
	// ---------------------
	// weak_mate_3ply()
	// ---------------------

	// 利きのある場所への取れない近接王手からの3手詰め
	Move weak_mate_3ply(const Position& pos, int ply)
	{
		// 1手詰めであるならこれを返す
		Move m = Mate::mate_1ply(pos);
		if (m)
			return m;

		// 詰まない
		if (ply <= 1)
			return Move::none();

		Color us = pos.side_to_move();
		Color them = ~us;
		Bitboard around8 = kingEffect(pos.king_square(them));

		// const剥がし
		Position* This = ((Position*)&pos);

		StateInfo si;
		StateInfo si2;

		// 近接王手で味方の利きがあり、敵の利きのない場所を探す。
		for (auto em : MoveList<CHECKS>(pos))
		{
			Move m = Move(em);

			// 近接王手で、この指し手による駒の移動先に敵の駒がない。
			Square to = m.to_sq();
			if ((around8 & to)

#if ! defined(LONG_EFFECT_LIBRARY)
				// toに利きがあるかどうか。mが移動の指し手の場合、mの元の利きを取り除く必要がある。
				&& (m.is_drop() ? pos.effected_to(us, to) : (bool)(pos.attackers_to(us, to, pos.pieces() ^ m.from_sq()) ^ m.from_sq()))

				// 敵玉の利きは必ずtoにあるのでそれを除いた利きがあるかどうか。
				&& (pos.attackers_to(them, to, pos.pieces()) ^ pos.king_square(them))
#else
				&& (m.is_drop() ? pos.effected_to(us, to) :
					pos.board_effect[us].effect(to) >= 2 ||
					(pos.long_effect.directions_of(us, m.from_sq()) & Effect8::directions_of(m.from_sq(), to)) != 0)

				// 敵玉の利きがあるので2つ以上なければそれで良い。
				&& (pos.board_effect[them].effect(to) <= 1)
#endif
				)
			{
				if (!pos.legal(m))
					continue;

				ASSERT_LV3(pos.gives_check(m));

				This->do_move(m, si, true);

				ASSERT_LV3(pos.in_check());

				// この局面ですべてのevasionを試す
				for (auto m2 : MoveList<EVASIONS>(pos))
				{
					if (!pos.legal(m2))
						continue;

					// この指し手で逆王手になるなら、不詰めとして扱う
					if (pos.gives_check(m2))
						goto NEXT_CHECK;

					This->do_move(m2, si2, false);

					ASSERT_LV3(!pos.in_check());

					if (!weak_mate_3ply(pos, ply - 2))
					{
						// 詰んでないので、m2で詰みを逃れている。
						This->undo_move(m2);
						goto NEXT_CHECK;
					}

					This->undo_move(m2);
				}

				// すべて詰んだ
				This->undo_move(m);

				// mによって3手で詰む。
				return m;

			NEXT_CHECK:;
				This->undo_move(m);
			}
		}
		return Move::none();
	}

	// 連続王手などの千日手判定を行う。
	// plies_from_root : 詰み探索開始局面からの手数
	// repetition_ply  : 千日手判定を行う最大手数 (16ぐらいでいいと思う)
	// or_node         : 攻め方のnodeであればtrue。受け方のnodeであればfalse。
	MateRepetitionState mate_repetition(const Position& pos, const int plies_from_root , const int repetition_ply, bool or_node)
	{
		// この判定、わりと難しい問題がある。何がどう難しいかは、以下のコードを読めばわかる。

		// ここで千日手に到達している可能性がある。
		// or nodeとand nodeとでわりと対称性があるが、REPETITION_DRAWはいずれも詰まない扱いにしないといけないなど、
		// 完全に対称とも言い切れない。
			
		// 与えられた局面以降で生じた劣等局面と、与えられた局面以前から生じた劣等局面では意味が異なる。
		// 前者は、それで詰むことはないので不詰扱いして良いが、後者は、一時的に劣等局面に入ったあと詰むことがあるので不詰扱いできない。
		// 例) XX歩打ち(王手)、XX玉(玉を逃げる)、の局面をdfpn_mate()で詰むかを判定する場合、この打った歩を成り捨てて、同玉から詰むことがある。
		// 　　このとき成り捨てた局面は劣等局面であるが、このように詰むことがあるので、不詰扱いしてはならない。

		if (or_node)
		{
			switch (pos.is_repetition(plies_from_root))
			{
			case REPETITION_DRAW:
			case REPETITION_LOSE:
			case REPETITION_INFERIOR: // 駒損したのでplies_from_root以降で生じているから、これは不詰扱いして良い。
				return MateRepetitionState::Mated; // 不詰 = 相手の勝ち = Mated

			case REPETITION_WIN:      // 連続王手の千日手で勝ちになった。(or nodeで普通これはないはずだが)
			case REPETITION_SUPERIOR: // 相手は駒損をしているが、即座に詰みとは言い切れない。
				// しかしこれが詰まないなら、相手は駒損せずに詰みを逃れる手段があるのでそちらを選択するだろうから、これは調べる必要がなく、詰み扱いで良い。
				return MateRepetitionState::Mate;

			case REPETITION_NONE:
				// 再度、普通の深さで千日手になっていないかチェックしないといけない。
				break;

			default: UNREACHABLE;
			}

			if (plies_from_root < repetition_ply)
			{
				// 普通の深さで再度千日手のチェックを行う。

				switch (pos.is_repetition(repetition_ply))
				{
				case REPETITION_DRAW: // 千日手模様で逃れられる。
				case REPETITION_LOSE:
					return MateRepetitionState::Mated; // 詰まない

				case REPETITION_WIN:      // 連続王手の千日手で勝ちになった。(or nodeで普通これはないはずだが、呼び出し元以前の局面によってはありうる)
					return MateRepetitionState::Mate;  // 詰み

				case REPETITION_NONE:
				case REPETITION_INFERIOR: // 駒損はしたが、それはroot以前の局面から見ると、であって、歩を成り捨てて劣等局面には突入するが、詰むようなケースかもしれない。
				case REPETITION_SUPERIOR: // 相手は駒損をしているが、即座に詰みとは言い切れない。
					break;

				default: UNREACHABLE;
				}
			}
		}
		else {
			// and node

			switch (pos.is_repetition(plies_from_root))
			{
			case REPETITION_DRAW:     // 詰まない(千日手で逃れている)
			case REPETITION_WIN:      // 連続王手の千日手で勝ちになった。
			case REPETITION_SUPERIOR: // 駒得をしている。即座に詰まないとは言い切れないが、これでこのあと詰まされるとしたら相手は駒損せずに詰ますことができるのでこの指し手は不詰扱いして良い。
				return MateRepetitionState::Mate;  // 不詰 = and node側から見ると勝ち

			case REPETITION_LOSE:     // 連続王手の千日手による反則負け。
			case REPETITION_INFERIOR: // 駒損したものの即座に詰まないとは言い切れないが、これで詰まないとしたら駒損しないほうの変化でも詰まないので、これは詰み扱いして良い。
				return MateRepetitionState::Mated; // 詰み = and node側から見ると負け

			case REPETITION_NONE:
				// 再度、普通の深さで千日手になっていないかチェックしないといけない。
				break;

			default: UNREACHABLE;
			}

			if (plies_from_root < repetition_ply)
			{
				// 普通の深さ(16手)で再度千日手のチェックを行う。
				switch (pos.is_repetition(repetition_ply))
				{
				case REPETITION_DRAW:
				case REPETITION_WIN:      // 連続王手の千日手で勝ちになった。
					return MateRepetitionState::Mate;  // 勝ち

				case REPETITION_LOSE:
					return MateRepetitionState::Mated; // 負け

				case REPETITION_NONE:
				case REPETITION_INFERIOR: // 駒損はしたが、今回は詰むとは言えない。
				case REPETITION_SUPERIOR: // 攻め方が駒損をしているが、即座にこの筋がないとは言い切れない。(MCTSの探索ならそんな指し手はしないかも知れないが)
					break;

				default: UNREACHABLE;
				}
			}
		}
		return MateRepetitionState::Unknown;
	}

#if defined(USE_MATE_SOLVER)|| defined(USE_MATE_DFPN)

	// ---------------------
	// class MateHashTable
	// ---------------------

	// このEntryに保存する。
	void MateHashEntry::save(Key board_key, Color root_color, /*Color side_to_move,*/ Hand hand, bool is_mate, u32 ply, Move move)
	{
		lock();

		// 書き出しの時に同一のboard_keyの情報があるなら、
		// 優劣関係を調べて、情報量が多いほうを書き出すべき。

		if (   this->board_key  == (board_key & 0xffffffffffff) /* 48bit/4 = fが12個 */
			&& this->root_color == root_color
			&& this->is_mate    == is_mate
			)
		{
			// 盤面が一致したので、手駒の優劣関係を調べる。

			// or_node(攻め方の局面)であるか
			bool or_node = root_color == side_to_move();

			// この置換表の情報のほうが優れているか
			bool entry_is_better;
			if (is_mate)
			{
				// 詰みの情報
				//  詰みが証明されている局面の情報があるとして、
				// 　攻め方は、それより手駒が同じか多ければ同様に詰む。
				//   受け方は、それより手駒が同じか少なければ同様に詰む(詰まされる)。
				// この時、このエントリーの情報は与えられた情報を包含しているので上書きする必要はない。
				entry_is_better =
					   ( or_node && hand_is_equal_or_superior(hand            , this->get_hand()))
					|| (!or_node && hand_is_equal_or_superior(this->get_hand(), hand            ));
			}
			else {
				// 不詰の情報
				entry_is_better =
					   ( or_node && hand_is_equal_or_superior(this->get_hand(), hand            ))
					|| (!or_node && hand_is_equal_or_superior(hand            , this->get_hand()));
			}

			if (entry_is_better)
			{
				unlock();
				return;
			}
		}

		this->board_key    = board_key;
		this->root_color   = root_color;
		//this->side_to_move = side_to_move;
		this->is_mate      = is_mate;
		this->ply          = ply;
		this->hand         = (u32)hand;
		this->set_move(move);
		unlock();
	}


	// このentryをlockする。
	void MateHashEntry::lock()
	{
		// 典型的なCAS lock
		while (true)
		{
			bool expected = false;
			if (mutex.compare_exchange_weak(expected,/* desired = */true))
				break;
		}

	}

	// このentryをunlockする。
	// lock済みであること。
	void MateHashEntry::unlock()
	{
		mutex = false;
	}

	// 与えられたboard_keyを持つMateHashEntryの先頭アドレスを返す。
	// (現状、1つしか該当するエントリーはない)
	MateHashEntry* MateHashTable::first_entry(const Key board_key) const {
		uint64_t index = mul_hi64((u64)board_key >> 1, entryCount);
		return &table[(index << 1) | ((u64)board_key & 1)];
	}

	// 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。MB単位。
	void MateHashTable::resize(size_t mbSize) {
		size_t size = mbSize * 1024 * 1024 / 16;

		if (entryCount != size)
		{
			table = new MateHashEntry[size];
			entryCount = size;

			//clear();
			// →　呼び出し元でクリアすること。
		}
	}

	// 置換表のエントリーの全クリア
	void MateHashTable::clear()
	{
		Tools::memclear("MateHash", table , entryCount * sizeof(MateHashEntry));
	}


#endif // #if defined(USE_MATE_SOLVER)|| defined(USE_MATE_DFPN)

} // namespace Mate


#if defined(USE_MATE_DFPN)

// dfpnの実装、以下の二通りをifdefで書き分けてあるので、DFPN32とDFPN64をそれぞれdefineして、二度includeする。

// Node数32bitまでしか扱えない版
#define DFPN32
#include "mate_dfpn.hpp"
#undef DFPN32

// Node数64bitまで扱える版
#define DFPN64
#include "mate_dfpn.hpp"
#undef DFPN64

namespace {

	std::unique_ptr<Mate::Dfpn::MateDfpnSolverInterface> BuildNode32bitSolver()
	{
		return std::make_unique<Mate::Dfpn32::MateDfpnPn<u32 , false /* 指し手Orderingなし*/, false /* no hash */>>();
	}

	std::unique_ptr<Mate::Dfpn::MateDfpnSolverInterface> BuildNode16bitOrderingSolver()
	{
		return std::make_unique<Mate::Dfpn32::MateDfpnPn<u32, true /* 指し手Orderingあり*/, false /* no hash */>>();
	}

	std::unique_ptr<Mate::Dfpn::MateDfpnSolverInterface> BuildNode64bitSolver()
	{
		return std::make_unique<Mate::Dfpn64::MateDfpnPn<u64 , false /* 指し手Orderingなし*/, false /* no hash */>>();
	}

	std::unique_ptr<Mate::Dfpn::MateDfpnSolverInterface> BuildNode48bitOrderingSolver()
	{
		return std::make_unique<Mate::Dfpn64::MateDfpnPn<u64,true /* 指し手Orderingあり*/, false /* no hash */ >>();
	}

	std::unique_ptr<Mate::Dfpn::MateDfpnSolverInterface> BuildNode32bitWithHashSolver()
	{
		return std::make_unique<Mate::Dfpn32::MateDfpnPn<u32 , false /* 指し手Orderingなし*/ , true /* with hash*/>>();
	}

	std::unique_ptr<Mate::Dfpn::MateDfpnSolverInterface> BuildNode16bitOrderingWithHashSolver()
	{
		return std::make_unique<Mate::Dfpn32::MateDfpnPn<u32, true /* 指し手Orderingあり*/, true /* with hash*/>>();
	}

	std::unique_ptr<Mate::Dfpn::MateDfpnSolverInterface> BuildNode64bitWithHashSolver()
	{
		return std::make_unique<Mate::Dfpn64::MateDfpnPn<u64 , false /* 指し手Orderingなし*/, true /* with hash*/>>();
	}

	std::unique_ptr<Mate::Dfpn::MateDfpnSolverInterface> BuildNode48bitOrderingWithHashSolver()
	{
		return std::make_unique<Mate::Dfpn64::MateDfpnPn<u64,true /* 指し手Orderingあり*/, true /* with hash*/>>();
	}
}

namespace Mate::Dfpn
{
	MateDfpnSolver::MateDfpnSolver(DfpnSolverType t)
	{
		ChangeSolverType(t);
	}

	// Solverのtypeをあとから変更する。
	void MateDfpnSolver::ChangeSolverType(DfpnSolverType t)
	{
		switch (t)
		{
		case DfpnSolverType::None             : impl = std::unique_ptr<MateDfpnSolverInterface>(); break;
		case DfpnSolverType::Node32bit        : impl = BuildNode32bitSolver();         break;
		case DfpnSolverType::Node16bitOrdering: impl = BuildNode16bitOrderingSolver(); break;
		case DfpnSolverType::Node64bit        : impl = BuildNode64bitSolver();         break;
		case DfpnSolverType::Node48bitOrdering: impl = BuildNode48bitOrderingSolver(); break;
		case DfpnSolverType::Node32bitWithHash        : impl = BuildNode32bitWithHashSolver();         break;
		case DfpnSolverType::Node16bitOrderingWithHash: impl = BuildNode16bitOrderingWithHashSolver(); break;
		case DfpnSolverType::Node64bitWithHash        : impl = BuildNode64bitWithHashSolver();         break;
		case DfpnSolverType::Node48bitOrderingWithHash: impl = BuildNode48bitOrderingWithHashSolver(); break;
		}
	}

}
#endif

#endif // defined (USE_MATE_1PLY)
