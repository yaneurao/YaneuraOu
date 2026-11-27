#ifndef SFEN_PACKER_H_INCLUDED
#define SFEN_PACKER_H_INCLUDED

#include <cstddef>

#include "../types.h"

namespace YaneuraOu {

// 将棋用の圧縮局面・PSVレコードの定義。変換処理はsfen_packer.cppに置く。
// Stockfishには対応する形式がないため、Position本体から分離する。

// packされたsfen
struct PackedSfen {
	u8 data[32];

	// 盤面を180度回転し、先後と手番を入れ替える。
	// SFEN文字列やPosition::set()を経由せず、packed sfen上で変換する。
	void flip();
	PackedSfen flipped() const;

	// 手番を返す。
	Color color() const {
		// これは、data[0]のbit0に格納されていることは保証されている。
		return Color(data[0] & 1);
	}

	// std::unordered_mapで使用できるように==と!=を定義しておく。

	bool operator==(const PackedSfen& rhs) const {
		static_assert(sizeof(PackedSfen) % sizeof(u64) == 0);

		for (size_t i = 0; i < sizeof(PackedSfen); i += sizeof(size_t))
		{
			// 8バイト単位で比較していく。一度でも内容が異なれば不一致。
			if (*(u64*)&data[i] != *(u64*)&rhs.data[i])
				return false;
		}
		return true;
	}

	bool operator!=(const PackedSfen& rhs) const {
		return !(this->operator==(rhs));
	}
};

// PSV形式(.psv)の1レコード。
// PackedSfenに、手番側視点のscoreなどを付与した40 bytes固定長形式。
struct PsvRecord {
	PackedSfen sfen;
	s16        score;
	u16        move;
	u16        gamePly;
	s8         game_result;
	u8         padding;
};

static_assert(sizeof(PsvRecord) == 40, "PsvRecord must be 40 bytes");

// std::unordered_mapで使用できるようにhash関数を定義しておく。
// std::unordered_map<PackedSfen,int,PackedSfenHash> packed_sfen_to_int;のようにtemplateの第3引数に指定する。
struct PackedSfenHash {
	size_t operator()(const PackedSfen& ps) const
	{
		static_assert(sizeof(PackedSfen) % sizeof(size_t) == 0);

		size_t s = 0;
		for (size_t i = 0; i < sizeof(PackedSfen) ; i+= sizeof(size_t))
		{
			// size_tのsize分ずつをxorしていき、それをhash keyとする。
			s ^= *(size_t*)&ps.data[i];
		}
		return s;
	}
};

}  // namespace YaneuraOu

#endif  // #ifndef SFEN_PACKER_H_INCLUDED
