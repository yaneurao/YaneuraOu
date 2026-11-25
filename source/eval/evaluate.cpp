#include "../evaluate.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <map>

#include "../misc.h"
#include "../position.h"
#include "../types.h"

namespace YaneuraOu {
namespace Eval
{


	// 駒の価値
    int PieceValue[PIECE_NB] =
    {
        0, PawnValue, LanceValue, KnightValue, SilverValue, BishopValue, RookValue,GoldValue,
        KingValue, ProPawnValue, ProLanceValue, ProKnightValue, ProSilverValue, HorseValue, DragonValue,0,

        0, PawnValue, LanceValue, KnightValue, SilverValue, BishopValue, RookValue,GoldValue,
        KingValue, ProPawnValue, ProLanceValue, ProKnightValue, ProSilverValue, HorseValue, DragonValue,0,
    };

	// 駒の価値、駒割計算用(後手の駒が負の値)
    int PieceValueM[PIECE_NB] =
    {
        0, PawnValue, LanceValue, KnightValue, SilverValue, BishopValue, RookValue,GoldValue,
        KingValue, ProPawnValue, ProLanceValue, ProKnightValue, ProSilverValue, HorseValue, DragonValue,0,

        0, -PawnValue, -LanceValue, -KnightValue, -SilverValue, -BishopValue, -RookValue,-GoldValue,
        -KingValue, -ProPawnValue, -ProLanceValue, -ProKnightValue, -ProSilverValue, -HorseValue, -DragonValue,0,
    };

    // KINGの価値はゼロとしておく。KINGを捕獲する指し手は非合法手なので、これがプラスとして評価されたとしても
    // そのあとlegal()で引っ掛かり、実際はその指し手で進めないからこれで良い。
    int CapturePieceValue[PIECE_NB] =
    {
        VALUE_ZERO             , PawnValue * 2   , LanceValue * 2   , KnightValue * 2   , SilverValue * 2  ,
        BishopValue * 2, RookValue * 2, GoldValue * 2, 0 /* SEEやfutilityで用いるため王の価値は0にしておかないといけない */ ,
        ProPawnValue + PawnValue, ProLanceValue + LanceValue, ProKnightValue + KnightValue, ProSilverValue + SilverValue,
        HorseValue + BishopValue, DragonValue + RookValue, VALUE_ZERO /* PRO_GOLD */,

        VALUE_ZERO             , PawnValue * 2   , LanceValue * 2   , KnightValue * 2   , SilverValue * 2  ,
        BishopValue * 2, RookValue * 2, GoldValue * 2, 0 ,
        ProPawnValue + PawnValue, ProLanceValue + LanceValue, ProKnightValue + KnightValue, ProSilverValue + SilverValue,
        HorseValue + BishopValue, DragonValue + RookValue, VALUE_ZERO /* PRO_GOLD */,
    };

    // 成った時の価値の上昇分
    int ProDiffPieceValue[PIECE_NB] =
    {
        VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
        VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
        VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
        VALUE_ZERO, ProPawnValue - PawnValue, ProLanceValue - LanceValue, ProKnightValue - KnightValue, ProSilverValue - SilverValue, HorseValue - BishopValue, DragonValue - RookValue, VALUE_ZERO ,
    };

#if defined(USE_BONA_PIECE)

	ExtBonaPiece kpp_board_index[PIECE_NB] = {
        { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
        { f_pawn, e_pawn },
        { f_lance, e_lance },
        { f_knight, e_knight },
        { f_silver, e_silver },
        { f_bishop, e_bishop },
        { f_rook, e_rook },
        { f_gold, e_gold },
        { f_king, e_king },

    // 通常の場合（金と小駒の成り駒を区別しない場合）
    #if !defined (DISTINGUISH_GOLDS)
        { f_gold, e_gold }, // 成歩
        { f_gold, e_gold }, // 成香
        { f_gold, e_gold }, // 成桂
        { f_gold, e_gold }, // 成銀

    // 金と小駒の成り駒を区別する場合
    #else
        { f_pro_pawn, e_pro_pawn },     // 成歩
        { f_pro_lance, e_pro_lance },   // 成香
        { f_pro_knight, e_pro_knight }, // 成桂
        { f_pro_silver, e_pro_silver }, // 成銀
    #endif

        { f_horse, e_horse }, // 馬
        { f_dragon, e_dragon }, // 龍
        { BONA_PIECE_ZERO, BONA_PIECE_ZERO }, // 金の成りはない

        // 後手から見た場合。fとeが入れ替わる。
        { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
        { e_pawn, f_pawn },
        { e_lance, f_lance },
        { e_knight, f_knight },
        { e_silver, f_silver },
        { e_bishop, f_bishop },
        { e_rook, f_rook },
        { e_gold, f_gold },
        { e_king, f_king },

    // 通常の場合（金と小駒の成り駒を区別しない場合）
    #if !defined (DISTINGUISH_GOLDS)
        { e_gold, f_gold }, // 成歩
        { e_gold, f_gold }, // 成香
        { e_gold, f_gold }, // 成桂
        { e_gold, f_gold }, // 成銀

    // 金と小駒の成り駒を区別する場合
    #else
        { e_pro_pawn, f_pro_pawn },     // 成歩
        { e_pro_lance, f_pro_lance },   // 成香
        { e_pro_knight, f_pro_knight }, // 成桂
        { e_pro_silver, f_pro_silver }, // 成銀
    #endif

        { e_horse, f_horse }, // 馬
        { e_dragon, f_dragon }, // 龍
        { BONA_PIECE_ZERO, BONA_PIECE_ZERO }, // 金の成りはない
    };

    ExtBonaPiece kpp_hand_index[COLOR_NB][KING] = {
    {
        { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
        { f_hand_pawn, e_hand_pawn },
        { f_hand_lance, e_hand_lance },
        { f_hand_knight, e_hand_knight },
        { f_hand_silver, e_hand_silver },
        { f_hand_bishop, e_hand_bishop },
        { f_hand_rook, e_hand_rook },
        { f_hand_gold, e_hand_gold },
    },
    {
        { BONA_PIECE_ZERO, BONA_PIECE_ZERO },
        { e_hand_pawn, f_hand_pawn },
        { e_hand_lance, f_hand_lance },
        { e_hand_knight, f_hand_knight },
        { e_hand_silver, f_hand_silver },
        { e_hand_bishop, f_hand_bishop },
        { e_hand_rook, f_hand_rook },
        { e_hand_gold, f_hand_gold },
    },
    };

    // BonaPieceの内容を表示する。手駒ならH,盤上の駒なら升目。例) HP3 (3枚目の手駒の歩)
    std::ostream& operator<<(std::ostream& os, BonaPiece bp)
    {
        if (bp < fe_hand_end)
        {
            for (auto c : COLOR)
            for (PieceType pc = PAWN; pc < KING; ++pc)
            {
                // この駒種の上限(e.g. 歩 = 18)
                int kind_num = kpp_hand_index[c][pc].fw - kpp_hand_index[c][pc].fb;
                int start = kpp_hand_index[c][pc].fb;
                if (start <= bp && bp < start+kind_num * 2)
                {
                bool is_black = bp < start + kind_num;
                if (!is_black) bp = (BonaPiece)(bp - kind_num);
        #if defined (PRETTY_JP)
                os << "手" << (is_black ? "先" : "後") << pretty(pc) << int(bp - start + 1); // ex.手先歩3
        #else
                os << "H" << (is_black ? "B" : "W") << pc << int(bp - kpp_hand_index[c][pc].fb + 1); // ex.HBP3
        #endif
                break;
                }
            }
        } else {
            for (auto pc : Piece())
            if (kpp_board_index[pc].fb <= bp && bp < kpp_board_index[pc].fb + SQ_NB)
            {
        #if defined (PRETTY_JP)
                os << Square(bp - kpp_board_index[pc].fb) << pretty(pc); // ex.32P
        #else
                os << Square(bp - kpp_board_index[pc].fb) << pc; // ex.32P
        #endif
                break;
            }
        }

        return os;
    }
#if defined(USE_AVX512)
namespace {
// 盤面の状態ではなく、駒種から入力番号への固定オフセットだけを保持する。
// kpp_board_indexの初期化後に同じ表から作るので、成り駒の区別設定にも追随する。
alignas(64) const auto BoardBonaOffsets = [] {
    std::array<std::array<i32, PIECE_NB>, COLOR_NB> offsets{};
    for (Color c : {BLACK, WHITE})
        for (int pc = 0; pc < PIECE_NB; ++pc)
            offsets[c][pc] = kpp_board_index[pc].from[c];
    return offsets;
}();
}
#endif

// 必要な時だけ盤面と手駒から入力番号を列挙する。駒番号・逆引き表は保持しない。
void collect_bona_pieces(const Position& pos, Color perspective, BonaPiece* pieces) {
    int size = 0;
#if defined(USE_AVX512)
    static_assert(sizeof(Piece) == 1);
    const auto& board = pos.piece_array();
    const auto offset0 = _mm512_load_si512(BoardBonaOffsets[perspective].data());
    const auto offset1 = _mm512_load_si512(BoardBonaOffsets[perspective].data() + 16);
    const auto zero = _mm512_setzero_si512();
    const auto types = _mm512_set1_epi32(15);
    const auto king = _mm512_set1_epi32(KING);
    const auto lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                       8, 9, 10, 11, 12, 13, 14, 15);
    int base = 0;
    for (; base + 16 <= SQ_NB; base += 16) {
        const auto pcs = _mm512_cvtepu8_epi32(_mm_loadu_si128(
          reinterpret_cast<const __m128i*>(board.data() + base)));
        const __mmask16 mask = _mm512_cmpneq_epi32_mask(pcs, zero)
          & _mm512_cmpneq_epi32_mask(_mm512_and_si512(pcs, types), king);
        if (!mask)
            continue;
        auto squares = _mm512_add_epi32(lanes, _mm512_set1_epi32(base));
        if (perspective == WHITE)
            squares = _mm512_sub_epi32(_mm512_set1_epi32(80), squares);
        const auto indices = _mm512_add_epi32(squares, _mm512_permutex2var_epi32(offset0, pcs, offset1));
        assert(size + _mm_popcnt_u32(mask) <= 38);
        _mm512_mask_compressstoreu_epi32(pieces + size, mask, indices);
        size += _mm_popcnt_u32(mask);
    }
    for (; base < SQ_NB; ++base) {
        const Piece pc = board[base];
        if (pc != NO_PIECE && type_of(pc) != KING)
            pieces[size++] = board_bona_piece(pc, Square(base)).from[perspective];
    }
#else
    Bitboard occupied = pos.pieces() ^ pos.pieces(KING);
    while (occupied) {
        const Square s = occupied.pop_lsb();
        const Piece pc = pos.piece_on(s);
        assert(size < 38);
        pieces[size++] = board_bona_piece(pc, s).from[perspective];
    }
#endif
    for (Color c : {BLACK, WHITE}) {
        const Hand hand = pos.hand_of(c);
        if (hand == Hand(0))
            continue;
        for (PieceType pt = PAWN; pt < KING; ++pt) {
            const int count = hand_count(hand, pt);
            const int base = kpp_hand_index[c][pt].from[perspective];
            for (int i = 0; i < count; ++i) {
                assert(size < 38);
                pieces[size++] = BonaPiece(base + i);
            }
        }
    }
    // 駒落ちで不足する非玉の枠は、従来どおり0特徴量として残す。
    std::fill(pieces + size, pieces + 38, BONA_PIECE_ZERO);
    for (Color c : {BLACK, WHITE}) {
        const Square s = pos.square<KING>(c);
        pieces[38 + c] = is_ok(s) ? board_bona_piece(make_piece(c, KING), s).from[perspective]
                                 : BONA_PIECE_ZERO;
    }
}

BonaPieceList collect_bona_pieces(const Position& pos) {
    BonaPieceList result;
    collect_bona_pieces(pos, BLACK, result.pieces[BLACK]);
    collect_bona_pieces(pos, WHITE, result.pieces[WHITE]);
    return result;
}
#endif // defined(USE_BONA_PIECE)


#if !defined(USE_EVAL)
	// 評価関数を用いない時用に、ここでダミーの評価関数を定義しておく。

	void init() {}
	Value compute_eval(const Position& pos) { return VALUE_ZERO; }
	void evaluate_with_no_return(const Position& pos) {}
	void print_eval_stat(Position& pos) {}
	void load_eval() {}
	Value evaluate(const Position& pos) { return VALUE_ZERO; }

#endif

	// 必要なときに駒割りを全計算する。
	// USE_MATERIAL_VALUEを有効にした構成では、局面設定時に計算し、以降は差分更新する。

	// 駒割りの計算
	// 先手側から見た評価値
	Value material(const Position& pos)
	{
		int v = VALUE_ZERO;

		for (auto i : SQ)
			v += PieceValueM[pos.piece_on(i)];

		// 手駒も足しておく
		for (auto c : COLOR)
			for (auto pt = PAWN; pt < PIECE_HAND_NB; ++pt)
				v += (c == BLACK ? 1 : -1) * Value(hand_count(pos.hand_of(c), pt) * PieceValue[pt]);

		return (Value)v;
	}

#if defined(EVAL_KPPT) || defined(EVAL_KPP_KKPT)

	// calc_check_sum()を呼び出して返ってきた値を引数に渡すと、ソフト名を表示してくれる。
	void print_softname(u64 check_sum)
	{
		// 評価関数ファイルの正体
		std::string softname = "unknown";

		// ソフト名自動判別
		std::map<u64, std::string> list = {
			// KPPT型
			{ 0x0000d6799926233 , "ShinYane(20161010)"    } , // 真やねうら王 = やねうら王(2016)
			{ 0x0000da0f36d1b4b , "Apery(WCSC26)"         } , // Apery(WCSC26)
			{ 0x0000d7213c45779 , "Ukamuse(sdt4)"         } , // 浮かむ瀬     = Apery(sdt4)

			{ 0x0000c9e81cef72b , "elmo(WCSC27)"          } , // elmo(WCSC27)
			{ 0x0000900f1fbe7a6 , "Yomita(WCSC27)"        } , // 読み太(WCSC27)   
			{ 0x0000d5214c4e6d8 , "Qhapaq(WCSC27)"        } , // Qhapaq(WCSC27)
			{ 0x0000d291a9942bb , "tanuki(WCSC27)"        } , // tanuki(WCSC27)

			{ 0x000000000000000 , "rezero epoch0"         } , // リゼロ評価関数epoch0
			{ 0x00008b66a4ecea9 , "rezero epoch0.1"       } , // リゼロ評価関数epoch0.1
			{ 0x000093a87cc91ad , "rezero epoch1"         } , // リゼロ評価関数epoch1
			{ 0x000096a5706b9ec , "rezero epoch2"         } , // リゼロ評価関数epoch2
			{ 0x000096b3df75de8 , "rezero epoch3"         } , // リゼロ評価関数epoch3
			{ 0x0000964a98dc84b , "rezero epoch4"         } , // リゼロ評価関数epoch4
			{ 0x000095f8443a079 , "rezero epoch5"         } , // リゼロ評価関数epoch5
			{ 0x0000953fbdc48d6 , "rezero epoch6"         } , // リゼロ評価関数epoch6
			{ 0x0000a4896775602 , "rezero epoch7"         } , // リゼロ評価関数epoch7
			{ 0x0000a516345897e , "rezero epoch8"         } , // リゼロ評価関数epoch8
			{ 0x0000a6eea7db958 , "yaseiyomita(20170626)" } , // 野生の読み太(20170626)
			{ 0x0000a6fbc5087ce , "yaseiyomita(20170703)" } , // 野生の読み太(20170703)
			{ 0x0000a63c328ca42 , "yaseiyomita(20170802)" } , // 野生の読み太(20170802)
			{ 0x0000c8e89414b88 , "elmoqhapaq ver1.0"     } , // elmo-qhapaq version 1.0
			{ 0x0000c8ac0b10c78 , "elmoqhapaq ver1.1"     } , // elmo-qhapaq version 1.1

			{ 0x0000c62cf4f6b91 , "relmo8"                } , // リゼロ評価関数epoch8とelmo(WCSC27)を1:1でブレンド
			{ 0x0000c85005252ee , "yaselmo(20170705)"     } , // 野生の読み太(20170703)とelmo(WCSC27)を1:1でブレンド

			// KPP_KKPT型
			{ 0x0000574a685513c , "kppkkpt rezero epoch4" } , // KPP_KKPT型リゼロ評価関数epoch4

			// ここに加えて欲しい評価関数があれば、
			// "isready"コマンドに対する応答時に表示されるcheck sumの値を連絡ください(｀･ω･´)ｂ
		};

		if (list.count(check_sum))
			softname = list[check_sum];

		sync_cout << "info string Eval Check Sum = " << std::hex << check_sum << std::dec
			<< " , Eval File = " << softname << sync_endl;
	}
#endif



} // namespace Eval
} // namespace YaneuraOu
