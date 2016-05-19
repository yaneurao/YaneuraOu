#ifndef _SHOGI_H_
#define _SHOGI_H_

//
//  やねうら王mini
//  公式サイト :  http://yaneuraou.yaneu.com/yaneuraou_mini/
//

// 思考エンジンのバージョンとしてUSIプロトコルの"usi"コマンドに応答するときの文字列
#define ENGINE_VERSION "2.89"

// --------------------
// コンパイル時の設定
// --------------------

// ※　extra/config.hのほうで行なうこと。

// --------------------
//  思考エンジンの種類
// --------------------

// やねうら王の思考エンジンとしてリリースする場合、以下から選択。(どれか一つは必ず選択しなければならない)
// オリジナルの思考エンジンをユーザーが作成する場合は、USER_ENGINE を defineして 他のエンジンのソースコードを参考に
//  engine/user-engine/ フォルダの中身を書くべし。

//#define YANEURAOU_NANO_ENGINE        // やねうら王nano        (完成2016/01/31)
//#define YANEURAOU_NANO_PLUS_ENGINE   // やねうら王nano plus   (完成2016/02/25)
//#define YANEURAOU_MINI_ENGINE        // やねうら王mini        (完成2016/02/29)
//#define YANEURAOU_CLASSIC_ENGINE     // やねうら王classic     (完成2016/04/03)
//#define YANEURAOU_CLASSIC_TCE_ENGINE // やねうら王classic tce (完成2016/04/15)
#define YANEURAOU_2016_MID_ENGINE    // やねうら王2016(MID)   (開発中)
//#define YANEURAOU_2016_LATE_ENGINE   // やねうら王2016(LATE)  (開発中)
//#define RANDOM_PLAYER_ENGINE         // ランダムプレイヤー
//#define MATE_ENGINE                  // 詰め将棋solverとしてリリースする場合。(開発中)
//#define HELP_MATE_ENGINE             // 協力詰めsolverとしてリリースする場合。協力詰めの最長は49909手。「寿限無3」 cf. http://www.ne.jp/asahi/tetsu/toybox/kato/fbaka4.htm
//#define LOCAL_GAME_SERVER            // 連続自動対局フレームワーク
//#define USER_ENGINE                  // ユーザーの思考エンジン

// --------------------
// release configurations
// --------------------

#include "extra/config.h"

// --------------------
//    bit operations
// --------------------

#include "extra/bitop.h"

// --------------------
//      手番
// --------------------

// 手番
enum Color { BLACK=0/*先手*/,WHITE=1/*後手*/,COLOR_NB /* =2 */ , COLOR_ALL = 2 /*先後共通の何か*/ , COLOR_ZERO = 0,};

// 相手番を返す
constexpr Color operator ~(Color c) { return (Color)(c ^ 1);  }

// 正常な値であるかを検査する。assertで使う用。
constexpr bool is_ok(Color c) { return COLOR_ZERO <= c && c < COLOR_NB; }

// 出力用(USI形式ではない)　デバッグ用。
std::ostream& operator<<(std::ostream& os, Color c);

// --------------------
//        筋
// --------------------

//  例) FILE_3なら3筋。
enum File { FILE_1, FILE_2, FILE_3, FILE_4, FILE_5, FILE_6, FILE_7, FILE_8, FILE_9 , FILE_NB , FILE_ZERO=0 };

// 正常な値であるかを検査する。assertで使う用。
constexpr bool is_ok(File f) { return FILE_ZERO <= f && f < FILE_NB; }

// USIの指し手文字列などで筋を表す文字列をここで定義されたFileに変換する。
inline File toFile(char c) { return (File)(c - '1'); }

// Fileを綺麗に出力する(USI形式ではない)
// "PRETTY_JP"をdefineしていれば、日本語文字での表示になる。例 → ８
// "PRETTY_JP"をdefineしていなければ、数字のみの表示になる。例 → 8
std::string pretty(File f);

// USI形式でFileを出力する
inline std::ostream& operator<<(std::ostream& os, File f) { os << (char)('1' + f); return os; }

// --------------------
//        段
// --------------------

// 例) RANK_4なら4段目。
enum Rank { RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9 , RANK_NB , RANK_ZERO = 0};

// 正常な値であるかを検査する。assertで使う用。
constexpr bool is_ok(Rank r) { return RANK_ZERO <= r && r < RANK_NB; }

// 移動元、もしくは移動先の升のrankを与えたときに、そこが成れるかどうかを判定する。
inline bool canPromote(const Color c, const Rank fromOrToRank) {
  ASSERT_LV1(is_ok(c) && is_ok(fromOrToRank));
  // 先手9bit(9段) + 後手9bit(9段) = 18bitのbit列に対して、判定すればいい。
  // ただし ×9みたいな掛け算をするのは嫌なのでbit shiftで済むように先手16bit、後手16bitの32bitのbit列に対して判定する。
  // このcastにおいて、VC++2015ではwarning C4800が出る。
  return static_cast<bool>(0x1c00007u & (1u << ((c << 4) + fromOrToRank)));
}

// 後手の段なら先手から見た段を返す。
// 例) relative_rank(WHITE,RANK_1) == RANK_9
inline Rank relative_rank(Color c, Rank r) { return c == BLACK ? r : (Rank)(8 - r); }

// USIの指し手文字列などで段を表す文字列をここで定義されたRankに変換する。
inline Rank toRank(char c) { return (Rank)(c - 'a'); }

// Rankを綺麗に出力する(USI形式ではない)
// "PRETTY_JP"をdefineしていれば、日本語文字での表示になる。例 → 八
// "PRETTY_JP"をdefineしていなければ、数字のみの表示になる。例 → 8
std::string pretty(Rank r);

// USI形式でRankを出力する
inline std::ostream& operator<<(std::ostream& os, Rank r) { os << (char)('a' + r); return os; }

// --------------------
//        升目
// --------------------

// 盤上の升目に対応する定数。
// 盤上右上(１一が0)、左下(９九)が80
enum Square : int32_t
{
  // 以下、盤面の右上から左下までの定数。
  // これを定義していなくとも問題ないのだが、デバッガでSquare型を見たときに
  // どの升であるかが表示されることに価値がある。
  SQ_11, SQ_12, SQ_13, SQ_14, SQ_15, SQ_16, SQ_17, SQ_18, SQ_19,
  SQ_21, SQ_22, SQ_23, SQ_24, SQ_25, SQ_26, SQ_27, SQ_28, SQ_29,
  SQ_31, SQ_32, SQ_33, SQ_34, SQ_35, SQ_36, SQ_37, SQ_38, SQ_39,
  SQ_41, SQ_42, SQ_43, SQ_44, SQ_45, SQ_46, SQ_47, SQ_48, SQ_49,
  SQ_51, SQ_52, SQ_53, SQ_54, SQ_55, SQ_56, SQ_57, SQ_58, SQ_59,
  SQ_61, SQ_62, SQ_63, SQ_64, SQ_65, SQ_66, SQ_67, SQ_68, SQ_69,
  SQ_71, SQ_72, SQ_73, SQ_74, SQ_75, SQ_76, SQ_77, SQ_78, SQ_79,
  SQ_81, SQ_82, SQ_83, SQ_84, SQ_85, SQ_86, SQ_87, SQ_88, SQ_89,
  SQ_91, SQ_92, SQ_93, SQ_94, SQ_95, SQ_96, SQ_97, SQ_98, SQ_99,

  // ゼロと末尾
  SQ_ZERO = 0, SQ_NB = 81,
  SQ_NB_PLUS1 = SQ_NB + 1, // 玉がいない場合、SQ_NBに移動したものとして扱うため、配列をSQ_NB+1で確保しないといけないときがあるのでこの定数を用いる。

  // 方角に関する定数。N=北=盤面の下を意味する。
  SQ_D  = +1, // 下(Down)
  SQ_R  = -9, // 右(Right)
  SQ_U  = -1, // 上(Up)
  SQ_L  = +9, // 左(Left)

  // 斜めの方角などを意味する定数。
  SQ_RU = int(SQ_U) + int(SQ_R), // 右上(Right Up)
  SQ_RD = int(SQ_D) + int(SQ_R), // 右下(Right Down)
  SQ_LU = int(SQ_U) + int(SQ_L), // 左上(Left Up)
  SQ_LD = int(SQ_D) + int(SQ_L), // 左下(Left Down)
  SQ_RUU = int(SQ_RU) + int(SQ_U), // 右上上
  SQ_LUU = int(SQ_LU) + int(SQ_U), // 左上上
  SQ_RDD = int(SQ_RD) + int(SQ_D), // 右下下
  SQ_LDD = int(SQ_LD) + int(SQ_D), // 左下下
};

// sqが盤面の内側を指しているかを判定する。assert()などで使う用。
// 駒は駒落ちのときにSQ_NBに移動するので、値としてSQ_NBは許容する。
constexpr bool is_ok(Square sq) { return SQ_ZERO <= sq && sq <= SQ_NB; }

// sqが盤面の内側を指しているかを判定する。assert()などで使う用。玉は盤上にないときにSQ_NBを取るのでこの関数が必要。
constexpr bool is_ok_plus1(Square sq) { return SQ_ZERO <= sq && sq < SQ_NB_PLUS1; }

extern File SquareToFile[SQ_NB];

// 与えられたSquareに対応する筋を返す。
// →　行数は長くなるが速度面においてテーブルを用いる。
inline File file_of(Square sq) { /* return (File)(sq / 9); */ ASSERT_LV2(is_ok(sq)); return SquareToFile[sq]; }

extern Rank SquareToRank[SQ_NB];

// 与えられたSquareに対応する段を返す。
// →　行数は長くなるが速度面においてテーブルを用いる。
inline Rank rank_of(Square sq) { /* return (Rank)(sq % 9); */ ASSERT_LV2(is_ok(sq)); return SquareToRank[sq]; }

// 筋(File)と段(Rank)から、それに対応する升(Square)を返す。
inline Square operator | (File f, Rank r) { Square sq = (Square)(f * 9 + r); ASSERT_LV2(is_ok(sq)); return sq; }

// ２つの升のfileの差、rankの差のうち大きいほうの距離を返す。sq1,sq2のどちらかが盤外ならINT_MAXが返る。
inline int dist(Square sq1, Square sq2) { return (!is_ok(sq1) || !is_ok(sq2)) ? INT_MAX : std::max(abs(file_of(sq1)-file_of(sq2)) , abs(rank_of(sq1) - rank_of(sq2))); }

// 移動元、もしくは移動先の升sqを与えたときに、そこが成れるかどうかを判定する。
inline bool canPromote(const Color c, const Square fromOrTo) {
  ASSERT_LV2(is_ok(fromOrTo));
  return canPromote(c, rank_of(fromOrTo));
}

// 盤面を180°回したときの升目を返す
inline Square Inv(Square sq) { return (Square)((SQ_NB - 1) - sq); }

// Squareを綺麗に出力する(USI形式ではない)
// "PRETTY_JP"をdefineしていれば、日本語文字での表示になる。例 → ８八
// "PRETTY_JP"をdefineしていなければ、数字のみの表示になる。例 → 88
inline std::string pretty(Square sq) { return pretty(file_of(sq)) + pretty(rank_of(sq)); }

// USI形式でSquareを出力する
inline std::ostream& operator<<(std::ostream& os, Square sq) { os << file_of(sq) << rank_of(sq); return os; }

// --------------------
//   壁つきの升表現
// --------------------

// This trick is invented by yaneurao in 2016.

// 長い利きを更新するときにある升からある方向に駒にぶつかるまでずっと利きを更新していきたいことがあるが、
// sqの升が盤外であるかどうかを判定する簡単な方法がない。そこで、Squareの表現を拡張して盤外であることを検出
// できるようにする。

// bit 0..7   : Squareと同じ意味
// bit 8      : Squareからのborrow用に1にしておく
// bit 9..13  : いまの升から盤外まで何升右に升があるか(ここがマイナスになるとborrowでbit13が1になる)
// bit 14..18 : いまの升から盤外まで何升上に(略
// bit 19..23 : いまの升から盤外まで何升下に(略
// bit 24..28 : いまの升から盤外まで何升左に(略
enum SquareWithWall : int32_t {
  // 相対移動するときの差分値
  SQWW_R  = SQ_R - (1 << 9) + (1 << 24) , SQWW_U = SQ_U - (1 << 14) + (1 << 19) , SQWW_D = -int(SQWW_U), SQWW_L = -int(SQWW_R),
  SQWW_RU = int(SQWW_R) + int(SQWW_U) , SQWW_RD = int(SQWW_R) + int(SQWW_D) , SQWW_LU = int(SQWW_L) + int(SQWW_U) , SQWW_LD = int(SQWW_L) + int(SQWW_D) ,

  // SQ_11の地点に対応する値(他の升はこれ相対で事前に求めテーブルに格納)
  SQWW_11 = SQ_11 | (1 << 8) /* bit8 is 1 */ | (0 << 9) /*右に0升*/| (0 << 14) /*上に0升*/ | (8 << 19) /*下に8升*/| (8 << 24) /*左に8升*/,

  // SQWW_RIGHTなどを足して行ったときに盤外に行ったときのborrow bitの集合
  SQWW_BORROW_MASK = (1 << 13) | (1 << 18) | (1 << 23) | (1 << 28) ,
};

// 型変換。下位8bit == Square
inline Square to_sq(SquareWithWall sqww) { return Square(sqww & 0xff); }

extern SquareWithWall sqww_table[SQ_NB_PLUS1];

// 型変換。Square型から。
inline SquareWithWall to_sqww(Square sq) { return sqww_table[sq]; }

// 盤内か。壁(盤外)だとfalseになる。
inline bool is_ok(SquareWithWall sqww) { return (sqww & SQWW_BORROW_MASK) == 0; }

// 単にSQの升を出力する。
inline std::ostream& operator<<(std::ostream& os, SquareWithWall sqww) { os << to_sq(sqww); return os; }

// --------------------
//        方角
// --------------------

// Long Effect Libraryの一部。これは8近傍、24近傍の利きを直列化したり方角を求めたりするライブラリ。
// ここではその一部だけを持って来た。(これらは頻繁に用いるので)
// それ以上を使いたい場合は、LONG_EFFECT_LIBRARYというシンボルをdefineして、extra/long_effect.hをincludeすること。
namespace Effect8
{
  // 方角を表す。遠方駒の利きや、玉から見た方角を表すのに用いる。
  // bit0..右上、bit1..右、bit2..右下、bit3..上、bit4..下、bit5..左上、bit6..左、bit7..左下
  // 同時に複数のbitが1であることがありうる。
  enum Directions : uint8_t { DIRECTIONS_ZERO = 0 , DIRECTIONS_RU = 1, DIRECTIONS_R = 2 , DIRECTIONS_RD = 4,
    DIRECTIONS_U = 8, DIRECTIONS_D = 16 , DIRECTIONS_LU = 32 , DIRECTIONS_L = 64 , DIRECTIONS_LD = 128 };

  // sq1にとってsq2がどのdirectionにあるか。
  extern Directions direc_table[SQ_NB_PLUS1][SQ_NB_PLUS1];
  inline Directions directions_of(Square sq1, Square sq2) { return direc_table[sq1][sq2]; }

  // Directionsをpopしたもの。複数の方角を同時に表すことはない。
  // おまけで桂馬の移動も追加しておく。
  enum Direct { DIRECT_RU, DIRECT_R, DIRECT_RD, DIRECT_U, DIRECT_D, DIRECT_LU, DIRECT_L, DIRECT_LD,
    DIRECT_NB, DIRECT_ZERO = 0, DIRECT_RUU=8,DIRECT_LUU,DIRECT_RDD,DIRECT_LDD,DIRECT_NB_PLUS4 };

  // Directionsに相当するものを引数に渡して1つ方角を取り出す。
  inline Direct pop_directions(Directions& d) { return (Direct)pop_lsb(d); }

  // DirectからDirectionsへの逆変換
  inline Directions to_directions(Direct d) { return Directions(1 << d); }

  inline bool is_ok(Direct d) { return DIRECT_ZERO <= d && d < DIRECT_NB_PLUS4; }

  // DirectをSquareWithWall型の差分値で表現したもの。
  const SquareWithWall DirectToDeltaWW_[DIRECT_NB] = { SQWW_RU,SQWW_R,SQWW_RD,SQWW_U,SQWW_D,SQWW_LU,SQWW_L,SQWW_LD, };
  inline SquareWithWall DirectToDeltaWW(Direct d) { ASSERT_LV3(is_ok(d));  return DirectToDeltaWW_[d]; }
}

// 与えられた3升が縦横斜めの1直線上にあるか。駒を移動させたときに開き王手になるかどうかを判定するのに使う。
// 例) 王がsq1, pinされている駒がsq2にあるときに、pinされている駒をsq3に移動させたときにis_aligned(sq1,sq2,sq3)であれば、
//  pinされている方向に沿った移動なので開き王手にはならないと判定できる。
// ただし玉はsq1として、sq2,sq3は同じ側にいるものとする。(玉を挟んでの一直線は一直線とはみなさない)
inline bool is_aligned(Square sq1 /* is ksq */, Square sq2, Square sq3)
{
  auto d1 = Effect8::directions_of(sq1, sq2);
  return d1 ? d1 == Effect8::directions_of(sq1, sq3) : false;
}

// --------------------
//     探索深さ
// --------------------

// 通常探索時の最大探索深さ
const int MAX_PLY = MAX_PLY_NUM;

// 探索深さを表現するためのenum
enum Depth : int32_t
{
  // 探索深さ0
  DEPTH_ZERO = 0,

  // Depthは1手をONE_PLY倍にスケーリングする。
  ONE_PLY = 2 ,

  // 最大深さ
  DEPTH_MAX = MAX_PLY*(int)ONE_PLY ,

  // 静止探索で王手がかかっているときにこれより少ない残り探索深さでの探索した結果が置換表にあってもそれは信用しない
  DEPTH_QS_CHECKS = 0*(int)ONE_PLY,
  // 静止探索で王手がかかっていないとき。
  DEPTH_QS_NO_CHECKS = -1*(int)ONE_PLY,
  // 静止探索でこれより深い(残り探索深さが少ない)ところではRECAPTURESしか生成しない。
  DEPTH_QS_RECAPTURES = -3*(int)ONE_PLY,

  // DEPTH_NONEは探索せずに値を求めたという意味に使う。
  DEPTH_NONE = -6 * (int)ONE_PLY
};

// --------------------
//     評価値の性質
// --------------------

// searchで探索窓を設定するので、この窓の範囲外の値が返ってきた場合、
// high fail時はこの値は上界(真の値はこれより小さい)、low fail時はこの値は下界(真の値はこれより大きい)
// である。
enum Bound {
  BOUND_NONE,  // 探索していない(DEPTH_NONE)ときに、最善手か、静的評価スコアだけを置換表に格納したいときに用いる。
  BOUND_UPPER, // 上界(真の評価値はこれより小さい) = 詰みのスコアや、nonPVで評価値があまり信用ならない状態であることを表現する。
  BOUND_LOWER, // 下界(真の評価値はこれより大きい)
  BOUND_EXACT = BOUND_UPPER | BOUND_LOWER // 真の評価値と一致している。PV nodeでかつ詰みのスコアでないことを表現する。
};

// --------------------
//        評価値
// --------------------

// 置換表に格納するときにあまりbit数が多いともったいないので値自体は16bitで収まる範囲で。
enum Value : int32_t
{
  VALUE_ZERO = 0,

  // 1手詰めのスコア(例えば、3手詰めならこの値より2少ない)
  VALUE_MATE = 32000,

  // Valueの取りうる最大値(最小値はこの符号を反転させた値)
  VALUE_INFINITE = 32001,

  // 無効な値
  VALUE_NONE = 32002,

  VALUE_MATE_IN_MAX_PLY  =   int(VALUE_MATE) - MAX_PLY,   // MAX_PLYでの詰みのときのスコア。
  VALUE_MATED_IN_MAX_PLY =  -int(VALUE_MATE_IN_MAX_PLY), // MAX_PLYで詰まされるときのスコア。

  // 勝ち手順が何らか証明されているときのスコア下限値
  VALUE_KNOWN_WIN        =   int(VALUE_MATE_IN_MAX_PLY) - 1000,

  // 千日手による優等局面への突入したときのスコア
  // これある程度離しておかないと、置換表に書き込んで、相手番から見て、これから
  // singularの判定なんかをしようと思ったときに
  // -VALUE_KNOWN_WIN - margin が、VALUE_MATED_IN_MAX_PLYを下回るとまずいので…。
  VALUE_SUPERIOR             = 28000,

  // 評価関数の返す値の最大値(2**14ぐらいに収まっていて欲しいところだが..)
  VALUE_MAX_EVAL             = 25000,
};

// ply手で詰ませるときのスコア
inline Value mate_in(int ply) {  return (Value)(VALUE_MATE - ply);}

// ply手で詰まされるときのスコア
inline Value mated_in(int ply) {  return (Value)(-VALUE_MATE + ply);}


// --------------------
//        駒
// --------------------

enum Piece : int32_t
{
  // 金の順番を飛の後ろにしておく。KINGを8にしておく。
  // こうすることで、成りを求めるときに pc |= 8;で求まり、かつ、先手の全種類の駒を列挙するときに空きが発生しない。(DRAGONが終端になる)
  NO_PIECE, PAWN/*歩*/, LANCE/*香*/, KNIGHT/*桂*/, SILVER/*銀*/, BISHOP/*角*/, ROOK/*飛*/, GOLD/*金*/ ,
  KING = 8/*玉*/, PRO_PAWN /*と*/, PRO_LANCE /*成香*/, PRO_KNIGHT /*成桂*/, PRO_SILVER /*成銀*/, HORSE/*馬*/, DRAGON/*龍*/, QUEEN/*未使用*/,
  // 以下、先後の区別のある駒(Bがついているのは先手、Wがついているのは後手)
  B_PAWN = 1 , B_LANCE, B_KNIGHT, B_SILVER, B_BISHOP, B_ROOK, B_GOLD , B_KING, B_PRO_PAWN, B_PRO_LANCE, B_PRO_KNIGHT, B_PRO_SILVER, B_HORSE, B_DRAGON, B_QUEEN,
  W_PAWN = 17, W_LANCE, W_KNIGHT, W_SILVER, W_BISHOP, W_ROOK, W_GOLD , W_KING, W_PRO_PAWN, W_PRO_LANCE, W_PRO_KNIGHT, W_PRO_SILVER, W_HORSE, W_DRAGON, W_QUEEN,
  PIECE_NB, // 終端
  PIECE_ZERO = 0,

  // --- 特殊な定数

  PIECE_PROMOTE = 8, // 成り駒と非成り駒との差(この定数を足すと成り駒になる)
  PIECE_WHITE = 16,  // これを先手の駒に加算すると後手の駒になる。
  PIECE_RAW_NB = 8,  // 非成駒の終端

  PIECE_HAND_ZERO = PAWN, // 手駒の開始位置
  PIECE_HAND_NB = KING  , // 手駒になる駒種の最大+1

  HDK = KING,       // Position::pieces()で使うときの定数。H=Horse,D=Dragon,K=Kingの合体したBitboardが返る。

  // 指し手生成(GeneratePieceMove = GPM)でtemplateの引数として使うマーカー的な値。変更する可能性があるのでユーザーは使わないでください。
  GPM_BR   = 100 ,     // Bishop Rook
  GPM_GBR  = 101 ,     // Gold Bishop Rook
  GPM_GHD  = 102 ,     // Gold Horse Dragon
  GPM_GHDK = 103 ,     // Gold Horse Dragon King
};

// USIプロトコルで駒を表す文字列を返す。
inline std::string usi_piece(Piece pc) { return std::string(". P L N S B R G K +P+L+N+S+B+R+G+.p l n s b r g k +p+l+n+s+b+r+g+k").substr(pc * 2, 2); }

// 駒に対して、それが先後、どちらの手番の駒であるかを返す。
constexpr Color color_of(Piece pc) { return (pc & PIECE_WHITE) ? WHITE : BLACK; }

// 後手の歩→先手の歩のように、後手という属性を取り払った駒種を返す
constexpr Piece type_of(Piece pc) { return (Piece)(pc & 15); }

// 成ってない駒を返す。後手という属性も消去する。
// 例) 成銀→銀 , 後手の馬→先手の角
// ただし、pc == KINGでの呼び出しはNO_PIECEが返るものとする。
constexpr Piece raw_type_of(Piece pc) { return (Piece)(pc & 7); }

// pcとして先手の駒を渡し、cが後手なら後手の駒を返す。cが先手なら先手の駒のまま。pcとしてNO_PIECEは渡してはならない。
inline Piece make_piece(Color c, Piece pt) { ASSERT_LV3(color_of(pt) == BLACK && pt!=NO_PIECE);  return (Piece)(pt + (c << 4)); }

// pcが遠方駒であるかを判定する。LANCE,BISHOP(5),ROOK(6),HORSE(13),DRAGON(14)
inline bool has_long_effect(Piece pc) { return (type_of(pc) == LANCE) || (((pc+1) & 6)==6); }

// Pieceの整合性の検査。assert用。
constexpr bool is_ok(Piece pc) { return NO_PIECE <= pc && pc < PIECE_NB; }

// Pieceを綺麗に出力する(USI形式ではない) 先手の駒は大文字、後手の駒は小文字、成り駒は先頭に+がつく。盤面表示に使う。
// "PRETTY_JP"をdefineしていれば、日本語文字での表示になる。
std::string pretty(Piece pc);

// ↑のpretty()だと先手の駒を表示したときに先頭にスペースが入るので、それが嫌な場合はこちらを用いる。
inline std::string pretty2(Piece pc) { ASSERT_LV1(color_of(pc) == BLACK); auto s = pretty(pc); return s.substr(1, s.length() - 1); }

// USIで、盤上の駒を表現する文字列
// ※　歩Pawn 香Lance 桂kNight 銀Silver 角Bishop 飛Rook 金Gold 王King
extern std::string PieceToCharBW;

// PieceをUSI形式で表示する。
std::ostream& operator<<(std::ostream& os, Piece pc);

// --------------------
//        駒箱
// --------------------

// Positionクラスで用いる、駒リスト(どの駒がどこにあるのか)を管理するときの番号。
enum PieceNo {
  PIECE_NO_PAWN = 0, PIECE_NO_LANCE = 18, PIECE_NO_KNIGHT = 22, PIECE_NO_SILVER = 26,
  PIECE_NO_GOLD = 30, PIECE_NO_BISHOP = 34, PIECE_NO_ROOK = 36, PIECE_NO_KING = 38, 
  PIECE_NO_BKING = 38, PIECE_NO_WKING = 39, // 先手、後手の玉の番号が必要な場合はこっちを用いる
  PIECE_NO_ZERO = 0, PIECE_NO_NB = 40, 
};

// PieceNoの整合性の検査。assert用。
constexpr bool is_ok(PieceNo pn) { return PIECE_NO_ZERO <= pn && pn < PIECE_NO_NB; }

// --------------------
//       指し手
// --------------------

// 指し手 bit0..6 = 移動先のSquare、bit7..13 = 移動元のSquare(駒打ちのときは駒種)、bit14..駒打ちか、bit15..成りか
enum Move : uint16_t {

  MOVE_NONE    = 0,             // 無効な移動

  MOVE_NULL    = (1 << 7) + 1,  // NULL MOVEを意味する指し手。Square(1)からSquare(1)への移動は存在しないのでここを特殊な記号として使う。
  MOVE_RESIGN  = (2 << 7) + 2,  // << で出力したときに"resign"と表示する投了を意味する指し手。
  MOVE_WIN     = (3 << 7) + 3,  // 入玉時の宣言勝ちのために使う特殊な指し手

  MOVE_DROP    = 1 << 14,       // 駒打ちフラグ
  MOVE_PROMOTE = 1 << 15,       // 駒成りフラグ
};

// 指し手の移動元の升を返す
constexpr Square move_from(Move m) { return Square((m >> 7) & 0x7f); }
  
// 指し手の移動先の升を返す
constexpr Square move_to(Move m) { return Square(m & 0x7f); }

// 指し手が駒打ちか？
constexpr bool is_drop(Move m){ return (m & MOVE_DROP)!=0; }

// 指し手が成りか？
constexpr bool is_promote(Move m) { return (m & MOVE_PROMOTE)!=0; }

// 駒打ち(is_drop()==true)のときの打った駒
constexpr Piece move_dropped_piece(Move m) { return (Piece)move_from(m); }

// fromからtoに移動する指し手を生成して返す
constexpr Move make_move(Square from, Square to) { return (Move)(to + (from << 7)); }

// fromからtoに移動する、成りの指し手を生成して返す
constexpr Move make_move_promote(Square from, Square to) { return (Move)(to + (from << 7) + MOVE_PROMOTE); }

// Pieceをtoに打つ指し手を生成して返す
// constexpr 
constexpr Move make_move_drop(Piece pt, Square to) { return (Move)(to + (pt << 7) + MOVE_DROP); }

// 指し手がおかしくないかをテストする
// ただし、盤面のことは考慮していない。MOVE_NULLとMOVE_NONEであるとfalseが返る。
// これら２つの定数は、移動元と移動先が等しい値になっている。このテストだけをする。
inline bool is_ok(Move m) {
  // return move_from(m)!=move_to(m);
  // とやりたいところだが、駒打ちでfromのbitを使ってしまっているのでそれだとまずい。
  // 駒打ちのbitも考慮に入れるために次のように書く。
  return (m >> 7) != (m & 0x7f);
}

// 見た目に、わかりやすい形式で表示する
std::string pretty(Move m);

// 移動させた駒がわかっているときに指し手をわかりやすい表示形式で表示する。
std::string pretty(Move m, Piece movedPieceType);

// USI形式の文字列にする。
std::string to_usi_string(Move m);

// USI形式で指し手を表示する
inline std::ostream& operator<<(std::ostream& os, Move m) { os << to_usi_string(m); return os; }

// --------------------
//   拡張された指し手
// --------------------

// 指し手とオーダリングのためのスコアがペアになっている構造体。
// オーダリングのときにスコアで並べ替えしたいが、一つになっているほうが並び替えがしやすいのでこうしてある。
struct ExtMove {

  Move move;   // 指し手
  Value value; // これはMovePickerが指し手オーダリングのために並び替えるときに用いる値(≠評価値)。

  // Move型とは暗黙で変換できていい。

  operator Move() const { return move; }
  void operator=(Move m) { move = m; }

#ifdef KEEP_PIECE_IN_COUNTER_MOVE
  // killerやcounter moveを32bit(Move32)で扱うときに、無理やりExtMoveに代入するためのhack
  void operator=(Move32 m) { *(Move32*)this = m; }
#endif
};

// ExtMoveの並べ替えを行なうので比較オペレーターを定義しておく。
inline bool operator<(const ExtMove& first, const ExtMove& second) {
  return first.value < second.value;
}

inline std::ostream& operator<<(std::ostream& os, ExtMove m) { os << m.move << '(' << m.value << ')'; return os; }

// --------------------
//       手駒
// --------------------

// 手駒
// 歩の枚数を8bit、香、桂、銀、角、飛、金を4bitずつで持つ。こうすると16進数表示したときに綺麗に表示される。(なのはのアイデア)
enum Hand : uint32_t { HAND_ZERO = 0, };
 
// 手駒のbit位置
constexpr int PIECE_BITS[PIECE_HAND_NB] = { 0, 0 /*歩*/, 8 /*香*/, 12 /*桂*/, 16 /*銀*/, 20 /*角*/, 24 /*飛*/ , 28 /*金*/ };

// Piece(歩,香,桂,銀,金,角,飛)を手駒に変換するテーブル
constexpr Hand PIECE_TO_HAND[PIECE_HAND_NB] = { (Hand)0, (Hand)(1 << PIECE_BITS[PAWN]) /*歩*/, (Hand)(1 << PIECE_BITS[LANCE]) /*香*/, (Hand)(1 << PIECE_BITS[KNIGHT]) /*桂*/,
(Hand)(1 << PIECE_BITS[SILVER]) /*銀*/,(Hand)(1 << PIECE_BITS[BISHOP]) /*角*/,(Hand)(1 << PIECE_BITS[ROOK]) /*飛*/,(Hand)(1 << PIECE_BITS[GOLD]) /*金*/ };

// その持ち駒を表現するのに必要なbit数のmask(例えば3bitなら2の3乗-1で7)
constexpr int PIECE_BIT_MASK[PIECE_HAND_NB] = { 0,31/*歩は5bit*/,7/*香は3bit*/,7/*桂*/,7/*銀*/,3/*角*/,3/*飛*/,7/*金*/ };

constexpr int PIECE_BIT_MASK2[PIECE_HAND_NB] = { 0,
  PIECE_BIT_MASK[PAWN]   << PIECE_BITS[PAWN]  , PIECE_BIT_MASK[LANCE]  << PIECE_BITS[LANCE] , PIECE_BIT_MASK[KNIGHT] << PIECE_BITS[KNIGHT],
  PIECE_BIT_MASK[SILVER] << PIECE_BITS[SILVER], PIECE_BIT_MASK[BISHOP] << PIECE_BITS[BISHOP], PIECE_BIT_MASK[ROOK]   << PIECE_BITS[ROOK]  ,
  PIECE_BIT_MASK[GOLD]   << PIECE_BITS[GOLD] };

// 駒の枚数が格納されているbitが1となっているMASK。(駒種を得るときに使う)
constexpr int32_t HAND_BIT_MASK = PIECE_BIT_MASK2[PAWN] | PIECE_BIT_MASK2[LANCE] | PIECE_BIT_MASK2[KNIGHT] | PIECE_BIT_MASK2[SILVER]
          | PIECE_BIT_MASK2[BISHOP] | PIECE_BIT_MASK2[ROOK] | PIECE_BIT_MASK2[GOLD];

// 余らせてあるbitの集合。
constexpr int32_t HAND_BORROW_MASK = (HAND_BIT_MASK << 1) & ~HAND_BIT_MASK;


// 手駒pcの枚数を返す。
inline int hand_count(Hand hand, Piece pr) { ASSERT_LV2(PIECE_HAND_ZERO <= pr && pr < PIECE_HAND_NB); return (hand >> PIECE_BITS[pr]) & PIECE_BIT_MASK[pr]; }

// 手駒pcを持っているかどうかを返す。
inline int hand_exists(Hand hand, Piece pr) { ASSERT_LV2(PIECE_HAND_ZERO <= pr && pr < PIECE_HAND_NB); return hand & PIECE_BIT_MASK2[pr]; }

// 手駒にpcをc枚加える
inline void add_hand(Hand &hand, Piece pr, int c = 1) { hand = (Hand)(hand + PIECE_TO_HAND[pr] * c); }

// 手駒からpcをc枚減ずる
inline void sub_hand(Hand &hand, Piece pr, int c = 1) { hand = (Hand)(hand - PIECE_TO_HAND[pr] * c); }


// 手駒h1のほうがh2より優れているか。(すべての種類の手駒がh2のそれ以上ある)
// 優等局面の判定のとき、局面のhash key(StateInfo::key() )が一致していなくて、盤面のhash key(StateInfo::board_key() )が
// 一致しているときに手駒の比較に用いるので、手駒がequalというケースは前提により除外されているから、この関数を以ってsuperiorであるという判定が出来る。
inline bool hand_is_equal_or_superior(Hand h1, Hand h2) { return ((h1-h2) & HAND_BORROW_MASK) == 0; }

// 手駒を表示する(USI形式ではない) デバッグ用
std::ostream& operator<<(std::ostream& os, Hand hand);

// --------------------
// 手駒情報を直列化したもの
// --------------------

// 特定種の手駒を持っているかどうかをbitで表現するクラス
// bit0..歩を持っているか , bit1..香 , bit2..桂 , bit3..銀 , bit4..角 , bit5..飛 , bit6..金 , bit7..玉(フラグとして用いるため)
enum HandKind : uint32_t { HAND_KIND_PAWN = 1 << (PAWN-1), HAND_KIND_LANCE=1 << (LANCE-1) , HAND_KIND_KNIGHT = 1 << (KNIGHT-1),
  HAND_KIND_SILVER = 1 << (SILVER-1), HAND_KIND_BISHOP = 1 << (BISHOP-1), HAND_KIND_ROOK = 1 << (ROOK-1) , HAND_KIND_GOLD = 1 << (GOLD-1) ,
  HAND_KIND_KING = 1 << (KING-1) , HAND_KIND_ZERO = 0,};

// Hand型からHandKind型への変換子
// 例えば歩の枚数であれば5bitで表現できるが、011111bを加算すると1枚でもあれば桁あふれしてbit5が1になる。
// これをPEXT32で回収するという戦略。
inline HandKind toHandKind(Hand h) {return (HandKind)PEXT32(h + HAND_BIT_MASK, HAND_BORROW_MASK);}

// 特定種類の駒を持っているかを判定する
inline bool hand_exists(HandKind hk, Piece pt) { ASSERT_LV2(PIECE_HAND_ZERO <= pt && pt < PIECE_HAND_NB);  return static_cast<bool>(hk & (1 << (pt - 1))); }

// 歩以外の手駒を持っているかを判定する
inline bool hand_exceptPawnExists(HandKind hk) { return hk & ~HAND_KIND_PAWN; }

// 手駒の有無を表示する(USI形式ではない) デバッグ用
std::ostream& operator<<(std::ostream& os, HandKind hk);

// --------------------
//    指し手生成器
// --------------------

// 将棋のある局面の合法手の最大数。593らしいが、保険をかけて少し大きめにしておく。
const int MAX_MOVES = 600;

// 生成する指し手の種類
enum MOVE_GEN_TYPE
{
  // LEGAL/LEGAL_ALL以外は自殺手が含まれることがある(pseudo-legal)ので、do_moveの前にPosition::legal()でのチェックが必要。

  NON_CAPTURES,	// 駒を取らない指し手
  CAPTURES,			// 駒を取る指し手

  CAPTURES_PRO_PLUS,      // CAPTURES + 価値のかなりあると思われる成り(歩だけ)
  NON_CAPTURES_PRO_MINUS, // NON_CAPTURES - 価値のかなりあると思われる成り(歩だけ)

  // BonanzaではCAPTURESに銀以外の成りを含めていたが、Aperyでは歩の成り以外は含めない。
  // あまり変な成りまで入れるとオーダリングを阻害する。
  // 本ソースコードでは、NON_CAPTURESとCAPTURESは使わず、CAPTURES_PRO_PLUSとNON_CAPTURES_PRO_MINUSを使う。
  
  // note : NON_CAPTURESとCAPTURESとの生成される指し手の集合は被覆していない。
  // note : CAPTURES_PRO_PLUSとNON_CAPTURES_PRO_MINUSとの生成される指し手の集合も被覆していない。
  // →　被覆させないことで、二段階に指し手生成を分解することが出来る。

  EVASIONS ,             // 王手の回避(指し手生成元で王手されている局面であることがわかっているときはこちらを呼び出す)
  EVASIONS_ALL,          // EVASIONS + 歩の不成なども含む。

  NON_EVASIONS,          // 王手の回避ではない手(指し手生成元で王手されていない局面であることがわかっているときのすべての指し手)
  NON_EVASIONS_ALL,      // NON_EVASIONS + 歩の不成などを含む。
  
  // 以下の2つは、pos.legalを内部的に呼び出すので生成するのに時間が少しかかる。棋譜の読み込み時などにしか使わない。
  LEGAL,                 // 合法手すべて。ただし、2段目の歩・香の不成や角・飛の不成は生成しない。
  LEGAL_ALL,             // 合法手すべて

  CHECKS,                // 王手となる指し手(歩の不成などは含まない)
  CHECKS_ALL,            // 王手となる指し手(歩の不成なども含む)

  QUIET_CHECKS,          // 王手となる指し手(歩の不成などは含まない)で、CAPTURESの指し手は含まない指し手
  QUIET_CHECKS_ALL,      // 王手となる指し手(歩の不成なども含む)でCAPTURESの指し手は含まない指し手

  RECAPTURES,            // 指定升への移動の指し手のみを生成する。(歩の不成などは含まない)
  RECAPTURES_ALL,        // 指定升への移動の指し手のみを生成する。(歩の不成なども含む)
};

struct Position; // 前方宣言

// 指し手を生成器本体
// gen_typeとして生成する指し手の種類をシてする。gen_allをfalseにすると歩の不成、香の8段目の不成は生成しない。通常探索中はそれでいいはず。
// mlist : 指し手を返して欲しい指し手生成バッファのアドレス
// 返し値 : 生成した指し手の終端
struct CheckInfo;
template <MOVE_GEN_TYPE gen_type> ExtMove* generateMoves(const Position& pos, ExtMove* mlist);
template <MOVE_GEN_TYPE gen_type> ExtMove* generateMoves(const Position& pos, ExtMove* mlist,Square recapSq); // RECAPTURES,RECAPTURES_ALL専用

// MoveGeneratorのwrapper。範囲forで回すときに便利。
template<MOVE_GEN_TYPE GenType>
struct MoveList {
  // 局面をコンストラクタの引数に渡して使う。すると指し手が生成され、lastが初期化されるので、
  // このclassのbegin(),end()が正常な値を返すようになる。
  // CHECKS,CHECKS_NON_PRO_PLUSを生成するときは、事前にpos.check_info_update();でCheckInfoをupdateしておくこと。
  explicit MoveList(const Position& pos) : last(generateMoves<GenType>(pos, mlist)){}
    
  // 内部的に持っている指し手生成バッファの先頭
  const ExtMove* begin() const { return mlist; }

  // 生成された指し手の末尾のひとつ先
  const ExtMove* end() const { return last; }

  // 生成された指し手のなかに引数で指定された指し手が含まれているかの判定。
  // ASSERTなどで用いる。遅いので通常探索等では用いないこと。
  bool contains(Move move) const {
    for (const auto& m : *this) if (m == move) return true;
    return false;
  }

  // 生成された指し手の数
  size_t size() const { return last - mlist; }

  // i番目の要素を返す
  const ExtMove at(size_t i) const { ASSERT_LV3(0<=i && i<size()); return begin()[i]; }

private:
  // 指し手生成バッファも自前で持っている。
  ExtMove mlist[MAX_MOVES], *last;
};

// --------------------
//       置換表
// --------------------

// 局面のハッシュキー
// 盤面(盤上の駒 + 手駒)に対して、Zobrist Hashでそれに対応する値を計算する。
typedef uint64_t Key;

// --------------------
//        探索
// --------------------

// 入玉ルール設定
enum EnteringKingRule
{
  EKR_NONE ,           // 入玉ルールなし
  EKR_24_POINT,        // 24点法(31点以上で宣言勝ち)
  EKR_27_POINT,        // 27点法 = CSAルール
  EKR_TRY_RULE,        // トライルール
};

// 千日手の状態
enum RepetitionState
{
  REPETITION_NONE,     // 千日手ではない
  REPETITION_WIN ,     // 連続王手の千日手による勝ち
  REPETITION_LOSE,     // 連続王手の千日手による負け
  REPETITION_DRAW,     // 連続王手ではない普通の千日手
  REPETITION_SUPERIOR, // 優等局面(盤上の駒が同じで手駒が相手より優れている)
  REPETITION_INFERIOR, // 劣等局面(盤上の駒が同じで手駒が相手より優れている)
  REPETITION_NB,
};

inline bool is_ok(RepetitionState rs) { return REPETITION_NONE <= rs && rs < REPETITION_NB; }

// 引き分け時のスコア
extern Value drawValueTable[REPETITION_NB][COLOR_NB];
inline Value draw_value(RepetitionState rs, Color c) { ASSERT_LV3(is_ok(rs)); return drawValueTable[rs][c]; }

// --------------------
//      評価関数
// --------------------

namespace Eval {
  enum BonaPiece : int16_t;

  Value evaluate(const Position& pos);
}

// --------------------
//     USI関連
// --------------------

namespace USI {
  struct Option;

  // USIのoption名と、それに対応する設定内容を保持しているclass
  typedef std::map<std::string, Option> OptionsMap;

  // USIプロトコルで指定されるoptionの内容を保持するclass
  struct Option {
    typedef void(*OnChange)(const Option&);

    Option(OnChange f = nullptr) : type("button"), min(0), max(0), on_change(f) {}

    // bool型のoption デフォルト値が v
    Option(bool v, OnChange f = nullptr) : type("check"),min(0),max(0),on_change(f)
    { defaultValue = currentValue = v ? "true" : "false"; }

    // int型で(min,max)でデフォルトがv
    Option(int v, int min_, int max_, OnChange f = nullptr) : type("spin"),min(min_),max(max_),on_change(f)
    { defaultValue = currentValue = std::to_string(v); }

    // combo型。内容的には、string型と同等。
    // list = コンボボックスに表示する値。v = デフォルト値かつ現在の値
    Option(const std::vector<std::string>&list, const std::string& v, OnChange f = nullptr) : type("combo"), on_change(f) ,list(list)
    { defaultValue = currentValue = v; }

    // USIプロトコル経由で値を設定されたときにそれをcurrentValueに反映させる。
    Option& operator=(const std::string&);
    Option& operator=(const char* ptr) { return *this = std::string(ptr); };

    // 起動時に設定を代入する。
    void operator<<(const Option&);

    // int,bool型への暗黙の変換子
    operator int() const {
      ASSERT_LV1(type == "check" || type == "spin");
      return type == "spin" ? stoi(currentValue) : currentValue == "true";
    }

    // string型への暗黙の変換子
    operator std::string() const { ASSERT_LV1(type == "string" || type == "combo");  return currentValue; }

  private:
    friend std::ostream& operator<<(std::ostream& os, const OptionsMap& om);

    // 出力するときの順番。この順番に従ってGUIの設定ダイアログに反映されるので順番重要！
    size_t idx;

    std::string defaultValue, currentValue, type;

    // int型のときの最小と最大
    int min, max;

    // combo boxのときの表示する文字列リスト
    std::vector<std::string> list;

    // 値が変わったときに呼び出されるハンドラ
    OnChange on_change;
  };

  // USIメッセージ応答部(起動時に、各種初期化のあとに呼び出される)
  void loop(int argc, char* argv[]);

  // optionのdefault値を設定する。
  void init(OptionsMap&);

  // pv(読み筋)をUSIプロトコルに基いて出力する。
  // iteration_depth = 反復深化のiteration深さ。
  std::string pv(const Position& pos, int iteration_depth, Value alpha, Value beta);

  // USIプロトコルで、idxの順番でoptionを出力する。
  std::ostream& operator<<(std::ostream& os, const OptionsMap& om);

  // USIプロトコルの形式でValue型を出力する。
  // 歩が100になるように正規化するので、operator <<()をこういう仕様にすると
  // 実際の値と異なる表示になりデバッグがしにくくなるから、そうはしていない。
  std::string score_to_usi(Value v);

  // USIに追加オプションを設定したいときは、この関数を定義すること。
  // USI::init()のなかからコールバックされる。
  void extra_option(USI::OptionsMap& o);
}

// USIのoption設定はここに保持されている。
extern USI::OptionsMap Options;

// 局面posとUSIプロトコルによる指し手を与えて
// もし可能なら等価で合法な指し手を返す。(合法でないときはMOVE_NONEを返す。"resign"に対してはMOVE_RESIGNを返す。)
Move move_from_usi(const Position& pos, const std::string& str);

// 合法かのテストはせずにともかく変換する版。
Move move_from_usi(const std::string& str);

// --------------------
//  operators and macros
// --------------------

#include "extra/macros.h"


#endif // of #ifndef _SHOGI_H_
