#ifndef __LEARN_WEIGHT_H__
#define __LEARN_WEIGHT_H__

// 評価関数の機械学習のときに用いる重み配列などに関する機械学習用ツール類一式

#include "learn.h"
#if defined (EVAL_LEARN)

#if defined(SGD_UPDATE)
#include "../misc.h"  // PRNG
#endif

#if V_FRACTION_BITS == 8
typedef s8 V_FRACTION_TYPE;
#elif V_FRACTION_BITS == 16
typedef s16 V_FRACTION_TYPE;
#elif V_FRACTION_BITS == 32
typedef s32 V_FRACTION_TYPE;
#elif V_FRACTION_BITS == 64
typedef s64 V_FRACTION_TYPE;
#endif

namespace EvalLearningTools
{
	// -------------------------------------------------
	//                     初期化
	// -------------------------------------------------

	// このEvalLearningTools名前空間にあるテーブル類を初期化する。
	// 学習の開始までに必ず一度呼び出すこと。
	void init();

	// -------------------------------------------------
	//       勾配等を格納している学習用の配列
	// -------------------------------------------------

#if defined(_MSC_VER)
#pragma pack(push,2)
#elif defined(__GNUC__)
#pragma pack(2)
#endif
	struct Weight
	{
		// mini-batch 1回分の勾配の累積値
		LearnFloatType g;

		// ADA_GRAD_UPDATEのとき。LearnFloatType == floatとして、
		// 合計 4*2 + 4*2 + 1*2 = 18 bytes
		// 1GBの評価関数パラメーターに対してその4.5倍のサイズのWeight配列が確保できれば良い。
		// ただし、構造体のアライメントが4バイト単位になっているとsizeof(Weight)==20なコードが生成されるので
		// pragma pack(2)を指定しておく。

		// SGD_UPDATE の場合、この構造体はさらに10バイト減って、8バイトで済む。

		// AdaGradなどの学習率η(eta)。
		// updateFV()が呼び出されるまでにeta1,2,3,eta1_epoch,eta2_epochは設定されているものとする。
		// update_weights()のepochが、eta1_epochまでeta1から徐々にeta2に変化する。
		// eta2_epoch以降は、eta2から徐々にeta3に変化する。
		static double eta;
		static double eta1;
		static double eta2;
		static double eta3;
		static u64 eta1_epoch;
		static u64 eta2_epoch;

		// etaの一括初期化。0が渡された場合、デフォルト値が設定される。
		static void init_eta(double eta1, double eta2, double eta3, u64 eta1_epoch, u64 eta2_epoch)
		{
			Weight::eta1 = (eta1 != 0) ? eta1 : 30.0;
			Weight::eta2 = (eta2 != 0) ? eta2 : 30.0;
			Weight::eta3 = (eta3 != 0) ? eta3 : 30.0;
			Weight::eta1_epoch = (eta1_epoch != 0) ? eta1_epoch : 0;
			Weight::eta2_epoch = (eta2_epoch != 0) ? eta2_epoch : 0;
		}

		// epochに応じたetaを設定してやる。
		static void calc_eta(u64 epoch)
		{
			if (Weight::eta1_epoch == 0) // eta2適用除外
				Weight::eta = Weight::eta1;
			else if (epoch < Weight::eta1_epoch)
				// 按分する
				Weight::eta = Weight::eta1 + (Weight::eta2 - Weight::eta1) * epoch / Weight::eta1_epoch;
			else if (Weight::eta2_epoch == 0) // eta3適用除外
				Weight::eta = Weight::eta2;
			else if (epoch < Weight::eta2_epoch)
				Weight::eta = Weight::eta2 + (Weight::eta3 - Weight::eta2) * (epoch - Weight::eta1_epoch) / (Weight::eta2_epoch - Weight::eta1_epoch);
			else
				Weight::eta = Weight::eta3;
		}

#if defined (ADA_GRAD_UPDATE) || defined(ADA_PROP_UPDATE)

		// AdaGradのg2
		LearnFloatType g2;

		// vの固定小数表現 8-bits。(vをfloatで持つのもったいないのでvの補助bitとして小数部を持つ)
		// 何bit持つかは、V_FRACTION_BITSの設定で変更できる。
		V_FRACTION_TYPE v_frac;

		// AdaGradでupdateする
		// この関数を実行しているときにgの値やメンバーが書き変わらないことは
		// 呼び出し側で保証されている。atomic演算である必要はない。
		template <typename T>
		void updateFV(T& v)
		{
			// AdaGradの更新式
			//   勾配ベクトルをg、更新したいベクトルをv、η(eta)は定数として、
			//     g2 = g2 + g^2
			//     v = v - ηg/sqrt(g2)

			constexpr double epsilon = 0.000001;

			if (g == LearnFloatType(0))
				return;

			g2 += g * g;

#if defined(ADA_PROP_UPDATE)
			// 少しずつ減衰させることで、学習が硬直するのを防ぐ。
			// (0.99)^100 ≒ 0.366
			g2 = LearnFloatType(g2 * 0.99);
#endif

			// v8は小数部8bit(V_FRACTION_BITS==8のとき)を含んでいるのでこれを復元する。
			// 128倍にすると、-1を保持できなくなるので127倍にしておく。
			// -1.0～+1.0を-127～127で保持している。
			// std::round()限定なら-0.5～+0.5の範囲なので255倍でも良いが、
			// どんな丸め方をするかはわからないので余裕を持たせてある。

			const double m = (s64)1 << (V_FRACTION_BITS - 1);

			double V = v + ((double)v_frac / m);

			V -= eta * (double)g / sqrt((double)g2 + epsilon);

			// Vの値を型の範囲に収まるように制限する。
			// ちなみに、windows.hがmin,maxマクロを定義してしまうのでそれを回避するために、
			// ここでは括弧で括ることで関数形式マクロとして扱われないようにしている。
			V = std::min((double)(std::numeric_limits<T>::max)() , V);
			V = std::max((double)(std::numeric_limits<T>::min)() , V);

			v = (T)round(V);
			v_frac = (V_FRACTION_TYPE)((V - v) * m);

			// この要素に関するmini-batchの1回分の更新が終わったのでgをクリア
			// g[i] = 0;
			// →次元下げの問題があるので、これは呼び出し側で行なうことにする。
		}

#elif defined(SGD_UPDATE)

		static AsyncPRNG prng;

		// 勾配の符号だけ見るSGDでupdateする
		// この関数を実行しているときにgの値やメンバーが書き変わらないことは
		// 呼び出し側で保証されている。atomic演算である必要はない。
		template <typename T>
		void updateFV(T & v)
		{
			if (g == 0)
				return;

			// gの符号だけ見てupdateする。
			// g < 0 なら vを少し足す。
			// g > 0 なら vを少し引く。

			// 整数しか足さないので小数部不要。

			// 0～5ぐらいずつ動かすのがよさげ。
			// ガウス分布っぽいほうが良いので5bitの乱数を発生させて(それぞれのbitは1/2の確率で1である)、
			// それをpop_count()する。このとき、二項分布になっている。
			s16 diff = (s16)POPCNT32((u32)prng.rand(31));

			auto V = v;
			if (g > 0.0)
				V-= diff;
			else
				V+= diff;

			V = std::min((double)(std::numeric_limits<T>::max)(), V);
			V = std::max((double)(std::numeric_limits<T>::min)(), V);

			v = (T)V;
		}

#endif

		// gradの設定
		template <typename T> void set_grad(const T& g_) { g = g_; }

		// gradの加算
		template <typename T> void add_grad(const T& g_) { g += g_; }

		LearnFloatType get_grad() const { return g; }
	};
#if defined(_MSC_VER)
#pragma pack(pop)
#elif defined(__GNUC__)
#pragma pack(0)
#endif

	// 手番つきのweight配列
	// 透過的に扱えるようにするために、Weightと同じメンバを持たせておいてやる。
	struct Weight2
	{
		Weight w[2];

		template <typename T> void updateFV(std::array<T, 2>& v) { for (int i = 0; i<2; ++i) w[i].updateFV(v[i]); }
		template <typename T> void set_grad(const std::array<T, 2>& g) { for (int i = 0; i<2; ++i) w[i].set_grad(g[i]); }
		template <typename T> void add_grad(const std::array<T, 2>& g) { for (int i = 0; i<2; ++i) w[i].add_grad(g[i]); }

		std::array<LearnFloatType, 2> get_grad() const { return std::array<LearnFloatType, 2>{w[0].get_grad(), w[1].get_grad()}; }
	};

	// -------------------------------------------------
	//                  tables
	// -------------------------------------------------

	// 	--- BonaPieceに対してMirrorとInverseを提供する。

	// これらの配列は、init();を呼び出すと初期化される。
	// これらの配列は、以下のKK/KKP/KPPクラスから参照される。

	// あるBonaPieceを相手側から見たときの値を返す
	extern Eval::BonaPiece inv_piece(Eval::BonaPiece p);

	// 盤面上のあるBonaPieceをミラーした位置にあるものを返す。
	extern Eval::BonaPiece mir_piece(Eval::BonaPiece p);

	// 次元下げしたときに、そのなかの一番小さなindexになることが
	// わかっているindexに対してtrueとなっているフラグ配列。
	// この配列もinit()によって初期化される。
	extern std::vector<bool> min_index_flag;

	// mir_piece/inv_pieceの初期化のときに呼び出されるcallback
	// fe_endをユーザー側で拡張するときに用いる。
	// この初期化のときに必要なのでinv_piece_とinv_piece_を公開している。
	// mir_piece_init_functionが呼び出されたタイミングで、fe_old_endまでは
	// これらのテーブルの初期化が完了していることが保証されている。
	extern std::function<void()> mir_piece_init_function;
	extern s16 mir_piece_[Eval::fe_end];
	extern s16 inv_piece_[Eval::fe_end];

	// -------------------------------------------------
	// Weight配列を直列化したときのindexを計算したりするヘルパー。
	// -------------------------------------------------

	// 注意 : 上記のinv_piece/mir_pieceを間接的に参照するので、
	// 最初にinit()を呼び出して初期化すること。

	struct KK
	{
		KK() {}
		KK(Square king0, Square king1) : king0_(king0), king1_(king1), inverse_(false) {}
		KK(Square king0, Square king1,bool inverse) : king0_(king0), king1_(king1) , inverse_(inverse) {}

		// KK,KKP,KPP配列を直列化するときの通し番号の、KKの最小値、最大値。
		static u64 min_index() { return 0; }
		static u64 max_index() { return min_index() + (u64)SQ_NB*(u64)SQ_NB; }

		// 与えられたindexが、min_index()以上、max_index()未満にあるかを判定する。
		static bool is_ok(u64 index) { return min_index() <= index && index < max_index(); }

		// indexからKKのオブジェクトを生成するbuilder
		static KK fromIndex(u64 index)
		{
			index -= min_index();
			int king1 = (int)(index % SQ_NB);
			index /= SQ_NB;
			int king0 = (int)(index  /* % SQ_NB */);
			ASSERT_LV3(king0 < SQ_NB);
			return KK((Square)king0, (Square)king1 , false);
		}

		// fromIndex()を用いてこのオブジェクトを構築したときに、以下のアクセッサで情報が得られる。
		Square king0() const { return king0_; }
		Square king1() const { return king1_; }

// 次元下げの数
#if defined(USE_KK_INVERSE_WRITE)
	#define KK_LOWER_COUNT 4
#elif defined(USE_KK_MIRROR_WRITE)
	#define KK_LOWER_COUNT 2
#else 
	#define KK_LOWER_COUNT 1
#endif

#if defined(USE_KK_INVERSE_WRITE) && !defined(USE_KK_MIRROR_WRITE) 
		// USE_KK_INVERSE_WRITEわ使うならUSE_KK_MIRROR_WRITEも定義して欲しい。
		static_assert(false, "define also USE_KK_MIRROR_WRITE!");
#endif

		// 低次元の配列のindexを得る。
		// USE_KK_INVERSE_WRITEが有効なときは、それらをinverseしたものが[2],[3]に入る。
		// この次元下げに関して、gradの符号は反転させないといけないので注意すること。
		// is_inverse()で判定できるのでこれを利用すると良い。
		void toLowerDimensions(/*out*/KK kk_[KK_LOWER_COUNT]) const {
			kk_[0] = KK(king0_, king1_,false);
#if defined(USE_KK_MIRROR_WRITE)
			kk_[1] = KK(Mir(king0_),Mir(king1_),false);
#if defined(USE_KK_INVERSE_WRITE)
			kk_[2] = KK(Inv(king1_), Inv(king0_),true);
			kk_[3] = KK(Inv(Mir(king1_)) , Inv(Mir(king0_)),true);
#endif
#endif
		}

		// 現在のメンバの値に基いて、直列化されたときのindexを取得する。
		u64 toIndex() const {
			return min_index() + (u64)king0_ * (u64)SQ_NB + (u64)king1_;
		}

		// toLowerDimensionsで次元下げしたものがinverseしたものであるかを返す。
		bool is_inverse() const {
			return inverse_;
		}

		// is_inverse() == trueのときに、gradの手番ではないほうの符号を反転させて返す。
		template <typename T>
		std::array<T, 2> adjust_grad(const std::array<T, 2>& rhs)
		{
			return !is_inverse() ? rhs : std::array<T, 2>{-rhs[0], rhs[1]};
		}

		// 比較演算子
		bool operator==(const KK& rhs) { return king0() == rhs.king0() && king1() == rhs.king1(); }
		bool operator!=(const KK& rhs) { return !(*this == rhs); }

	private:
		Square king0_, king1_ ;
		bool inverse_;
	};

	// デバッグ用出力。
	static std::ostream& operator<<(std::ostream& os, KK rhs)
	{
		os << "KK(" << rhs.king0() << "," << rhs.king1() << ")";
		return os;
	}

	struct KKP
	{
		KKP() {}
		KKP(Square king0, Square king1, Eval::BonaPiece p) : king0_(king0), king1_(king1), piece_(p), inverse_(false) {}
		KKP(Square king0, Square king1, Eval::BonaPiece p,bool inverse) : king0_(king0), king1_(king1), piece_(p),inverse_(inverse) {}

		// KK,KKP,KPP配列を直列化するときの通し番号の、KKPの最小値、最大値。
		static u64 min_index() { return KK::max_index(); }
		static u64 max_index() { return min_index() + (u64)SQ_NB*(u64)SQ_NB*(u64)Eval::fe_end; }

		// 与えられたindexが、min_index()以上、max_index()未満にあるかを判定する。
		static bool is_ok(u64 index) { return min_index() <= index && index < max_index(); }

		// indexからKKPのオブジェクトを生成するbuilder
		static KKP fromIndex(u64 index)
		{
			index -= min_index();
			int piece = (int)(index % Eval::fe_end);
			index /= Eval::fe_end;
			int king1 = (int)(index % SQ_NB);
			index /= SQ_NB;
			int king0 = (int)(index  /* % SQ_NB */);
			ASSERT_LV3(king0 < SQ_NB);
			return KKP((Square)king0, (Square)king1, (Eval::BonaPiece)piece,false);
		}

		// fromIndex()を用いてこのオブジェクトを構築したときに、以下のアクセッサで情報が得られる。
		Square king0() const { return king0_; }
		Square king1() const { return king1_; }
		Eval::BonaPiece piece() const { return piece_; }

		// KKPの次元下げの数
#if defined(USE_KKP_INVERSE_WRITE)
		#define KKP_LOWER_COUNT 4
#elif defined(USE_KKP_MIRROR_WRITE)
		#define KKP_LOWER_COUNT 2
#else
		#define KKP_LOWER_COUNT 1
#endif

#if defined(USE_KKP_INVERSE_WRITE) && !defined(USE_KKP_MIRROR_WRITE) 
		// USE_KKP_INVERSE_WRITEわ使うならUSE_KKP_MIRROR_WRITEも定義して欲しい。
		static_assert(false, "define also USE_KKP_MIRROR_WRITE!");
#endif

		// 低次元の配列のindexを得る。ミラーしたものがkkp_[1]に返る。
		// USE_KKP_INVERSE_WRITEが有効なときは、それらをinverseしたものが[2],[3]に入る。
		// この次元下げに関して、gradの符号は反転させないといけないので注意すること。
		// is_inverse()で判定できるのでこれを利用すると良い。
		void toLowerDimensions(/*out*/ KKP kkp_[KKP_LOWER_COUNT]) const {
			kkp_[0] = KKP(king0_, king1_, piece_,false);
#if defined(USE_KKP_MIRROR_WRITE)
			kkp_[1] = KKP(Mir(king0_), Mir(king1_), mir_piece(piece_),false);
#if defined(USE_KKP_INVERSE_WRITE)
			kkp_[2] = KKP( Inv(king1_), Inv(king0_), inv_piece(piece_),true);
			kkp_[3] = KKP( Inv(Mir(king1_)), Inv(Mir(king0_)) , inv_piece(mir_piece(piece_)),true);
#endif
#endif
		}

		// 現在のメンバの値に基いて、直列化されたときのindexを取得する。
		u64 toIndex() const {
			return min_index() + ((u64)king0_ * (u64)SQ_NB + (u64)king1_) * (u64)Eval::fe_end + (u64)piece_;
		}

		// toLowerDimensionsで次元下げしたものがinverseしたものであるかを返す。
		bool is_inverse() const {
			return inverse_;
		}

		// is_inverse() == trueのときに、gradの手番ではないほうの符号を反転させて返す。
		template <typename T>
		std::array<T, 2> adjust_grad(const std::array<T, 2>& rhs)
		{
			return !is_inverse() ? rhs : std::array<T, 2>{-rhs[0], rhs[1]};
		}

		// 比較演算子
		bool operator==(const KKP& rhs) { return king0() == rhs.king0() && king1() == rhs.king1() && piece() == rhs.piece(); }
		bool operator!=(const KKP& rhs) { return !(*this == rhs); }

	private:
		Square king0_, king1_;
		Eval::BonaPiece piece_;
		bool inverse_;
	};

	// デバッグ用出力。
	static std::ostream& operator<<(std::ostream& os, KKP rhs)
	{
		os << "KKP(" << rhs.king0() << "," << rhs.king1() << "," << rhs.piece() << ")";
		return os;
	}


	struct KPP
	{
		KPP() {}
		KPP(Square king, Eval::BonaPiece p0, Eval::BonaPiece p1) : king_(king), piece0_(p0), piece1_(p1) {}

		// KK,KKP,KPP配列を直列化するときの通し番号の、KPPの最小値、最大値。
		static u64 min_index() { return KKP::max_index(); }
#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
		static u64 max_index() { return min_index() + (u64)SQ_NB*(u64)Eval::fe_end*(u64)Eval::fe_end; }
#else
		// kpp[SQ_NB][fe_end][fe_end]の[fe_end][fe_end]な正方配列の部分を三角配列化する。
		// kpp[SQ_NB][triangle_fe_end]とすると、この三角配列の1行目は要素1個、2行目は2個、…。
		// ゆえに、triangle_fe_end = 1 + 2 + .. + fe_end = fe_end * (fe_end + 1) / 2
		static const u64 triangle_fe_end = (u64)Eval::fe_end*((u64)Eval::fe_end + 1) / 2;
		static u64 max_index() { return min_index() + (u64)SQ_NB*triangle_fe_end; }
#endif

		// 与えられたindexが、min_index()以上、max_index()未満にあるかを判定する。
		static bool is_ok(u64 index) { return min_index() <= index && index < max_index(); }

		// indexからKPPのオブジェクトを生成するbuilder
		static KPP fromIndex(u64 index)
		{
			index -= min_index();

#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
			int piece1 = (int)(index % Eval::fe_end);
			index /= Eval::fe_end;
			int piece0 = (int)(index % Eval::fe_end);
			index /= Eval::fe_end;
#else
			u64 index2 = index % triangle_fe_end;

			// ここにindex2からpiece0,piece1を求める式を書く。
			// これは index2 = i * (i+1) / 2 + j の逆関数となる。
			// j = 0 の場合、i^2 + i - 2 * index2 == 0なので
			// 2次方程式の解の公式から i = (sqrt(8*index2+1) - 1) / 2である。
			// iを整数化したのちに、j = index2 - i * (i + 1) / 2としてjを求めれば良い。

			// BonaPieceは32bit(16bitに収まらない可能性)を想定しているのでこの掛け算は64bitでないといけない。
			int piece1 = int(sqrt(8 * index2 + 1) - 1) / 2;
			int piece0 = int(index2 - (u64)piece1*((u64)piece1 + 1) / 2);

			ASSERT_LV3(piece1 < (int)Eval::fe_end);
			ASSERT_LV3(piece0 < (int)Eval::fe_end);

			index /= triangle_fe_end;
#endif
			int king = (int)(index  /* % SQ_NB */);
			ASSERT_LV3(king < SQ_NB);
			return KPP((Square)king, (Eval::BonaPiece)piece0, (Eval::BonaPiece)piece1);
		}

		// fromIndex()を用いてこのオブジェクトを構築したときに、以下のアクセッサで情報が得られる。
		Square king() const { return king_; }
		Eval::BonaPiece piece0() const { return piece0_; }
		Eval::BonaPiece piece1() const { return piece1_; }


		// 次元下げの数
#if defined(USE_KPP_MIRROR_WRITE)
	#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
		#define KPP_LOWER_COUNT 4
	#else
		#define KPP_LOWER_COUNT 2
	#endif
#else
	#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)
		#define KPP_LOWER_COUNT 2
	#else
		#define KPP_LOWER_COUNT 1
	#endif
#endif

		// 低次元の配列のindexを得る。p1,p2を入れ替えたもの、ミラーしたものなどが返る。
		void toLowerDimensions(/*out*/ KPP kpp_[KPP_LOWER_COUNT]) const {

#if defined(USE_TRIANGLE_WEIGHT_ARRAY)
			// 三角配列を用いる場合は、piece0とpiece1を入れ替えたものは返らないので注意。
			kpp_[0] = KPP(king_, piece0_, piece1_);
#if defined(USE_KPP_MIRROR_WRITE)
			kpp_[1] = KPP(Mir(king_), mir_piece(piece0_), mir_piece(piece1_));
#endif

#else
			// 三角配列を用いない場合
			kpp_[0] = KPP(king_, piece0_, piece1_);
			kpp_[1] = KPP(king_, piece1_, piece0_);
#if defined(USE_KPP_MIRROR_WRITE)
			kpp_[2] = KPP(Mir(king_), mir_piece(piece0_), mir_piece(piece1_));
			kpp_[3] = KPP(Mir(king_), mir_piece(piece1_), mir_piece(piece0_));
#endif
#endif
		}

		// 現在のメンバの値に基いて、直列化されたときのindexを取得する。
		u64 toIndex() const
		{
#if !defined(USE_TRIANGLE_WEIGHT_ARRAY)

			return min_index() + ((u64)king_ * (u64)Eval::fe_end + (u64)piece0_) * (u64)Eval::fe_end + (u64)piece1_;

#else
			// Bonanza6.0で使われているのに似せたマクロ
			auto PcPcOnSq = [](Square k, Eval::BonaPiece i, Eval::BonaPiece j)
			{
				// この三角配列の(i,j)は、i行目のj列目の要素。
				// i行目0列目は、そこまでの要素の合計であるから、1 + 2 + ... + i = i * (i+1) / 2
				// i行目j列目は、これにjを足したもの。i * (i + 1) /2 + j

				// BonaPiece型は、32bitを想定しているので掛け算には気をつけないとオーバーフローする。
				return (u64)k * triangle_fe_end + (u64)(u64(i)*(u64(i)+1) / 2 + u64(j));
			};

			auto k = king_;
			auto i = piece0_;
			auto j = piece1_;

			return min_index() + ( (i >= j) ? PcPcOnSq(k, i, j) : PcPcOnSq(k, j, i));
#endif
		}

		// toLowerDimensionsで次元下げしたものがinverseしたものであるかを返す。
		// KK,KKPとinterfaceを合せるために用意してある。このKPPクラスでは、このメソッドは常にfalseを返す。
		bool is_inverse() const {
			return false;
		}

		// 比較演算子
		bool operator==(const KPP& rhs) {
			return king() == rhs.king() &&
				((piece0() == rhs.piece0() && piece1() == rhs.piece1())
#if defined(USE_TRIANGLE_WEIGHT_ARRAY)
					// 三角配列を用いるときはpiece0とpiece1の入れ替わりを許容する。
				|| (piece0() == rhs.piece1() && piece1() == rhs.piece0())
#endif
					); }
		bool operator!=(const KPP& rhs) { return !(*this == rhs); }


	private:
		Square king_;
		Eval::BonaPiece piece0_, piece1_;
	};

	// デバッグ用出力。
	static std::ostream& operator<<(std::ostream& os, KPP rhs)
	{
		os << "KPP(" << rhs.king() << "," << rhs.piece0() << "," << rhs.piece1() << ")";
		return os;
	}

	// KPPPの4駒関係。ただし、手番ありでミラー等を考慮しないと学習に2TB以上のメモリが必要…。
	// 三角配列を使っても学習のために50GB×12バイト = 600GB必要。
	// ミラーしたもののみを格納するようにしてもの半分ぐらい必要。
	// ここでは、三角配列は必ず用いて、かつミラーしたものを格納するものとする。
	//
	// 事前に以下の定数を定義すること。
	//
	// 定数)
	//   KPPP_KING_SQ
	//
	struct KPPP
	{
		KPPP() {}
		KPPP(Square king, Eval::BonaPiece p0, Eval::BonaPiece p1, Eval::BonaPiece p2) :
			king_(king), piece0_(p0), piece1_(p1), piece2_(p2) { sort_piece(); }

		// KK,KKP,KPP,KPPP配列を直列化するときの通し番号の、KPPPの最小値、最大値。
		static u64 min_index() { return KPP::max_index(); }

		// KPPPのときに扱う玉の升の数。
		// 3段×ミラーなら3段×5筋 = 15みたいな感じ。
		// 2段×ミラーなしなら2×9筋 = 18みたいな感じ。
		static u64 king_sq() {
#define KPPP_KING_SQ 15
			return KPPP_KING_SQ;
		}

		// kppp[king_sq][fe_end][fe_end][fe_end]の[fe_end][fe_end][fe_end]な正方配列の部分を三角配列化する。
		// kppp[king_sq][triangle_fe_end]とすると、この三角配列の0行目から要素数は、0,0,1,3,…,n行目はn(n-1)/2個。
		// ゆえに、
		// triangle_fe_end = Σn(n-1)/2 , n=0..fe_end-1
		//                 =  fe_end * (fe_end - 1) * (fe_end - 2) / 6
		static const u64 triangle_fe_end = ((u64)Eval::fe_end)*((u64)Eval::fe_end - 1)*((u64)Eval::fe_end - 2) / 6;
		static u64 max_index() { return min_index() + (u64)king_sq()*triangle_fe_end; }

		// 与えられたindexが、min_index()以上、max_index()未満にあるかを判定する。
		static bool is_ok(u64 index) { return min_index() <= index && index < max_index(); }

		// 次元下げの数
#define KPPP_LOWER_COUNT 1

		// 低次元の配列のindexを得る。
		// ミラーしたものも、p0,p1,p2を入れ替えたものも返らないので注意。
		void toLowerDimensions(/*out*/ KPPP kppp_[KPPP_LOWER_COUNT]) const
		{
			kppp_[0] = KPPP(king_, piece0_, piece1_,piece2_);
		}

		// indexからKPPPのオブジェクトを生成するbuilder
		static KPPP fromIndex(u64 index)
		{
			ASSERT_LV3(index >= min_index());

			index -= min_index();

			u64 index2 = index % triangle_fe_end;

			// ここにindex2からpiece0,piece1,piece2を求める式を書く。
			// これは index2 = i(i-1)(i-2)/6-1 + j(j+1)/2 + k の逆関数となる。
			// j = k = 0 の場合、3次方程式の解の公式から実根は、 i = ...である。(以下式)
			// iを整数化したのちに、最初の式に入れてKPPのとき同様にjを求めれば良い。

			int piece2;
			if (index2 <= 1)
			{
				// index == 0,1のときだけ実数解が複数ある。
				piece2 = (int)index2 + 2;

			} else {

				//double t = pow(sqrt((243 *index2 * index2 - 1) * 3) + 27 * index2, 1.0 / 3);
				// →　これだとindex2が大きくなるとsqrt()の中身、オーバーフローする。

				double t;
				
				if (index2 < 100000000)
					t = pow(sqrt((243.0 *index2 * index2 - 1)) * sqrt(3.0) + 27 * index2, 1.0 / 3);
				else
					// index2が非常に大きいとき、sqrtの中身、近似的に √243 * index2とみなせるだろう。
					t = pow( index2 * sqrt(243 * 3.0) + 27 * index2, 1.0 / 3);
				
				// 丸めのときに計算誤差でわずかに足りないのを防ぐためのデルタ
				// 大きすぎると1大きい数になってしまう時があるので調整が必要。
				
				const double delta = 0.000000001;

				piece2 = int(t / pow(3.0, 2.0 / 3) + 1.0 / (pow(3.0, 1.0 / 3) * t) + delta) + 1;
				// ううう。ほんまにこんなことせんとあかんのか？(´ω｀)
			}

			// 上式のi(i-1)(i-2)/6(=aとする)のiにpiece2を代入。また、k = 0を代入。
			// j(j+1)/2 = index2 - a
			// 2次方程式の解の公式より..

			u64 a = (u64)piece2*((u64)piece2 - 1)*((u64)piece2 - 2) / 6;
			int piece1 = int((1 + sqrt(8.0 * (index2 - a ) + 1)) / 2);
			u64 b = (u64)piece1 * (piece1 - 1) / 2;
			int piece0 = int(index2 - a - b);

#if 0
			if (!((piece0 < piece1 && piece1 < piece2)))
			{
				std::cout << index << " , " << index2 << "," << a << "," << sqrt(8.0 * (index2 - a) + 1);
			}
#endif

			ASSERT_LV3(piece0 < piece1 && piece1 < piece2);
			ASSERT_LV3(piece2 < (int)Eval::fe_end);
			ASSERT_LV3(piece1 < (int)Eval::fe_end);
			ASSERT_LV3(piece0 < (int)Eval::fe_end);

			index /= triangle_fe_end;

			int king = (int)(index  /* % SQ_NB */);
			ASSERT_LV3(king < SQ_NB);
			return KPPP((Square)king, (Eval::BonaPiece)piece0, (Eval::BonaPiece)piece1 , (Eval::BonaPiece)piece2);
		}

		// 現在のメンバの値に基いて、直列化されたときのindexを取得する。
		u64 toIndex() const
		{
			// Bonanza 6.0で使われているのに似せたマクロ
			// 前提条件) i > j > k であること。
			// i==j,j==kのケースはNG。
			auto PcPcPcOnSq = [](Square king, Eval::BonaPiece i, Eval::BonaPiece j , Eval::BonaPiece k)
			{
				// この三角配列の(i,j,k)は、i行目のj列目の要素。
				// i行目0列0番目は、そこまでの要素の合計であるから、0 + 0 + 1 + 3 + 6 + ... + (i)*(i-1)/2 = i*(i-1)*(i-2)/ 6
				// i行目j列0番目は、そこにjを加味したもの。 + j*(j-1) / 2
				// i行目j列k番目は、そこにkを足したもの。   + k
				ASSERT_LV3(i > j && j > k);

				// BonaPiece型は、32bitを想定しているので掛け算には気をつけないとオーバーフローする。
				return (u64)king * triangle_fe_end + (u64)(
						  u64(i)*(u64(i) - 1) * (u64(i) - 2) / 6
						+ u64(j)*(u64(j) - 1) / 2
						+ u64(k)
					);
			};

			return min_index() + PcPcPcOnSq(king_, piece0_, piece1_, piece2_);
		}

		// fromIndex()を用いてこのオブジェクトを構築したときに、以下のアクセッサで情報が得られる。
		Square king() const { return king_; }
		Eval::BonaPiece piece0() const { return piece0_; }
		Eval::BonaPiece piece1() const { return piece1_; }
		Eval::BonaPiece piece2() const { return piece2_; }

		// toLowerDimensionsで次元下げしたものがinverseしたものであるかを返す。
		// KK,KKPとinterfaceを合せるために用意してある。このKPPPクラスでは、このメソッドは常にfalseを返す。
		bool is_inverse() const {
			return false;
		}

		// 比較演算子
		bool operator==(const KPPP& rhs) {
			return king() == rhs.king() &&
				((piece0() == rhs.piece0() && piece1() == rhs.piece1() && piece2() == rhs.piece2())
#if defined(USE_TRIANGLE_WEIGHT_ARRAY)
					// 三角配列を用いるときはpiece0とpiece1の入れ替わりを許容する。
					|| (piece0() == rhs.piece0() && piece1() == rhs.piece2() && piece2() == rhs.piece1())
					|| (piece0() == rhs.piece1() && piece1() == rhs.piece0() && piece2() == rhs.piece2())
					|| (piece0() == rhs.piece1() && piece1() == rhs.piece2() && piece2() == rhs.piece0())
					|| (piece0() == rhs.piece2() && piece1() == rhs.piece0() && piece2() == rhs.piece1())
					|| (piece0() == rhs.piece2() && piece1() == rhs.piece1() && piece2() == rhs.piece0())
#endif
					);
		}
		bool operator!=(const KPPP& rhs) { return !(*this == rhs); }

	private:

		// piece0,1,2を降順に並び替える。
		void sort_piece()
		{
			auto king = king_;
			auto i = piece0_;
			auto j = piece1_;
			auto k = piece2_;

			// ミラーの処理(これはメモリが足りなくなるので強制で行なう)
			if (file_of(king) > FILE_5)
			{
				king = Mir(king);
				i = mir_piece(i);
				j = mir_piece(j);
				k = mir_piece(k);
			}

			// 並び替え
			std::array<int,3> a = { i,j,k };
			std::sort(a.begin(), a.end());

			piece2_ = (Eval::BonaPiece)a[0];
			piece1_ = (Eval::BonaPiece)a[1];
			piece0_ = (Eval::BonaPiece)a[2];

			ASSERT_LV3(piece0_ > piece1_ && piece1_ > piece2_);
		}

		Square king_;
		Eval::BonaPiece piece0_, piece1_,piece2_;

	};

	// デバッグ用出力。
	static std::ostream& operator<<(std::ostream& os, KPPP rhs)
	{
		os << "KPPP(" << rhs.king() << "," << rhs.piece0() << "," << rhs.piece1() << "," << rhs.piece2() << ")";
		return os;
	}
}

#endif // defined (EVAL_LEARN)
#endif
