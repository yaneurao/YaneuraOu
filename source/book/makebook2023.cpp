#include "../types.h"

#if defined (ENABLE_MAKEBOOK_CMD)

// ------------------------------
//    ペタショック化コマンド
// ------------------------------

// 定跡自体は、別の何らかのスクリプトによって、やねうら王形式の定跡ファイルが書き出されているものとする。
// その定跡ファイルに対してそれぞれの局面に対してmin-max探索を行った結果の定跡ファイルを書き出すのが、
// このペタショック化である。

// コマンド例)
//    makebook petashock book1.db user_book1.db
// 
// book1.dbをmin-max探索してuser_book1.dbを書き出す。
// 
// エンジンオプションのFlippedBookがtrueなら、先手番の局面しか書き出さない。(後手番の局面はそれをflipした局面が書き出されているはずだから)
//
/*
	やねうら王のペタショックコマンドは後退解析をしている。
	疑似コードで書くと以下のようになる。

	MAX_PLY回繰り返す:
		for node in nodes:
			v = nodeのなかで一番良い指し手の評価値
			for parent in node.parents:
				parentからnodeに行く指し手の評価値 = v

	nodesは定跡DB上のすべての定跡局面を意味します。
	node.parentは、このnodeに遷移できる親nodeのlistです。
	また、子nodeを持っている指し手の評価値は0(千日手スコア)で初期化されているものとする。
*/

#include <sstream>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>
#include <limits>
#include <random>
#include <utility> // For std::forward

#include "book.h"
#include "../thread.h"
#include "../position.h"
#include "../misc.h"

using namespace std;
using namespace Book;

// ある局面の指し手の配列、std::vector<Move>だとsize_t(64bit環境で8バイト)でcapacityとかsizeとか格納するので
// 非常にもったいない。定跡のある局面の指し手が255手を超えることはないだろうから1byteでも十分。
// そこで、size_tではなくint16_tでサイズなどを保持しているvectorのsubsetを用意する。
// dataポインタが8バイトなのももったいないが…。これは仕方がないか…。

template <typename T>
class SmallVector {
public:
    SmallVector() : data(nullptr), count(0), capacity(0) {}

    // ムーブコンストラクタ
    SmallVector(SmallVector&& other) noexcept 
        : data(nullptr), count(0), capacity(0) {
        swap(other);
    }

	// ムーブ代入演算子
    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            clear();
            swap(other);
        }
        return *this;
    }

	~SmallVector() {
        clear();
		release();
    }

    void push_back(const T& value) {
        emplace_back(value);
    }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        //if (count == UINT16_MAX) {
        //    throw std::length_error("LimitedVector cannot exceed UINT16_MAX elements");
        //}

        if (count == capacity) {
            increase_capacity();
        }

        new (&get()[count]) T(std::forward<Args>(args)...);
        count++;
    }

    T& operator[](size_t idx) {
        //if (idx < 0 || idx >= count) {
        //    throw std::out_of_range("Index out of range");
        //}
        //return data[idx];

		return get()[idx];
    }

    const T& operator[](size_t idx) const {
        //if (idx < 0 || idx >= count) {
        //    throw std::out_of_range("Index out of range");
        //}
        //return data[idx];

		return get()[idx];
    }

    size_t size() const {
        return count;
    }

    T* begin() {
        return get();
    }

    T* end() {
        return get() + count;
    }

    void clear() {
        for (int i = 0; i < count; ++i) {
            (*this)[i].~T();
        }
        count = 0;
    }

    void erase(T* position) {
        //if (position < data || position >= data + count) {
        //    throw std::out_of_range("Iterator out of range");
        //}

        position->~T(); // Call the destructor for the element to be erased

        // Shift elements after position one place to the left
        for (T* it = position; it != get() + count - 1; ++it) {
            new (it) T(std::move(*(it + 1))); // Move construct the next element

			(it + 1)->~T(); // Destroy the moved-from object
        }

        --count;
    }

    void swap(SmallVector& other) noexcept {
        using std::swap;
        swap(data, other.data);
        swap(count, other.count);
        swap(capacity, other.capacity);
    }

private:

    void increase_capacity() {
        size_t new_capacity = capacity == 0 ? 1 : capacity * 2;
        if (new_capacity > UINT16_MAX) new_capacity = UINT16_MAX;

		// 小さいなら、dataポインターが配置されているメモリ領域を使う。
		bool embedded = new_capacity * sizeof(T) <= sizeof(T*);
		T* new_data = embedded
			? reinterpret_cast<T*>(&data)
			: reinterpret_cast<T*>(new char[new_capacity * sizeof(T)]);

		// 両方がembeddedである時、同じオブジェクトを指しているため、
		// moveのあとのデストラクタ呼び出しによって壊してしまうのでやらない。
		if (!embedded)
		{
			for (int i = 0; i < count; ++i) {
				new (&new_data[i]) T(std::move((*this)[i])); // Move existing elements
				(*this)[i].~T(); // Call destructor for moved elements
			}
		}

		release();
		if (!embedded)
	        data = new_data;

		capacity = int16_t(new_capacity);
    }

	// メンバー変数のdataはポインターではなく要素の格納用に使っているのか？
	bool is_embedded() const { return sizeof(T) * capacity <= sizeof(T*);}

	// dataポインターの値を返す。ただし、節約のためにここを要素に使っているなら、このアドレスを返す。
	T* get()
	{
		return is_embedded() ? reinterpret_cast<T*>(&data) : data;
	}

	// dataを解放する。ただし、newしてなければ解放しない。
	void release() {
		if (!is_embedded())
			delete[] reinterpret_cast<char*>(data);
	}

	// このポインターもったいないので節約のために要素が1つだけとかで
	// それがsize_of(T*)に収まる場合はここをデータとして用いる。
    T* data;

    int16_t count;
    int16_t capacity;
};

// ADLを利用したswapの非メンバ関数バージョン
// SmallVectorの名前空間内で定義する
namespace std {
    template <typename T>
    void swap(SmallVector<T>& a, SmallVector<T>& b) noexcept {
        a.swap(b);
    }
}


namespace MakeBook2023
{
	// BookMoveのポインターみたいなやつ。これで局面の行き来を行う。
	// しばらくは42億を超えることはないと思うので32bitでいいや。
	typedef u32 BookNodeIndex;
	// BookNodeIndexのnullptrの時の値。
	const BookNodeIndex BookNodeIndexNull = numeric_limits<BookNodeIndex>::max();

	// BoonNodeの評価値で∞を表現する定数。
	const int BOOK_VALUE_INF  = numeric_limits<int>::max();

	// ペタショック前の定跡DBに指し手の評価値を99999にして書き出しておくと、
	// これは指し手は存在するけど評価値は不明の指し手である。(という約束にする)
	// これは棋譜の指し手などを定跡DBに登録する時に評価値が確定しないのでそういう時に用いる。
	const int BOOK_VALUE_NONE = -99999;

	// 定跡で千日手手順の時のdepth。∞であることがわかる特徴的な定数にしておくといいと思う。
	const int BOOK_DEPTH_INF = 999;

	// 千日手の状態
	// 最終的には書き出した定跡DBのdepthに反映させる。
	// depth
	//  +1000 : 先手は千日手を打開できない。
	//  +2000 : 後手は千日手を打開できない。
	//  +3000 : 先手・後手ともに千日手を打開できない。
	struct DrawState
	{
		DrawState(u8 state):state(state){}

		// bit0 : 先手は千日手を打開する権利を持っていない。
		// bit1 : 後手は千日手を打開する権利を持っていない。

		// つまり、
		//   00b : 先後千日手を打開する権利を持っている。
		//   01b : 先手は(後手が千日手を選んだ時に)千日手を打開できない。(後手は打開できる)
		//   10b : 後手は(先手が千日手を選んだ時に)千日手を打開できない。(先手は打開できる)
		//   11b : 先手・後手ともに千日手を打開できない。

		u8 state;

		/*
			後手番で、11bと00bとの指し手があった時に 後手は00bの指し手を選択するが、親には01b を伝播する。
			この時、01bの指し手を選択すると、これがループである可能性がある。

			A→B→C→D→A
				  D→E
			となっている場合、後手が誤ったほうを選択するとループになる。

			だから、bestの選出は、後手は
			  00b > 01b > 10b > 11b
			先手は、
			  00b > 10b > 01b > 11b
			の順番でなければならない。
		*/

		// 比較オペレーター
		bool operator==(const DrawState& v) const { return state==v.state;}
		bool operator!=(const DrawState& v) const { return !(*this==v); }

		// 手番cにおいて評価値が同じ時に this のほうが yより勝るか。
		bool is_superior(DrawState y, Color c) const
		{
			// 先手にとってのbestを選出する順番
			constexpr int black_[] = {1,3,2,4};
			// 後手にとってのbestを選出する順番
			constexpr int white_[] = {1,2,3,4};

			if (c==BLACK)
				return black_[this->state] < black_[y.state];
			else
				return white_[this->state] < white_[y.state];
		}

		// 手番cの時、評価値が同じでDrawStateだけ異なる指し手がある時に
		// このnodeのDrawStateを求める。
		// 
		// 例)
		//   後手番で評価値が同じである 00bと11bの指し手があるとして
		//   後手はどちらかを選べるので、後手には千日手の権利があることになる。(先手はそれを決める権利がない)
		//   ゆえに、このとき、このnodeの draw_stateは、01b となる。
		//
		//   つまり、
		//     手番側のbitは、0と1があるなら0(bit and)
		//     非手番側のbitは、0と1があるなら1(bit or)
		//   をすればいいということである。
		void select(DrawState y, Color c)
		{
			*this = select_static(*this, y , c);
		}

		// ↑のstatic版
		static DrawState select_static(DrawState x , DrawState y , Color c)
		{
			u8 our_bit  =  c == BLACK ? 1 : 2;
			u8 them_bit = ~c == BLACK ? 1 : 2;
			
			return ((x.state & our_bit ) & (y.state & our_bit ))
				 | ((x.state & them_bit) | (y.state & them_bit));
		}

		// depthに変換する。
		// stateを1000倍しておく。
		u16 to_depth() const
		{
			return state * 1000;
		}
	};

	// 定跡の評価値とその時のdepth、千日手の状態をひとまとめにした構造体
	struct ValueDepth
	{
		ValueDepth()
			: value(BOOK_VALUE_NONE), depth(0), draw_state(0){}

		ValueDepth(int value, int depth)
			: value(value) , depth(depth) , draw_state(0){}

		ValueDepth(int value, int depth, DrawState state)
			: value(value) , depth(depth) , draw_state(state){}

		int value;
		u16 depth;
		DrawState draw_state;

		// 比較オペレーター
		bool operator==(const ValueDepth& v) const { return value==v.value && depth==v.depth && draw_state==v.draw_state;}
		bool operator!=(const ValueDepth& v) const { return !(*this==v); }

		// 優れているかの比較
		// 1. 評価値が優れている。
		// 2. 評価値が同じ場合、DrawStateが優れている
		// 3. 評価値とDrawStateも同じ場合、評価値が正ならdepthが小さいほうが優れている。
		// 4. 評価値とDrawStateも同じ場合、評価値が負ならdepthが大きいほうが優れている。
		//
		// 3., 4.の理屈としては、同じ評価値を持つ２つのノード(two nodes with the same value)問題があることを発見したからだ。
		// いま 
		//     A→B→C→D→A
		//     A→E→F→G
		//     B→X
		// のような経路があるとする。
		//
		// X,Gは同じ評価値でAの手番側から見て100であり、Aの手番側はここを目指すとする。
		// A→B→XはAからXに2手でvalue=100にいける。
		// A→E→F→GはAからGに3手でvalue=100にいける。
		// Aの手番を持たされた側は、depthが小さいからと言ってA→Bを選ぶとBの手番側がXに行く手を選択せずにCに行く手を選択してしまう。
		// そうするとC→D→AとAに戻ってきて千日手になる。
		// Bの手番側は、B→C→D→A→E→F→Gのコースを選んだほうが得なので、depthの大きい側を選ぶべきなのである。
		// この理屈から、評価値が正(千日手スコアより大きい)のほうは、千日手を回避するためにdepthが小さいほうを目指すが(親にそれを伝播する)、
		// 評価値が負(千日手スコアより小さい)のほうは、千日手にするためにdepthが大きいほうを目指す(親にそれを伝播する)べきなのである。
		// ゆえに、評価値の正負によって、どちらのdepthの指し手を選ぶかが変わるのである。
		// 
		// ※ PVが同じleafに到達する２つの指し手があるとして、depthが大きいほうは循環を含んでいる可能性が高いので、
		//    千日手にしたい側はdepthが大きいほうを、したくない側はdepthが小さいほうを選ぶという理屈。
		//
		bool is_superior(ValueDepth& v, Color color) const
		{
			// 評価値ベースの比較
			if (this->value != v.value)
				return this->value > v.value;

			// DrawStateベースの比較

			// 値が同じならdepthの低いほうを採用する。
			// なぜなら、循環してきて、別の枝に進むことがあり、それはdepthが高いはずであるから。
			if (this->draw_state != v.draw_state)
				return this->draw_state.is_superior(v.draw_state, color);

			// depthベースの比較。評価値の符号で場合分けが生ずる。
			// 一貫性をもたせるためにvalueが千日手スコアの場合は、先手ならdepthの小さいほうを目指すことにしておく。
			// →　省メモリ化のため、定跡読み込み時に先手の局面に変換してメモリに格納することにしたため、
			//    先後の区別ができなくなってしまった。
			auto dv = draw_value(REPETITION_DRAW, color);
			if ((this->value > dv) || (this->value == dv && (color == BLACK)))
				return this->depth < v.depth;
			else
				return this->depth > v.depth;
		}

		// depthにdraw_stateの状態を反映させる。
		void draw_state_to_depth() { depth += draw_state.to_depth(); }
	};

	// 定跡の1つの指し手を表現する構造体
	// 高速化のために、BookNodeIndexで行き来する。
	struct BookMove
	{
		// moveの指し手がleafである場合。
		BookMove(Move move,int value,int depth):
			move(move),vd(ValueDepth(value,depth)),next(BookNodeIndexNull){}

		// moveの指し手がleafではない場合。
		BookMove(Move move,ValueDepth vd, BookNodeIndex next):
			move(move),vd(vd),next(next){}

		// move(4) + value(4) + depth(4) + next(4) = 16 bytes

		// 指し手
		Move move;

		// ↑の指し手を選んだ時の定跡ツリー上の評価値
		// (定跡ツリー上でmin-max探索をした時の評価値)
		ValueDepth vd;

		// ↑のmoveで局面を進めた時の次の局面のBookMoveポインター。
		// これが存在しない時は、BookMoveIndexNull。
		BookNodeIndex next;
	};


	// あるnodeに対してその親nodeとその何番目の指し手がこのnodeに接続されているのかを保持する構造体。
	struct ParentMove
	{
		ParentMove(BookNodeIndex parent,size_t move_index):parent(parent),move_index((u32)move_index){}

		BookNodeIndex parent;
		u32 move_index;
	};

	// 定跡の1つの局面を表現する構造体。
	// 高速化のために、hashkeyで行き来をする。
	struct BookNode
	{
		// 引数のcは棋譜に出現したこの局面の手番。
		// 後手の場合、packed_sfenとしてこれをflipしさせて先手番の局面を登録する。
		BookNode(): out_count(0){}

		// このnodeからの出次数
		// 合法手が最大でMAX_MOVESしかないのでこれ以上このnodeから
		// 出ていくことはありえない。つまり、10bitにも収まる。
		u16 out_count : 15;
		// 元の手番。
		u16 color_ : 1;

		// 元の手番(flipして先手番の局面としてpacked_sfenを登録する前の手番)を設定する。
		void set_color(Color c) { color_ = u16(c); }

		// このnodeへの入次数
		//u64 in_count = 0;
		// これは、↓このlistを見ればいいから、あえて持つ必要はなさそう。

		// このnodeの親のlist
		// (つまり、その局面から指し手で1手進めてこの局面に到達できる)
		SmallVector<ParentMove> parents;

		// 指し手
		SmallVector<BookMove> moves;

		// 後退解析IIの時の前回の親に伝播したValueDepthの値。
		// 全ノード、これが前回と変わっていないなら後退解析を終了できる。
		//ValueDepth lastParentVd = ValueDepth(BOOK_VALUE_INF, 0);
		// ⇨　メモリもったいないし大きな定跡だと循環しつづけて値が変わり続けて
		//    役に立たないっぽいので削除。

		// key(この局面からhash keyを逆引きしたい時に必要になるので仕方なく追加してある)
		//HASH_KEY key;
		// ⇨　packed sfenから復元したらいいから削除。

		// 初期局面からの手数
		//u16 ply = 0;
		// ⇨　もったいない。使わないから削除。

		// 手番を返す。
		// これはメンバーのpacked_sfenから情報を取り出す。
		Color color() const { return Color(color_); }

		// 局面図
		PackedSfen packed_sfen;
	};

	// hashkeyのbit数をチェックする。
	void hashbit_check()
	{
		// 高速化のために、HASH_KEYだけで局面を処理したいので、hashの衝突はあってはならない。
		// そのため、HASH_KEY_BITSは128か256が望ましい。
		if (HASH_KEY_BITS < 128)
		{
			cout << "WARNING! : HASH_KEY_BITS = " << HASH_KEY_BITS << " is too short." << endl;
			cout << "    Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
		}
	}

	// ペタショック化
	class PetaShock
	{
	public:

		// 定跡をペタショック化する。
		// next : これが非0の時はペタショック化ではなく次に思考対象とすべきsfenをファイルに出力する。
		void make_book(Position& pos , istringstream& is, bool next)
		{
			hashbit_check();

			// 千日手の遡り手数を初手まで遡ることにする。
			pos.set_max_repetition_ply(MAX_PLY);

			string readbook_path;
			string writebook_path;
			string root_sfens_path;

			// 次の思考対象とすべきsfenを書き出す時のその局面の数。
			u64 next_nodes = 0;

			// leaf nodeの指し手に加える乱数の大きさ
			int eval_noise = 0;

			// 書き出す時にメモリを超節約する。
			bool memory_saving = false;

			// peta_shock_nextで局面を"startpos moves.."の形式で出力する。
			bool from_startpos = false;

			is >> readbook_path >> writebook_path;

			readbook_path  = Path::Combine("book",readbook_path );
			writebook_path = Path::Combine("book",writebook_path);

			if (next)
			{
				is >> next_nodes;
				string token;
				while (is >> token)
				{
					if (token == "eval_noise")
						is >> eval_noise;
					else if (token == "from_startpos")
						from_startpos = true;
				}

			} else {
				string token;
				while (is >> token)
				{
					if (token == "memory_saving")
						memory_saving = true;
				}
			}

			cout << "[ PetaShock makebook CONFIGURATION ]" << endl;

			if (next)
			{
				// 書き出すsfenの数
				cout << "write next_sfens   : " << next_nodes    << endl;
				cout << "eval_noise         : " << eval_noise    << endl;
				cout << "from_startpos      : " << from_startpos << endl;

				// これは現状ファイル名固定でいいや。
				root_sfens_path = Path::Combine("book","root_sfens.txt");
				cout << "root_sfens_path    : " << root_sfens_path << endl;

			} else {

				cout << "memory_saving      : " << memory_saving << endl;

			}

			cout << "readbook_path      : " << readbook_path  << endl;
			cout << "writebook_path     : " << writebook_path << endl;

			// DrawValueBlackの反映(DrawValueWhiteは無視)
			drawValueTable[REPETITION_DRAW][BLACK] =   Value((int)Options["DrawValueBlack"]);
			drawValueTable[REPETITION_DRAW][WHITE] = - Value((int)Options["DrawValueBlack"]);

			// 引き分けのスコアを変更したいなら先に変更しておいて欲しい。
			cout << "draw_value_black   : " << draw_value(REPETITION_DRAW, BLACK) << endl;
			cout << "draw_value_white   : " << draw_value(REPETITION_DRAW, WHITE) << endl;

			cout << endl;

			// === helper function ===

			// あるnodeのbestと親に伝播すべきparent_vdとを得るヘルパー関数。
			// best_index : 何番目の指し手がbestであったのかを返す。
			// すべてのnode.movesがMOVE_NONEなら、best_index == size_maxを返す。
			auto get_bestvalue = [&](BookNode& node , ValueDepth& parent_vd , size_t& best_index)
			{
				// まずこのnodeのbestを得る。
				ValueDepth best(-BOOK_VALUE_INF, BOOK_DEPTH_INF);
				// 親に伝播するbest
				parent_vd = ValueDepth(-BOOK_VALUE_INF,BOOK_DEPTH_INF,DrawState(3));

				best_index = size_max;
				for(size_t i = 0 ; i< node.moves.size() ; ++i)
				{
					const auto& book_move = node.moves[i];

					// MOVE_NONEならこの枝はないものとして扱う。
					if (book_move.move == MOVE_NONE)
						continue;

					if (book_move.vd.is_superior(best, node.color()))
					{
						best = book_move.vd;
						best_index = i;
					}

					if (parent_vd.value < book_move.vd.value)
						parent_vd = book_move.vd;
					else if (parent_vd.value == book_move.vd.value)
					{
						// valueが同じなのでdraw_stateはORしていく必要がある。
						parent_vd.draw_state.select(best.draw_state, node.color());

						// depthに関しては、bestのdepthを採用すればbestの指し手を追いかけていった時の手数になる。
						parent_vd.depth = best.depth;
					}
				}

				// 親に伝播するほうはvalueを反転させておく。
				parent_vd.value =        - parent_vd.value;
				parent_vd.depth = std::min(parent_vd.depth + 1 , BOOK_DEPTH_INF);

				return best;
			};

			// 評価値が同じでdepth違いの枝があると、実戦で、そっちが選ばれ続けて千日手になりかねないので
			// (評価値が同じで)depthが高いほうの指し手は残りの枝の価値をわずかにマイナスする処理を行うヘルパー関数。
			auto adjust_second_bestvalue = [&](BookNode& node)
			{
				ValueDepth vd;
				size_t _;
				auto best = get_bestvalue(node, vd , _);

				for(auto& book_move : node.moves)
				{

					if (   best.value == book_move.vd.value
						&& (
								// draw_state違い。これはbestが優先されるべき。
							   ( best.draw_state != book_move.vd.draw_state)
								// あるいは、draw_stateは同じだが、経路の長さが違う場合。これはbestのほうのみが選ばれるべき。
							|| ( best.draw_state == book_move.vd.draw_state && best.depth != book_move.vd.depth)
							)
						)
					{
						// depthが最小でない指し手の評価値を1だけ減らしておく。

						// bestの値しか親には伝播しないので、ここで引いたところで
						// このnodeにしか影響はない。
						book_move.vd.value--;
					}

					// depthにdraw stateを反映させる。
					book_move.vd.depth += book_move.vd.draw_state.to_depth();
				}
			};

			// 定跡の開始局面。
			// "book/root_sfens.txt"からrootのsfen集合を読み込むことにする。
			// これはUSIのposition文字列の"position"を省略した文字列であること。
			// つまり、
			//   startpos
			//   startpos moves ...
			//   sfen ...
			// のような文字列である。
			vector<string> root_sfens;
			if (next)
			{
				if (SystemIO::ReadAllLines(root_sfens_path,root_sfens).is_not_ok())
					// root_sfens.txtが存在しないなら、startposを突っ込んでおく。
					root_sfens.emplace_back("startpos");
			}


			// progress表示用
			Tools::ProgressBar progress;

			cout << "Read a book DB      : " << endl;

			// MemoryBookに読み込むと時間かかる + メモリ消費量が大きくなるので
			// 直接自前でbook_nodesに読み込む。

			SystemIO::TextReader reader;
			// ReadLine()の時に行の末尾のスペース、タブを自動トリム。空行は自動スキップ。
			reader.SetTrim(true);
			reader.SkipEmptyLine(true);

			auto result = reader.Open(readbook_path);
			if (result.is_not_ok())
			{
				sync_cout << "info string Error! : can't read file : " + readbook_path << sync_endl;
				return;
			}

			progress.reset(reader.GetSize());

			StateInfo si;
			std::string line;

			// 指し手を無視するモード
			bool ignoreMove = false;

			while(reader.ReadLine(line).is_ok())
			{
				progress.check(reader.GetFilePos());

				// バージョン識別文字列(とりあえず読み飛ばす)
				if (line.length() >= 1 && line[0] == '#')
					continue;

				// コメント行(とりあえず読み飛ばす)
				if (line.length() >= 2 && line.substr(0, 2) == "//")
					continue;

				// "sfen "で始まる行は局面のデータであり、sfen文字列が格納されている。
				if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
				{
					// 5文字目から末尾までをくり抜く。
					// 末尾のゴミは除去されているはずなので、Options["IgnoreBookPly"] == trueのときは、手数(数字)を除去。

					string sfen = line.substr(5); // 新しいsfen文字列を"sfen "を除去して格納

					// if (ignoreBookPly)
					StringExtension::trim_number_inplace(sfen); // 末尾の数字除去

					// この局面の手番
					// ⇨ "w"の文字は駒には使わないので"w"があれば後手番であることが確定する。
					Color stm = (sfen.find('w') != std::string::npos) ? WHITE : BLACK;

					// 先手番にしたsfen、後手番にしたsfen。
					string black_sfen = stm == BLACK ? sfen : Position::sfen_to_flipped_sfen(sfen);
					string white_sfen = stm == WHITE ? sfen : Position::sfen_to_flipped_sfen(sfen);

					// hashkey_to_indexには後手番の局面のhash keyからのindexを登録する。
					pos.set(white_sfen, &si, Threads.main());
					auto white_hash_key = pos.hash_key();
					if (hashkey_to_index.count(white_hash_key) > 0)
					{
						// 手番まで同じであるかを調べる。
						auto book_node_index = hashkey_to_index[white_hash_key];
						if (book_nodes[book_node_index].color() == stm)
						{
							// 重複局面 hash key衝突したのか？
							cout << "Error! : Hash Conflict! Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
							return ;
						}

						// 単に先後反転した局面がDB上に存在しただけであったので無視する。(先に出現した局面を優先)
						// この直後にやってくる指し手をこの局面の指し手を無視する。
						ignoreMove = true;
						continue;
					}
					hashkey_to_index[white_hash_key] = BookNodeIndex(book_nodes.size()); // emplace_back()する前のsize()が今回追加されるindex

					// BookNode.packed_sfenには先手番の局面だけを登録する。
					pos.set(black_sfen, &si, Threads.main());

					book_nodes.emplace_back(BookNode());
					auto& book_node = book_nodes.back();

					book_node.set_color(stm); // 元の手番
					pos.sfen_pack(book_node.packed_sfen);

					// この直後にやってくる指し手をこの局面の指し手として取り込む。
					ignoreMove = false;
					continue;
				}

				// いま指し手は無視する
				if (ignoreMove)
					continue;

				auto& book_node = book_nodes.back();

				// この行に Move PonderMove value depthが書かれている。これをそのままparseする。

				Parser::LineScanner scanner(line);
				auto move_str   = scanner.get_text();
				auto ponder_str = scanner.get_text();
				auto value = (int)scanner.get_number(0);
				auto depth = (int)scanner.get_number(0);
				Move16 move16   = (move_str   == "none" || move_str   == "None" || move_str   == "resign") ? MOVE_NONE : USI::to_move16(move_str  );
				//Move16 ponder = (ponder_str == "none" || ponder_str == "None" || ponder_str == "resign") ? MOVE_NONE : USI::to_move16(ponder_str);

				// 後手番であるなら、先手の局面として登録しないといけないので指し手もflipする。
				// posは上でblack_sfenが設定されているので先手番になるようにflipされている。
				if (book_node.color() == WHITE)
					move16 = flip_move(move16);
				Move move = pos.to_move(move16);

				// 合法手チェック。
				if (!pos.pseudo_legal_s<true>(move) || !pos.legal(move))
				{
					cout << "\nError! Illegal Move : sfen = " << pos.sfen() << " , move = " << move_str << endl;
					continue;
				}

				book_node.moves.emplace_back(BookMove(move, value, depth));
			}

			// 局面の合流チェック

			cout << "Convergence Check   :" << endl;

			// sfen nextの時はこの処理端折りたいのだが、parent局面の登録などが必要で
			// この工程を端折るのはそう簡単ではないからやめておく。

			// 合流した指し手の数
			u64 converged_moves = 0;

			progress.reset(book_nodes.size() - 1);
			for(BookNodeIndex book_node_index = 0 ; book_node_index < BookNodeIndex(book_nodes.size()) ; ++book_node_index)
			{
				auto& book_node = book_nodes[book_node_index];

				StateInfo si,si2;
				pos.set_from_packed_sfen(book_node.packed_sfen, &si, Threads.main());

				// 定跡DBに登録されていた指し手
				SmallVector<BookMove> book_moves;
				std::swap(book_node.moves, book_moves); // swapしていったんbook_move.movesはクリアしてしまう。

				// ここから全合法手で一手進めて既知の(DB上の他の)局面に行くかを調べる。
				for(auto move:MoveList<LEGAL_ALL>(pos))
				{
					pos.do_move(move, si2);

					// moveで進めた局面が存在する時のhash値。
					HASH_KEY next_hash = pos.state()->hash_key();

					if (this->hashkey_to_index.count(next_hash) > 0)
					{
						// 定跡局面が存在した。

						// 元のnodeの出次数と、next_nodeへの入次数をインクリメントしてやる。
						// (後退解析みたいなことをしたいので)
						book_node.out_count++;
						BookNodeIndex next_book_node_index = this->hashkey_to_index[next_hash];
						BookNode&     next_book_node       = this->book_nodes[next_book_node_index];

						// parentのlistに、元のnodeの何番目の指し手であるかを追加しておく。
						next_book_node.parents.emplace_back(ParentMove(book_node_index, book_node.moves.size()));

						// どうせmin-maxして、ここの評価値とdepthは上書きされるが、後退解析するので千日手の時のスコアで初期化する。

						// 千日手の時のvalueとdepth。
						// これは、
						//	value = draw_value
						//	depth = ∞
						//  draw_state = 先後ともに回避できない
						BookMove book_move(move,
							ValueDepth(
								draw_value(REPETITION_DRAW, book_node.color()),
								BOOK_DEPTH_INF,
								DrawState(3)
							),
							next_book_node_index);

						book_node.moves.emplace_back(book_move);

						// これが定跡DBのこの局面の指し手に登録されていないなら、
						// これは(定跡DBにはなかった指し手で進めたら既知の局面に)合流したということだから
						// 合流カウンターをインクリメントしておく。
						if (std::find_if(book_node.moves.begin(),book_node.moves.end(),[&](auto& book_move){ return book_move.move == move; })== book_node.moves.end())
							converged_moves++;
					}

					pos.undo_move(move);
				}

				// 定跡DB上のこの局面の指し手も登録しておく。
				for(auto& book_move : book_moves)
				{
					Move move = book_move.move;

					// これがbook_nodeにすでに登録されているか？
					if (std::find_if(book_node.moves.begin(),book_node.moves.end(),[&](auto& book_move){ return book_move.move == move; })== book_node.moves.end())
						// 登録されてなかったので登録する。(登録されていればどうせmin-max探索によって値が上書きされるので登録しなくて良い。)
						// 登録されていなかったということは、ここから接続されているnodeはないので、出次数には影響を与えない。
						book_node.moves.emplace_back(book_move);
				}

				progress.check(book_node_index);
			}

			//cout << "converged_moves : " << converged_moves << endl;

			// やねうら王の定跡DBの構築
			//cout << "build yaneuraou book : " << endl;

			// 下準備がすべて終わったのでbuildしてみる。

			// sfen_to_hashkeyとhashkey_to_book_nodeを頼りに、
			// 定跡上のmin-max探索する。

			// まず後退解析みたいなことをして、ループ以外はある程度処理しておく。
			// アルゴリズム的には、
			//   queue = 出次数0のnodeの集合
			//   while queue:
			//     node = queue.pop()
			//     eval  = max(node.各定跡の指し手の評価値)
			//     depth = evalがmaxだった時の定跡のdepth
			//     foreach parent in node.parents:
			//       parentのnodeへの指し手の評価値 = - eval     # min-maxなので評価値は反転する
			//       parentのnodeへの指し手のdepth  =   depth+1  # 1手深くなる
			//       if --parent.出次数 == 0:                    # 出次数が0になったのでこの局面もqueueに追加する。
			//         queue.push(parent)


			// leaf nodeの指し手の評価値に乱数を加える。
			if (eval_noise != 0)
			{
				cout << "add random bonus for every leaf move." << endl;
				progress.check(book_nodes.size());

				// 乱数生成器
				std::random_device rd;
				std::mt19937 gen(rd());

				// 正規分布を定義（平均 = 0 , 標準偏差 = eval_noise）
				std::normal_distribution<> d(0, eval_noise);

				for(size_t i = 0 ; i < book_nodes.size() ; i++)
				{
					BookNode& book_node = book_nodes[i];

					// leaf nodeでbest move以外の指し手が展開されるのは嫌だ。
					// 
					// そこで、このnodeの指し手すべてに同一のノイズを加算する。
					// こうすることでbest valueを持つmoveが展開される。
					// (その指し手がleaf nodeでないなら、それが伝播してきて置き換わるから問題なし)

					int noise = int(d(gen));
					for(auto& move : book_node.moves)
						move.vd.value += noise;

					progress.check(i);
				}
			}

			// 後退解析その1 : 出次数0の局面を削除
			cout << "Retrograde Analysis : Step I   -> delete nodes with zero out-degree." << endl;
			u64 retro_counter1 = 0;
			if (next)
			{
				// 次に掘るsfenを探す時は、出次数0の局面を削除してしまうとleafまで到達できなくなるのでまずい。
				// この工程をskipする。
				cout << "..skip" << endl;

			} else {

				// 作業対象nodeが入っているqueue
				// このqueueは処理順は問題ではないので両端queueでなくて良いからvectorで実装しておく。
				vector<BookNodeIndex> queue;

				progress.reset((u64)book_nodes.size());

				// 出次数0のnodeをqueueに追加。
				for(size_t i = 0 ; i < book_nodes.size() ; i++)
				{
					BookNode& book_node = book_nodes[i];
					if (book_node.out_count == 0)
						queue.emplace_back((BookNodeIndex)i);
				}

				// 出次数0のnodeがなくなるまで繰り返す。
				while (queue.size())
				{
					progress.check(++retro_counter1);

					auto node_index = queue.back();
					queue.pop_back();
					auto& book_node = book_nodes[node_index];

					ValueDepth parent_vd;
					size_t _;
					auto best = get_bestvalue(book_node , parent_vd , _);

					for(auto& pm : book_node.parents)
					{
						auto  parent_index      = pm.parent;
						auto& parent            = book_nodes[parent_index];
						auto  parent_move_index = pm.move_index;

						auto& m = parent.moves[parent_move_index];
						m.vd = parent_vd;
						// →　出自数0なのでdepthがBOOK_DEPTH_INFであることは無いからそのチェックは不要。

						m.next  = BookNodeIndexNull; // この指し手はもう辿る必要がないのでここがnodeがleaf nodeに変わったことにしておく。

						// parentの出次数が1減る。parentの出次数が0になったなら、処理対象としてqueueに追加。
						if (--parent.out_count == 0)
							queue.emplace_back(parent_index);
					}

					// 元のnodeの入次数 = 0にするためparentsをクリア。(もう親を辿ることはない)
					book_node.parents.clear();
				}
				// progress barを100%にする。
				progress.check(book_nodes.size());

				// 処理したノード数
				//cout << "processed nodes  : " << counter << endl;

			}

			// 出次数0以外の残りのnodeの数をカウント。
			// これらは何らかloopしているということなのできちんとしたmin-max探索が必要となる。
			u64 retro_counter2 = 0;
			if (next)
				retro_counter2 = book_nodes.size();
			else
			{
				for(size_t i = 0 ; i < book_nodes.size() ; i++)
				{
					if (book_nodes[i].out_count !=0)
						retro_counter2++;
				}
			}

			// 後退解析その2 : すべてのノードの親に評価値を伝播。MAX_PLY回行われる。

			cout << "Retrograde Analysis : Step II  -> Propagate the eval to the parents of all nodes." << endl;

			// ループがあって処理できていないnodeの数。
			//cout << "loop nodes       : " << counter << endl;

			// これらのnodeに対してはmin-max探索を行わなければならない。
			// ただし、min-maxだと組み合わせ爆発するので、non leafな指し手の値を
			// draw_valueで初期化して、全ノードに対してMAX_PLY回だけparentに評価値を伝播する。
			// これでループから抜け出せないところはdraw_valueになり、そうでないところは、正しい値が伝播されるということである。

			progress.reset(book_nodes.size() * MAX_PLY);
			u64 counter = 0;

			// MAX_PLY回だけ評価値を伝播させる。
			for(size_t loop = 0 ; loop < MAX_PLY ; ++loop)
			{
				for(auto& book_node : book_nodes)
				{
					// 入次数1以上のnodeなのでparentがいるから、このnodeの情報をparentに伝播。
					// これがmin-maxの代わりとなる。
					if (book_node.parents.size() != 0)
					{
						ValueDepth parent_vd;
						size_t _;
						auto best = get_bestvalue(book_node , parent_vd, _);

						// このnodeの評価が決まり、更新が確定したので、これをparentに伝達する。
						// 子:親は1:Nだが、親のある指し手によって進める子は高々1つしかいないので
						// 子の評価値は、それぞれの親のこの局面に進む指し手の評価値としてそのまま伝播される。
						for(auto& parent : book_node.parents)
						{
							BookNodeIndex parent_index = parent.parent;
							BookNode&     parent_node  = book_nodes[parent_index];
							BookMove&     parent_move  = parent_node.moves[parent.move_index];

							// 親に伝播させている以上、前回のvalueの値と異なることは確定している。

							parent_move.vd = parent_vd;
						}
					}

					progress.check(++counter);
				}
			}
			progress.check(book_nodes.size() * MAX_PLY);

			// 後退解析その3 : 

			// 評価値が同じでdepth違いの枝があると、実戦で、そっちが選ばれ続けて千日手になりかねないので
			// (評価値が同じで)depthが高いほうの指し手は残りの枝の価値をわずかにマイナスしておく。

			cout << "Retrograde Analysis : step III -> Adjust the bestvalue at all nodes." << endl;

			if (next)
			{
				cout << "..skip" << endl;
			} else {
				progress.reset(book_nodes.size() - 1);
				for(size_t i = 0 ; i < book_nodes.size() ; ++i)
				{
					auto& book_node = book_nodes[i];
					adjust_second_bestvalue(book_node);

					progress.check(i);
				}
			}

			// 書き出したsfenの個数
			u64 write_counter = 0;

			if (next)
			{
				// 書き出すsfen
				unordered_set<string> write_sfens;

				// 次に探索すべき定跡局面についてsfenを書き出していく。
				// これはmin-max探索した時のPVのleaf node。
				cout << "Retrograde Analysis : step IV  -> pick up next sfens to search." << endl;

				// rootから辿っていきPV leafに到達したらそのsfenを書き出す。
				// そのPV leaf nodeを削除して後退解析により、各局面の評価値を更新する。
				// これを繰り返す。

				progress.reset(next_nodes * root_sfens.size());

				// それぞれのroot_sfenに対して。
				// ⇨ この、root_sfen文字列は、"startpos moves ..."みたいな文字列でありうるので
				//   書き出すなら、これを普通のsfen文字列にしたものにしないといけないことに注意。
				for(auto root_sfen : root_sfens)
				{
					deque<StateInfo> si0;
					BookTools::feed_position_string(pos, root_sfen, si0);

					// 普通のsfen文字列にしたroot_sfen。
					string root_sfen0 = pos.sfen();
					// root_sfenの元の手番
					Color root_stm = pos.side_to_move();
					// root局面のgame ply
					int root_ply = pos.game_ply();
					if (root_stm == BLACK)
					{
						// 後手番の局面になるようにflipする。(hash key調べたいので)
						auto white_sfen = Position::sfen_to_flipped_sfen(pos.sfen());
						StateInfo si;
						pos.set(white_sfen, &si, Threads.main());
					}

					// このroot_sfenの局面が定跡DB上に存在しない
					if (hashkey_to_index.count(pos.hash_key()) == 0)
					{
						if (from_startpos)
							write_sfens.emplace(root_sfen);
						else
							write_sfens.emplace(root_sfen0);

						continue;
					}

					// 以下、Positionを用いずにBookNodeIndexで行き来する。
					BookNodeIndex root_book_node_index = hashkey_to_index[pos.hash_key()];

					u64 timeup_counter = 0;

					// 今回のroot_sfenに対して書き出した局面数
					u64 write_counter2 = 0;

					// 所定の行数のsfenを書き出すまで回る。
					// ただし、局面が尽きることがあるのでrootが存在しなければループは抜ける。
					while (true)
					{
						// このroot_sfenに対して規定の行数を書き出した
						if (write_counter2 >= next_nodes)
							break;

						// 書き出す指し手がすべてのrootでなくなっていることを証明するロジックを書くのわりと面倒なので
						// timeup_counterをカウントすることにする。
						// next_nodesの10倍も回ったらもうあかんやろ…。
						if (++timeup_counter > next_nodes * 10)
							break;

						// 現在の手番
						Color stm = root_stm;

						// leafの局面までの手順
						string sfen_path = root_sfen;
						if (!StringExtension::Contains(sfen_path,"moves"))
							sfen_path += " moves";

						// 開始局面
						BookNodeIndex book_node_index = root_book_node_index;
						int ply = root_ply;

						// いままで訪問したnode。千日手チェック用。
						unordered_set<BookNodeIndex> visited_nodes;

						// まずPV leafまで辿る。
						while (true)
						{
							auto& book_node = book_nodes[book_node_index];
							visited_nodes.insert(book_node_index);

							auto& moves     = book_node.moves;

							ValueDepth parent_vd;
							size_t best_index;
							auto best = get_bestvalue(book_node, parent_vd , best_index);

							// このnodeにはすでに指し手がないはずなのにどこからやってきたのだ…。
							// このnode、DB上で元から指し手がなかったのか？あるいはroot_sfenの局面の指し手が尽きたのか？
							if (best_index == size_max)
								break;

							// このbestのやつのindexが必要なので何番目にあるか保存しておく。(あとで切断する用)
							auto move_index = best_index;

							// 次のnode
							BookNodeIndex next_book_node_index = book_node.moves[best_index].next;

							// leafに到達したか？
							if (next_book_node_index == BookNodeIndexNull)
							{
								// book_nodeから指し手 mで進めた局面を書き出したいので、実際に局面を復元してやってみる。
								StateInfo si,si2;
								pos.set_from_packed_sfen(book_node.packed_sfen, &si, Threads.main());
								Move m = book_node.moves[best_index].move;
								pos.do_move(m, si2);
								// ⇨　packed_sfenは先手の局面であり1手進めているから、ここでは後手の局面になっている。

								sfen_path += ' ' + to_usi_string(stm == BLACK ? m : flip_move(m));

								// 現在の手番が先手であれば、flipして先手の盤面として書き出す。
								stm = ~stm;
								string sfen = (stm == WHITE) ? pos.sfen(ply + 1) : pos.flipped_sfen(ply + 1);

								// write_sfensのinsertはここでしか行わないので、ここでcheck()すれば十分。
								if (from_startpos)
									write_sfens.insert(sfen_path);
								else
									write_sfens.insert(sfen);
								progress.check(write_sfens.size());
								write_counter2++;

								// この手はないものとして、この book_node_index を起点として上流に更新していけばOK。
								book_node.moves[best_index].move = MOVE_NONE;

								break;
							}

							// 千日手がPVになっているか？

							// たまに循環がひたすら回避されながらMAX_PLYを超えて手数が増えることがあるのでgame_ply()の判定必須。
							if (ply >= MAX_PLY || visited_nodes.count(next_book_node_index))
							{
								// このnodeから、next_book_node_indexに至る経路を切っておく。
								// MAX_PLY超えたほうは切るのは微妙かも知れないが、仕方がない。

								// next_book_nodeの親のリストからbook_node_indexを消去。(この経路がなくなるので)
								// この親リストに必ず1つだけ存在することは保証されているので、存在チェックと重複チェックを省くことができる。

								auto& next_book_node = book_nodes[next_book_node_index];
								auto& p = next_book_node.parents;
								p.erase(std::find_if(p.begin(), p.end(), [&](auto& pm){ return pm.parent == book_node_index; }));

								// この手はないものとして、この book_node_index を起点として上流に更新していけばOK。
								book_node.moves[best_index].move = MOVE_NONE;

								break;
							}

							Move m = book_node.moves[best_index].move;
							// 格納されているのは先手化した指し手なので、後手の手番であるなら、先手化する必要がある。
							sfen_path += ' ' + to_usi_string(stm == BLACK ? m : flip_move(m));
							stm = ~stm;

							// 次のnodeを辿る。
							book_node_index = next_book_node_index;
							ply++;
						}

						/*
							PV leaf nodeまで到達したので、ここからrootまで遡ってbest move,best valueの更新を行う。
							いま見ているnodeをnとする。nの親を辿るのだが、nの指し手がすべて無効になった時は、nの親のnに行く指し手を無効化する必要がある。

							また、rootまで遡る時にparentsが複数あったりloopがあったりするので単純に遡ると組み合わせ爆発を起こす。
							そこで、次のアルゴリズムを用いる。

							queue = [処理すべき局面]
							while queue:
								n = queue.pop_left()
								vd = nのbest
								deleted = nの指し手がすべて消滅したのか？
								for parent in n.parents:
									parentからnに至る指し手 = MOVE_NONE if deleted else vd
									// ⇨　これによるparentでのbestに変化が生じた時だけqueueに詰めばいいのだが、
									//    循環してるとどうせ永久ループになるので気にせず積む。
									if (not parentが処理済み)
										queue.push_right(parent)
								if deleted:
									n.parents.clear()

						*/

						//deque<BookNodeIndex> queue;
						// ⇨　これ、queueにした方がleaf nodeからの距離の順に伝播して良いと思うのだが、
						//    循環してて300万局面のleaf nodeから親が5万ノードほどあるので増えてくるとここの
						//    オーバーヘッドが許容できない。
						//    仕方なくvectorにする。

						vector<BookNodeIndex> queue;
						queue.emplace_back(book_node_index);

						// update済みノード
						unordered_set<BookNodeIndex> already_updated_node;

						while (queue.size())
						{
							//auto book_node_index = queue[0];
							//queue.pop_front();

							auto book_node_index = queue.back();
							queue.pop_back();

							auto& book_node = book_nodes[book_node_index];

							ValueDepth parent_vd;
							size_t best_index;
							auto best_vd = get_bestvalue(book_node, parent_vd, best_index);
							// すべてがMOVE_NONEなら、best_index == size_maxになることが保証されている。
							bool delete_flag = best_index == size_max;

							for(auto& pm : book_node.parents)
							{
								auto& parent_book_node = book_nodes[pm.parent];
								ValueDepth parent_parent_vd;
								size_t     parent_best_index;
								auto parent_best_vd    = get_bestvalue(book_node, parent_parent_vd, parent_best_index);

								if (delete_flag)
									// 1a. この局面に至る親からの指し手をすべて削除。
									book_nodes[pm.parent].moves[pm.move_index].move = MOVE_NONE;
								else
									// 1b. この局面に至る親からの指し手の評価値を更新
									book_nodes[pm.parent].moves[pm.move_index].vd = best_vd;

								// これ⇑によって親のbestに変化が生じたのか？
								auto parent_best_vd2   = get_bestvalue(book_node, parent_parent_vd, parent_best_index);
								if (parent_best_vd != parent_best_vd2)
								{
									// 2. 親を更新対象に追加
									// すでに一度でも追加しているならこれ以上は追加しない。(ループ防止)
									if (already_updated_node.count(pm.parent) == 0)
									{
										already_updated_node.insert(pm.parent);
										queue.push_back(pm.parent);
									}
								}
							}
							// 3. 親を丸ごと削除
							if (delete_flag)
								book_node.parents.clear();
						}
					}

				//NEXT_ROOT:;

				}

				progress.check(next_nodes * root_sfens.size());

				// write_sfensのなかにある局面とそれをflipした局面の組が含まれないかを
				// チェックする。
				// flipした局面に対しても辿っているので、これはわりとありうる。

				SystemIO::TextWriter writer;
				writer.Open(writebook_path);
				for(auto& write_sfen : write_sfens)
					writer.WriteLine(write_sfen);
				write_counter = write_sfens.size();

			} else {
				// 通常のpeta_shockコマンド時の処理。(peta_shock_nextコマンドではなく)

				// メモリ上の定跡DBを再構成。
				// この時点でもうhash_key_to_index不要なので解放する。
				// (clear()では解放されないので、swap trickを用いる。)
				HashKey2Index().swap(this->hashkey_to_index);

				// MemoryBookを用いるとオーバーヘッドが大きいので自前で直接ファイルに書き出す。

				size_t n = book_nodes.size();

				if (memory_saving)
				{
					// メモリ超絶節約モード

					cout << "Sorting a book      : " << endl;

					// 並び替えを行う。
					// ただしbook_nodes直接並び替えるのはメモリ移動量が大きいのでindexのみをsortする。
					// ⇨　あー、これ、sfen文字列でsortしないといけないのか…わりと大変か…。
					vector<BookNodeIndex> book_indices(n);
					for(size_t i = 0 ; i < n ; ++i)
						book_indices[i] = BookNodeIndex(i);

					// nの64倍ぐらいで終わるんちゃうんか？
					progress.reset(n * 64);

					// 進捗出力用カウンター
					size_t c = 0;
					StateInfo si;

					// カスタム比較関数
					auto customCompare = [&](int i, int j){
						// packed sfenをunpackして文字列として比較。unpackがN * log(N)回ぐらい走るのでわりとキツイか…。
						auto sfen_i = pos.sfen_unpack(book_nodes[i].packed_sfen);
						auto sfen_j = pos.sfen_unpack(book_nodes[j].packed_sfen);

						// 進捗出力用
						c = min(c + 1 , n * 64);
						progress.check(c);
						return sfen_i < sfen_j;
					};
					sort(book_indices.begin(), book_indices.end(), customCompare);
					progress.check(n * 64);

					cout << "Write book directly : " << endl;

					SystemIO::TextWriter writer;
					if (writer.Open(writebook_path).is_not_ok())
					{
						cout << "Error! : open file erro , path = " << writebook_path << endl;
						return;
					}

					progress.reset(n - 1);

					// バージョン識別用文字列
					writer.WriteLine(Book::BookDBHeader2016_100);

					for(size_t i = 0 ; i < n ; ++i)
					{
						auto& book_node = book_nodes[book_indices[i]];
						auto  sfen      = pos.sfen_unpack(book_node.packed_sfen);

						// sfenを出力。上でsortしているのでsfen文字列順で並び替えされているはず。
						writer.WriteLine("sfen " + sfen);

						// 指し手を出力
						for(auto& move : book_node.moves)
							writer.WriteLine(to_usi_string(move.move) + " None " + to_string(move.vd.value) + " " + to_string(move.vd.depth));

						progress.check(i);
						write_counter++;
					}
				} else {

					// ⇑ sort中にpacked sfenのunpackをしてメモリ節約するのは無謀であったか…。

					cout << "Unpack packed sfens : " << endl;

					// 並び替えを行う。
					// ただしbook_nodes直接並び替えるのはメモリ移動量が大きいのでindexのみをsortする。
					// ⇨　あー、これ、sfen文字列でsortしないといけないのか…わりと大変か…。
					using BookNodeIndexString = pair<BookNodeIndex,string>;
					vector<BookNodeIndexString> book_indices(n);
					progress.reset(n - 1);
					for(size_t i = 0 ; i < n ; ++i)
					{
						auto& book_node = book_nodes[i];
						auto sfen = pos.sfen_unpack(book_node.packed_sfen);
						// 元のDBで後手の局面は後手の局面として書き出す。(ので、ここでsfenをflipしてからsortする)
						if (book_node.color() == WHITE)
							sfen = Position::sfen_to_flipped_sfen(sfen);

						book_indices[i] = BookNodeIndexString(BookNodeIndex(i), sfen);
						progress.check(i);
					}

					cout << "Sorting book_nodes  : " << endl;
					sort(book_indices.begin(), book_indices.end(),
						[&](BookNodeIndexString& i, BookNodeIndexString& j){
						return i.second < j.second;
					});

					// ⇑ここでsfenをunpackしてしまうなら、最初からsfenで持っておいたほうがいい気がするし、
					// あるいは、このままsortなしで書き出したほうがいいような気もする。
					// (sortは改めてやるとして)

					// しかしsortするのも丸読みしないといけないから大変か…。
					// この時点で要らないものをいったん解放できると良いのだが…。

					cout << "Write to a book DB  : " << endl;

					SystemIO::TextWriter writer;
					if (writer.Open(writebook_path).is_not_ok())
					{
						cout << "Error! : open file erro , path = " << writebook_path << endl;
						return;
					}

					progress.reset(n - 1);

					// バージョン識別用文字列
					writer.WriteLine(Book::BookDBHeader2016_100);

					for(size_t i = 0 ; i < n ; ++i)
					{
						auto& book_node = book_nodes[book_indices[i].first];
						auto& sfen = book_indices[i].second;

						// sfenを出力。上でsortしているのでsfen文字列順で並び替えされているはず。
						writer.WriteLine("sfen " + sfen);

						// 指し手を出力
						for(auto& move : book_node.moves)
						{
							// 元のDB上で後手の局面なら後手の局面として書き出したいので、
							// 後手の局面であるなら指し手を反転させる。
							Move16 m16 = (book_node.color() == WHITE) ? flip_move(move.move) : move.move;
							writer.WriteLine(to_usi_string(m16) + " None " + to_string(move.vd.value) + " " + to_string(move.vd.depth));
						}

						progress.check(i);
						write_counter++;
					}
				}

				cout << "write " + writebook_path << endl;
			}

			cout << "[ PetaShock Result ]" << endl;

			// 合流チェックによって合流させた指し手の数。
			if (!next)
				cout << "converged_moves  : " << converged_moves << endl;

			// 後退解析において判明した、leafから見てループではなかったノード数
			cout << "retro_counter1   : " << retro_counter1 << endl;
			// 後退解析において判明した、ループだったノード数
			cout << "retro_counter2   : " << retro_counter2 << endl;

			// 書き出したrecord数
			if (next)
			{
				cout << "write sfen nodes : " << write_counter << endl << endl;
				cout << "Making peta-shock next sfens has been completed." << endl;
			}
			else
			{
				cout << "write book nodes : " << write_counter << endl << endl;
				cout << "Making a peta-shock book has been completed." << endl;
			}

		}

	private:

		// 定跡本体
		vector<BookNode> book_nodes;

		// 同様に、HASH_KEYからBookMoveIndexへのmapper
		// ただし、flipして後手番にしたhashkeyを登録してある。
		// ⇨　後手の局面はflipして先手の局面として格納している。ゆえに、格納されているのはすべて先手の局面であり、
		// 　そこから1手進めると後手の局面となる。この時に、hash keyから既存の局面かどうかを調べたいので…。
		using HashKey2Index = unordered_map<HASH_KEY,BookNodeIndex>;
		HashKey2Index hashkey_to_index;
	};
}

namespace Book
{
	// 2023年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2023(Position& pos, istringstream& is, const string& token)
	{
		if (token == "peta_shock")
		{
			// ペタショックコマンド
			// やねうら王の定跡ファイルに対して定跡ツリー上でmin-max探索を行い、その結果を別の定跡ファイルに書き出す。
			//   makebook peta_shock book.db user_book1.db
			// 先手の局面しか書き出さない。後手の局面はflip(盤面を180°回転させる)して、先手の局面として書き出す。
			// エンジンオプションの FlippedBook を必ずオンにして用いること。
			MakeBook2023::PetaShock ps;
			ps.make_book(pos, is, false);
			return 1;

		} else if (token == "peta_shock_next"){

			// ペタショックnext
			// ペタショック手法と組み合わせてmin-maxして、有望な局面をsfen形式でテキストファイルに書き出す。
			//   makebook peta_shock_next book.db sfens.txt 1000
			//   makebook peta_shock_next book.db sfens.txt 1000 eval_noise 20
			// ⇨　1000局面を書き出す。20はleaf nodeの指し手の評価値に加える乱数の大きさ。
			// 　 この場合、評価値に、平均 = 0 , 標準偏差 = 20 ガウスノイズを加算する。
			// 　　(これを加えることで序盤の指し手を開拓しやすくなる)
			//   makebook peta_shock_next book.db sfens.txt 1000 from_startpos
			// ⇨　startpos moves ... の形式で局面を出力する。
			// 
			//   makebook peta_shock_next book.db sfens.txt 1000 minimum
			// ⇨ memory_savingをつけるとpacked sfenのままsortするので書き出しの時にメモリがさらに節約できる。
			//   (でも書き出すのに時間1時間ぐらいかかる)

			MakeBook2023::PetaShock ps;
			ps.make_book(pos, is , true);
			return 1;
		}

		return 0;
	}
}


#endif // defined (ENABLE_MAKEBOOK_CMD)
