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
	// peta_shockコマンド実行時にsfen文字列を一時保存するファイル名
	string SFEN_TEMP_FILENAME = "sfen_tmp.txt";

	// BookMoveのポインターみたいなやつ。これで局面の行き来を行う。
	// しばらくは42億を超えることはないと思うので32bitでいいや。
	typedef u32 BookNodeIndex;
	// BookNodeIndexのnullptrの時の値。
	const BookNodeIndex BookNodeIndexNull = numeric_limits<BookNodeIndex>::max();

	// BoonNodeの評価値で∞を表現する定数。
	const int BOOK_VALUE_INF  = numeric_limits<s16>::max();

	// ペタショック前の定跡DBに指し手の評価値をminにして書き出しておくと、
	// これは指し手は存在するけど評価値は不明の指し手である。(という約束にする)
	// これは棋譜の指し手などを定跡DBに登録する時に評価値が確定しないのでそういう時に用いる。
	const int BOOK_VALUE_NONE = numeric_limits<s16>::min();

	const int BOOK_VALUE_MAX  = numeric_limits<s16>::max()-1;
	const int BOOK_VALUE_MIN  = numeric_limits<s16>::min()+1;

	// 定跡で千日手手順の時のdepth。∞であることがわかる特徴的な定数にしておくといいと思う。
	const u16 BOOK_DEPTH_INF = 999;


	// 定跡の評価値とその時のdepth、千日手の状態をひとまとめにした構造体
	struct ValueDepth
	{
		ValueDepth()
			: value(BOOK_VALUE_NONE), depth(0){}

		ValueDepth(s16 value, u16 depth)
			: value(value) , depth(depth){}

		s16 value;
		u16 depth;

		// 比較オペレーター
		bool operator==(const ValueDepth& v) const { return value==v.value && depth==v.depth;}
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

			// depthベースの比較。評価値の符号で場合分けが生ずる。
#if 0
			// 一貫性をもたせるためにvalueが千日手スコアの場合は、先手ならdepthの小さいほうを目指すことにしておく。
			auto dv = draw_value(REPETITION_DRAW, color);
			if ((this->value > dv) || (this->value == dv && (color == BLACK)))
				return this->depth < v.depth;
			else
				return this->depth > v.depth;
#endif
			// ⇨　省メモリ化のため、定跡読み込み時に先手の局面に変換してメモリに格納することにしたため、
			//    先後の区別ができなくなってしまった。
			auto dv = draw_value(REPETITION_DRAW, color);
			if (this->value >= dv)
				return this->depth < v.depth;
			else
				return this->depth > v.depth;
		}
	};

	// 定跡の1つの指し手を表現する構造体
	// 高速化のために、BookNodeIndexで行き来する。
	struct BookMove
	{
		// moveの指し手がleafである場合。
		BookMove(Move16 move,s16 value,s16 depth):
			move(move),vd(ValueDepth(value,depth)),next(BookNodeIndexNull){}

		// moveの指し手がleafではない場合。
		BookMove(Move16 move,ValueDepth vd, BookNodeIndex next):
			move(move),vd(vd),next(next){}

		// move(2) + value(2) + depth(2) + next(4) = 10 bytes → 12

		// 指し手
		Move16 move;

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
		//PackedSfen packed_sfen;
		// ⇨　メモリがもったいないから元のsfenをファイルに書き出す。(どうせ元のsfen文字列はそのまま使うので)
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


	enum class PETA_SHOCK_TYPE
	{
		PetaShock          , /* 通常のpeta shock化 */
		PetaShockNextPV    , /* 次に掘るべき局面を求める。初期局面から最善手を辿った局面 */
		PetaShockNextHalfPV, /* 次に掘るべき局面を求める。初期局面から先手は最善手、後手はすべての指し手、みたいな感じで辿った時の局面。 */
	};

	// SFEN棋譜とそのleaf nodeでの評価値(評価値は先手から見て)
	typedef pair<string, s16> KIF_EVAL;


	// ペタショック化
	class PetaShock
	{
	public:
		// 定跡をペタショック化する。
		// next : これが非0の時はペタショック化ではなく次に思考対象とすべきsfenをファイルに出力する。
		void make_book(Position& pos , istringstream& is, PETA_SHOCK_TYPE type)
		{
			// 実行するコマンドは、PetaShock Next(1,2問わず)であるか
			bool next = (type == PETA_SHOCK_TYPE::PetaShockNextPV    )
				     || (type == PETA_SHOCK_TYPE::PetaShockNextHalfPV);
			// 実行するコマンドは、PetaShock Next2であるか？
			bool next2 = type == PETA_SHOCK_TYPE::PetaShockNextHalfPV;

			hashbit_check();

			// 千日手の遡り手数を初手まで遡ることにする。
			pos.set_max_repetition_ply(MAX_PLY);

			string readbook_path;
			string writebook_path;
			string root_sfens_path;

			// 次の思考対象とすべきsfenを書き出す時のその局面の数。
			u64 next_nodes = 0;

			// peta_shock_nextの時のleaf nodeの指し手に加える乱数の大きさ
			int eval_noise = 0;

			// peta_shock_next2の時の指し手を辿る評価値のlimit
			int eval_limit = 400;

			// peta_shock_nextで局面を"startpos moves.."の形式で出力する。
			bool from_startpos = false;

			// 書き出す時に最善手と同じ評価値の指し手以外は削除する。
			bool shrink = false;

			is >> readbook_path >> writebook_path;

			string BOOK_DIR = Options["BookDir"];
			this->sfen_temp_path = Path::Combine(BOOK_DIR, SFEN_TEMP_FILENAME);
			readbook_path  = Path::Combine(BOOK_DIR, readbook_path );
			writebook_path = Path::Combine(BOOK_DIR, writebook_path);

			if (next)
			{
				is >> next_nodes;
			}

			string token;
			while (is >> token)
			{
				if (next)
				{
					if (token == "eval_noise")
						is >> eval_noise;
					else if (token == "from_startpos")
						from_startpos = true;
					else if (token == "eval_limit")
						is >> eval_limit;
				} else {
					if (token == "shrink")
						shrink = true;
				}
			}

			cout << "[ PetaShock makebook CONFIGURATION ]" << endl;

			if (next)
			{
				// 書き出すsfenの数
				cout << "write next_sfens   : " << next_nodes    << endl;
				if (next2)
					cout << "eval_limit         : " << eval_limit    << endl;
				else if (next)
					cout << "eval_noise         : " << eval_noise    << endl;
				cout << "from_startpos      : " << from_startpos << endl;

				// これは現状ファイル名固定でいいや。
				root_sfens_path = Path::Combine(BOOK_DIR, "root_sfens.txt");
				cout << "root_sfens_path    : " << root_sfens_path << endl;

			} else {

				cout << "shrink             : " << shrink << endl;

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

			// 行番号
			size_t line_no = 0;

			// sfen文字列はファイルに書き出す。
			SystemIO::TextWriter sfen_writer;
			sfen_writer.Open(sfen_temp_path);

			while(reader.ReadLine(line).is_ok())
			{
				progress.check(reader.GetFilePos());
				line_no ++;

				// バージョン識別文字列など(とりあえず読み飛ばす)
				if (line.length() >= 1 && line[0] == '#')
				{
					if (line_no == 1)
					{
						if (line != ::Book::BookDBHeader2016_100)
							cout << "WARNING : illegal header" << endl;
					} else if (line_no == 2)
					{
						// 2行目には
						// # NOE:258
						// よって、この文字列をparseする必要がある。
						auto splited = StringExtension::Split(line.substr(2),",");
						for(auto command : splited)
						{
							auto splited2 = StringExtension::Split(command,":");
							if (splited2.size() >= 1)
							{
								auto& token = splited2[0];
								if (token == "NOE" && splited2.size() == 2) // numbers of entires
								{
									size_t noe = StringExtension::to_int(string(splited2[1]), 0);
									cout << "Number of Sfen Entries = " << noe << endl;

									// エントリー数が事前にわかったので、その分だけそれぞれの構造体配列を確保する。
									book_nodes.reserve(noe);
									hashkey_to_index.reserve(noe);
								}
							}
						}
					}
					continue;
				}

				// コメント行(とりあえず読み飛ばす)
				if (line.length() >= 2 && line.substr(0, 2) == "//")
					continue;

				// "sfen "で始まる行は局面のデータであり、sfen文字列が格納されている。
				if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
				{
					// 5文字目から末尾までをくり抜く。
					// 末尾のゴミは除去されているはずなので、Options["IgnoreBookPly"] == trueのときは、手数(数字)を除去。

					string sfen = line.substr(5); // 新しいsfen文字列を"sfen "を除去して格納

					// sfen文字列はテンポラリファイルに書き出しておく。(もし末尾に手数があるなら、それも含めてそのまま書き出す)
					sfen_writer.WriteLine(sfen);

					// if (ignoreBookPly)
					StringExtension::trim_number_inplace(sfen); // 末尾の数字除去

					// この局面の(元の)手番
					// ⇨ "w"の文字は駒には使わないので"w"があれば後手番であることが確定する。
					Color stm = (sfen.find('w') != std::string::npos) ? WHITE : BLACK;

					// 先手番にしたsfen、後手番にしたsfen。
					//string black_sfen = stm == BLACK ? sfen : Position::sfen_to_flipped_sfen(sfen);
					string white_sfen = stm == WHITE ? sfen : Position::sfen_to_flipped_sfen(sfen);

					// hashkey_to_indexには後手番の局面のhash keyからのindexを登録する。
					pos.set(white_sfen, &si, Threads.main());
					HASH_KEY white_hash_key = pos.hash_key();

#if 0
					// 念のためにエラーチェック(この時間もったいないか…)
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
						// この直後にやってくる指し手(この局面の指し手)を無視する。
						ignoreMove = true;
						continue;
					}
#endif

					hashkey_to_index[white_hash_key] = BookNodeIndex(book_nodes.size()); // emplace_back()する前のsize()が今回追加されるindex

					// BookNode.packed_sfenには先手番の局面だけを登録する。
					//pos.set(black_sfen, &si, Threads.main());

					book_nodes.emplace_back(BookNode());
					auto& book_node = book_nodes.back();

					book_node.set_color(stm); // 元の手番。これを維持してファイルに書き出さないと、sfen文字列でsortされていたのが狂う。

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
				auto value = (s16)std::clamp((int)scanner.get_number(0), BOOK_VALUE_MIN , BOOK_VALUE_MAX);
				auto depth = (s16)scanner.get_number(0);
				Move16 move16   = (move_str   == "none" || move_str   == "None" || move_str   == "resign") ? Move16::none() : USI::to_move16(move_str  );
				//Move16 ponder = (ponder_str == "none" || ponder_str == "None" || ponder_str == "resign") ? Move16::none() : USI::to_move16(ponder_str);

				// 後手番であるなら、先手の局面として登録しないといけないので指し手もflipする。
				// posは上でblack_sfenが設定されているので先手番になるようにflipされている。
				if (book_node.color() == WHITE)
					move16 = flip_move(move16);
				//Move move = pos.to_move(move16);
				// ⇨　この復元のためだけに先手番のposition構築するの嫌だな…。

#if 0
				// 合法手チェック。
				if (!pos.pseudo_legal_s<true>(move) || !pos.legal(move))
				{
					cout << "\nError! Illegal Move : sfen = " << pos.sfen() << " , move = " << move_str << endl;
					continue;
				}
				// ⇨　この合法性チェックのためにわざわざ先手番にflipした局面をセットしなおすのは
				//   時間もったいないな…。
#endif

				book_node.moves.emplace_back(BookMove(move16, value, 0 /*depth*/));
				// ⇨ leaf nodeのdepthは0扱いで良いのでは…。
			}
			sfen_writer.Close();

			// 局面の合流チェック

			cout << "Convergence Check   :" << endl;

			// sfen nextの時はこの処理端折りたいのだが、parent局面の登録などが必要で
			// この工程を端折るのはそう簡単ではないからやめておく。

			// 合流した指し手の数
			u64 converged_moves = 0;

			SystemIO::TextReader sfen_reader;
			sfen_reader.Open(sfen_temp_path);

			progress.reset(book_nodes.size() - 1);
			for(BookNodeIndex book_node_index = 0 ; book_node_index < BookNodeIndex(book_nodes.size()) ; ++book_node_index)
			{
				auto& book_node = book_nodes[book_node_index];

				StateInfo si,si2;
				string sfen;
				sfen_reader.ReadLine(sfen);

				// この局面の手番(書き出す前に判定してbook_node.colorに格納しているから、それを参照する。)
				if (book_node.color() == WHITE)
					sfen = Position::sfen_to_flipped_sfen(sfen);

				pos.set(sfen, &si, Threads.main());
				ASSERT_LV3(pos.side_to_move() == BLACK);

				// 定跡DBに登録されていた指し手
				SmallVector<BookMove> book_moves;
				std::swap(book_node.moves, book_moves); // swapしていったんbook_move.movesはクリアしてしまう。

				// ここから全合法手で一手進めて既知の(DB上の他の)局面に行くかを調べる。
				for(auto move:MoveList<LEGAL_ALL>(pos))
				{
					// moveで進めた局面が存在する時のhash値。
					HASH_KEY next_hash = pos.hash_key_after(move);

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
						BookMove book_move(move.to_move16(),
							ValueDepth(
								draw_value(REPETITION_DRAW, book_node.color()),
								BOOK_DEPTH_INF
							),
							next_book_node_index);

						// これが定跡DBのこの局面の指し手に登録されていないなら、
						// これは(定跡DBにはなかった指し手で進めたら既知の局面に)合流したということだから
						// 合流カウンターをインクリメントしておく。
						if (std::find_if(book_node.moves.begin(),book_node.moves.end(),[&](auto& book_move){ return book_move.move == move; })== book_node.moves.end())
							converged_moves++;

						book_node.moves.emplace_back(book_move);
					}
				}

				// 定跡DB上のこの局面の指し手も登録しておく。
				for(auto& book_move : book_moves)
				{
					Move16 move = book_move.move;

					// これがbook_nodeにすでに登録されているか？
					if (std::find_if(book_node.moves.begin(),book_node.moves.end(),[&](auto& book_move){ return book_move.move == move; })== book_node.moves.end())
						// 登録されてなかったので登録する。(登録されていればどうせmin-max探索によって値が上書きされるので登録しなくて良い。)
						// 登録されていなかったということは、ここから接続されているnodeはないので、出次数には影響を与えない。
						book_node.moves.emplace_back(book_move);
				}

				progress.check(book_node_index);
			}
			sfen_reader.Close();

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
				// すべてのnodeに更新がなければ終了したいので、そのためのフラグ。
				bool node_updated = false;

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

			// --- 書き出し ---

			// 書き出したsfenの個数
			u64 write_counter = 0;

			if (next)
			{
				if (next2)
					// peta_shock_next2コマンド
					write_peta_shock_next2_sfen(writebook_path, write_counter, root_sfens, next_nodes, from_startpos, eval_limit);
				else
					// peta_shock_nextコマンド
					write_peta_shock_next_sfen(writebook_path, write_counter, root_sfens, next_nodes, from_startpos);
			}
			else
			{
				// peta_shockコマンド
				write_peta_shock_book(writebook_path, write_counter, book_nodes, shrink);
			}

			// --- 結果出力 ---

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

	protected:

		// === helper function ===

		// あるnodeのbestと親に伝播すべきparent_vdとを得るヘルパー関数。
		// best_index : 何番目の指し手がbestであったのかを返す。
		// すべてのnode.movesがMOVE_NONEなら、best_index == size_maxを返す。
		ValueDepth get_bestvalue(BookNode& node , ValueDepth& parent_vd , size_t& best_index)
		{
			// まずこのnodeのbestを得る。
			ValueDepth best(-BOOK_VALUE_INF, BOOK_DEPTH_INF);
			// 親に伝播するbest
			parent_vd = ValueDepth(-BOOK_VALUE_INF,BOOK_DEPTH_INF);

			best_index = size_max;
			for(size_t i = 0 ; i< node.moves.size() ; ++i)
			{
				const auto& book_move = node.moves[i];

				// MOVE_NONEならこの枝はないものとして扱う。
				if (book_move.move.to_u16() == MOVE_NONE)
					continue;

				if (book_move.vd.is_superior(best, node.color()))
				{
					best = book_move.vd;
					best_index = i;
				}
			}

			// 親に伝播するほうはvalueを反転させておく。depthは+1しておく。
			parent_vd.value =            - best.value;
			parent_vd.depth = std::min(u16(best.depth + 1) , BOOK_DEPTH_INF);

			return best;
		};

		// 評価値が同じでdepth違いの枝があると、実戦で、そっちが選ばれ続けて千日手になりかねないので
		// (評価値が同じで)depthが高いほうの指し手は残りの枝の価値をわずかにマイナスする処理を行うヘルパー関数。
		void adjust_second_bestvalue(BookNode& node)
		{
			ValueDepth vd;
			size_t _;
			auto best = get_bestvalue(node, vd , _);

			for(auto& book_move : node.moves)
			{

				if (   best.value == book_move.vd.value
					&& best.depth != book_move.vd.depth
					)
				{
					// depthが最小でない指し手の評価値を1だけ減らしておく。

					// bestの値しか親には伝播しないので、ここで引いたところで
					// このnodeにしか影響はない。
					book_move.vd.value--;
				}
			}
		};


		// ペタショック化した定跡ファイルを書き出す。
		void write_peta_shock_book(std::string writebook_path, u64& write_counter, std::vector<BookNode>& book_nodes, bool shrink)
		{
			// 通常のpeta_shockコマンド時の処理。(peta_shock_nextコマンドではなく)

			// メモリ上の定跡DBを再構成。
			// この時点でもうhash_key_to_index不要なので解放する。
			// (clear()では解放されないので、swap trickを用いる。)
			HashKey2Index().swap(this->hashkey_to_index);

			// progress表示用
			Tools::ProgressBar progress;

			// MemoryBookを用いるとオーバーヘッドが大きいので自前で直接ファイルに書き出す。

			size_t n = book_nodes.size();

			cout << "Write to a book DB  : " << endl;

			SystemIO::TextWriter writer;
			if (writer.Open(writebook_path).is_not_ok())
			{
				cout << "Error! : open file error , path = " << writebook_path << endl;
				return;
			}

			progress.reset(n - 1);

			// バージョン識別用文字列
			writer.WriteLine(::Book::BookDBHeader2016_100);

			SystemIO::TextReader sfen_reader;
			sfen_reader.Open(sfen_temp_path);

			for(size_t i = 0 ; i < n ; ++i)
			{
				auto& book_node = book_nodes[i];
				string sfen;
				sfen_reader.ReadLine(sfen); // 元のsfen(手番を含め)通りにしておく。

				writer.WriteLine("sfen " + sfen);
				writer.Flush(); // ⇦ これ呼び出さないとメモリ食ったままになる。

				// 評価値順で降順sortする。
				std::sort(book_node.moves.begin(), book_node.moves.end(),
					[](const BookMove& x, const BookMove& y) {
						return x.vd.value > y.vd.value;
					});

				// 指し手を出力
				for(auto& move : book_node.moves)
				{
					// shrinkモードなら、最善手と異なる指し手は削除。
					if (shrink && book_node.moves[0].vd.value != move.vd.value)
						continue;

					// 元のDB上で後手の局面なら後手の局面として書き出したいので、
					// 後手の局面であるなら指し手を反転させる。
					Move16 m16 = (book_node.color() == WHITE) ? flip_move(move.move) : move.move;
					writer.WriteLine(to_usi_string(m16) + " None " + to_string(move.vd.value) + " " + to_string(move.vd.depth));
				}

				progress.check(i);
				write_counter++;

			}
			sfen_reader.Close();

			cout << "write " + writebook_path << endl;
		}

		// peta_shock_nextコマンドによるSFEN棋譜の書き出し。
		void write_peta_shock_next_sfen(std::string writebook_path, u64& write_counter, const std::vector<std::string>& root_sfens, u64 next_nodes, bool from_startpos)
		{
			// 書き出すsfen
			unordered_set<string> write_sfens;

			// 次に探索すべき定跡局面についてsfenを書き出していく。
			// これはmin-max探索した時のPVのleaf node。
			cout << "Retrograde Analysis : step IV  -> pick up next sfens to search." << endl;

			// rootから辿っていきPV leafに到達したらそのsfenを書き出す。
			// そのPV leaf nodeを削除して後退解析により、各局面の評価値を更新する。
			// これを繰り返す。

			// progress表示用
			Tools::ProgressBar progress;
			progress.reset(next_nodes * root_sfens.size());

			Position pos;

			// それぞれのroot_sfenに対して。
			// ⇨ この、root_sfen文字列は、"startpos moves ..."みたいな文字列でありうるので
			//   書き出すなら、これを普通のsfen文字列にしたものにしないといけないことに注意。
			for(auto root_sfen : root_sfens)
			{
				deque<StateInfo> si0;
				BookTools::feed_position_string(pos, root_sfen, si0);

				// 普通のsfen文字列にしたroot_sfen。
				string root_sfen0 = pos.sfen();

				// root局面のgame ply
				int root_ply = pos.game_ply();
				if (pos.side_to_move() == BLACK)
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
				// ⇨　最後に局面を書き出す時にpacked_sfenが必要になるから良くなかった。
				// ⇨　どうせここでPositionを用いて辿るのがボトルネックになっているわけではないので気にしないことにする。
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

					deque<StateInfo> si1;
					BookTools::feed_position_string(pos, root_sfen, si1);

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
							// 格納されているのは先手化した指し手なので、後手の手番であるなら、先手化する必要がある。
							Move16 m      = book_node.moves[best_index].move;
							Move16 move16 = (pos.side_to_move() == BLACK) ? m : flip_move(m);

							// 局面を進める
							si1.push_back(StateInfo());
							pos.do_move(pos.to_move(move16), si1.back());

							sfen_path += ' ' + to_usi_string(move16);

							string sfen = pos.sfen(ply + 1);

							// write_sfensのinsertはここでしか行わないので、ここでcheck()すれば十分。
							if (from_startpos)
								write_sfens.insert(sfen_path);
							else
								write_sfens.insert(sfen);
							progress.check(write_sfens.size());
							write_counter2++;

							// この手はないものとして、この book_node_index を起点として上流に更新していけばOK。
							book_node.moves[best_index].move = Move16::none();

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
							book_node.moves[best_index].move = Move16::none();

							break;
						}

						// 格納されているのは先手化した指し手なので、後手の手番であるなら、先手化する必要がある。
						Move16 m      = book_node.moves[best_index].move;
						Move16 move16 = (pos.side_to_move() == BLACK) ? m : flip_move(m);

						// 局面を進める
						si1.push_back(StateInfo());
						pos.do_move(pos.to_move(move16), si1.back());

						// 棋譜も進める
						sfen_path += ' ' + to_usi_string(move16);

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
								book_nodes[pm.parent].moves[pm.move_index].move = Move16::none();
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
		}

		// peta_shock_next2コマンドによるSFEN棋譜の書き出し。
		void write_peta_shock_next2_sfen(std::string writebook_path, u64& write_counter, const std::vector<std::string>& root_sfens, u64 next_nodes, bool from_startpos, s16 eval_limit)
		{
			// 書き出すsfen
			unordered_set<string> write_sfens;

			// 次に探索すべき定跡局面についてsfenを書き出していく。
			// これはmin-max探索した時のPVのleaf node。
			cout << "Retrograde Analysis : step IV  -> pick up next sfens to search." << endl;

			// rootから辿っていきPV leafに到達したらそのsfenを書き出す。
			// そのPV leaf nodeを削除して後退解析により、各局面の評価値を更新する。
			// これを繰り返す。

			// progress表示用
			Tools::ProgressBar progress;
			progress.reset(next_nodes * root_sfens.size());

			Position pos;

			// 一回のroot_sfenの、ある手番側について書き出す局面数
			// ⇨　next_nodesとして1を指定されることがあるので繰り上げ算で2で割る。
			u64 next_nodes0 = (next_nodes + 1) / 2;

			// それぞれのroot_sfenに対して。
			// ⇨ この、root_sfen文字列は、"startpos moves ..."みたいな文字列でありうるので
			//   書き出すなら、これを普通のsfen文字列にしたものにしないといけないことに注意。
			for(auto root_sfen : root_sfens)
			{
				deque<StateInfo> si0;
				BookTools::feed_position_string(pos, root_sfen, si0);

				// 普通のsfen文字列にしたroot_sfen。
				string root_sfen0 = pos.sfen();

				// root局面のgame ply
				int root_ply = pos.game_ply();
				if (pos.side_to_move() == BLACK)
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

				BookNodeIndex root_book_node_index = hashkey_to_index[pos.hash_key()];

				// PVを辿る手番側
				for(Color pv_color : COLOR)
				{
					deque<StateInfo> si1;
					BookTools::feed_position_string(pos, root_sfen, si1);

					// 現在の局面までの棋譜。あとで' '.join(kif)してSFEN文字列として書き出す。
					vector<string> sfen_path;
					sfen_path.emplace_back(root_sfen);
					if (!StringExtension::Contains(root_sfen, "moves"))
						sfen_path.emplace_back("moves");

					// leaf nodeまでの棋譜とその時の評価値。
					vector<KIF_EVAL> kifs;

					// 今回の探索で辿ったことのあるnode
					unordered_set<BookNodeIndex> visited_nodes;

					// leafまで辿る。
					peta_next_search(pos, pv_color, root_book_node_index, sfen_path, eval_limit, kifs, visited_nodes);

					// これですべて辿ったことになる。kifsを評価値で昇順sortして、
					// 評価値の良い順にsortして、上位からnext_nodes0 個、write_sfensとして書き出す。
					// ⇨　相手がすべての指し手(≒ランダム)なので、PVのevalより良くなるはず。
					//  だから、(自分から見た評価値として)昇順にして悪いほうから調べていくべき。
					// ⇨　安定sortでないと、一度遭遇した局面だから、棋譜がそこまでしか得られなくなってしまう。
					// ⇨　合流した時、棋譜を破棄した方がいいか…。

					u64 sort_num = std::min(next_nodes0, u64(kifs.size()));
					std::partial_sort(kifs.begin(), kifs.begin() + sort_num , kifs.end(),
						[pv_color](const KIF_EVAL& x, const KIF_EVAL& y) {
							// 格納されている評価値は先手から見た評価値となっている。
							// pv_colorが先手であるなら、評価値を昇順に並び替えて前からnext_nodes0 個取り出す。
							// pv_colorが後手であるなら、評価値を降順に並び替えて前からnext_nodes0 個取り出す。
							if (pv_color == BLACK)
								return x.second < y.second;
							else
								return x.second > y.second;
						});

					// ここで得られた棋譜、あとでまとめて書き出す。
					for(u64 i = 0 ; i < sort_num ; ++i)
					{
						write_sfens.insert(kifs[i].first);
						//cout << kifs[i].first << " , " << kifs[i].second << endl;
					}
				}
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
		}

		// 与えられた局面から再帰的に局面を辿る。
		// 現在の局面の手番がpv_colorであった場合は、最善手を辿り、さもなくば、すべての指し手を辿る。
		// (ただし評価値制限はある。絶対値が eval_limit 以内の指し手のみ)
		//
		// 今回の探索ですでに辿ったことのあるnodeに到達した場合、
		// 棋譜はそこまでとする。
		void peta_next_search(Position& pos, Color pv_color, BookNodeIndex book_node_index, vector<string>& sfen_path, s16 eval_limit, vector<KIF_EVAL>& kifs, unordered_set<BookNodeIndex>& visited_nodes )
		{
			// この局面に訪問した。
			visited_nodes.insert(book_node_index);

			BookNode& node = book_nodes[book_node_index];
			// 辿るやつの候補
			vector<u16> candidates;

			if (pos.side_to_move() == pv_color)
			{
				// 最善手のみ辿る。ただしabs(eval) <= eval_limitのものだけ。
				ValueDepth parent_vd;
				size_t best_index;
				s16 best_eval = get_bestvalue(node, parent_vd, best_index).value;

				for(u16 i = 0 ; i < node.moves.size() ; ++i)
				{
					auto& move = node.moves[i];
					if (move.vd.value == best_eval && abs(move.vd.value) <= eval_limit)
						candidates.emplace_back(i);
				}

			} else {

				// abs(eval) <= eval_limit 以上のものをすべて辿る。

				for(u16 i = 0 ; i < node.moves.size() ; ++i)
				{
					auto& move = node.moves[i];
					if (abs(move.vd.value) <= eval_limit)
						candidates.emplace_back(i);
				}

			}

			for(auto candidate : candidates)
			{
				auto& move = node.moves[candidate];

				Move16 m = move.move;
				// 格納されているのは先手化した指し手なので、後手の手番であるなら、先手化する必要がある。
				Move16 move16 = (pos.side_to_move() == BLACK) ? m : flip_move(m);

				// 棋譜も1手進める。
				sfen_path.emplace_back(to_usi_string(move16));

				BookNodeIndex next_book_node_index = move.next;

				// 1. この指し手で進めた局面が書き出し済みの局面である。
				// 2. この指し手で進めた局面が定跡DBから外れるか。
				// 3. そうでない。
				if (visited_nodes.count(next_book_node_index)){

					// 書き出し済みの局面に遭遇したということは、合流したと言うことで、
					// そのleafは書き出し済みであるから、今回の棋譜は破棄することにする。
					// (ここまでの手順を書き出してもいいがこれを書き出すとキリがないので書き出さないことにする。)

					;

				} else if (next_book_node_index == BookNodeIndexNull) {

					// 定跡DBから外れた。

					// その指し手で進めた局面を書き出す。
					// evalは、先手から見た評価値にしておく。

					s16 eval = pos.side_to_move() == BLACK ? move.vd.value : -move.vd.value;
					kifs.emplace_back(KIF_EVAL(StringExtension::Join(sfen_path, " "), eval));

				} else {

					// 局面を1手進めて再帰的に辿る。
					Move move32 = pos.to_move(move16);
					StateInfo si;
					pos.do_move(move32, si);
					peta_next_search(pos, pv_color, next_book_node_index, sfen_path, eval_limit , kifs, visited_nodes);
					pos.undo_move(move32);
				}

				// 棋譜を1手戻す。
				sfen_path.pop_back();
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

		// sfenファイルの一時ファイルを書き出すpath
		string sfen_temp_path;
	};
}

using namespace MakeBook2023;

namespace Book
{
	// 2023年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2023(Position& pos, istringstream& is, const string& token)
	{
		if (token == "peta_shock") {

			// ペタショックコマンド
			// 
			// やねうら王の定跡ファイルに対して定跡ツリー上でmin-max探索を行い、その結果を別の定跡ファイルに書き出す。
			//   makebook peta_shock book.db user_book1.db
			// 　⇨　先手か後手か、片側の局面しか書き出さない。エンジンオプションの FlippedBook を必ずオンにして用いること。
			//   makebook peta_shock book.db user_book1.db shrink
			//   ⇨  "shrink"を指定すると、その局面の最善手と同じ評価値の指し手のみを書き出す。
			PetaShock ps;
			ps.make_book(pos, is, PETA_SHOCK_TYPE::PetaShock);
			return 1;

		} else if (token == "peta_shock_next") {

			// ペタショックNextPV
			//
			// ペタショック手法と組み合わせてmin-maxして、有望な局面をSFEN形式でテキストファイルに書き出す。
			//   makebook peta_shock_next book.db sfens.txt 1000
			//   makebook peta_shock_next book.db sfens.txt 1000 eval_noise 20
			// ⇨　1000局面を書き出す。20はleaf nodeの指し手の評価値に加える乱数の大きさ。
			// 　 この場合、評価値に、平均 = 0 , 標準偏差 = 20 ガウスノイズを加算する。
			// 　　(これを加えることで序盤の指し手を開拓しやすくなる)
			//   makebook peta_shock_next book.db sfens.txt 1000 from_startpos
			// ⇨　startpos moves ... の形式で局面を出力する。
			// 
			//   makebook peta_shock_next book.db sfens.txt 1000

			PetaShock ps;
			ps.make_book(pos, is , PETA_SHOCK_TYPE::PetaShockNextPV);
			return 1;

		} else if (token == "peta_shock_next2") {

			// ペタショックNextHalfPV
			//
			// ペタショックNextPVの改良版。
			// root局面から、先手なら最善手、後手ならすべての指し手、というように辿っていった時の末端の局面を
			// SFEN形式でファイルに書き出す。(後手なら最善手、先手ならすべての指し手　に関しても同様)
			// 
			//  makebook peta_shock_next2 book.db sfens.txt 1000 eval_limit 400 from_startpos
			//
			// ⇨　先手が最善手を辿るパターンと後手が最善手を辿るパターンとで500局面ずつSFEN棋譜を書き出す。
			// 　 すべての指し手とは言え、評価値制限はする。⇑のように指定してあれば評価値の絶対値が400まで。
			//

			PetaShock ps;
			ps.make_book(pos, is , PETA_SHOCK_TYPE::PetaShockNextHalfPV);
			return 1;
		}

		return 0;
	}
}


#endif // defined (ENABLE_MAKEBOOK_CMD)
