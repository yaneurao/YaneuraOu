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

	BOOK_MAX_PLY回繰り返す:
		for node in nodes:
			v = nodeのなかで一番良い指し手の評価値
			for parent in node.parents:
				parentからnodeに行く指し手の評価値 = v

	nodesは定跡DB上のすべての定跡局面を意味します。
	node.parentは、このnodeに遷移できる親nodeのlistです。
	また、子nodeを持っている指し手の評価値は0(千日手スコア)で初期化されています。

	上のように実装すると、parentsのポインターが必要なので、これが要らないように改良して実装しています。
	あと、連続王手の千日手の処理については、その実装箇所にコメントがあるようにそこだけDFSしています。
*/

#include <sstream>
#include <vector>
#include <unordered_map>
#include <deque>
#include <algorithm>
#include <limits>
#include <random>
#include <utility> // For std::forward
#include <new>
#include <stdexcept>

#include "book.h"
#include "../thread.h"
#include "../position.h"
#include "../movegen.h"
#include "../misc.h"

using namespace std;
namespace YaneuraOu {

// ある局面の指し手の配列、std::vector<Move>だとsize_t(64bit環境で8バイト)でcapacityとかsizeとか格納するので
// 非常にもったいない。定跡のある局面の指し手が255手を超えることはないだろうから1byteでも十分。
// そこで、size_tではなくint16_tでサイズなどを保持しているvectorのsubsetを用意する。
// dataポインタが8バイトなのももったいないが…。これは仕方がないか…。

template <typename T>
class SmallVector {
public:
	SmallVector() noexcept : data(nullptr), count(0), capacity(0) {

		// 定跡の指し手、MultiPVで探索していて、
		// 4の倍数なのでcapacityは4を初期値にしておく。
		reserve(4);
	}

	// コピーコンストラクタ
	SmallVector(const SmallVector& other)
		: data(nullptr), count(0), capacity(0) {
		if (other.count > 0) {
			reserve(other.count);
			for (uint16_t i = 0; i < other.count; ++i) {
				new (&data[i]) T(other.data[i]);
			}
			count = other.count;
		}
	}

	// コピー代入演算子
	SmallVector& operator=(const SmallVector& other) {
		if (this != &other) {
			clear();
			release();
			if (other.count > 0) {
				reserve(other.count);
				for (uint16_t i = 0; i < other.count; ++i) {
					new (&data[i]) T(other.data[i]);
				}
				count = other.count;
			}
		}
		return *this;
	}

	// ムーブコンストラクタ
	SmallVector(SmallVector&& other) noexcept
		: data(nullptr), count(0), capacity(0) {
		swap(other);
	}

	// ムーブ代入演算子
	SmallVector& operator=(SmallVector&& other) noexcept {
		if (this != &other) {
			clear();
			release();
			swap(other);
		}
		return *this;
	}

	~SmallVector() noexcept {
		clear();
		release();
	}

	void push_back(const T& value) {
		emplace_back(value);
	}

	template <typename... Args>
	void emplace_back(Args&&... args) {
		if (count == capacity) {
			increase_capacity();
		}
		new (data + count) T(std::forward<Args>(args)...);
		++count;
	}

	void erase(T* position) {
		position->~T();
		for (T* it = position; it != data + count - 1; ++it) {
			new (it) T(std::move(*(it + 1)));
			(it + 1)->~T();
		}
		--count;
	}

	T& operator[](size_t idx) {
		return data[idx];
	}

	const T& operator[](size_t idx) const {
		return data[idx];
	}

	size_t size() const noexcept {
		return count;
	}

	T* begin() noexcept { return data; }
	T* end() noexcept { return data + count; }
	const T* begin() const noexcept { return data; }
	const T* end() const noexcept { return data + count; }

	void clear() noexcept {
		for (uint16_t i = 0; i < count; ++i) {
			data[i].~T();
		}
		count = 0;
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
		if (new_capacity > UINT16_MAX) {
			std::cout << "Error! : SmallVector exceeds maximum capacity" << std::endl;
			Tools::exit();
		}
		reserve(static_cast<uint16_t>(new_capacity));
	}

	void reserve(uint16_t new_capacity) {
		T* new_data = reinterpret_cast<T*>(new char[new_capacity * sizeof(T)]);
		for (uint16_t i = 0; i < count; ++i) {
			new (&new_data[i]) T(std::move(data[i]));
			data[i].~T();
		}
		release();
		data = new_data;
		capacity = new_capacity;
	}

	void release() noexcept {
		delete[] reinterpret_cast<char*>(data);
		data = nullptr;
		capacity = 0;
	}

	T* data;
	uint16_t count;
	uint16_t capacity;
};

// ADLを利用したswapの非メンバ関数バージョン
// SmallVectorの名前空間内で定義する
template <typename T>
void swap(SmallVector<T>& a, SmallVector<T>& b) noexcept {
    a.swap(b);
}

namespace MakeBook2025
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

	// MATEより大きなスコアを使うと、表示のときにバグる可能性がある。
	const int BOOK_VALUE_MAX  =  VALUE_MATE;
	const int BOOK_VALUE_MIN  = -VALUE_MATE;

	// 定跡で千日手手順の時のdepth。∞であることがわかる特徴的な定数にしておくといいと思う。
	const u16 BOOK_DEPTH_MAX = 9999;
	// 定跡で連続王手の千日手絡みの特殊な定数。連続王手の千日手で王手している側と王手されている側。
	// BOOK_DEPTH_PERPUTUAL_CHECKEDは使わないが、BOOK_DEPTH_PERPUTUAL_CHECKのほうは、ValueDepthで
	// 大小比較に用いるので、+1して偶然この値になってはまずい。
	// そこで他の定数を+1してもこの値にならない値に設定しておく必要がある。
	const u16 BOOK_DEPTH_PERPUTUAL_CHECKED = BOOK_DEPTH_MAX - 1;
	const u16 BOOK_DEPTH_PERPUTUAL_CHECK   = BOOK_DEPTH_MAX - 2;

	// 定跡の最大長さ(これ以上長いものは掘らなくていいという考え)
	// 乱数を入れているので、実際の手数はこれより少し少ないかも。
	const u16 BOOK_MAX_PLY = 256;

	// 定跡の評価値とその時のdepthをひとまとめにした構造体
	struct ValueDepth
	{
		ValueDepth()
			: value(BOOK_VALUE_NONE), depth(0) {}

		ValueDepth(s16 value, u16 depth)
			: value(value) , depth(depth) {}

		// 比較オペレーター
		bool operator==(const ValueDepth& v) const { return value==v.value && depth==v.depth;}
		bool operator!=(const ValueDepth& v) const { return !(*this==v); }

		// 優れているかの比較
		bool operator > (const ValueDepth& v) const
		{
			// 評価値ベースの比較
			if (this->value != v.value)
				return this->value > v.value;

			// BOOK_DEPTH_PERPUTUAL_CHECK を含むのであれば、これは特別扱いする。
			if (this->depth == BOOK_DEPTH_PERPUTUAL_CHECK)
				return false;
			if (v.depth == BOOK_DEPTH_PERPUTUAL_CHECK)
				return true;

			// depthベースの比較。評価値の符号で場合分けが生ずる。
			// valueが同じだとしても、負けているほうは手数を少しでも伸ばしたいし、勝っているほうは短い手数で勝ちたいため。

			if (this->value >= 0)
				return this->depth < v.depth;
			else
				return this->depth > v.depth;
		}

		s16 value;
		// depthは最大値 BOOK_DEPTH_MAXまでなので2bitほど余ってはいるか…。
		u16 depth;
	};

	std::ostream& operator<<(std::ostream& os, ValueDepth vd)
	{
		os << "( " << vd.value << ", " << vd.depth << ")";
		return os;
	}

	// 定跡の1つの指し手を表現する構造体
	// 高速化のために、BookNodeIndexで行き来する。
	// 📝 : immutableにしたいのだが、メンバ変数にconstをつけるとコピー/代入ができなくて困る。
	struct BookMove
	{
		// moveの指し手がleaf nodeである場合。
		BookMove(Move16 move, s16 value, s16 depth) :
			move(move), vd(ValueDepth(value, depth)), leaf(true) {
		}

		// moveの指し手がleaf nodeである場合。
		BookMove(Move16 move, ValueDepth vd):
			move(move), vd(vd), leaf(true) { }

		// moveの指し手がleaf nodeではない場合。
		BookMove(Move16 move, BookNodeIndex next) :
			move(move), next(next), leaf(false) {
		}

		// 指し手
		Move16 move;

		// ここがleaf nodeであるか？
		bool leaf;

		union
		{
			// 1. leaf_node == trueの時。
			//    moveを選んだ時の評価値
			ValueDepth vd;

			// 2. leaf_node == falseの時。
			//	  moveで局面を進めた時の次の局面のBookMoveポインター。
			BookNodeIndex next;
		};

		// move(2) + leaf_node(1) + { ValueDepth(4) or next(4) } = 7 bytes → 8
	};

	// 定跡の1つの局面を表現する構造体。
	// 高速化のために、BookNodeIndexで行き来をする。
	struct BookNode
	{
		// この局面での指し手
		// 📝 : read_book(), convergence_check()で初期化している。
		// ⚠ この局面は先手化された局面であるから、movesは先手番としての指し手
		SmallVector<BookMove> moves;

		// この局面の親に伝播させる時のValueとDepth。
		//   value = -best_value
		//   depth =  best_valueのdepth + 1
		// 💡 : 前回計算時のものと、今回の計算に用いるものとで2面用意して、交互に役割を入れ替えて用いるような実装が考えられるが、
		//       その方法だと、千日手サイクル(通常偶数手)が２つ接続されていたりすると、２つの値が交互に循環し続けたりすることがある。
		//       なので、何も考えずに同じ面に対して更新し続けたほうが良いようである。
		// 📝 : init_cycle_nodes()で初期化されている。
		ValueDepth vd;

		// 棋譜に出現したこの局面の手番。(書き出す時にこれを再現する)
		// 📝 : read_book()で初期化している。
		u8 color;

		// このNodeはconst nodeとみなせるか。
		// (movesすべてがconst(leaf node)もしくはconst nodeであるか)
		// 📝 : convergence_check()で初期化している。
		bool const_node;

		// この局面で王手されているか
		// 📝 : read_book()で初期化している。
		bool checked;

		// 連続王手の千日手のループ上の局面である。
		// 📝 : read_book()で初期化している。そのあとextract_check_loop()で求めている。
		bool check_loop;

		//u8 reserved;

		// 16(moves) + 4(vd) + 1(color) + 1*3(flags) = 24 bytes
	};

	// ペタショック化
	class PetaShock
	{
	public:
		// 定跡をペタショック化する。
		void make_book(istringstream& is, string book_dir)
		{
			// 初期化等
			initialize(is, book_dir);

			// ペタショック化する定跡ファイルの読み込み
			read_book();

			// 局面の合流チェック
			convergence_check();

			// 後退解析その1 : 出次数0の局面を定跡ツリーから削除
			remove_const_nodes();

			// 後退解析その2 : 連続王手の千日手のループを抽出
			extract_check_loop();

			// 千日手スコアで各ノードを初期化する。
			init_cycle_nodes();

			// 後退解析その3 : 評価値の親ノードへの伝播
			propagate_all_nodes();

			// ペタショック化した定跡の書き出し
			write_peta_shock_book(writebook_path, book_nodes);

			// 結果出力
			output_result();
		}

	protected:

		// === helper function ===

		// 初期化
		void initialize(istringstream& is, string book_dir)
		{
			// hashkeyのbit数をチェックして、128bit未満であれば警告を出す。

			// 高速化のために、HASH_KEYだけで局面を処理したいので、hashの衝突はあってはならない。
			// そのため、HASH_KEY_BITSは128か256が望ましい。
			if (HASH_KEY_BITS < 128)
			{
				std::cout << "WARNING! : HASH_KEY_BITS = " << HASH_KEY_BITS << " is too short." << endl;
				std::cout << "    Rebuild with a set HASH_KEY_BITS == 128 or 256." << endl;
			}

			is >> readbook_path >> writebook_path;

			// コマンドラインオプションの読み込み
			string token;
			shrink = fast = false;
			while (is >> token)
			{
				if (token == "shrink")
					shrink = true;

				else if (token == "fast")
					fast = true;
			}

			string BOOK_DIR = book_dir;

			readbook_path  = Path::Combine(BOOK_DIR, readbook_path);
			writebook_path = Path::Combine(BOOK_DIR, writebook_path);
			sfen_temp_path = Path::Combine(BOOK_DIR, SFEN_TEMP_FILENAME);

			cout << "[ PetaShock makebook CONFIGURATION ]" << endl;

			cout << "readbook_path      : " << readbook_path << endl;
			cout << "writebook_path     : " << writebook_path << endl;

			cout << "shrink             : " << shrink << endl;
			cout << "fast               : " << fast << endl;

			/*
				note: DrawValueの変更について。

				引き分けのスコアを0から変更するには、各局面について、初期局面が{ 先手, 後手 }の2通りを持つ必要がある。
				ここでは、その処理をしていないので、DrawValueの変更した場合の動作は未定義。
			*/

			drawValueTable[REPETITION_DRAW][BLACK] = 0;
			drawValueTable[REPETITION_DRAW][WHITE] = 0;

			// 統計値の初期化

			converged_moves    = 0;
			const_nodes        = 0;
			in_check_counter   = 0;
			check_loop_counter = 0;

			// 各種配列のクリア

			original_sfens.clear();
			check_loop_nodes.clear();
		}

		// ペタショック化前の定跡ファイルの読み込み。
		Tools::Result read_book()
		{
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
				return Tools::ResultCode::FileNotFound;
			}

			// 行バッファ
			std::string line;

			// headerとnoeを調べる。
			while (reader.ReadLine(line).is_ok())
			{
				auto line_no = reader.GetLineNumber();
				if (line_no == 1)
				{
					// バージョン識別文字列がなければwarningを出す。
					if (line.length() < 1 || line[0] != '#' || line != YaneuraOu::Book::BookDBHeader2016_100)
						cout << "WARNING : illegal YaneuraOu Book header 2016" << endl;
				}
				else if (line_no == 2)
				{
					// 2行目には
					// # NOE:258
					// よって、この文字列をparseする必要がある。
					if (line.length() >= 1 && line[0] == '#')
					{
						// いったん変数に入れておかないと、temporary objectのstring_viewを作ることになってそのあと不正になる。
						auto line2 = line.substr(2);
						auto splited = StringExtension::Split(line2, ",");
						for (auto command : splited)
						{
							auto splited2 = StringExtension::Split(command, ":");
							if (splited2.size() >= 1)
							{
								auto& token = splited2[0];
								if (token == "NOE" && splited2.size() == 2) // numbers of entires
								{
									size_t noe = StringExtension::to_int(string(splited2[1]), 0);
									cout << "Number Of Elements : " << noe << endl;

									// エントリー数が事前にわかったので、その分だけそれぞれの構造体配列を確保する。
									book_nodes.reserve(noe);
									hashkey_to_index.reserve(noe);
									if (fast)
										original_sfens.reserve(noe);
								}
							}
						}
					}
				}
				else
					break;
			}
			reader.Close();
			reader.Open(readbook_path);

			progress.reset(reader.GetSize() - 1);

			// sfen文字列はファイルに書き出す。
			SystemIO::TextWriter sfen_writer;
			if (!fast)
				sfen_writer.Open(sfen_temp_path);

			Position pos;

			while (reader.ReadLine(line).is_ok())
			{
				progress.check(reader.GetFilePos());

				// #で始まる行は読み飛ばす。
				if (line.length() >= 1 && line[0] == '#')
					continue;

				// コメント行(とりあえず読み飛ばす)
				if (line.length() >= 2 && line.substr(0, 2) == "//")
					continue;

				// "sfen "で始まる行は局面のデータであり、sfen文字列が格納されている。
				if (line.length() >= 5 && line.substr(0, 5) == "sfen ")
				{
					string sfen = line.substr(5); // 新しいsfen文字列を"sfen "を除去して格納

					// sfen文字列はテンポラリファイルに書き出しておく。(もし末尾に手数があるなら、それも含めてそのまま書き出す)
					// あとで局面を書き出す時に用いる。
					if (fast)
						original_sfens.push_back(sfen);
					else
						sfen_writer.WriteLine(sfen);

					// 末尾のplyは使わないので除去。
					StringExtension::trim_number_inplace(sfen);

					// この局面の(元の)手番
					// ⇨ "w"の文字は駒には使わないので"w"があれば後手番であることが確定する。
					Color stm = (sfen.find('w') != std::string::npos) ? WHITE : BLACK;

					// 後手番化したsfen。
					// hashkeyは、すべて後手番の局面で考えるから、hashkeyを求めるときに後手番の局面にしておく。
					string white_sfen = stm == WHITE ? sfen : Position::sfen_to_flipped_sfen(sfen);

					// hashkey_to_indexには後手番の局面のhash keyからのindexを登録する。
					StateInfo si;
					pos.set(white_sfen, &si);
					Key white_hash_key = pos.key();
					// 元の定跡ファイルにflipした局面は登録されていないものとする。
					// ⇨  登録されていたら、あとから出現した局面を優先する。

					auto book_node_index = BookNodeIndex(book_nodes.size()); // emplace_back()する前のsize()が今回追加されるindex
					hashkey_to_index[white_hash_key] = book_node_index;

					// この局面は王手されているのか？
					bool checked = pos.checkers();

					book_nodes.emplace_back(BookNode());
					auto& book_node = book_nodes.back();

					book_node.color        = stm; // 元の手番。これを維持してファイルに書き出さないと、sfen文字列でsortされていたのが狂う。
					book_node.checked      = checked;
					book_node.check_loop   = checked;
					in_check_counter      += checked;

					// この直後にやってくる指し手をこの局面の指し手として取り込む。
					continue;
				}

				// 先に"sfen .."が出現しているはずなので、back()は有効なはず。
				// ("sfen"が出現せずに指し手が出現するような定跡ファイルはおかしいわけで…)
				auto& book_node = book_nodes.back();

				// この行に Move PonderMove value depthが書かれている。これをそのままparseする。

				Parser::LineScanner scanner(line);
				auto move_str   = scanner.get_text();
				auto ponder_str = scanner.get_text(); // 使わないがskipはしないといけない。
				auto value      = (s16)std::clamp((int)scanner.get_number(0), BOOK_VALUE_MIN, BOOK_VALUE_MAX);
				auto depth      = (s16)scanner.get_number(0);
				Move16 move16 = (move_str == "none" || move_str == "None" || move_str == "resign") ? Move16::none() : USIEngine::to_move16(move_str);
				//Move16 ponder = (ponder_str == "none" || ponder_str == "None" || ponder_str == "resign") ? Move16::none() : USI::to_move16(ponder_str);

				// 先手の局面として登録しないといけないので後手番であるなら指し手もflipする。
				// posは上でblack_sfenが設定されているので先手番になるようにflipされている。
				if (book_node.color == WHITE)
					move16 = flip_move(move16);

				book_node.moves.emplace_back(BookMove(move16, value, 0 /*depth*/));
				// あとで合流のチェックをしてleaf nodeであるかを確認する。
			}
			if (!fast)
				sfen_writer.Close();

			return Tools::Result::Ok();
		}

		// 局面の合流チェック
		void convergence_check()
		{
			cout << "Convergence Check   :" << endl;

			// テンポラリとして書き出していたsfenファイル。
			// これは、元の定跡ファイルに出現したsfen文字列がそのまま書き出されている。
			SystemIO::TextReader sfen_reader;
			if (!fast)
				sfen_reader.Open(sfen_temp_path);

			Tools::ProgressBar progress;
			progress.reset(book_nodes.size() - 1);

			Position pos;

			// note : ここ、スレッド並列にして高速化すべきだが、
			//  テンポラリファイルにsfen文字列を書き出してしまっているのでそれができない。
			//  ⇨  fastオプションが指定されている時はその限りではないか..

			for (BookNodeIndex i = 0; i < BookNodeIndex(book_nodes.size()); ++i)
			{
				auto& book_node = book_nodes[i];

				StateInfo si, si2;
				string sfen;
				if (fast)
					sfen = original_sfens[i];
				else
					sfen_reader.ReadLine(sfen);

				// この局面が後手番なら、sfenを先手の局面化する。
				// 💡: BookNodeは先手の局面で考えている。hashkeyは後手の局面で考えている。
				if (book_node.color == WHITE)
					sfen = Position::sfen_to_flipped_sfen(sfen);

				pos.set(sfen, &si);
				ASSERT_LV3(pos.side_to_move() == BLACK);

				// 元ファイルの定跡DBに登録されていた指し手
				SmallVector<BookMove> book_moves;
				std::swap(book_node.moves, book_moves); // swapしていったんbook_move.movesはクリアしてしまう。

				// ここから全合法手で一手進めて既知の(定跡ツリー上の他の)局面に行くかを調べる。
				for (auto move : MoveList<LEGAL_ALL>(pos))
				{
					// moveで進めた局面が存在する時のhash値。
					Key next_hash = pos.key_after(move);

					if (hashkey_to_index.count(next_hash) > 0)
					{
						// 定跡局面が存在した。

						BookNodeIndex next_book_node_index = hashkey_to_index[next_hash];
						BookNode&     next_book_node       = book_nodes[next_book_node_index];

						BookMove book_move(move.to_move16(), next_book_node_index);

						// これが定跡DBのこの局面の指し手に登録されていないなら、
						// これは(定跡DBにはなかった指し手で進めたら既知の局面に)合流したということだから
						// 合流カウンターをインクリメントしておく。
						if (std::find_if(book_node.moves.begin(), book_node.moves.end(), [&](auto& book_move) { return book_move.move == move; }) == book_node.moves.end())
							converged_moves++;

						book_node.moves.emplace_back(book_move);
					}
				}

				// どこにも合流していなければ、これは定跡ツリー上で、leaf nodeしか存在しないnodeである。
				book_node.const_node = book_node.moves.size() == 0;

				// 元ファイルの定跡DB上のこの局面の指し手も登録しておく。
				for (auto& book_move : book_moves)
				{
					Move16 move = book_move.move;
					if (move == Move16::none())
						continue;

					// これがbook_nodeにすでに登録されているか？
					// 登録されているということは合流する( = 子局面がある)ということだから、評価値は子局面のものを使うので
					// ここで評価値を反映させる必要はない。
					if (std::find_if(book_node.moves.begin(), book_node.moves.end(), [&](auto& book_move) { return book_move.move == move; }) == book_node.moves.end())
						// 登録されてなかったので登録する。(登録されていればどうせmin-max探索によって値が上書きされるので元の定跡ファイルの評価値は反映させなくて良い。)
						book_node.moves.emplace_back(book_move);
				}

				progress.check(i);
			}
			if (fast)
				sfen_reader.Close();

			//cout << "converged_moves : " << converged_moves << endl;
		}

		// 親に伝播するためのVDを作る。(評価値を反転させて、depthを1加算)
		ValueDepth make_vd_for_parent(ValueDepth vd)
		{
			// 親に伝播するValueDepthなので、valueを反転させ、depthは+1しておく。
			return ValueDepth(-vd.value, std::min(u16(vd.depth + 1), BOOK_DEPTH_MAX));
		}

		// あるnodeについて、leaf nodeと子nodeを調べ、そのnodeの親に伝播すべきValueDepthを得るヘルパー関数。
		ValueDepth bestvd_for_parent(const BookNode& node)
		{
			ValueDepth best(-BOOK_VALUE_INF, BOOK_DEPTH_MAX);

			for (const auto& book_move : node.moves)
			{
				// leaf nodeであるなら、このbook_moveのvdが有効。
				// leaf nodeでないなら、子のvdを見る。
				// 💡 右辺はtemporary objectではないので、左辺はauto&で問題ない。
				const auto& vd = book_move.leaf ? book_move.vd : book_nodes[book_move.next].vd;
				if (vd > best)
					best = vd;
			}

			return make_vd_for_parent(best);
		}

		// 評価値の親ノードへの伝播を1回だけ行う。
		// const nodeの一つ上のnodeだけが処理対象。
		// 返し値 : 今回const nodeにしたnodeの数
		u64 remove_const_nodes_once()
		{
			/*
			  sfen_to_hashkeyとhashkey_to_book_nodeを頼りに、
			  定跡上のmin-max探索する。

			  まず後退解析みたいなことをして、ループ以外はある程度処理しておく。
			  アルゴリズム的には、
			   queue = 出次数0のnodeの集合
			   while queue:
				 node = queue.pop()
				 eval  = max(node.各定跡の指し手の評価値)
				 depth = evalがmaxだった時の定跡のdepth
				 foreach parent in node.parents:
				   parentのnodeへの指し手の評価値 = - eval     # min-maxなので評価値は反転する
				   parentのnodeへの指し手のdepth  =   depth+1  # 1手深くなる
				   if --parent.出次数 == 0:                    # 出次数が0になったのでこの局面もqueueに追加する。
					 queue.push(parent)

			  とやることが考えられる。

			  しかし、これだと親ノードへのpointerが必要になって、working memoryが増える。

			  ここでは親ノードを使わないで求める。
			*/

			// 子がすべてleafもしくはconst nodeであるなら、それはconst nodeにできる。

			// TODO : ここの処理は並列化できる。ここの速度はさほど問題ではないので、まあいいか…。
			u64 node_count = 0;
			for (BookNodeIndex book_node_index = 0; book_node_index < BookNodeIndex(book_nodes.size()); ++book_node_index)
			{
				auto& node = book_nodes[book_node_index];

				// const node以外を処理対象とする。
				if (node.const_node)
					continue;

				// このnodeのすべての指し手がleafもしくはconst nodeか？
				// ⇨ 子nodeがあって、そこがconst nodeでなければ、このnodeは処理対象ではない。
				for (auto& move : node.moves)
					if (!move.leaf && !book_nodes[move.next].const_node)
						goto Next;

				// すべてがconst nodeだったので、このnodeをconst node化できる。

				// 子のbestをnode.vdに反映。これは次回以降にこのnodeの親が用いる。
				node.vd = bestvd_for_parent(node);
				node.const_node = true;
				node_count ++;

			Next: ;
			}
			return node_count;
		}

		// 評価値の親ノードへの伝播して出次数0のnodeを定跡ツリーから削除していく。(最終的に定跡ファイルには書き出す)
		void remove_const_nodes()
		{
			for (auto& node : book_nodes)
				node.const_node = false;

			cout << "Retrograde Analysis : Step I   -> delete nodes with zero out-degree." << endl;

			Tools::ProgressBar progress;
			progress.reset(BOOK_MAX_PLY);

			for(int i = 0; i < BOOK_MAX_PLY; ++i)
			{
				u64 count = remove_const_nodes_once();
				const_nodes += count;

				if (count == 0)
					break;
					// const nodeにできるところはすべてそうなった。
					// 残りは非const node(leaf nodeから辿ったときにループを含むnode)

				progress.check(i);
			}

			progress.check(BOOK_MAX_PLY);
		}

		// 後退解析その2 : 連続王手の千日手のループを抽出
		void extract_check_loop()
		{
			/*
				note : ペタショック化アルゴリズム、連続王手の処理のアルゴリズム

				連続王手からなる閉経路をいまcheck loopと書く。

				1. check loopを発見できれば良い。
				2. あとはそのループの指し手の評価値を -INF(マイナス無限大 = 自分負け)で初期化すると、
					通常のペタショック化アルゴリズムを適用できる。
				3. ただし、そのcheck loopから他のループに遷移できると、そのループのvalueの初期値(0)が
					最初のpropagateで流れこんできて、これがcheck loopのなかで伝播されていってしまう。
				4. そこで、2.3.を考慮して、check loopのnodeだけ記憶しておいて、そこだけpropagateのときにdfsする。

				・check loopの発見方法アルゴリズムについて

					ある局面が王手がかかっている局面とする。

					そうすると、それがcheck loop上の局面であるなら、
					この2手先の局面も王手がかかっているはずである。
					逆に2手先の局面に王手がかかっている局面がないなら、それはcheck loop上の局面ではない。

				これを利用すると次のように書ける。

				```
				checks = 王手のかかっている局面の集合
				for ＿ in range(BOOK_MAX_PLY):
					for check in checks:
						if checkの2手先に王手がかかっている局面がない:
							checks.remove(check) # この局面を取り除く
				```

				たったこれだけである。これにより、このループを抜けるとchecksにcheck loop上の局面だけが残る。
				(これを数珠つなぎにする処理は必要)

				数珠つなぎにする処理は、以下のようにcheck loopの局面取り出して調べると手っ取り早い。

				```
				checked_pos = []
				for check in checks:
					for checked in check.children:
						if checkedの子 in checks:
							checked_pos.append(checked)
				```

				これでcheck loop上の局面が得られたので、これをpropagateのときに利用する。
			*/

			cout << "Retrograde Analysis : Step II  -> Extract perpetual check repetition loops." << endl;

			// 📒 アルゴリズム
			//   あるnode Aがcheck_loopだとする。
			//   この2手先にcheck_loopであるnodeがなければ、node Aはcheck loopではない。
			//   そこで、このようにしてcheck loop集合から取り除いていき、取り除けないようになった残りがcheck loop集合の王手されている局面。
			// 
			//   ここで得られたcheck loop集合を数珠つなぎにしたものが、check loop集合。

			Tools::ProgressBar progress;
			progress.reset(BOOK_MAX_PLY - 1);

			unordered_set<BookNodeIndex> check_loop_nodes_set;

			for(int i = 0; i < BOOK_MAX_PLY ; ++i)
			{
				// 今回更新されたnodeの個数
				u64 updated = 0;
				for (auto& node : book_nodes)
				{
					if (!node.check_loop)
						continue;

					// 2手先がcheck_loopか調べる
					for (auto& move : node.moves)
					{
						if (move.leaf)
							continue;

						auto& next_node = book_nodes[move.next];
						for (auto& move2 : next_node.moves)
						{
							if (move2.leaf)
								continue;

							auto& next_next_node = book_nodes[move2.next];
							if (next_next_node.check_loop)
								goto Next;
						}
					}
					// 2手先にcheck loop上の局面が見つからなかった。
					// ゆえに、元のnodeはcheck loop上の局面ではない。
					node.check_loop = false;
					updated++;
					continue;

				Next:;
				}
				progress.check(i);
				if (updated == 0)
					break;
			}

			// 最終的に何個残ったのか？
			for (BookNodeIndex i = 0; i < BookNodeIndex(book_nodes.size()) ; ++i)
			{
				auto& node = book_nodes[i];
				if (node.check_loop)
					check_loop_nodes_set.insert(BookNodeIndex(i));
			}

			// check_loop上で王手がかかっていた局面の数
			check_loop_counter = check_loop_nodes_set.size();

			// check_loop_nodesの局面を数珠つなぎにする。
			// 
			// 📒 アルゴリズム
			// check_loop上の局面の2手先の局面がcheck_loop上の局面であるなら、この間にある局面もcheck_loop上の局面である。

			std::vector<BookNodeIndex> nodes(check_loop_nodes_set.begin(), check_loop_nodes_set.end());
			for (auto index: nodes)
			{
				auto& node = book_nodes[index];

				// 2手先がcheck_loopか調べる
				for (auto& move : node.moves)
				{
					if (move.leaf)
						continue;

					auto next_node_index = move.next;
					auto& next_node = book_nodes[next_node_index];
					for (auto& move2 : next_node.moves)
					{
						if (move2.leaf)
							continue;

						auto& next_next_node = book_nodes[move2.next];
						if (next_next_node.check_loop)
						{
							// この間のnodeもcheck_loopの仲間入り。
							next_node.check_loop = true;
							check_loop_nodes_set.insert(next_node_index);
						}
					}
				}
			}

			// this->check_loop_nodes(これはvector)にコピーしておく。
			check_loop_nodes.reserve(check_loop_nodes_set.size());
			std::copy(check_loop_nodes_set.begin(), check_loop_nodes_set.end(), std::back_inserter(check_loop_nodes));

			progress.check(BOOK_MAX_PLY);
		}

		// 千日手スコアで各ノードを初期化する。
		void init_cycle_nodes()
		{
			cout << "Retrograde Analysis : Step III -> Initialize the cycle nodes." << endl;

			Tools::ProgressBar progress;
			progress.reset(book_nodes.size() - 1);

			// サイクルになっているノードのみを千日手スコアで初期化する。
			// サイクルになっていなければ、remove_const_nodes()でconst node化されているはず。
			for (size_t i = 0 ; i < book_nodes.size() ; ++i)
			{
				auto& node = book_nodes[i];
				if (!node.const_node)
				{
					if (!node.check_loop)
						// 通常の(連続王手の千日手ではない)千日手なら0で初期化。
						node.vd = ValueDepth(0, BOOK_DEPTH_MAX);
					else
						// 連続王手の千日手であるなら、王手されているなら(parent用のvdは)-INF,王手されてないなら+INFで初期化。
						node.vd = ValueDepth(node.checked ? BOOK_VALUE_MIN : BOOK_VALUE_MAX, BOOK_DEPTH_MAX);
				}
				progress.check(i);
			}
		}

		// このnodeの内容を出力する。(debug用)
		void dump_node(BookNodeIndex index)
		{
			cout << "--- BookNodes[" << index << "] ---" << endl;
			const auto& node = book_nodes[index];

			cout << "const_node = " << node.const_node << ", check_loop = " << node.check_loop << endl;
			cout << "vd = " << node.vd << endl;

			for (auto& move : node.moves)
			{
				cout << "is leaf = " << move.leaf << ", move = " << to_usi_string(move.move) << " : ";
				if (move.leaf)
					cout << "vd = " << move.vd << endl;
				else
				{
					cout << "next index = " << move.next << ", book_nodes[next].vd = ";
					const auto& next_node = book_nodes[move.next];
					cout << next_node.vd << endl;
				}
			}
		}

		// 各ノードのbestvalueを親ノードに伝播させる。
		// 
		// 返し値
		//   今回更新されたノード数。
		u64 propagate_all_nodes_once()
		{
			// 今回更新されたnode数
			u64 nodes_count = 0;
			for (BookNodeIndex i = 0 ; i < BookNodeIndex(book_nodes.size()) ; ++i)
			{
				auto& node = book_nodes[i];

				// const node　⇨　vdの値が変わらないので更新は無駄
				// check loop  ⇨  このあとdfsで更新するのでここで更新するとおかしくなる
				if (node.const_node || node.check_loop)
					continue;

				auto best = bestvd_for_parent(node);

				// これは循環ではないものが絡んだためにBOOK_DEPTH_MAXになっていないだけで、
				// 実際は循環であると思う。
				if (best.depth > BOOK_MAX_PLY)
					best.depth = BOOK_DEPTH_MAX;

				// 前回からvdが変化した箇所のカウント。
				nodes_count += node.vd != best;

				// vdを更新する。
				node.vd = best;
			}

			//cout << nodes_count << endl;

			return nodes_count;
		}

		// check loopのあるnodeからdfsして親node用のValueDepthを返す。
		ValueDepth dfs_for_check_loop_node(BookNodeIndex node_index, unordered_set<BookNodeIndex>& trajectory)
		{
			auto& node = book_nodes[node_index];

			// check loop上の局面ではないのでこれ以上探索しない。この局面のvdをそのまま返す。
			if (!node.check_loop)
				return node.vd;

			// 連続王手の千日手が成立。
			if (trajectory.count(node_index))
			{
				// 王手されている側の手番なら+MAX,王手している側なら-MAX を親node用に評価値を反転させて返す。
				// BOOK_DEPTH_PERPUTUAL_CHECK/CHECKEDは、連続王手の千日手を表す特殊な定数
				if (node.checked)
					return ValueDepth(-BOOK_VALUE_MAX, BOOK_DEPTH_PERPUTUAL_CHECK);
				else
					return ValueDepth(+BOOK_VALUE_MAX, BOOK_DEPTH_PERPUTUAL_CHECKED);
			}

			// check loop上の局面で、まだ連続王手の千日手は成立していないのでさらにdfsで探索する。

			trajectory.insert(node_index);

			ValueDepth best(-BOOK_VALUE_INF, 0);

			for (const auto& move : node.moves)
			{
				// leaf nodeであるなら、このbook_moveのvdが有効。
				// leaf nodeでないなら、次のnodeを再帰的に辿ってvdを得る。
				// ⚠ 右辺はtemporary objectなので左辺はauto&だとまずい。
				const auto vd = move.leaf ? move.vd : dfs_for_check_loop_node(move.next, trajectory);
				if (vd > best)
					best = vd;
			}

			trajectory.erase(node_index);

			return make_vd_for_parent(best);
		}

		// check loop上の局面だけdfsする。
		void dfs_for_check_loop_nodes()
		{
			// 局面集合 : check_loop_nodes
			// ここからdfsする。check loop上の局面は丁寧に辿る。

			unordered_set<BookNodeIndex> trajectory; // dfsでいま訪問した局面(循環の検出用)
			for (auto node_index : check_loop_nodes)
			{
				auto& node = book_nodes[node_index];
				node.vd = dfs_for_check_loop_node(node_index, trajectory);
			}
		}


		// 後退解析その3 : 評価値の親ノードへの伝播
		// 「各ノードのbestvalueを親ノードに伝播させる」をBOOK_MAX_PLY回繰り返す。
		void propagate_all_nodes()
		{
			cout << "Retrograde Analysis : Step IV  -> Propagate the eval to the parents of all nodes." << endl;

			Tools::ProgressBar progress;
			progress.reset(BOOK_MAX_PLY + 100);

			// 最大でBOOK_MAX_PLY + 100回だけ評価値を伝播させる。
			for (int ply = 0; ply < BOOK_MAX_PLY + 100; ++ply)
			{
				// 親nodeにValueDepthを伝播させる。
				u64 updating_nodes = propagate_all_nodes_once();

				// check loop上の局面だけdfsする。
				dfs_for_check_loop_nodes();

				progress.check(ply);

				// 更新がすべて完了したので早期終了。
				if (updating_nodes == 0)
					break;
			}
			progress.check(BOOK_MAX_PLY + 100);
		}

		// ペタショック化した定跡ファイルを書き出す。
		//	shrink : bestvalueの指し手のみを書き出す。
		void write_peta_shock_book(std::string writebook_path, std::vector<BookNode>& book_nodes)
		{
			// 通常のpeta_shockコマンド時の処理。(peta_shock_nextコマンドではなく)

			// メモリ上の定跡DBを再構成。
			// この時点でもうhash_key_to_index不要なので解放する。
			// (clear()では解放されないので、swap trickを用いる。)
			HashKey2Index().swap(this->hashkey_to_index);

			// progress表示用
			Tools::ProgressBar progress;

			// MemoryBookを用いるとオーバーヘッドが大きいので自前で直接ファイルに書き出す。

			cout << "Write to a book DB  : " << endl;

			SystemIO::TextWriter writer;
			if (writer.Open(writebook_path).is_not_ok())
			{
				cout << "Error! : open file error , path = " << writebook_path << endl;
				return;
			}

			progress.reset(book_nodes.size() - 1);

			// バージョン識別用文字列
			writer.WriteLine(YaneuraOu::Book::BookDBHeader2016_100);

			SystemIO::TextReader sfen_reader;
			if (!fast)
				sfen_reader.Open(sfen_temp_path);

			for(BookNodeIndex i = 0 ; i < BookNodeIndex(book_nodes.size()) ; ++i)
			{
				auto& book_node = book_nodes[i];
				string sfen;
				if (fast)
					sfen = original_sfens[i];
				else
					sfen_reader.ReadLine(sfen); // 元のsfen(手番を含め)通りにしておく。

				writer.WriteLine("sfen " + sfen);
				writer.Flush(); // ⇦ これ呼び出さないとメモリ食ったままになる。

				// いったんコピー。
				SmallVector<BookMove> moves;
				for (auto& move : book_node.moves)
					if (move.leaf)
						moves.emplace_back(move);
					else
					{
						auto& next_node = book_nodes[move.next];
						auto vd = next_node.vd;
						// この指し手を選ぶとcheck loop(連続王手の千日手サイクル)に突入して
						// かつ王手されている局面になる。この指し手と同じ評価値がbestであるなら、
						// こちらの指し手は選びたくないので選ばれないようにdepthを調整する。
						if (next_node.check_loop && next_node.checked)
							vd.depth = BOOK_DEPTH_PERPUTUAL_CHECK;
						moves.emplace_back(BookMove(move.move, vd));
					}

				// 評価値順で降順sortする。
				std::sort(moves.begin(), moves.end(),
					[](const BookMove& x, const BookMove& y) {
						return x.vd > y.vd;
					});

				// 指し手を出力
				for(size_t i = 0 ; i < moves.size() ; ++i)
				{
					auto& move = moves[i];

					// shrinkモードなら、最善手と異なる指し手は削除。
					if (shrink && moves[0].vd.value != move.vd.value)
						continue;

					// 1.
					// valueがbestmoveと同じだが、depthが異なるなら、valueを-1しておく。
					// (千日手絡みで手順が伸びている/縮んでいるのかも知れないから)

					// 2.
					// あと、連続王手のループから(王手している側が)そこから抜ける指し手があるとき、
					// ループ回る指し手と抜ける指し手が同じ評価値であることがある。
					// この時、depthを比較してループを回るほうの指し手を選ぶといつまでもループが抜けられなくて
					// 連続王手の千日手が成立してしまう。
					// よって、check loopでかつ!checkのときで同じスコアのときにはdepthを見てはならない。
					// そのため、depth == BOOK_DEPTH_PERPUTUAL_CHECKはValueDepthのoperator >() で特殊な処理をしている。

					if (i > 0 && moves[0].vd.value == move.vd.value && moves[0].vd.depth != move.vd.depth)
						move.vd.value--;

					// 元のDB上で後手の局面なら後手の局面として書き出したいので、
					// 後手の局面であるなら指し手を反転させる。
					Move16 m16 = (book_node.color == WHITE) ? flip_move(move.move) : move.move;
					writer.WriteLine(to_usi_string(m16) + " none " + to_string(move.vd.value) + " " + to_string(move.vd.depth));
				}

				progress.check(i);
			}
			if (!fast)
				sfen_reader.Close();

			cout << "write " + writebook_path << endl;
		}

		// 結果出力(統計値など)
		void output_result()
		{
			cout << "[ PetaShock Result ]" << endl;

			// 書き出したrecord数
			cout << "write book nodes   : " << book_nodes.size() << endl;

			// 後退解析1.において判明した、const node化したnodeの数。(leaf nodeから見てループしていなかったnodeの数)
			cout << "const_nodes        : " << const_nodes << endl;
			cout << "non-const_nodes    : " << book_nodes.size() - const_nodes << endl;

			// 王手されていた局面の数
			cout << "in-check nodes     : " << in_check_counter << endl;

			// check loop上の王手されていた局面の数
			cout << "check-loops nodes1 : " << check_loop_counter << endl;

			// check loop上の局面の数
			cout << "check-loops nodes2 : " << check_loop_nodes.size() << endl;

			// 合流チェックによって合流させた指し手の数。
			cout << "converged_moves    : " << converged_moves << endl;

			cout << endl << "Making a peta-shock book has been completed." << endl;
		}

	private:

		// -- 定跡データ

		// 定跡本体
		vector<BookNode> book_nodes;

		// 局面のHASH_KEYからBookMoveIndexへのmapper
		// ただし、flipして後手番にしたhashkeyを登録してある。
		// ⇨　後手の局面はflipして先手の局面として格納している。ゆえに、格納されているのはすべて先手の局面であり、
		// 　そこから1手進めると後手の局面となる。この時に、hash keyから既存の局面かどうかを調べたいので…。
		using HashKey2Index = unordered_map<Key, BookNodeIndex>;
		HashKey2Index hashkey_to_index;

		// fast == trueのときは、テンポラリファイルではなくここに元の定跡ファイル上のSFEN文字列を溜めておく。
		vector<string> original_sfens;

		// check loop上の局面のBookNodeIndex
		vector<BookNodeIndex> check_loop_nodes;

		// -- path

		// 読み込む定跡ファイルのpath(ペタショック化前の定跡ファイル)
		string readbook_path;

		// 書き出す定跡ファイルのpath(ペタショック化後の定跡ファイル)
		string writebook_path;

		// sfenの一時ファイルを書き出すpath
		string sfen_temp_path;

		// -- 統計値

		// 合流した指し手の数
		u64 converged_moves;

		// 後退解析1.において判明した、const nodeの数。(leaf nodeから見てループしていなかったnodeの数)
		u64 const_nodes;

		// 王手していた局面の数
		u64 in_check_counter;

		// check loop上の王手されていた局面の数
		u64 check_loop_counter;


		// -- option設定

		// その局面の最善手しか書き出さない。
		bool shrink;

		// テンポラリファイルを書き出さない。
		bool fast;

	};

} // namespace MakeBook2025

namespace Book
{
	// 2025年以降に作ったmakebook拡張コマンド
	// "makebook XXX"コマンド。XXXの部分に"build_tree"や"extend_tree"が来る。
	// この拡張コマンドを処理したら、この関数は非0を返す。
	int makebook2025(std::istringstream& is, const std::string& token, const OptionsMap& options)
    {
		if (token == "peta_shock") {

			// ペタショックコマンド
			// 
			// やねうら王の定跡ファイルに対して定跡ツリー上でmin-max探索を行い、その結果を別の定跡ファイルに書き出す。
			//   makebook peta_shock book.db user_book1.db
			// 　⇨　先手か後手か、片側の局面しか書き出さない。エンジンオプションの FlippedBook を必ずオンにして用いること。
			//   オプション指定
			//		shrink : 最善手しか書き出さない
			//      fast   : テンポラリファイルを書き出さない。(メモリ上に格納するのでその分だけメモリを消費する。)
			//     ⇨  fastの時は、思考エンジンオプションのThreadsで指定したスレッド数で並列化して合流チェックなどを行う。
			//		事前に "Threads 32"などとしてスレッド数を指定しておいてください。
			MakeBook2025::PetaShock ps;
			auto book_dir = options["BookDir"];
			ps.make_book(is, book_dir);
			return 1;
		}

		return 0;
	}

} // namespace Book
} // namespace YaneuraOu

#endif // defined (ENABLE_MAKEBOOK_CMD)
