#include "../../shogi.h"
#ifdef MATE_ENGINE

#include <unordered_set>

#include "../../extra/all.h"
#include "mate-search.h" 

using namespace std;
using namespace Search;

// --- 詰み将棋探索

// df-pn with Threshold Controlling Algorithm (TCA)の実装。
// 岸本章宏氏の "Dealing with infinite loops, underestimation, and overestimation of depth-first
// proof-number search." に含まれる擬似コードを元に実装しています。
//
// TODO(someone): 優越関係の実装
// TODO(someone): 証明駒の実装
// TODO(someone): Source Node Detection Algorithm (SNDA)の実装
// 
// リンク＆参考文献
//
// Ayumu Nagai , Hiroshi Imai , "df-pnアルゴリズムの詰将棋を解くプログラムへの応用",
// 情報処理学会論文誌,43(6),1769-1777 (2002-06-15) , 1882-7764
// http://id.nii.ac.jp/1001/00011597/
//
// Nagai, A.: Df-pn algorithm for searching AND/OR trees and its applications, PhD thesis,
// Department of Information Science, The University of Tokyo (2002)
//
// Ueda T., Hashimoto T., Hashimoto J., Iida H. (2008) Weak Proof-Number Search. In: van den Herik
// H.J., Xu X., Ma Z., Winands M.H.M. (eds) Computers and Games. CG 2008. Lecture Notes in Computer
// Science, vol 5131. Springer, Berlin, Heidelberg
//
// Toru Ueda, Tsuyoshi Hashimoto, Junichi Hashimoto, Hiroyuki Iida, Weak Proof - Number Search,
// Proceedings of the 6th international conference on Computers and Games, p.157 - 168, September 29
// - October 01, 2008, Beijing, China
//
// Kishimoto, A.: Dealing with infinite loops, underestimation, and overestimation of depth-first
// proof-number search. In: Proceedings of the AAAI-10, pp. 108-113 (2010)
//
// A. Kishimoto, M. Winands, M. Müller and J. Saito. Game-Tree Search Using Proof Numbers: The First
// Twenty Years. ICGA Journal 35(3), 131-156, 2012. 
//
// A. Kishimoto and M. Mueller, Tutorial 4: Proof-Number Search Algorithms
// 
// df-pnアルゴリズム学習記(1) - A Succulent Windfall
// http://caprice-j.hatenablog.com/entry/2014/02/14/010932
//
// IS将棋の詰将棋解答プログラムについて
// http://www.is.titech.ac.jp/~kishi/pdf_file/csa.pdf
//
// df-pn探索おさらい - 思うだけで学ばない日記
// http://d.hatena.ne.jp/GMA0BN/20090520/1242825044
//
// df-pn探索のコード - 思うだけで学ばない日記
// http://d.hatena.ne.jp/GMA0BN/20090521/1242911867
//

namespace MateEngine
{
  // 詰将棋エンジン用のMovePicker
  struct MovePicker
  {
    MovePicker(Position& pos, bool or_node) {
      // たぬき詰めであれば段階的に指し手を生成する必要はない。
      // 自分の手番なら王手の指し手(CHECKS)、
      // 相手の手番ならば回避手(EVASIONS)を生成。
      endMoves = or_node ?
        generateMoves<CHECKS_ALL>(pos, moves) :
        generateMoves<EVASIONS_ALL>(pos, moves);
      endMoves = std::remove_if(moves, endMoves, [&pos](const auto& move) {
        return !pos.legal(move);
      });
    }

    bool empty() {
      return moves == endMoves;
    }

    ExtMove* begin() { return moves; }
    ExtMove* end() { return endMoves; }
    const ExtMove* begin() const { return moves; }
    const ExtMove* end() const { return endMoves; }

  private:
    ExtMove moves[MAX_MOVES], *endMoves = moves;
  };

  // 置換表
  // 通常の探索エンジンとは置換表に保存したい値が異なるため
  // 詰め将棋専用の置換表を用いている
  // ただしSmallTreeGCは実装せず、Stockfishの置換表の実装を真似ている
  struct TranspositionTable {
    static const constexpr uint32_t kInfiniteDepth = 1000000;
    static const constexpr int CacheLineSize = 64;
    struct TTEntry {
      // ハッシュの上位32ビット
      uint32_t hash_high; // 0
                          // TTEntryのインスタンスを作成したタイミングで先端ノードを表すよう1で初期化する
      int pn; // 1
      int dn; // 1
      uint32_t generation : 8; // 0
                               // ルートノードからの最短距離
                               // 初期値を∞として全てのノードより最短距離が長いとみなす
      int minimum_distance : 24; // UINT_MAX
                                 // TODO(nodchip): 指し手が1手しかない場合の手を追加する
      int num_searched; // 0
    };
    static_assert(sizeof(TTEntry) == 20, "");

    struct Cluster {
      TTEntry entries[3];
      int padding;
    };
    static_assert(sizeof(Cluster) == 64, "");
    static_assert(CacheLineSize % sizeof(Cluster) == 0, "");

    virtual ~TranspositionTable() {
      if (tt_raw) {
        std::free(tt_raw);
        tt_raw = nullptr;
        tt = nullptr;
      }
    }

    TTEntry& LookUp(Key key) {
      auto& entries = tt[key & clusters_mask];
      uint32_t hash_high = key >> 32;
      // 検索条件に合致するエントリを返す
      for (auto& entry : entries.entries) {
        if (entry.hash_high == 0) {
          // 空のエントリが見つかった場合
          entry.hash_high = hash_high;
          entry.pn = 1;
          entry.dn = 1;
          entry.generation = generation;
          entry.minimum_distance = kInfiniteDepth;
          entry.num_searched = 0;
          return entry;
        }

        if (hash_high == entry.hash_high) {
          // keyが合致するエントリを見つけた場合
          entry.generation = generation;
          return entry;
        }
      }

      // 合致するエントリが見つからなかったので
      // 世代が一番古いエントリをつぶす
      TTEntry* best_entry = nullptr;
      uint32_t best_generation = UINT_MAX;
      for (auto& entry : entries.entries) {
        uint32_t temp_generation;
        if (generation < entry.generation) {
          temp_generation = 256 - entry.generation + generation;
        }
        else {
          temp_generation = generation - entry.generation;
        }

        if (best_generation > temp_generation) {
          best_entry = &entry;
          best_generation = temp_generation;
        }
      }
      best_entry->hash_high = hash_high;
      best_entry->pn = 1;
      best_entry->dn = 1;
      best_entry->generation = generation;
      best_entry->minimum_distance = kInfiniteDepth;
      best_entry->num_searched = 0;
      return *best_entry;
    }

    TTEntry& LookUp(Position& n) {
      return LookUp(n.key());
    }

    // moveを指した後の子ノードの置換表エントリを返す
    TTEntry& LookUpChildEntry(Position& n, Move move) {
      return LookUp(n.key_after(move));
    }

    void Resize() {
      int64_t hash_size_mb = (int)Options["Hash"];
      if (hash_size_mb == 16) {
        hash_size_mb = 4096;
      }
      int64_t new_num_clusters = 1LL << MSB64((hash_size_mb * 1024 * 1024) / sizeof(Cluster));
      if (new_num_clusters == num_clusters) {
        return;
      }

      num_clusters = new_num_clusters;

      if (tt_raw) {
        std::free(tt_raw);
        tt_raw = nullptr;
        tt = nullptr;
      }

      tt_raw = std::calloc(new_num_clusters * sizeof(Cluster) + CacheLineSize, 1);
      tt = (Cluster*)((uintptr_t(tt_raw) + CacheLineSize - 1) & ~(CacheLineSize - 1));
      clusters_mask = num_clusters - 1;
    }

    void NewSearch() {
      generation = (generation + 1) & 0xff;
    }

    int tt_mask = 0;
    void* tt_raw = nullptr;
    Cluster* tt = nullptr;
    int64_t num_clusters = 0;
    int64_t clusters_mask = 0;
    uint32_t generation = 0; // 256で一周する
  };

  static const constexpr int kInfinitePnDn = 100000000;
  static const constexpr int kMaxDepth = MAX_PLY;

  TranspositionTable transposition_table;

  // TODO(tanuki-): ネガマックス法的な書き方に変更する
  void DFPNwithTCA(Position& n, int thpn, int thdn, bool inc_flag, bool or_node, int depth) {
    if (Signals.stop.load(std::memory_order_relaxed)) {
      return;
    }

    auto nodes_searched = n.this_thread()->nodes.load(memory_order_relaxed);
    if (nodes_searched && nodes_searched % 10000000 == 0) {
      sync_cout << "info string nodes_searched=" << nodes_searched << sync_endl;
    }

    auto& entry = transposition_table.LookUp(n);

    if (depth > kMaxDepth) {
      entry.pn = kInfinitePnDn;
      entry.dn = 0;
      entry.minimum_distance = std::min(entry.minimum_distance, depth);
      return;
    }

    // if (n is a terminal node) { handle n and return; }

    // 1手読みルーチンによるチェック
    if (or_node && !n.in_check() && n.mate1ply()) {
      entry.pn = 0;
      entry.dn = kInfinitePnDn;
      entry.minimum_distance = std::min(entry.minimum_distance, depth);
      return;
    }

    MovePicker move_picker(n, or_node);
    if (move_picker.empty()) {
      // nが先端ノード

      if (or_node) {
        // 自分の手番でここに到達した場合は王手の手が無かった、
        entry.pn = kInfinitePnDn;
        entry.dn = 0;
      }
      else {
        // 相手の手番でここに到達した場合は王手回避の手が無かった、
        entry.pn = 0;
        entry.dn = kInfinitePnDn;
      }

      entry.minimum_distance = std::min(entry.minimum_distance, depth);
      return;
    }

    // minimum distanceを保存する
    // TODO(nodchip): このタイミングでminimum distanceを保存するのが正しいか確かめる
    entry.minimum_distance = std::min(entry.minimum_distance, depth);

    bool first_time = true;
    while (!Signals.stop.load(std::memory_order_relaxed)) {
      ++entry.num_searched;

      // determine whether thpn and thdn are increased.
      // if (n is a leaf) inc flag = false;
      if (entry.pn == 1 && entry.dn == 1) {
        inc_flag = false;
      }

      // if (n has an unproven old child) inc flag = true;
      for (const auto& move : move_picker) {
        // unproven old childの定義はminimum distanceがこのノードよりも小さいノードだと理解しているのだけど、
        // 合っているか自信ない
        const auto& child_entry = transposition_table.LookUpChildEntry(n, move);
        if (entry.minimum_distance > child_entry.minimum_distance &&
          child_entry.pn != kInfinitePnDn &&
          child_entry.dn != kInfinitePnDn) {
          inc_flag = true;
          break;
        }
      }

      // expand and compute pn(n) and dn(n);
      if (or_node) {
        entry.pn = kInfinitePnDn;
        entry.dn = 0;
        for (const auto& move : move_picker) {
          const auto& child_entry = transposition_table.LookUpChildEntry(n, move);
          entry.pn = std::min(entry.pn, child_entry.pn);
          entry.dn += child_entry.dn;
        }
        entry.dn = std::min(entry.dn, kInfinitePnDn);
      }
      else {
        entry.pn = 0;
        entry.dn = kInfinitePnDn;
        for (const auto& move : move_picker) {
          const auto& child_entry = transposition_table.LookUpChildEntry(n, move);
          entry.pn += child_entry.pn;
          entry.dn = std::min(entry.dn, child_entry.dn);
        }
        entry.pn = std::min(entry.pn, kInfinitePnDn);
      }

      // if (first time && inc flag) {
      //   // increase thresholds
      //   thpn = max(thpn, pn(n) + 1);
      //   thdn = max(thdn, dn(n) + 1);
      // }
      if (first_time && inc_flag) {
        thpn = std::max(thpn, entry.pn + 1);
        thpn = std::min(thpn, kInfinitePnDn);
        thdn = std::max(thdn, entry.dn + 1);
        thdn = std::min(thdn, kInfinitePnDn);
      }

      // if (pn(n) ≥ thpn || dn(n) ≥ thdn)
      //   break; // termination condition is satisfied
      if (entry.pn >= thpn || entry.dn >= thdn) {
        break;
      }

      // first time = false;
      first_time = false;

      // find the best child n1 and second best child n2;
      // if (n is an OR node) { /* set new thresholds */
      //   thpn child = min(thpn, pn(n2) + 1);
      //   thdn child = thdn - dn(n) + dn(n1);
      // else {
      //   thpn child = thpn - pn(n) + pn(n1);
      //   thdn child = min(thdn, dn(n2) + 1);
      // }
      Move best_move;
      int thpn_child;
      int thdn_child;
      if (or_node) {
        // ORノードでは最も証明数が小さい = 玉の逃げ方の個数が少ない = 詰ましやすいノードを選ぶ
        int best_pn = kInfinitePnDn;
        int second_best_pn = kInfinitePnDn;
        int best_dn = 0;
        int best_num_search = INT_MAX;
        for (const auto& move : move_picker) {
          const auto& child_entry = transposition_table.LookUpChildEntry(n, move);
          if (child_entry.pn < best_pn ||
            child_entry.pn == best_pn && best_num_search > child_entry.num_searched) {
            second_best_pn = best_pn;
            best_pn = child_entry.pn;
            best_dn = child_entry.dn;
            best_move = move;
            best_num_search = child_entry.num_searched;
          }
          else if (child_entry.pn < second_best_pn) {
            second_best_pn = child_entry.pn;
          }
        }

        thpn_child = std::min(thpn, second_best_pn + 1);
        thdn_child = std::min(thdn - entry.dn + best_dn, kInfinitePnDn);
      }
      else {
        // ANDノードでは最も反証数の小さい = 王手の掛け方の少ない = 不詰みを示しやすいノードを選ぶ
        int best_dn = kInfinitePnDn;
        int second_best_dn = kInfinitePnDn;
        int best_pn = 0;
        int best_num_search = INT_MAX;
        for (const auto& move : move_picker) {
          const auto& child_entry = transposition_table.LookUpChildEntry(n, move);
          if (child_entry.dn < best_dn ||
            child_entry.dn == best_dn && best_num_search > child_entry.num_searched) {
            second_best_dn = best_dn;
            best_dn = child_entry.dn;
            best_pn = child_entry.pn;
            best_move = move;
          }
          else if (child_entry.dn < second_best_dn) {
            second_best_dn = child_entry.dn;
          }
        }

        thpn_child = std::min(thpn - entry.pn + best_pn, kInfinitePnDn);
        thdn_child = std::min(thdn, second_best_dn + 1);
      }

      StateInfo state_info;
      n.do_move(best_move, state_info);
      DFPNwithTCA(n, thpn_child, thdn_child, inc_flag, !or_node, depth + 1);
      n.undo_move(best_move);
    }
  }

  // 詰み手順を1つ返す
  // 最短の詰み手順である保証はない
  bool dfs(bool or_node, Position& pos, std::vector<Move>& moves, std::unordered_set<Key>& visited) {
    // 一度探索したノードを探索しない
    if (visited.find(pos.key()) != visited.end()) {
      return false;
    }
    visited.insert(pos.key());

    MovePicker move_picker(pos, or_node);
    Move mate1ply = pos.mate1ply();
    if (mate1ply || move_picker.empty()) {
      if (mate1ply) {
        moves.push_back(mate1ply);
      }
      //std::ostringstream oss;
      //oss << "info string";
      //for (const auto& move : moves) {
      //  oss << " " << move;
      //}
      //sync_cout << oss.str() << sync_endl;
      //if (mate1ply) {
      //  moves.pop_back();
      //}
      return true;
    }

    const auto& entry = transposition_table.LookUp(pos);

    for (const auto& move : move_picker) {
      const auto& child_entry = transposition_table.LookUpChildEntry(pos, move);
      if (child_entry.pn != 0) {
        continue;
      }

      StateInfo state_info;
      pos.do_move(move, state_info);
      moves.push_back(move);
      if (dfs(!or_node, pos, moves, visited)) {
        pos.undo_move(move);
        return true;
      }
      moves.pop_back();
      pos.undo_move(move);
    }

    return false;
  }

  // 詰将棋探索のエントリポイント
  void dfpn(Position& r) {
    if (r.in_check()) {
      sync_cout << "info string The king is checked... df-pn is skipped..." << sync_endl;
      sync_cout << "bestmove None" << sync_endl;
      return;
    }

    transposition_table.Resize();
    // キャッシュの世代を進める
    transposition_table.NewSearch();

    auto start = std::chrono::system_clock::now();

    DFPNwithTCA(r, kInfinitePnDn, kInfinitePnDn, false, true, 0);
    const auto& entry = transposition_table.LookUp(r);

    auto nodes_searched = r.this_thread()->nodes.load(memory_order_relaxed);
    sync_cout << "info string" <<
      " pn " << entry.pn <<
      " dn " << entry.dn <<
      " nodes_searched " << nodes_searched << sync_endl;

    std::vector<Move> moves;
    std::unordered_set<Key> visited;
    dfs(true, r, moves, visited);

    auto end = std::chrono::system_clock::now();
    if (!moves.empty()) {
      auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      time_ms = std::max(time_ms, 1LL);
      int64_t nps = nodes_searched * 1000LL / time_ms;
      std::ostringstream oss;
      oss << "info depth " << moves.size() << " time " << time_ms << " nodes " << nodes_searched << " pv";
      for (const auto& move : moves) {
        oss << " " << move;
      }
      oss << " score mate + nps " << nps;
      sync_cout << oss.str() << sync_endl;
    }

    // "stop"が送られてきたらSignals.stop == trueになる。
    // "ponderhit"が送られてきたらLimits.ponder == 0になるので、それを待つ。(stopOnPonderhitは用いない)
    //    また、このときSignals.stop == trueにはならない。(この点、Stockfishとは異なる。)
    // "go infinite"に対してはstopが送られてくるまで待つ。
    while (!Signals.stop && (Limits.ponder || Limits.infinite))
      sleep(1);
    //	こちらの思考は終わっているわけだから、ある程度細かく待っても問題ない。
    // (思考のためには計算資源を使っていないので。)

    if (moves.empty()) {
      sync_cout << "bestmove None" << sync_endl;
    }
    else if (moves.size() == 1) {
      sync_cout << "bestmove " << moves[0] << sync_endl;
    }
    else {
      sync_cout << "bestmove " << moves[0] << " ponder " << moves[1] << sync_endl;
    }

    Signals.stop = true;
  }
}

void USI::extra_option(USI::OptionsMap & o) {}

// --- Search

void Search::init() {}
void Search::clear() { }
void MainThread::think() {
  Thread::search();
}
void Thread::search() {
  MateEngine::dfpn(rootPos);
}

#endif
