#include "../../shogi.h"
#ifdef HELP_MATE_ENGINE

#include <atomic>
#include "../../position.h"

// --- 協力詰め探索

namespace HelpMate
{
  struct TranspositionTable;

  struct TTEntry {

    // この深さにおいて詰まない
    uint32_t depth() const { return ((uint32_t)depth_high8 << 16) + depth16; }

    // この局面で指し手が1つしかないときに指し手生成処理を端折るための指し手
    Move move() const { return (Move)move16; }

    int64_t key() const { return key64; }

    // TTEntryの世代(置換表側で世代カウンターで管理)
    uint16_t generation() const { return gen16; }
    void set_generation(uint16_t g) { gen16 = g; }

    // 置換表のエントリーに対して与えられたデータを保存する。上書き動作
    void save(const Key128& key_, uint32_t depth, Move move)
    { depth16 = depth & 0xffff; depth_high8 = depth >> 16; key64 = key_.p(1); move16 = (uint16_t)move; }

    // 与えられたkey_がこのTTEntryに格納されているかを判定する。
    bool found(const Key128& key_) const { return key64 == key_.p(1); }

  private:
    friend struct TranspositionTable;

    // この残り探索深さにおいて詰まない(24bit)
    uint16_t depth16;
    uint8_t depth_high8;
    std::atomic<uint8_t> lock; // entry lock用

    uint16_t move16;     // 1手しかないときの指し手
    uint16_t gen16;      // 置換表の世代
    uint64_t key64;
    // 3 + 1 + 2 + 2 + 8 = 16

  };

  struct TranspositionTable {

    // 置換表のなかから与えられたkeyに対応するentryを探す。
    // 置換表により深いdepthのentryがあればそのdepthを引数のdepthに反映させてtrueを返す。
    // 置換表により深いdepthのentryはなかったけどこのnodeのentryがあったならその指し手を
    //   引数のtt_moveに反映させてtrueを返す。
    // 置換表にこのnodeのentryが見つからない場合はfalseを返す。
    bool probe(const Key128& key, uint32_t& depth , Move& tt_move)
    {
      auto& cluster = table[(size_t)key.p(0) % clusterCount];
      TTEntry* const tte = &cluster.entry[0];

      lock(cluster);
      for (int i = 0; i < ClusterSize; ++i)
      {
        // 空のエントリーを見つけた(そこまではkeyが合致していないので見つからなかったものとして終了)
        if (!tte[i].key())
          break;
        
        // keyが合致しているentryを見つけた。
        if (tte[i].found(key))
        {
          // 置換表に登録されているdepthがそれ以上深いなら枝刈りできる。
          if (tte[i].depth() >= depth)
          {
            depth = tte[i].depth();
            tte[i].set_generation(generation16); // Refresh
            unlock(cluster);
            return true;
          }
          tt_move = tte[i].move();
          // いまからこのnodeを更新するのでいますぐのrefreshは不要。
          unlock(cluster);
          return false;
        }
      }
      tt_move = MOVE_NONE;
      unlock(cluster);
      return false;
    }

    void save(const Key128& key,uint32_t depth,Move move)
    {
      auto& cluster = table[(size_t)key.p(0) % clusterCount];
      TTEntry* const tte = &cluster.entry[0];
      lock(cluster);

      TTEntry* replace;
      for (int i = 0; i < ClusterSize; ++i)
      {
        // 空のentryがあったのでここに上書き
        if (!tte[i].key())
        {
          replace = &tte[i];
          goto WriteBack;
        }

        // 同じhash keyのentryがあったのでここ以外に書き込むわけにはいかない
        if (tte[i].found(key))
        {
          if (tte[i].depth() >= depth)
          {
            // 現在のdepthのほうが深いなら書き込まずに終了。たぶん他のスレッドが書き込んだのだろう。
            unlock(cluster);
            return;
          }
          replace = &tte[i];
          goto WriteBack;
        }
      }

      replace = tte;

      // 一致するhash keyが格納されているTTEntryが見つからなかったし、空のエントリーも見つからなかったのでどれか一つ犠牲にする。
      for (int i = 1; i < ClusterSize; ++i)

        // ・(残り探索深さが)深いときの探索の結果であるものほど価値があるので残しておきたい。depth × 重み1.0
        // ・generationがいまの探索generationに近いものほど価値があるので残しておきたい。generation×重み 8.0
        // 以上に基いてスコアリングする。
        // 以上の合計が一番小さいTTEntryを使う。

        // 並列化の影響により、自分より未来のgenerationでありうるので1024を足しておく。
        // (スレッドごとにgenerationを変えたほうがいいかも知れないが現状そうはしていないので1024は過剰ではあるが。)

        if ( (int32_t)replace->depth() - (uint16_t)(1024 + generation16 - replace->generation()) * 8
             >(int32_t)tte[i].depth()  - (uint16_t)(1024 + generation16 - tte[i].generation()  ) * 8 )
          replace = &tte[i];

    WriteBack:;
      replace->set_generation(generation16);
      replace->save(key, depth, move);

      unlock(cluster);
    }

    // 置換表のサイズを変更する。mbSize == 確保するメモリサイズ。MB単位。
    void resize(size_t mbSize) {
      free(mem);

      // 2のべき乗にはしない。entryCountは偶数であることだけ保証する。
      // 先手と後手との局面はhash keyの下位1bitで判別しているので、
      // 先手用の局面と後手用の局面とでTTEntryは別のところになって欲しいから。
      clusterCount = (mbSize * 1024 * 1024 / sizeof(Cluster)) & ~UINT64_C(1);
      
      mem = calloc(clusterCount * sizeof(Cluster) + CacheLineSize - 1, 1);
      if (!mem)
      {
        std::cout << "failed to calloc\n";
		my_exiT();
      }
      table = (Cluster*)((uintptr_t(mem) + CacheLineSize - 1) & ~(CacheLineSize - 1));
    }

    // 置換表のエントリーの全クリア
    void clear() { memset(table, 0, clusterCount * sizeof(Cluster)); }

    // 世代カウンターをインクリメントする。
    void new_search() { ++generation16; }

    // 置換表使用率を調べる。世代が同じエントリーの数をサンプリングして調べる。
    int hashfull() const
    {
      // すべてのエントリーにアクセスすると時間が非常にかかるため、先頭から1000エントリーだけ
      // サンプリングして使用されているエントリー数を返す。
      int cnt = 0;
      for (int i = 0; i < 1000 / ClusterSize; ++i)
      {
        const auto tte = &table[i].entry[0];
        for (int j = 0; j < ClusterSize; ++j)
          if ((tte[j].generation() == generation16))
            ++cnt;
      }
      return cnt;
    }

    TranspositionTable() { mem = nullptr; generation16 = 0; resize(16); }
    ~TranspositionTable() { free(mem); }

    // CPUのcache line size(この単位でClusterを配置しないといけない)
    const int CacheLineSize = 64;

    // 1クラスターにおけるTTEntryの数
    static const int ClusterSize = 4;

    struct Cluster {
      TTEntry entry[ClusterSize];
    };

    void lock(Cluster& cluster)
    {
      // このClusterの1つ目のTTEntry::lockをこのClusterのlock用に使う。
      auto& lk = cluster.entry[0].lock;
      while (true)
      {
        uint8_t expected = 0;
        if (lk.compare_exchange_weak(expected, 1))
          break;
        // 0なら1にして、これができたらlockできたとみなす。
      }
    }
    void unlock(Cluster& cluster)
    {
      auto& lk = cluster.entry[0].lock;
      uint8_t expected = 1;
      // 自分がlockしたのだから1になっているはず..
      if (!lk.compare_exchange_weak(expected, 0))
        ASSERT_LV1(false);
    }

    // 確保されているClusterの先頭
    Cluster* table;
    // callocで確保したもの。64byteでalignしたのが↑のtable
    void* mem;

    size_t clusterCount;
    int16_t generation16;
  };

  // 協力詰めを解く。反復深化のループ。
  // thread_id : 0...thread_num-1
  // thread_num : スレッド数
  void id_loop(Position& root,int thread_id,int thread_num);

  // 協力詰め関係の初期化
  void init();

  // 全スレッド終了後にmain threadから呼び出される。
  void finalize();

  // 協力詰め用のglobalな置換表。
  extern TranspositionTable TT;

} // end of namespace

#endif
