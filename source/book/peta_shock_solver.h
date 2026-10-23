#ifndef PETA_SHOCK_SOLVER_H_INCLUDED
#define PETA_SHOCK_SOLVER_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace YaneuraOu::Book::PetaRetrograde {

using Index = uint32_t;
constexpr Index None = std::numeric_limits<Index>::max();
constexpr int Mate = 32000;

// 通常の千日手は0、連続王手の千日手は王手側の負けとして、定跡グラフの値と保証手を求める。
// Graphは正規化した1局面を先後2状態に分けて扱うが、指し手配列自体は複製しない。
// child(v,j)==Noneは評価値付きの定跡外への手。parents(w,f)は逆辺ごとにf(親,指し手番号)を呼ぶ。
// 入力グラフはsolve()から後続の検証が終わるまで変更しないこと。
template<class Graph> class Solver {
    const Graph& g;
    Index n;
    std::vector<Index> cores[2]; // cores[p]: pが自分の手番で王手を続けられる循環の全頂点。
    std::vector<int> local_index; // 全体の頂点番号から局所番号への対応。対象外は-1。

    // 許可された内部辺だけで強連結成分を求める。再帰を使わないKosaraju法でO(V+E)。
    // cyclic_componentsを渡す場合は、各成分が循環を持つかも同じ走査で判定する。
    template<class View, class Allow> static std::vector<Index> components_of(
        const View& g, Index n, Allow allow,
        std::vector<uint8_t>* cyclic_components = nullptr) {
        std::vector<uint8_t> seen(n, 0);
        std::vector<Index> order, result(n, None), stack;
        std::vector<std::pair<Index, Index>> dfs;
        order.reserve(n);
        // 順方向DFSの帰りがけ順を記録する。定跡外への手はグラフの辺として辿らない。
        for (Index root = 0; root < n; ++root) {
            if (seen[root]) continue;
            seen[root] = 1;
            dfs.emplace_back(root, 0);
            while (!dfs.empty()) {
                auto v = dfs.back().first;
                auto j = dfs.back().second;
                if (j == g.degree(v)) {
                    order.push_back(v);
                    dfs.pop_back();
                } else {
                    ++dfs.back().second;
                    auto w = g.child(v, j);
                    if (w != None && allow(v, j) && !seen[w]) {
                        seen[w] = 1;
                        dfs.emplace_back(w, 0);
                    }
                }
            }
        }
        Index id = 0;
        // 帰りがけ順の逆順に逆辺を辿ると、一度の探索で一つの強連結成分が得られる。
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            if (result[*it] != None) continue;
            result[*it] = id;
            stack.push_back(*it);
            Index count = 0;
            bool self_loop = false;
            while (!stack.empty()) {
                auto w = stack.back();
                stack.pop_back();
                ++count;
                g.parents(w, [&](Index v, Index j) {
                    if (result[v] == None && allow(v, j)) {
                        result[v] = id;
                        stack.push_back(v);
                    }
                    if (cyclic_components && v == w && allow(v, j)) self_loop = true;
                });
            }
            // 2頂点以上なら必ず循環がある。1頂点の場合だけ自己ループが必要。
            // 成分計算後に全辺をもう一度走査する必要はない。
            if (cyclic_components) (*cyclic_components)[id] = count > 1 || self_loop;
            ++id;
        }
        return result;
    }

    template<class Allow> std::vector<Index> components(Allow allow) const {
        return components_of(g, n, allow);
    }

    // 循環成分に属する頂点だけを、頂点番号順で返す。経路や個々の閉路は列挙しない。
    template<class View, class Allow> static std::vector<Index> cyclic_of(
        const View& g, Index n, Allow allow) {
        std::vector<uint8_t> cycle(n, 0);
        auto c = components_of(g, n, allow, &cycle);
        std::vector<Index> result;
        for (Index v = 0; v < n; ++v)
            if (cycle[c[v]]) result.push_back(v);
        return result;
    }

    template<class Allow> std::vector<Index> cyclic(Allow allow) const {
        return cyclic_of(g, n, allow);
    }

    // 王手循環部分だけの小さな作業用グラフ。辺の許可条件は構築時に適用済み。
    struct CoreGraph {
        std::vector<std::vector<Index>> next, previous;
        explicit CoreGraph(size_t size) : next(size), previous(size) {}
        Index degree(Index v) const { return Index(next[v].size()); }
        Index child(Index v, Index j) const { return next[v][j]; }
        template<class Visit> void parents(Index w, Visit visit) const {
            // cyclic_of()には常にtrueの条件を渡すため、元の指し手番号はここでは不要。
            for (auto v : previous[w]) visit(v, Index(0));
        }
    };

    template<class Allow> std::vector<Index> checking_cycles(int player, Allow allow) {
        // 方策・評価値条件で辺を削っても、元の王手循環の外に新しい循環は生まれない。
        // よって検証時はcores[player]内だけを調べれば、全体を調べた場合と同じ循環頂点が得られる。
        const auto& core = cores[player];
        if (core.empty()) return {};
        CoreGraph subgraph(core.size());
        for (Index i = 0; i < core.size(); ++i) local_index[core[i]] = int(i);
        for (Index i = 0; i < core.size(); ++i) {
            auto v = core[i];
            for (Index j = 0; j < g.degree(v); ++j) {
                auto w = g.child(v, j);
                if (w == None || local_index[w] < 0 || !allow(v, j)) continue;
                auto k = Index(local_index[w]);
                subgraph.next[i].push_back(k);
                subgraph.previous[k].push_back(i);
            }
        }
        // 全頂点の対応表を初期化し直さず、今回使った要素だけを戻す。
        for (auto v : core) local_index[v] = -1;
        auto result = cyclic_of(subgraph, Index(core.size()), [](Index, Index) { return true; });
        for (auto& v : result) v = core[v];
        return result;
    }

    // leafの評価値は手番側視点なので、保証を計算するplayerの視点へ揃える。
    int payoff(Index v, Index j, int player) const {
        return g.owner(v) == player ? g.score(v, j) : -g.score(v, j);
    }

    struct Positive {
        std::vector<int> value; // playerが保証できる正の値。0は正の保証が未確定。
        std::vector<Index> choice; // 自分の手番で保証を達成する手。相手の手番では不要。
    };

    // 相手の王手循環内で、閾値level以上を保証できる最大の集合を求める。
    // 閾値以上が確定した出口へ進むか、相手が王手を続けて反則負けするなら保証を満たす。
    // 反復処理はこの小領域に限定し、その外への伝播はpositive()のqueueで行う。
    std::vector<std::pair<Index, Index>> safety(int player, int level,
                                              const std::vector<int>& won) {
        const auto& core = cores[1-player];
        std::vector<uint8_t> alive(core.size(), 0), bad(core.size(), 0);
        std::vector<Index> count(core.size(), 0);
        std::vector<std::vector<Index>> incoming(core.size());
        std::deque<Index> queue;
        for (Index i = 0; i < core.size(); ++i) {
            local_index[core[i]] = int(i);
            alive[i] = !won[core[i]];
        }
        auto good = [&](Index v, Index j) {
            auto w = g.child(v, j);
            return w == None ? payoff(v, j, player) >= level : won[w] != 0;
        };
        auto internal = [&](Index v, Index j) {
            auto w = g.child(v, j);
            return w != None && local_index[w] >= 0 && alive[local_index[w]]
                && (g.owner(v) == player || g.checked(w));
        };
        // 自分の手番は安全な手が一つあればよい(OR)。相手の手番は全応手が安全でなければならない(AND)。
        // incomingは候補を除外した影響を局所的に親へ返すための逆辺。
        for (Index i = 0; i < core.size(); ++i) {
            if (!alive[i]) continue;
            auto v = core[i];
            for (Index j = 0; j < g.degree(v); ++j) {
                ++local_edges;
                if (good(v,j)) ++count[i];
                else if (internal(v,j)) {
                    ++count[i];
                    incoming[local_index[g.child(v,j)]].push_back(i);
                } else if (g.owner(v) != player) bad[i] = 1;
            }
            if ((g.owner(v) == player && !count[i]) || bad[i]) queue.push_back(i);
        }
        // 条件を満たせない頂点を除外し、残りの手がなくなる親などへ除外を伝播する。
        while (!queue.empty()) {
            auto i = queue.front(); queue.pop_front();
            if (!alive[i]) continue;
            alive[i] = 0;
            for (auto parent : incoming[i]) {
                if (!alive[parent]) continue;
                --count[parent];
                if (g.owner(core[parent]) != player || !count[parent]) queue.push_back(parent);
            }
        }
        std::vector<std::pair<Index, Index>> result;
        for (Index i = 0; i < core.size(); ++i) {
            if (!alive[i]) continue;
            auto v = core[i];
            Index choice = None;
            if (g.owner(v) == player) {
                // 確定済みの出口を優先する。なければ残った安全集合内の手を保証手にする。
                for (Index j = 0; j < g.degree(v); ++j)
                    if (good(v,j)) { choice = j; break; }
                if (choice == None)
                    for (Index j = 0; j < g.degree(v); ++j)
                        if (internal(v,j)) { choice = j; break; }
            }
            result.emplace_back(v, choice);
        }
        for (auto v : core) local_index[v] = -1;
        return result;
    }

    // 高い閾値から順に、playerがその値以上を保証できる頂点を確定する。
    // 通常部分では各頂点を一度だけqueueへ入れ、対応する逆辺も一度だけ処理する。
    Positive positive(int player) {
        Positive out{std::vector<int>(n, 0), std::vector<Index>(n, None)};
        std::vector<Index> left(n); // 相手の頂点で、閾値以上とまだ確認できていない応手数。
        std::vector<std::vector<std::pair<Index, Index>>> leaves(Mate + 1);
        for (Index v = 0; v < n; ++v) {
            left[v] = g.degree(v);
            for (Index j = 0; j < g.degree(v); ++j)
                if (g.child(v,j) == None && payoff(v,j,player) > 0)
                    leaves[payoff(v,j,player)].emplace_back(v,j);
        }
        std::deque<Index> queue;
        auto admit = [&](Index v, Index j, int level) {
            if (out.value[v]) return;
            out.value[v] = level;
            out.choice[v] = j;
            queue.push_back(v);
        };
        auto hit = [&](Index v, Index j, int level) {
            if (out.value[v]) return;
            if (g.owner(v) == player) admit(v,j,level);
            else if (--left[v] == 0) admit(v,None,level);
        };
        // leafの値が変わる閾値でのみ再計算する。Mateでは連続王手による勝ちも拾う。
        for (int level = Mate; level > 0; --level) {
            if (level != Mate && leaves[level].empty()) continue;
            for (auto edge : leaves[level]) hit(edge.first,edge.second,level);
            for (;;) {
                while (!queue.empty()) {
                    auto w = queue.front(); queue.pop_front();
                    g.parents(w, [&](Index v, Index j) {
                        ++propagation_edges;
                        hit(v,j,level);
                    });
                }
                ++local_calls;
                auto added = safety(player,level,out.value);
                if (added.empty()) break;
                // 王手循環側で新たに確定した頂点も通常のqueueへ戻し、必ず親へ伝播してから終了判定する。
                // 通常伝播が一度0件だっただけで終了すると、issue #334のように未伝播の0が残る。
                for (auto entry : added) admit(entry.first,entry.second,level);
            }
        }
        return out;
    }

    // 正の保証手が未設定の領域でも、値を保ちながら自分の王手を中断できる手を構成する。
    // すでに-Mateの局面は反則負けより良い値を約束していないので、この回避条件の対象外。
    bool avoid_checks(int player) {
        std::vector<Index> active;
        for (auto v : cores[player]) {
            int own = g.owner(v) == player ? values[v] : -values[v];
            if (own > -Mate && own <= 0) {
                local_index[v] = int(active.size());
                active.push_back(v);
            }
        }
        std::vector<Index> left(active.size(), 0), choice(active.size(), None);
        std::vector<uint8_t> reached(active.size(), 0);
        std::vector<std::vector<std::pair<Index,Index>>> pred(active.size());
        std::deque<Index> queue;
        auto hit = [&](Index i, Index j) {
            if (reached[i]) return;
            if (g.owner(active[i]) == player || --left[i] == 0) {
                reached[i] = 1;
                choice[i] = j;
                queue.push_back(i);
            }
        };
        for (Index i = 0; i < active.size(); ++i) {
            auto v = active[i];
            for (Index j = 0; j < g.degree(v); ++j)
                if (g.owner(v) != player || edge_value(v,j) == values[v]) ++left[i];
        }
        // 自分は同評価値の手だけを選べるが、相手には評価値の低い手も含め全応手を残す。
        // leaf・対象領域の外・自分の非王手を出口とし、そこへ到達できる頂点を逆向きに確定する。
        for (Index i = 0; i < active.size(); ++i) {
            auto v = active[i];
            for (Index j = 0; j < g.degree(v); ++j) {
                if (g.owner(v) == player && edge_value(v,j) != values[v]) continue;
                auto w = g.child(v,j);
                if (w == None || local_index[w] < 0 || (g.owner(v) == player && !g.checked(w))) hit(i,j);
                else pred[local_index[w]].emplace_back(i,j);
            }
        }
        while (!queue.empty()) {
            auto i = queue.front(); queue.pop_front();
            for (auto edge : pred[i]) hit(edge.first,edge.second);
        }
        bool ok = true;
        for (Index i = 0; i < active.size(); ++i) {
            auto v = active[i];
            if (!reached[i]) ok = false;
            else if (g.owner(v) == player) policy[v] = choice[i];
            local_index[v] = -1;
        }
        return ok;
    }

public:
    std::vector<int> values; // 各頂点の手番側視点の評価値。
    std::vector<Index> policy; // 保証を満たす選択手の番号。単に同評価値なら何を選んでもよいわけではない。
    std::string error;
    uint64_t propagation_edges = 0, local_edges = 0, local_calls = 0;
    explicit Solver(const Graph& graph) : g(graph), n(graph.size()), local_index(n, -1) {}

    // 内部手は子局面の値を符号反転し、leaf手は入力値をそのまま使う。
    int edge_value(Index v, Index j) const {
        auto w = g.child(v,j);
        return w == None ? g.score(v,j) : -values[w];
    }

    // 許可した手を任意に選んでも値の保証を壊さないか、相手の全応手を残して検査する。
    // restrictなしなら違反で即失敗。ありなら制限が必要な頂点に1を立て、候補を絞る側へ返す。
    // valuesはpublicなので、数値の整合性も毎回確認し、過去の検証結果を使い回さない。
    template<class Permits> bool certify_with(Permits permits, std::vector<uint8_t>* restrict = nullptr) {
        bool verified = true;
        for (Index v = 0; v < n; ++v) {
            if (policy[v] >= g.degree(v) || edge_value(v,policy[v]) != values[v] || !permits(v,policy[v]))
                return fail("Selected move does not preserve the value");
            for (Index j = 0; j < g.degree(v); ++j)
                if (edge_value(v,j) > values[v] || (permits(v,j) && edge_value(v,j) != values[v]))
                    return fail("Local minimax mismatch");
        }
        for (int player = 0; player < 2; ++player) {
            auto own = [&](Index v) { return g.owner(v) == player ? values[v] : -values[v]; };
            auto follows = [&](Index v, Index j) { return g.owner(v) != player || permits(v,j); };
            // -Mateより良い値を約束しているのに、自分の連続王手で負ける循環を残してはいけない。
            auto bad = checking_cycles(player, [&](Index v, Index j) {
                auto w = g.child(v,j);
                return follows(v,j) && own(v) > -Mate && own(w) > -Mate
                    && (g.owner(v) != player || g.checked(w));
            });
            if (!bad.empty()) {
                if (!restrict) return fail("Policy permits own perpetual-check loss");
                verified = false;
                for (auto v : bad) if (g.owner(v) == player) (*restrict)[v] = 1;
            }
            // 正の値を約束する領域で、相手が王手を中断しながら循環を続けられると0へ逃げられる。
            // 通常千日手も対象なので、この検査は王手循環部分に限定せず全グラフで行う。
            auto c = components([&](Index v, Index j) {
                return follows(v,j) && own(v) > 0 && own(g.child(v,j)) > 0;
            });
            std::vector<uint8_t> bad_components(restrict ? n : 0, 0);
            for (Index v = 0; v < n; ++v) {
                if (g.owner(v) == player || own(v) <= 0) continue;
                for (Index j = 0; j < g.degree(v); ++j) {
                    auto w = g.child(v,j);
                    if (w != None && own(w) > 0 && !g.checked(w) && c[v] == c[w]) {
                        if (!restrict) return fail("Policy promises a positive value on a nonwinning cycle");
                        verified = false;
                        bad_components[c[v]] = 1;
                    }
                }
            }
            if (restrict)
                for (Index v = 0; v < n; ++v)
                    if (g.owner(v) == player && bad_components[c[v]]) (*restrict)[v] = 1;
        }
        return verified;
    }

    bool certify() {
        return certify_with([&](Index v, Index j) { return policy[v] == j; });
    }

    // 値を保つ手だけで距離を計算し、非負なら短い出口、負なら長い出口を優先する。O(V+E)。
    // 非負の頂点は最初に確定した子で最短距離が得られ、負の頂点は全ての子の確定を待てば最長になる。
    // このためFIFOだけで計算できる。循環で距離が未確定の部分の大域的なdepth最適性は保証しない。
    bool prefer_signed_depth(bool canonical_pairs = false) {
        std::vector<Index> distance(n, None), pending(n, 0);
        std::deque<Index> queue;
        for (Index v = 0; v < n; ++v) {
            bool leaf = false;
            for (Index j = 0; j < g.degree(v); ++j) {
                if (edge_value(v,j) != values[v]) continue;
                if (g.child(v,j) == None) leaf = true;
                else ++pending[v];
            }
            if (leaf && (values[v] >= 0 || pending[v] == 0)) {
                distance[v] = 0;
                queue.push_back(v);
            }
        }
        while (!queue.empty()) {
            auto w = queue.front(); queue.pop_front();
            g.parents(w, [&](Index v, Index j) {
                if (distance[v] != None || edge_value(v,j) != values[v]) return;
                if (values[v] < 0 && --pending[v] != 0) return;
                distance[v] = distance[w] + 1;
                queue.push_back(v);
            });
        }
        const auto guaranteed = policy; // depthの選好よりも優先すべき、検証済みの元の保証手。
        for (Index v = 0; v < n; ++v) {
            // 距離を確定できない王手循環などでは、検証済みの元の手を維持する。
            if (distance[v] == None) continue;
            auto matches = [&](Index j) {
                auto w = g.child(v,j);
                auto d = w == None ? Index(0) : distance[w] == None ? None : distance[w] + 1;
                return edge_value(v,j) == values[v] && d == distance[v];
            };
            if (matches(policy[v])) continue;
            for (Index j = 0; j < g.degree(v); ++j)
                if (matches(j)) { policy[v] = j; break; }
        }
        if (canonical_pairs)
            for (Index v = 0; v < n; v += 2) policy[v+1] = policy[v];
        if (certify()) return true;

        // 新旧の手を混ぜた場合の循環も検査してから、危険な領域を元の保証手へ戻す。
        // この和集合から辺を削るだけなら、新たな循環を作らずに候補を制限できる。
        std::vector<uint8_t> restricted(n, 0);
        certify_with([&](Index v, Index j) {
            return j == policy[v] || j == guaranteed[v];
        }, &restricted);
        if (canonical_pairs)
            for (Index v = 0; v < n; v += 2)
                restricted[v] = restricted[v+1] = restricted[v] || restricted[v+1];
        for (Index v = 0; v < n; ++v)
            if (restricted[v]) policy[v] = guaranteed[v];
        return certify();
    }

    // 値の計算、保証手の構成、保証検査を行う。depthの選好と保存時の-1補正は別段階で扱う。
    bool solve() {
        for (Index v = 0; v < n; ++v) {
            if (!g.degree(v)) return fail("Position without moves");
            for (Index j = 0; j < g.degree(v); ++j)
                if (g.child(v,j) == None && (g.score(v,j) < -Mate || g.score(v,j) > Mate))
                    return fail("Leaf value outside [-32000,32000]");
        }
        // 双方が毎回王手する循環は、この解法の対応範囲外なので明示的に拒否する。
        if (!cyclic([&](Index v, Index j) { return g.checked(g.child(v,j)); }).empty())
            return fail("Mutual continuous-check cycle is not supported; no book written");
        for (int p = 0; p < 2; ++p)
            cores[p] = cyclic([&](Index v, Index j) { return g.owner(v) != p || g.checked(g.child(v,j)); });
        auto a = positive(0), b = positive(1);
        values.resize(n);
        policy.assign(n, None);
        // 両者の正の保証を手番側の値へまとめる。どちらも正を保証しない場合は引き分け値0。
        for (Index v = 0; v < n; ++v) {
            if (a.value[v] && b.value[v]) return fail("Conflicting positive guarantees");
            values[v] = g.owner(v) == 0 ? a.value[v] - b.value[v] : b.value[v] - a.value[v];
            if (values[v] > 0) policy[v] = g.owner(v) == 0 ? a.choice[v] : b.choice[v];
        }
        if (!avoid_checks(0) || !avoid_checks(1)) return fail("Cannot interrupt own checks at the computed value");
        for (Index v = 0; v < n; ++v)
            if (policy[v] == None)
                for (Index j = 0; j < g.degree(v); ++j)
                    if (edge_value(v,j) == values[v]) { policy[v] = j; break; }
        return certify();
    }

    size_t core_size(int p) const { return cores[p].size(); }
    bool fail(const char* message) { error = message; return false; }
};

} // namespace YaneuraOu::Book::PetaRetrograde
#endif
