# やねうら王のソースコードのルール

ここに書いてあることは、やねうらお(やねうら王開発者)が、やねうら王のソースコードを書く時にこのルールで書いているというルールの説明であり、プルリクするときにこれらのルールを守る必要はありません。(守ってもらえていると嬉しいですが…)

ソースコードを読んで動作について理解しようと思う時に、ここに書いてあるルールを知っていると役に立つと思います。

## 絵文字について

やねうら王では、以下のルールでこれらの絵文字を使っていますが、厳密なルールではないです。

|絵文字|読み方|意味|備考|
|-|-|-|-|
|🌈|にじ| やねうら王独自|独自の改良、改造など|
|🤔|かんがえ| やねうらおの考え |深い洞察に基づく|
|📓|のーと | 解説記事 | 長文 |
|📝|えんぴつ| メモ書き | 備忘録 |
|📌|おしぴん| 強調したいメモ |目立たせたいメモ書き|
|💡|でんきゅう| 読む人のためのヒント| 読む人への配慮 |
|⚠|ちゅうい| 注意事項 | 致命的なバグになりうる |

### ブロックを示す絵文字

> // 🌈 やねうら王独自 🌈

のように両端に絵文字がある場合、そこ以降、次のブロックまで(通例5～20行)それが継続することを意味します。(ブロックの定義は曖昧ですが、ここでは何かのまとまりをブロックと呼んでいます。)

## Stockfishのソースコードを持ってくる場合

Stockfishからソースコードをコピーしてきたものを改変して使用する場合、

```
#if STOCKFISH
    元のStockfishのコード
#else
    改変したStockfishのコード
#endif
```

のように、`#if STOCKFISH`～`#else`～`#endif`で書いて、元のソースコードに対して、どう改変されているのかがわかるように示します。

Stockfishのコメントに書かれている英文については、日本語に適宜翻訳しています。

```
例)
    // Non-main threads go directly to iterative_deepening()
    // メインスレッド以外は直接 iterative_deepening() へ進む

    if (!is_mainthread())
    {
        iterative_deepening();
        return;
    }
```

## namespaceについて

ただし、`namespace Stockfish`に関しては、上のルールは無視して、`namespace YaneuraOu`と変更します。

また、StockfishからNNUEのコードを持ってくる(これは逆輸入)する場合、やねうら王では、この逆輸入されたNNUEのことをSFNN(Stockfish NNUE)と呼んでいるので、`namespace NNUE`は`namespace SFNN`と変更します。

## でスコープを閉じるとき

namespaceを閉じるとき以下のように名前空間名を`// namespace`のあとに書きます。
```
namespace YaneuraOu {
namespace {

} // namespace
} // namespace YaneuraOu
```

同様に、中身の行数が長いwhileを閉じる時もコメントでそのことを示すことがあります。

```
while (...) {

// すごく長い行数

} // while
```

また、`#if`～`#endif`に関しても長くなる時は、コメントでそのことを示すことがあります。

```
#if defined(USE_CLASSIC_EVAL)

// すごく長い行数

#endif // defined(USE_CLASSIC_EVAL)
```

## ソースコードのフォーマット

基本的には`source/.clang-format`を適用してください。

Stockfishでは、
```
    stop     = false
    starting = false
```
のように`=`などのindentを揃えることが多いです。

また、namespaceはindentなしとなっています。(Visual Studioが勝手にindentする場合、Visual StudioのC++の書式設定で調整してください。)

.clang-formatを適用すると半自動的に以上のようになりますが、ならないことがあるので、意識してみてください。
