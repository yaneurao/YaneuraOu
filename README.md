# About this project

shogi engine(AI player), stronger than Bonanza6 , educational and tiny code(about 2500 lines) , USI compliant engine , capable of being compiled by VC++2015 Update2

将棋の思考エンジンで、Bonanza6より強く、教育的で短いコード(2500行程度)で書かれたUSIプロトコル準拠の思考エンジンで、VC++2015 Update2でコンパイル可能です。

[やねうら王mini 公式サイト (解説記事等)](http://yaneuraou.yaneu.com/YaneuraOu_Mini/)

[やねうら王公式 ](http://yaneuraou.yaneu.com/)

# Sub-projects

## やねうら王nano

やねうら王nanoは1500行程度で書かれた将棋AIの基本となるプログラムです。探索部は150行程度で、非常にシンプルなコードで、αβ以外の枝刈りを一切していません。(R2000程度)

## やねうら王nano plus

やねうら王nano plusは、探索部300行程度で、オーダリングなどを改善した非常にシンプルでかつそこそこ強い思考エンジンです。(R2500程度)
	
## やねうら王mini

やねうら王miniは、やねうら王nano plusを並列化して、将棋ソフトとしての体裁を整えたものです。Bonanza6より強く、教育的かつ短いコードで書かれています。全体で3000行程度、探索部500行程度。(R2700程度)

## やねうら王classic 

やねうら王classicは、やねうら王miniのソースコードを改良する形で、Apery(WCSC 2015)ぐらいの強さを目指しました。入玉宣言機能も追加しました。(R3000程度)

## やねうら王classic-tce

やねうら王classic-tceは、やねうら王classicのソースコードに持ち時間制御(秒読み、フィッシャールールに対応)、ponderの機能を追加したものです。(R3300程度の予定)

## やねうら王2016 Mid

やねうら王 思考エンジン 2016年5月版(公開予定)
Hyperopt等を用いて各種ハイパーパラメーターの調整の自動化を行ない自動調整します。(R3400程度の予定)

## やねうら王2016 Late

やねうら王 思考エンジン 2016年11月版(非公開予定)
利きを利用した評価関数にして、NDFの学習メソッドを用いることで、技巧(2015)と同等以上の強さを目指します。(R3600程度の予定)

## 連続自動対局フレームワーク

連続自動対局を自動化できます。 

## やねうら王協力詰めsolver
	
『寿限無3』(49909手)も解ける協力詰めsolver →　[解説ページ](http://yaneuraou.yaneu.com/2016/01/02/%E5%8D%94%E5%8A%9B%E8%A9%B0%E3%82%81solver%E3%82%92%E5%85%AC%E9%96%8B%E3%81%97%E3%81%BE%E3%81%99/)

## やねうら王詰め将棋solver (気が向いたら製作します)

長手数の詰将棋が解けるsolverです。

## やねうら王評価関数バイナリ

CSAのライブラリの[ダウンロードページ](http://www.computer-shogi.org/library/)からダウンロードできます。


#　俺の作業メモ(2016/04/14 5:00現在)

- [ ] ・やねうら王classic-TCE(TimeControlEnabled)版を正式に公開する。
- [ ] ・clangでコンパイル通るようにしてUbuntu×AWSで動くようにする。
- [ ] ・やねうら王2016用の定跡を自動生成する。
- [ ] ・local-game-serverに、持ち時間の管理機能入れる。
- [ ] ・定跡の選択機能入れる。


※　括弧のなかの+Rは、自己対局時の勝率から計算されるもので、0.5手延長などは顕著に勝率が上がりますが、自己対局以外では効果に乏しいです。

- [x] 2016/04/14・自己対戦フレームワークで定跡データベース中の対局の手数が指定した手数より少ない場合クラッシュするっぽいバグ修正。(thx. tanuki-)
- [x] 2016/04/14・やねうら王2016 MidとLateに分離。
- [x] 2016/04/13・singularのときkiller,historyを更新しない。(+R10)(thx.読み太の作者さん)
- [x] 2016/04/13・ehashの仕組み追加したが速くならなかった。
- [x] 2016/04/12・historyとか、成りの指し手のときは、成り駒のhistoryを見ないといけないのに
　そういう処理になってなかった。(thx. 読み太の作者さん)
- [x] 2016/04/11・classic-tce、aspiration depth間違っていたの修正＆調整。(+R8)
- [x] 2016/04/08・王手がかかっていないときはstaticEvalをVALUE_NONEに。(+R4)
- [x] 2016/04/07・history等に駒打ちの情報を追加。(+R20)
- [x] 2016/04/07・dynamic margin実験。
- [x] 2016/04/07・go infiniteでstopが送られて来ていないのにbestmoveを返すことがあったのを修正。
- [x] 2016/04/07・優等局面のVALUE変更。
- [x] 2016/04/07・counter moveには、先手用の指し手に後手の指し手が混じらないように変更。
- [x] 2016/04/07・自己対戦サーバーでStateStack、1局ごとにクリアするように修正。
- [x] 2016/04/07・MaxMovesToDrawの設定が無視されていた問題を修正。
- [x] 2016/04/06・やねうら王classic-tceにsingular extension追加。(+R40)
- [x] 2016/04/06・やねうら王classic-tce。counter moveに駒種を入れる。
- [x] 2016/04/06・やねうら王classic-tceに。bestmove changed のカウンター、インクリメントしてなかったの修正。(+R20)
- [x] 2016/04/06・やねうら王classic-tceに。alpha更新のタイミングでeasy moveをクリアしていなかったの修正。
- [x] 2016/04/06・やねうら王classic-tceに。qsearch()のfutilityでcaptureを考慮するように変更。
- [x] 2016/04/06・VALUE_WINまわりの処理、色々修正。
- [x] 2016/04/06・byoyomiが指定されていないときの動作修正。
- [x] 2016/04/04・やねうら王classic-tceに、Options["NetworkDelay2"]を追加。
- [x] 2016/04/04・やねうら王twig→やねうら王classic-tceに名前変更。
- [x] 2016/04/04・やねうら王twigに持ち時間制御入れる。(+R50)
- [x] 2016/04/04・やねうら王twigにEasyMoveの判定追加。
- [x] 2016/04/03・やねうら王twig、終局までの手数に対応。
- [x] 2016/04/03・やねうら王twig、フィッシャールールに対応
- [x] 2016/04/03・やねうら王twigにponder対応。(+R50)
- [x] 2016/04/03・思考時間の計算用のクラス追加。
- [x] 2016/04/03・benchコマンド時に2個目以降の局面で正しくnpsが表示されなかった問題を修正。
- [x] 2016/04/03・やねうら王twigからタイマースレッド削除。
- [x] 2016/04/03・やねうら王classicからやねうら王twigにfork。
- [x] 2016/04/03・mate1ply修正。(thx. tさん) (+R3)
- [x] 2016/04/02・やねうら王classicのnonPV時のreduction定数、調整。(+R10)
- [x] 2016/04/02・kppファイル読み込み時の変換するためのメモリを動的に確保するように変更。
- [x] 2016/04/02・やねうら王classic、local-game-server、入玉宣言勝ちに対応。
- [x] 2016/04/02・MovePickerのevasion、capture時のordering調整。(+R13)
- [x] 2016/04/02・やねうら王classicのPV時のreduction定数、調整。(+R20)
- [x] 2016/04/01・MovePickerにfollowup move追加。
- [x] 2016/04/01・やねうら王classicのhistory bonus考えなおす。
- [x] 2016/04/01・やねうら王classicにsingular extension調整してみる。
- [x] 2016/04/01・やねうら王classicのRecaptureのオーダリング調整。
- [x] 2016/03/31・see()高速化。(thx. tさん) (+R5)
- [x] 2016/03/31・mate1ply修正。(thx. woodyringさん) (+R15)
- [x] 2016/03/30・やねうら王2015から評価関数ファイルの変換に失敗していたので読み込み時に修正するコード追加した。(+R40)
- [x] 2016/03/30・classicでevaluate()を毎node呼び出しておかないとevaluate()の差分計算がなされないので修正。(thx. tさん、わるおさん) (+R40)
- [x] 2016/03/30・mini,classicでNULL MOVEの前のcheck_info_update()が抜けていたので修正。
- [x] 2016/03/30・classicでcheckinfoのupdateが抜けている経路があったので修正。
- [x] 2016/03/30・depth 4手浅いものより大きいなら置換表のエントリー書き換えるように変更。
- [x] 2016/03/30・置換表格納のときにkeyのcastが間違っていたの修正。(棋力に影響なし)
- [x] 2016/03/30・Positionクラス、eval_list()関数は、戻り値はEval::EvalList型ではなくEval::EvalList*型を返すように修正。(thx. わるおさん)(+R15)
- [x] 2016/03/25・CounterMoveHistoryStateの更新と取得に失敗していた件、修正。(thx. tさん) (+R55)
- [x] 2016/03/25・置換表にDEPTH_MAX書き出せなくなってた件、修正。(thx. tさん)
- [x] 2016/03/25・mate1ply()でpinnedを使っているので先行してcheck_info_update()が必要なので修正。(thx. tさん)
- [x] 2016/03/25・pseudo-legalのチェックで、counter moveの手は手番に関係ない(ような実装になっている)ので、
	違法手のチェックが必要だったのでその修正。(thx. woodyringさん)
- [x] 2016/03/25・mate1ply()でpinnedを使っているので先行してcheck_info_update()が必要。(thx. tさん)
	・classic、mini、nano plus修正。
- [x] 2016/03/25・classicで1手詰めを呼び出すときに王手がかかっているかのチェックするの忘れていたの修正。(thx. woodyringさん)
- [x] 2016/03/25・古いほうの置換表実装削除。(置換表が弱くなる原因ではなかったようなので)
- [x] 2016/03/25・nanoで常にfull depth searchになっていたバグを修正。(thx. kazuさん)
- [x] 2016/03/25・seeのバグ修正。(thx. tさん、woodyringさん) (+R15)
- [x] 2016/03/04・打ち歩詰めの判定修正。(thanks to tanuki-さん)
- [x] 2016/03/01・やねうら王classic、悪いhistoryにreduction量を増やす枝刈り追加。
- [x] 2016/03/01・やねうら王classic、引き分け時のスコア、value_from_tt()を通すようにした。
- [x] 2016/03/01・やねうら王mini、引き分け時のスコア、value_from_tt()を通すようにした。
- [x] 2016/02/29・fail lowを引き起こした直前のcounter moveに加点するようにした。
- [x] 2016/02/29・1手詰めを見つけたときのスコアがおかしかったの修正。
- [x] 2016/02/29・やねうら王miniに定跡のnarrow book機能入れた。
- [x] 2016/02/29・やねうら王classicに定跡のnarrow book機能入れた。
- [x] 2016/02/29・思考エンジンごとにUSIのOptionを追加できるようにした。
- [x] 2016/02/29・やねうら王miniの静止探索で駒取りにならないevasionの枝刈り追加。(+R100)
- [x] 2016/02/29・やねうら王classicの静止探索で駒取りにならないevasionの枝刈り追加。(+R100)
- [x] 2016/02/29・やねうら王miniの静止探索にfutilityによる枝刈り追加。(+R70)
- [x] 2016/02/29・やねうら王classicの静止探索にfutilityによる枝刈り追加。(+R70)
- [x] 2016/02/29・やねうら王classicに親nodeでSEE負の指し手を枝刈り追加。(+R40)
- [x] 2016/02/29・やねうら王classicに親nodeでのfutility枝刈り追加。(+R40)
- [x] 2016/02/29・やねうら王miniにhistoryに基づく枝刈り追加。(+R20)
- [x] 2016/02/29・やねうら王miniで通常探索時の1手詰め判定削除。
- [x] 2016/02/29・やねうら王classicにhistoryに基づく枝刈り追加。(+R150)
- [x] 2016/02/29・やねうら王miniのソースコード、整理。効果の薄い枝刈り削除。
- [x] 2016/02/29・やねうら王miniにmoveCountベースのfutility追加。(+R150)
- [x] 2016/02/29・やねうら王classicにmoveCountベースのfutility追加。(+R150)
- [x] 2016/02/28・やねうら王classicに王手延長追加。(+R50)
- [x] 2016/02/28・やねうら王classicに多重反復深化追加。(+R12)
- [x] 2016/02/28・やねうら王classicにProbCut追加。(+R70)
- [x] 2016/02/28・やねうら王miniにProbCut追加。(+R70)
- [x] 2016/02/28・MovePickerにProbCut用の指し手生成を追加。
- [x] 2016/02/28・やねうら王classicの開発開始。
- [x] 2016/02/28・やねうら王miniの思考エンジンの実行ファイルを公開。
- [x] 2016/02/28・やねうら王miniの開発終了。
- [x] 2016/02/28・local game serverでCreateProcessに失敗したときに復帰できるように修正。
- [x] 2016/02/28・やねうら王miniで定跡の指し手が指せていなかったの修正。
- [x] 2016/02/28・やねうら王miniにrazoring追加。(+R20)
- [x] 2016/02/28・やねうら王miniにnull move search追加。(+R60)
- [x] 2016/02/28・Position::do_null_move()/undo_null_move()追加。
- [x] 2016/02/28・MovePickerで置換表の指し手とcounter moveにおいて歩や大駒の不成などを除外するようにした。
- [x] 2016/02/28・Position::moved_piece()で後手の駒打ちのときに後手の駒が返るように変更。
- [x] 2016/02/28・QUITE_CHECKで歩の不成が生成されていた問題を修正。
- [x] 2016/02/27・Position::pseudo_legal()修正。
- [x] 2016/02/26・やねうら王miniでmain threadでfail high/lowが起きたときにGUIに読み筋を出力するようにした。
- [x] 2016/02/26・やねうら王miniのaspirationのdelta調整。(+R20)
- [x] 2016/02/26・やねうら王miniにlazy SMP実装。(4コア時+R220程度)
- [x] 2016/02/26・やねうら王miniにPV line実装。(-R10)
- [x] 2016/02/26・やねうら王miniにss->moveCount追加。これによるhistoryへのbonus追加。(+R30)
- [x] 2016/02/26・やねうら王miniにaspiration windows search実装。(+R10)
- [x] 2016/02/25・やねうら王miniの思考オプションにMultiPV追加。
- [x] 2016/02/25・やねうら王miniの思考オプションにContempt(引き分け時スコアの設定)追加。
- [x] 2016/02/25・nano plusをベースにしてやねうら王miniの探索部書いていく。
- [x] 2016/02/25・nano plus、開発終了。(R2500相当)
- [x] 2016/02/25・nano plus、bad captureをkillerの直後に。(+R25)
- [x] 2016/02/25・nano plusのMovePickerにEVASIONSのオーダリング追加。
- [x] 2016/02/25・nano plusのMovePickerにCAPTURESのオーダリング追加。(+R30)
- [x] 2016/02/25・nano plusにhistory,counter move,counter move historyを追加。(+R120)
- [x] 2016/02/24・nano plusのMovePicker、別ファイルに分離。
- [x] 2016/02/24・劣等局面の判定追加。(+R5)
- [x] 2016/02/24・nano plusの枝刈りにfutility pruning追加。
- [x] 2016/02/24・nano plusのMovePickerで静止探索時に置換表の指し手がpseudo-legalでないときに落ちていたの修正。
- [x] 2016/02/23・nano plusのMovePickerで置換表の指し手がpseudo-legalでないときに落ちていたの修正。
- [x] 2016/02/23・local game serverの子プロセスの起動タイミングをばらつかせる。(乱数seedをばらけさせるため)
- [x] 2016/02/22・nano plus、"go infinite"に対応させる。
- [x] 2016/02/22・PRNGのデフォルトの乱数seedの精度を上げる。
- [x] 2016/02/22・Position::is_draw()→is_repetition()に名前変更。2手遡るの忘れていたの修正。
- [x] 2016/02/22・test autoplayコマンド追加。
- [x] 2016/02/22・nano plusにサイレントモード追加。
- [x] 2016/02/22・打ち歩詰めの指し手生成、生成条件が間違っていたの修正。
- [x] 2016/02/22・やねうら王nano plus、search()に一手詰め判定追加。(+R10)
- [x] 2016/02/22・やねうら王nano plusに千日手判定追加。(+R20)
- [x] 2016/02/22・Position::is_draw()実装
- [x] 2016/02/21・打ち歩詰め関係、ソース整理。
- [x] 2016/02/21・指し手生成で打ち歩詰め除外したときに、pseudo_legal()に打ち歩詰め判定入れるの忘れていたので修正。
- [x] 2016/02/21・Position::capture()の処理間違っていたの修正。
- [x] 2016/02/21・やねうら王nano plus、full depth searchの処理修正と調整。(+R100)
- [x] 2016/02/21・やねうら王nano plus、improvingフラグ用意。(+R5)
- [x] 2016/02/21・やねうら王nano plus、通常探索でevaluate()を呼び出すように変更。(+R50)
- [x] 2016/02/21・local game server、エンジン側から非合法手が送られてきたときにMOVE_RESIGN扱いにするように。
- [x] 2016/02/21・打ち歩詰めの判定、高速化。(+R10)
- [x] 2016/02/21・pos.legal()、玉の影の利きがあるときの処理間違っていたので修正。
- [x] 2016/02/21・local game serverでserver側の負荷が高かったのを修正。
- [x] 2016/02/21・local game serverで思考エンジンがbestmoveを返さないとハングしていたでタイムアウト処理追加。
- [x] 2016/02/21・やねうら王nano plusのMovePickerのRECAPTUREの処理、修正。
- [x] 2016/02/20・やねうら王nano plusの静止探索の指し手生成を色々調整。(+R180)。
- [x] 2016/02/20・やねうら王nano plusの静止探索で置換表絡みの処理追加。
- [x] 2016/02/19・やねうら王nano plusにLMRの導入。(+R140程度)
- [x] 2016/02/11・起動時にファイルからコマンドを受け付ける機能追加。
- [x] 2016/02/11・起動時にargvとして複数行を実行する機能追加。
- [x] 2016/02/10・自己対局サーバー、1スレッド×同時対局(マルチスレッド)に対応させる。
- [x] 2016/02/09・Threadクラス大改修。→ thread絡みの問題が起きなくなった。(ようだ)
- [x] 2016/02/09・_mm_mallocが失敗する件、さらに調査。→ std::thread絡みの問題くさい。
- [x] 2016/02/09・_mm_malloc()自作した。→これでも失敗する。newに失敗しているのか。ランタイム腐ってるのか…。
- [x] 2016/02/09・ローカルサーバー機能、ときどき対戦が開始しないので原因を調査する。→_mm_mallocが高負荷状態で失敗するようだ…。
- [x] 2016/02/09・やねうら王nano plusにkiller moveの導入。
- [x] 2016/02/09・nano plusで1手詰めなのに探索局面が多すぎるのでmate distance pruning追加した。
- [x] 2016/02/09・ベンチマークコマンド書けた。
- [x] 2016/02/08・やねうら王nano plusでMovePickerでの指し手生成を逐次生成に変更。
- [x] 2016/02/08・やねうら王nano plusで1手詰め判定を呼び出すように。
- [x] 2016/02/08・gcc/Clangでコンパイル通るようにする→Clangは無理。gccは行けそうだが、時間かかりそうなので保留。
- [x] 2016/02/08・やねうら王nano plusの開発開始。
- [x] 2016/02/08・やねうら王nano、floodgateでR2000を超えたので、nanoの開発終了。
- [x] 2016/02/08・実行ファイルをプロジェクトの一部として配布する。
- [x] 2016/02/07・やねうら王nanoをnonPV/PVの処理をきちんと書く。(これでR2000ぐらいになっていればnanoの開発は終了)
- [x] 2016/02/07・KPPの評価関数の差分実装完了。
- [x] 2016/02/06・やねうら王nanoで定跡の採択確率を出力するようにした。
- [x] 2016/02/06・やねうら王nanoの定跡を"book.db"から読み込むように変更。
- [x] 2016/02/06・定跡の読み込み等をこのフレームワークの中に入れる
- [x] 2016/02/06・定跡生成コマンド"makebook"
- [x] 2016/02/05・やねうら王nanoの探索部
- [x] 2016/02/05・やねうら王nano秒読みに対応。
- [x] 2016/02/04・やねうら王nanoに定跡追加。
- [x] 2016/02/01・新しい形式に変換した評価関数バイナリを用意する →　用意した → CSAのサイトでライブラリとして公開された。
- [x] 2016/01/31・やねうら王nanoの探索部