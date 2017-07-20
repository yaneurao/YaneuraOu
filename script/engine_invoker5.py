# coding: UTF-8
import argparse
import datetime
import math
import os.path
import random
import subprocess
import sys
import time

# -----------------------------------------------------------------

# subprocessでnon blockingなreadにするhack。
# fcntlはLinuxでは動くのだがWindowsでは動かない。
# スレッドを数だけ作るのはオーバーヘッドがあるのでそれはしたくない。

# Non-blocking read on a subprocess.PIPE in python
# http://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python

def pipe_non_blocking_set(fd):
	# Constant could define globally but avoid polluting the name-space
	# thanks to: http://stackoverflow.com/questions/34504970
	import msvcrt

	from ctypes import windll, byref, wintypes, WinError, POINTER
	from ctypes.wintypes import HANDLE, DWORD, BOOL

	LPDWORD = POINTER(DWORD)

	PIPE_NOWAIT = wintypes.DWORD(0x00000001)

	def pipe_no_wait(pipefd):
		SetNamedPipeHandleState = windll.kernel32.SetNamedPipeHandleState
		SetNamedPipeHandleState.argtypes = [HANDLE, LPDWORD, LPDWORD, LPDWORD]
		SetNamedPipeHandleState.restype = BOOL

		h = msvcrt.get_osfhandle(pipefd)

		res = windll.kernel32.SetNamedPipeHandleState(h, byref(PIPE_NOWAIT), None, None)
		if res == 0:
		    print(WinError())
		    return False
		return True

	return pipe_no_wait(fd)

# -----------------------------------------------------------------

win = lose = draw = 0
win_black = lose_black = 0

# レーティングの出力
def output_rating(win,draw,lose,win_black,win_white,opt2):
	total = win + lose
	if total != 0 :
		# 普通の勝率
		win_rate = win / float(win+lose)
		# 先手番/後手番のときの勝率内訳
		win_rate_black = win_black / float(win + lose)
		win_rate_white = win_white / float(win + lose)
	else:
		win_rate = 0
		win_rate_black = 0
		win_rate_white = 0

	if win_rate == 0 or win_rate == 1:
		rating = ""
	else:
		rating = " R" + str(round(-400*math.log(1/win_rate-1,10),2))

	print opt2 + "," + str(win) + " - " + str(draw) + " - " + str(lose) + \
		"(" + str(round(win_rate*100,2)) + "%" + rating + ")" + \
		" win black : white = " + \
		str(round(win_rate_black*100,2)) + "% : " + \
		str(round(win_rate_white*100,2)) + "%"
	sys.stdout.flush()


# 思考エンジンに対するオプションを生成する。
def create_option(engines,engine_threads,evals,times,hashes,numa,PARAMETERS_LOG_FILE_PATH):

	# 思考エンジンに対するコマンド列を保存する。
	options = []
	# 対局時間設定を保存する。
	options2 = []

	# 時間は.で連結できる。
	times = times.split(".")
	if len(times)==1:
		times.append(times[0])

	for i in range(2):

		rtime = 0
		byoyomi = 0
		inc_time = 0
		total_time = 0
		depth_time = 0

		nodes_time = False

		for b in times[i].split("/"):

			c = b[0]
			# 大文字で指定されていたらnodes_timeモード。
			if c != c.lower():
				c = c.lower()
				nodes_time = True

			t = int(b[1:])
			if c == "r":
				rtime = t
			elif c == "b":
				byoyomi = t
			elif c == "i":
				inc_time = t
			elif c == "t":
				total_time = t
			elif c == "d":
				depth_time = t

		option = []
		if ("Yane" in engines[i]):
			if rtime:
				option.append("go rtime " + str(rtime))
			elif inc_time:
				option.append("go btime REST_TIME wtime REST_TIME inc " + str(inc_time))
			elif depth_time:
				option.append("go depth " + str(depth_time))
			else:
				option.append("go btime REST_TIME wtime REST_TIME byoyomi " + str(byoyomi))

			option.append("setoption name Threads value " + str(engine_threads))
			option.append("setoption name EvalDir value " + evals[i])
			option.append("setoption name Hash value " + str(hashes[i]))
			option.append("setoption name BookFile value no_book")
			option.append("setoption MinimumThinkingTime value 1000")
			option.append("setoption name NetworkDelay value 0")
			option.append("setoption name NetworkDelay2 value 0")

#			option.append("setoption name EvalShare value false")
			option.append("setoption name EvalShare value true")
#			if i==0:
#				option.append("setoption name EvalShare value false")
#			else:
#				option.append("setoption name EvalShare value true")
			if nodes_time:
				option.append("setoption name nodestime value 600")
			if PARAMETERS_LOG_FILE_PATH :
				option.append("setoption name PARAMETERS_LOG_FILE_PATH value " + PARAMETERS_LOG_FILE_PATH + "_%%THREAD_NUMBER%%.log")
				# →　仕方ないので%THREAD_NUMBER%のところはのちほど置換する。
		else:
			# ここで対応しているengine一覧
			#  ・技巧(20160606)
			#  ・Silent Majority(V1.1.0)
			if rtime:
				option.append("go rtime " + str(rtime))
				print "Error! " + engines[i] + " doesn't support rtime "
			elif inc_time:
				option.append("go btime REST_TIME wtime REST_TIME inc " + str(inc_time))
			elif depth_time:
				option.append("go depth " + str(depth_time))
			else:
				option.append("go btime REST_TIME wtime REST_TIME byoyomi " + str(byoyomi))

			option.append("setoption name Threads value " + str(engine_threads))
			option.append("setoption name USI_Hash value " + str(hashes[i]))
#			option.append("setoption name EvalDir value " + evals[i])

			if "SILENT_MAJORITY" in engines[i]:
				option.append("setoption name Byoyomi_Margin value 0")
				option.append("setoption name Minimum_Thinking_Time value 0")
				option.append("setoption name Eval_Dir value " + evals[i])

		options.append(option)

		options2.append([total_time,inc_time,byoyomi,rtime,depth_time])

	options.append(options2[0])
	options.append(options2[1])

	return options

# engine1とengine2とを対戦させる
#  threads    : この数だけ並列対局
#  cpu        : 実行するプロセッサグループの数
#               1つのプロセッサには threads/cpu だけスレッドを割り当てる
#  book_sfens : 定跡
#  opt2       : 勝敗の表示の先頭にT2,b2000 のように対局条件を文字列化して突っ込む用。
#  book_moves : 定跡の手数
def vs_match(engines_full,options,threads,loop,cpu,book_sfens,fileLogging,opt2,book_moves):

	global win,lose,draw
	win = lose = draw = 0

	# engine0側にとって、
	# 先手番での勝ちと後手番での勝ち(先手/後手のときの勝率計算用)
	# 引き分けは上のdrawと同じ値になるから不要。
	global win_black,win_white
	win_black = win_white = 0

	# home + "book/records1.sfen

	# 定跡ファイルは1行目から順番に読む。次に読むべき行番号
	# 定跡ファイルは重複除去された、互角の局面集であるものとする。
	sfen_no = 0;

	cmds = []
	for i in range(2*threads):
		# working directoryを実行ファイルのあるフォルダ直下としてやる。
		# 最後のdirectory separatorを探す
		engine_path = engines_full[i % 2]
		pos = max(engine_path.rfind('\\') , engine_path.rfind('/'))
		if pos <= 0:
			working_dir = ""
		else:
			working_dir = engine_path[:pos]
		# print "working_dir = " + working_dir

		cmd = "cmd.exe /c start /B /WAIT /D " + working_dir + " /NODE " + str(int(i * cpu / (threads*2))) + " " + engines_full[i % 2]
		cmds.append(cmd)

	# 棋譜
	sfens = [""]*threads
	# 棋譜に対する、探索のときの評価値
	eval_values = [""]*threads
	# 探索のときに最後にそのスレッドから送られてきた評価値
	eval_value_from_thread = [""]*threads*2

	# 対局開始局面からの手数
	moves = [0]*threads

	# 今回の対局での先手番のplayer(0 or 1)
	turns = [0]*threads

	# process handle
	procs = [0]*threads*2
	# process state
	states = ["init"]*threads*2
	# 初回のreadyokに対するwait
	initial_waits = [True]*threads*2

	# 残り時間
	rest_times = [0]*threads*2
	# goコマンドを送った時刻
	go_times = [0]*threads*2
	# nodes計測用
	nodes_str = [""]*threads*2
	nodes = [0]*threads*2
	# 終了したプロセスの監視用
	term_procs = [False]*threads*2

	for i in range(threads*2):
#		proc = subprocess.Popen(cmds[i] , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE , universal_newlines=True , bufsize=1)
		proc = subprocess.Popen(cmds[i] , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE )
		pipe_non_blocking_set(proc.stdout.fileno())

		procs[i] = proc


	# これをTrueにするとコンソールに思考エンジンとのやりとりを出力する。
#	Logging = True
	Logging = False

	# これをTrueにするとログファイルに思考エンジンとのやりとりを出力する。
#	FileLogging = True
	FileLogging = fileLogging

	# これをTrueにすると棋譜をファイルに書き出すようになる。
	KifOutput = True

	# 現在時刻。ログファイルと棋譜ファイルを同じ名前にしておく。
	now = datetime.datetime.today()
	if FileLogging:
		log_file = open("script_log"+now.strftime("%Y%m%d%H%M%S")+".txt","w")

	if KifOutput:
		kif_file = open(now.strftime("%Y%m%d%H%M%S") + opt2.replace(",","_") + ".sfen","w")

	def send_cmd(i,s):
		p = procs[i]
		if Logging:
			print "[" + str(i) + "]<" + s
		if FileLogging:
			log_file.write( "[" + str(i) + "]<" + s + "\n")
		p.stdin.write(s+"\n")

	def isready_cmd(i):
		p = procs[i]
		send_cmd(i,"isready")
		states[i] = "wait_for_readyok"
		# rest_time = total_time
		rest_times[i] = options[2][0]

	def go_cmd(i):
		p = procs[i]
		# USI "position"
		s = "position startpos"
		if sfens[i/2] != "":
			s += " moves " + sfens[i/2]
		send_cmd(i,s)

		# USI "go"
		cmd = options[i & 1][0]
		cmd = cmd.replace("REST_TIME",str(rest_times[i]))
		send_cmd(i,cmd)

		# changes state
		states[i]   = "wait_for_bestmove"
		states[i^1] = "wait_for_another_player"

		go_times[i] = time.time()

	def usinewgame_cmd(i,sfen_no):
		p = procs[i]
		send_cmd(i,"usinewgame")
		sfens[i/2] = book_sfens[sfen_no]
		moves[i/2] = 0
		# 定跡の評価値はよくわからんので0にしとくしかない。
		eval_values[i/2] = "0 "*book_moves

	# ゲームオーバーのハンドラ
	# i = threads number
	# g = result : 1..1P勝ち , 2..2P勝ち , 3..引き分け
	def gameover_cmd(i,g):
		p = procs[i]
		if g == 3:
			result = "draw"
		elif g == (i % 2)+1:
			result = "win"
		else:
			result = "lose"

		send_cmd(i,"gameover " + result)
		states[i] = "init"

	def outlog(i,line):
		if Logging:
			print "[" + str(i) + "]>" + line.strip()
		if FileLogging:
			log_file.write("[" + str(i) + "]>" + line.strip() + "\n")

	def outstd(i,line):
		print "["+str(i)+"]>" + line.strip()
		sys.stdout.flush()


	# set options for each engine
	for i in range(len(states)):
		for j in range(len(options[i % 2])):
			if j != 0 :
				opt = options[i % 2][j]

				# 置換対象文字列が含まれているなら置換しておく。
				opt = opt.replace("%%THREAD_NUMBER%%",str(i))
				send_cmd(i,opt)

	# loop for playing games
	while True:

		receive_something = False

		for i in range(len(states)):

			proc = procs[i]

			retcode = proc.poll()
			if retcode is not None:
				# process finished unexpectedly
				if not term_procs[i]:
					print "[" + str(i) + "]:Error! process terminated."
					term_procs[i] = True
				continue

			update = False

			# このスレッドから何か受け取ったかを計測しておき、
			# time outの判定を行なう。
			receive_something_for_this_process = False

			for line in iter(proc.stdout.readline,b''):

#			line = proc.stdout.readline()
#			if line != "":

				receive_something = True
				receive_something_for_this_process = True

				outlog(i,line)

				# "Error"か"Display"の文字列が含まれていればそれをそのまま出力する。
				# "Failed"は技巧が出力してくる。
				if ("Error" in line) or ("Display" in line) or ("Failed" in line):
					outstd(i,line)

				# node数計測用
				if "nodes" in line :
					nodes_str[i] = line
				# 評価値計測用
				if "score" in line :
					eval_value_from_thread[i] = line

				gameover = 0

				if ("readyok" in line) and (states[i] == "wait_for_readyok"):

					# 初回のみこの応答に対して1秒待つことにより、
					# プロセスの初期化タイミングが重複しないようにする。
					if initial_waits[i]:
						initial_waits[i] = False
						time.sleep(1)

					states[i] = "start"
					if states[i^1] == "start":
						usinewgame_cmd(i  ,sfen_no)
						usinewgame_cmd(i^1,sfen_no)
						sfen_no = (sfen_no + 1) % len(book_sfens)

						# send go_cmd to player1 or player2 randomly.
						# go_cmd((i & ~1) + random.randint(0,1) )
						# outlog(i,"turn = " + str((i & ~1) + turns[i/2]))

						# 先手→後手、交互に行う。
						go_cmd((i & ~1) + turns[i/2])

						# send go_cmd to player1
						# go_cmd((i & ~1) )

				elif ("bestmove" in line) and (states[i] == "wait_for_bestmove"):

					# node数計測用(60手目までのみ)
					if moves[i/2] < 60 :
						# 最後に受け取ったnodesを含んだ文字列の次の数値がnodes数。それを加算しておく。
						ns = nodes_str[i].split()
						for j in range(len(ns)):
							if ns[j] == "nodes":
								if j+1 < len(ns):
									nodes[i] += int(ns[j+1])
								break

					# 評価値の計測用
					vs = eval_value_from_thread[i].split()
					for j in range(len(vs)):
						if vs[j] == "score":
							if j+2 < len(vs):
								# 技巧が変な文字送ってくるときがある。
								# "mate + string Nyugyoku"みたいなの。無視する。
								try:
									v = int(vs[j+2])
								except:
									print "Error : score = " + eval_value_from_thread[i]
									eval_value_from_thread[i] = ""

								if vs[j+1] == "cp" :
									eval_value_from_thread[i] = str(v)
								elif vs[j+1] == "mate" :
								# mate表記なら32000を0手詰めのスコアとして計算しなおして返す。
									if v >= 0:
										eval_value_from_thread[i] = str(32000 - v)
									else:
										eval_value_from_thread[i] = str(-32000 + v)
							else:
								eval_value_from_thread[i] = "?" # 評価値わからん。思考エンジンおかしい。
							break

					# if (not random time)
					if options[2][3]==0:

						# elapsed time
						elapsed_time = int(math.ceil(time.time() - go_times[i])*1000)

						# rest_time += inc_time - elapsed_time
						r = rest_times[i] + options[2][1] - elapsed_time

						if r < 0:

							# if (rest_time + byoyomi < 0) then time_over
							r += options[2][2]

#							if r < 0:
							if False:
								elapsed_time2 = int((time.time() - go_times[i])*1000)
								r = rest_times[i] + options[2][1] + options[2][2] - elapsed_time2
								mes = "Error : TimeOver = " + engines[i & 1] + " overtime = " + str(-r)
								outlog(i,mes)
								outstd(i,mes)
								line = "bestmove resign"

							rest_times[i] = 0
						else:
							rest_times[i] = r

					if "resign" in line:
						if (i%2)==1:
							win += 1
							gameover = 1 # 1P勝ち
						else:
							lose += 1
							gameover = 2 # 2P勝ち
						if (moves[i/2] & 1 == 1):
							win_black += 1
						else:
							win_white += 1
						update = True

					elif "win" in line:
						if (i%2)==0:
							win += 1
							gameover = 1 # 1P勝ち
						else:
							lose += 1
							gameover = 2 # 2P勝ち
						if (moves[i/2] & 1 == 0):
							win_black += 1
						else:
							win_white += 1
						update = True

					else:
						# bestmove XXX
						ss = line.split()

						if sfens[i/2]!="":
							sfens[i/2]+=" "

						# bestmoveとしておかしい文字列を送ってくるエンジン対策
						try:
							sfens[i/2] += ss[1]
							if KifOutput:
								eval_values[i/2] += eval_value_from_thread[i]+" "
						except:
							outlog(i,"Error!" + line)

						moves[i/2] += 1
						if moves[i/2] >= 256:
							draw += 1
							gameover = 3 # Draw
							update = True
						else:
							go_cmd(i^1)

				if gameover :
					gameover_cmd(i  ,gameover)
					gameover_cmd(i^1,gameover)
					if KifOutput:
						kif_file.write("startpos moves " + sfens[i/2] + "\n")
						kif_file.write(eval_values[i/2]+"\n")
					# 手番を変更する。
					turns[i/2] = turns[i/2] ^ 1

			if update:
				loop_count = win + lose + draw
				if loop_count >= loop :
					# 終了した対局が指定のloop回数に達したので終了する。

					"""
					# トータルのnodes数の出力
					for j in range(threads*2):
						mes = "["+str(j)+"]:nodes = " + str(nodes[j])
						outlog(j,mes)
						outstd(j,mes)
					"""

					for p in procs:
						p.terminate()
					if FileLogging:
						f.close()
					if KifOutput:
						kif_file.close()
					return

				# output result at stated periods
				if loop_count % 10 == 0 :
					output_rating(win,draw,lose,win_black,win_white,opt2)
					if FileLogging:
						for i in range(len(states)):
							log_file.write("["+str(i)+"] State = " + states[i] + "\n")
					if FileLogging:
						f.flush()
					if KifOutput:
						kif_file.flush()

			if states[i] == "init":
				isready_cmd(i)

			# goコマンドを送信してから1分経過しているならtime out処理
			# ただし持ち時間がtで指定されているなら、5分までは待つ。
			if states[i] == "wait_for_bestmove" \
				and not receive_something_for_this_process \
				and time.time() - go_times[i] >= (300 if "t" in opt2 else 60):

				# sys.float_info.maxにするとオーバーフローしかねん..
				go_times[i] = sys.maxint
				mes = "[" + str(i) + "]: Error! Engine Timeout"
				outlog(i,mes)
				outstd(i,mes)

		# process is not done, wait a bit and check again.

		if not receive_something :
#			time.sleep(1.0/1000)
			time.sleep(0)


# 省略されたエンジン名に対して、フルパス名を返す
def engine_to_full(e):
	# 技巧
	if e == "gikou":
		e = "gikou_win_20160606/gikou.exe"
	# Silent Majority
	elif e == "SM":
		e = "SM_V110/SILENT_MAJORITY_AVX2_x64.exe"
	# やねうら王2016(Mid)
	elif e == "mid":
		e = "YaneuraOuV357mid.exe"

	return e

# ここからmain()

# sample 
#   > c:\python27\python.exe \\WS2012_860C_YAN\yanehome\script\engine_invoker2.py ^
#        --home \\WS2012_860C_YAN\yanehome\ ^
#        --engine1 YaneuraOuV350.exe ^
#        --eval1 Apery20160505 ^
#        --engine2 YaneuraOuV350.exe ^
#        --eval2 Apery20160505 ^
#        --cores 8 ^
#        --loop 1000 ^
#        --cpu 2 ^
#        --engine_threads 1 ^
#        --hash1 16 ^
#        --hash2 16 ^
#        --time r100

# 持ち時間(--time)の書式サンプル
#  r100    : random time 100
#  b1000   : byoyomi time 1000
#  t300000 : total time 300000
#  i3000   : inc time 3000
#  t300000/i3000 : t300000 and i3000
#  大文字で書くとnodes as timeモード
#  R100    : random time 100 and nodes as time
#  r100,r300   : ,で併記可能(それぞれの時間で対局する)
#  b1000.b2000 : .で連結するとengine1とengine2とでそれぞれの持ち時間になる。
#  d6      : depth 6

# パラメーターのparse
parser = argparse.ArgumentParser("engine_invoker5.py")
parser.add_argument('--home', type=str, default="", help=u"ホームディレクトリ");
parser.add_argument('--engine1', type=str, default="", help=u"エンジン1のpath");
parser.add_argument('--eval1', type=str, default="", help=u"評価関数フォルダ1 (ホームディレクトリ配下のevalフォルダ内にあるものとする)");
parser.add_argument('--engine2', type=str, default="", help=u"エンジン2のpath");
parser.add_argument('--eval2', type=str, default="", help=u"評価関数フォルダ2 (ホームディレクトリ配下のevalフォルダ内にあるものとする)");
parser.add_argument('--cores', type=int, default=1, help=u"コアの数(これをengine_threadsで割った数だけ並列対局)");
parser.add_argument('--loop', type=int, default=1, help=u"対局回数");
parser.add_argument('--cpu', type=int, default=1, help=u"PCに搭載されているCPUの数 1つのプロセッサには threads/cpu だけスレッドを割り当てる (256を指定すると1の意味になり、かつファイルロギングをする)");
parser.add_argument('--engine_threads', type=int, default=1, help=u"思考エンジンのスレッド数");
parser.add_argument('--hash1', type=str, default="", help=u"思考エンジン1のhashサイズ");
parser.add_argument('--hash2', type=str, default="", help=u"思考エンジン2のhashサイズ");
parser.add_argument('--time', type=str, default="", help=u"time");
parser.add_argument('--rand_book', type=int, default=0, help=u"定跡の順番をランダム化(rand_book=1を指定したとき)");
parser.add_argument('--book_moves', type=int, default=24, help=u"何手目まで定跡で指させるか");
parser.add_argument('--PARAMETERS_LOG_FILE_PATH', type=str, default="", help=u"同optionのpath指定 (ここに\"_2.log\"のような文字列が自動的に付与される。)");
args = parser.parse_args()

home = args.home
threads = args.cores
loop = args.loop
cpu = args.cpu
engine_threads = args.engine_threads
hashes = [args.hash1, args.hash2]
engine1_path = args.engine1
engine2_path = args.engine2
eval1_path = args.eval1
eval2_path = args.eval2
book_moves = args.book_moves
play_time_list = args.time.split(",")
PARAMETERS_LOG_FILE_PATH = args.PARAMETERS_LOG_FILE_PATH
rand_book = args.rand_book

# cpuに256が指定されているときは、FileLoggingを有効にする。
fileLogging = False
if cpu == 256:
	cpu = 1
	fileLogging = True

# expand eval_dir

evaldirs = []
if not os.path.exists(os.path.join(home, eval2_path, "0")) :
	evaldirs.append(eval2_path)
else:
	i = 0
	while os.path.exists(os.path.join(home, eval2_path, str(i))):
		evaldirs.append(os.path.join(eval2_path, str(i)))
		i += 1

print "home           : " , home
print "play_time_list : " , play_time_list
print "evaldirs       : " , evaldirs
print "hash size      : " , hashes
print "book_moves     : " , book_moves
print "engine_threads : " , engine_threads
print "rand_book      : " , rand_book
print "PARAMETERS_LOG_FILE_PATH : " , PARAMETERS_LOG_FILE_PATH

book_file = open(os.path.join(home, "book", "records2016_10818.sfen"),"r")
book_sfens = []
count = 1
for sfen in book_file:
	s = sfen.split()
	sf = ""
	for i in range(book_moves):
		try:
			# skip "startpos moves"
			sf += s[i+2]+" "
		except:
			print "Error! " + " in records2016.sfen line = " + str(count)
	book_sfens.append(sf)
	count += 1
	if count % 100 == 0:
		sys.stdout.write(".")
		sys.stdout.flush()
book_file.close()
print

# 定跡をシャッフルする
if rand_book:
	random.shuffle(book_sfens)

threads = threads / engine_threads

for evaldir in evaldirs:

	engine1 = engine_to_full(engine1_path)
	engine2 = engine_to_full(engine2_path)

	engines = ( engine1 , engine2 )
	engines_full = ( os.path.join(home, "exe", engines[0]) , os.path.join(home, "exe", engines[1]) )
	evals   = ( eval1_path , evaldir )
	evals_full   = ( os.path.join(home, "eval", eval1_path) , os.path.join(home, "eval", evaldir) )

	for i in range(2):
		print "engine" + str(i+1) + " = " + engines[i] + " , eval = " + evals[i]

	for play_time in play_time_list:
		print "\nthreads = " + str(threads) + " , loop = " + str(loop) + " , max_cpu = " + str(cpu) + " , play_time = " + play_time

		options = create_option(engines,engine_threads,evals_full,play_time,hashes,cpu,PARAMETERS_LOG_FILE_PATH)

		for i in range(2):
			print "option " + str(i+1) + " = " + ' / '.join(options[i])
			print "time_setting = (total_time,inc_time,byoyomi,rtime,depth_time) = " + str(options[i+2])

		sys.stdout.flush()

		# 短くスレッド数と秒読み条件を文字列化
		opt2 = "T"+str(engine_threads) + "," + play_time

		vs_match(engines_full,options,threads,loop,cpu,book_sfens,fileLogging,opt2,book_moves)

		# output final result
		print "\nfinal result : "
		for i in range(2):
			print "engine" + str(i+1) + " = " + engines[i] + " , eval = " + evals[i]
#		print "play_time = " + play_time + " , " ,
		output_rating(win,draw,lose,win_black,win_white,opt2)

