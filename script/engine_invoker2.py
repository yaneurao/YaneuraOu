# coding: UTF-8
import time
import sys
import subprocess
import os.path
import math
import random
import datetime

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

# レーティングの出力
def output_rating(win,draw,lose):
	total = win + lose
	if total != 0 :
		win_rate = win / float(win+lose)
	else:
		win_rate = 0

	if win_rate == 0 or win_rate == 1:
		rating = ""
	else:
		rating = " R" + str(round(-400*math.log(1/win_rate-1,10),2))

	print str(win) + " - " + str(draw) + " - " + str(lose) + "(" + str(round(win_rate*100,2)) + "%" + rating + ")"
	sys.stdout.flush()


# 思考エンジンに対するオプションを生成する。
def create_option(engines,engine_threads,evals,times,hashes,numa):

	options = []

	rtime = 0
	byoyomi = 0
	inc_time = 0
	total_time = 0

	nodes_time = False

	for b in times.split("/"):

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

	for i in range(2):
		option = []
		if ("Yane" in engines[i]):
			if rtime:
				option.append("go rtime " + str(rtime))
			elif inc_time:
				option.append("go btime REST_TIME wtime REST_TIME inc " + str(inc_time))
			else:
				option.append("go btime REST_TIME wtime REST_TIME byoyomi " + str(byoyomi))

			option.append("setoption name Threads value " + str(engine_threads))
			option.append("setoption name EvalDir value " + evals[i])
			option.append("setoption name Hash value " + str(hashes[i]))
			option.append("setoption name BookFile value no_book")
			option.append("setoption name NetworkDelay value 0")
			option.append("setoption name NetworkDelay2 value 0")

#			option.append("setoption name EvalShare value false")
			option.append("setoption name EvalShare value true")
#			if i==0:
#				option.append("setoption name EvalShare value false")
#			else:
#				option.append("setoption name EvalShare value true")
			option.append("setoption name EngineNuma value " + str(numa))
			if nodes_time:
				option.append("setoption name nodestime value 600")
		else:
			# ここで対応しているengine一覧
			#  ・技巧(20160606)
			#  ・Silent Majority(V1.1.0)
			if rtime:
				option.append("go rtime " + str(rtime))
				print "Error! " + engines[i] + " doesn't support rtime "
			elif inc_time:
				option.append("go btime REST_TIME wtime REST_TIME inc " + str(inc_time))
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

	options.append([total_time,inc_time,byoyomi,rtime])

	return options

# engine1とengine2とを対戦させる
#  threads    : この数だけ並列対局
#  numa       : 実行するプロセッサグループ
#  book_sfens : 定跡
def vs_match(engines_full,options,threads,loop,numa,book_sfens,fileLogging):

	global win,lose,draw
	win = lose = draw = 0

	# home + "book/records1.sfen

	cmds = []
	for i in range(2):
		# working directoryを実行ファイルのあるフォルダ直下としてやる。
		# 最後のdirectory separatorを探す
		engine_path = engines_full[i]
		pos = max(engine_path.rfind('\\') , engine_path.rfind('/'))
		if pos <= 0:
			working_dir = ""
		else:
			working_dir = engine_path[:pos]
		# print "working_dir = " + working_dir

		cmds.append("cmd.exe /c start /B /WAIT /D " + working_dir + " /NODE " + str(numa) + " " + engines_full[i])

	# 棋譜
	sfens = [""]*threads
	# 対局開始局面からの手数
	moves = [0]*threads
	# 次の対局で先手番のplayer(0 or 1)
	turns = [0]*threads

	# process handle
	procs = [0]*threads*2
	# process state
	states = ["init"]*threads*2

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
#		proc = subprocess.Popen(cmds[i & 1] , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE , universal_newlines=True , bufsize=1)
		proc = subprocess.Popen(cmds[i & 1] , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE )
		pipe_non_blocking_set(proc.stdout.fileno())

		procs[i] = proc


	# これをTrueにするとコンソールに思考エンジンとのやりとりを出力する。
#	Logging = True
	Logging = False

	# これをTrueにするとログファイルに思考エンジンとのやりとりを出力する。
#	FileLogging = True
	FileLogging = fileLogging

	if FileLogging:
		now = datetime.datetime.today()
		f = open("script_log"+now.strftime("%Y%m%d%H%M%S")+".txt","w")

	def send_cmd(i,s):
		p = procs[i]
		if Logging:
			print "[" + str(i) + "]<" + s
		if FileLogging:
			f.write( "[" + str(i) + "]<" + s + "\n")
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

	def usinewgame_cmd(i):
		p = procs[i]
		send_cmd(i,"usinewgame")
		sfens[i/2] = book_sfens[random.randint(0,len(book_sfens)-1)]
		moves[i/2] = 0

	def gameover_cmd(i):
		p = procs[i]
		send_cmd(i,"gameover win")
		states[i] = "init"

	def outlog(i,line):
		if Logging:
			print "[" + str(i) + "]>" + line.strip()
		if FileLogging:
			f.write("[" + str(i) + "]>" + line.strip() + "\n")

	def outstd(i,line):
		print "["+str(i)+"]>" + line.strip()
		sys.stdout.flush()


	# set options for each engine
	for i in range(len(states)):
		for j in range(len(options[i % 2])):
			if j != 0 :
				opt = options[i % 2][j]
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

			for line in iter(proc.stdout.readline,b''):

#			line = proc.stdout.readline()
#			if line != "":

				receive_something = True

				outlog(i,line)

				# "Error"か"Display"の文字列が含まれていればそれをそのまま出力する。
				# "Failed"は技巧が出力してくる。
				if ("Error" in line) or ("Display" in line) or ("Failed" in line):
					outstd(i,line)

				# node数計測用
				if "nodes" in line :
					nodes_str[i] = line

				gameover = False

				if ("readyok" in line) and (states[i] == "wait_for_readyok"):
					states[i] = "start"
					if states[i^1] == "start":
						usinewgame_cmd(i)
						usinewgame_cmd(i^1)

						# send go_cmd to player1 or player2 randomly.
						# go_cmd((i & ~1) + random.randint(0,1) )
						# outlog(i,"turn = " + str((i & ~1) + turns[i/2]))

						# 先手→後手、交互に行う。
						go_cmd((i & ~1) + turns[i/2])
						turns[i/2] = turns[i/2] ^ 1

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
						else:
							lose += 1
						gameover = True
						update = True

					elif "win" in line:
						if (i%2)==0:
							win += 1
						else:
							lose += 1
						gameover = True
						update = True

					else:
						# bestmove XXX
						ss = line.split()

						if sfens[i/2]!="":
							sfens[i/2]+=" "

						# bestmoveとしておかしい文字列を送ってくるエンジン対策
						try:
							sfens[i/2] += ss[1]
						except:
							outlog(i,"Error!" + line)

						moves[i/2] += 1
						if moves[i/2] >= 256:
							draw += 1
							gameover = True
							update = True
						else:
							go_cmd(i^1)

				if gameover :
					gameover_cmd(i)
					gameover_cmd(i^1)

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
					return

				# output result at stated periods
				if loop_count % 10 == 0 :
					output_rating(win,draw,lose)
					if FileLogging:
						for i in range(len(states)):
							f.write("["+str(i)+"] State = " + states[i] + "\n")
					if FileLogging:
						f.flush()

			if states[i] == "init":
				isready_cmd(i)

		# process is not done, wait a bit and check again.

		if not receive_something :
#			time.sleep(1.0/1000)
			time.sleep(0)

param = sys.argv

# args format
# 	HOMEPATH engine1 evaldir1 engine2 evaldir2 threads loop numa engine_threads hash1 hash2 { time1 ... timeN }

# sample 
#   > c:\python27\python.exe \\WS2012_860C_YAN\yanehome\script\engine_invoker2.py \\WS2012_860C_YAN\yanehome\ YaneuraOuV350.exe Apery20160505 YaneuraOuV350.exe Apery20160505 8 1000 0 1 16 16 { r100 }

# HOMEPATH          : ホームディレクトリ
# engine1,engine2   : エンジン1,2のpath
# evaldir1,evaldir2 : 評価関数フォルダ1,2 (ホームディレクトリ配下のevalフォルダ内にあるものとする)
# threads           : 並列対局数
# loop              : 対局回数
# numa              : 実行するプロセッサグループ(256を指定すると0の意味になり、かつファイルロギングをする)
# engine_threads    : 思考エンジンのスレッド数
# hash1,hash2       : 思考エンジンのhashサイズ
# time1…timeN      : 持ち時間の指定

# 持ち時間の書式サンプル
#  r100    : random time 100
#  b1000   : byoyomi time 1000
#  t300000 : total time 300000
#  i3000   : inc time 3000
#  t300000/i3000 : t300000 and i3000
#  大文字で書くとnodes as timeモード
#  R100    : random time 100 and nodes as time

home = param[1]
if not (home.endswith('/') or home.endswith('\\')):
	home += '\\'

# 論理コアの数を物理コア数の数に変更する。
threads = int(param[6])
loop = int(param[7])
numa = int(param[8])
# numaに256が指定されているときは、FileLoggingを有効にする。
fileLogging = False
if numa == 256:
	numa = 0
	fileLogging = True

# threads number for an each engine
engine_threads = int(param[9])

# hash size for an each engine
hashes = [16,16]
hashes[0] = int(param[10])
hashes[1] = int(param[11])

if param[12] != "{" :
	play_time_list = [ param[12] ]
else:
	play_time_list = []
	for i in range (13,len(param)-1):
		play_time_list.append(param[i])

# expand eval_dir

evaldirs = []
if not os.path.exists(home + param[5] + "/0") :
	evaldirs.append(param[5])
else:
	i = 0
	while os.path.exists(home + param[5] + "/" + str(i)):
		evaldirs.append(param[5] + "/" + str(i) )
		i += 1

book_moves = 16

print "home           : " , home
print "play_time_list : " , play_time_list
print "evaldirs       : " , evaldirs
print "hash size      : " , hashes
print "book_moves     : " , book_moves
print "engine_threads : " , engine_threads

book_file = open(home+"/book/records2016.sfen","r")
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

threads = threads / engine_threads

for evaldir in evaldirs:

	engines = ( param[2] , param[4] )
	engines_full = ( home + "exe\\" + engines[0] , home + "exe\\" + engines[1] )
	evals   = ( param[3] , evaldir )
	evals_full   = ( home + "eval\\" + param[3]  , home + "eval\\" + evaldir )

	for i in range(2):
		print "engine" + str(i+1) + " = " + engines[i] + " , eval = " + evals[i]

	for play_time in play_time_list:
		print "\nthreads = " + str(threads) + " , loop = " + str(loop) + " , numa = " + str(numa) + " , play_time = " + play_time

		options = create_option(engines,engine_threads,evals_full,play_time,hashes,numa)

		for i in range(2):
			print "option " + str(i+1) + " = " + ' / '.join(options[i])
		print "time_setting(total_time,inc_time,byoyomi,rtime) = " + str(options[2])

		sys.stdout.flush()

		vs_match(engines_full,options,threads,loop,numa,book_sfens,fileLogging)

		# output final result
		print "\nfinal result : "
		for i in range(2):
			print "engine" + str(i+1) + " = " + engines[i] + " , eval = " + evals[i]
		print "play_time = " + play_time + " , " ,
		output_rating(win,draw,lose)

