import time
import sys
import subprocess
import os.path
import math
import random
import datetime

# -----------------------------------------------------------------

# Non-blocking read on a subprocess.PIPE in python
# http://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
# fcntl indeed doesn't work on Windows

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

# output rating.
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


# create option for an each engine.
def create_option(engines,engine_threads,evals,byoyomi,hash):
	options = []

	for i in range(2):
		option = []
		if ("Yane" in engines[i]):
			option.append("go rtime " + byoyomi)
			option.append("setoption name Threads value " + str(engine_threads))
			option.append("setoption name EvalDir value " + evals[i])
			option.append("setoption name Hash value " + str(hash))
			option.append("setoption name BookFile value no_book")
			option.append("setoption name NetworkDelay value 0")
			option.append("setoption name NetworkDelay2 value 0")
#			option.append("setoption name EvalShare value false")
			option.append("setoption name EvalShare value true")
		else:
			option.append("go btime 0 wtime 0 byoyomi " + byoyomi)
			option.append("setoption name Threads value " + str(engine_threads))
			option.append("setoption name EvalDir value " + evals[i])
			option.append("setoption name USI_Hash value " + str(hash))

		options.append(option)

	return options

# play engine1 vs engine2
def vs_match(engines_full,options,threads,loop,numa):

	global win,lose,draw
	win = lose = draw = 0

	# home + "book/records1.sfen

	cmds = []
	for i in range(2):
		cmds.append("cmd.exe /c start /B /WAIT /NODE " + numa + " " + engines_full[i])

	# process state
	states = []
	# process handle
	procs = []
	# sfen
	sfens = []
	moves = []
	for t in range(threads):
		sfens.append("")
		moves.append(0)
		for i in range(2):
			proc = subprocess.Popen(cmds[i], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE)
			pipe_non_blocking_set(proc.stdout.fileno())

			procs.append( proc )
			states.append("init")


	# logging flag for console
#	Logging = True
	Logging = False

	# logging flag for file
	FileLogging = True
#	FileLogging = False

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

	def go_cmd(i):
		p = procs[i]
		# USI "position"
		s = "position startpos"
		if sfens[i/2] != "":
			s += " moves " + sfens[i/2]
		send_cmd(i,s)

		# USI "go"
		cmd = options[i & 1][0]
		send_cmd(i,cmd)

		# changes state
		states[i]   = "wait_for_bestmove"
		states[i^1] = "wait_for_another_player"

	def usinewgame_cmd(i):
		p = procs[i]
		send_cmd(i,"usinewgame")
		sfens[i/2] = ""
		moves[i/2] = 0

	def gameover_cmd(i):
		p = procs[i]
		send_cmd(i,"gameover win")
		states[i] = "init"

	# set options for each engine
	for i in range(len(states)):
		for j in range(len(options[i % 2])):
			if j != 0 :
				opt = options[i % 2][j]
				send_cmd(i,opt)

	# loop for playing games
	while True:

		update = False
		receive_something = False

		for i in range(len(states)):
			proc = procs[i]

			retcode = proc.poll()
			if retcode is not None:
				# process finished unexpectedly
#				print "[" + str(i) + "]:Error! process terminated."
				continue

			for line in iter(proc.stdout.readline, b''):

				receive_something = True

				if Logging:
					print "[" + str(i) + "]>" + line.strip()
				if FileLogging:
					f.write("[" + str(i) + "]>" + line.strip() + "\n")

				if "Error" in line:
					print line

				gameover = False

				if ("readyok" in line) and (states[i] == "wait_for_readyok"):
					states[i] = "start"
					if states[i^1] == "start":
						usinewgame_cmd(i)
						usinewgame_cmd(i^1)

						# send go_cmd to player1 or player2 randomly.
						go_cmd((i & ~1) + random.randint(0,1) )

						# send go_cmd to player1
						# go_cmd((i & ~1) )

				elif ("bestmove" in line) and (states[i] == "wait_for_bestmove"):

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
						sfens[i/2] += ss[1]
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
					for p in procs:
						p.terminate()
					if FileLogging:
						f.close()
					return

			if states[i] == "init":
				isready_cmd(i)

		# process is not done, wait a bit and check again.

		if not receive_something :
#			time.sleep(1.0/1000)
			time.sleep(0)

		# output result at stated periods
		if update and (loop_count % 10) == 0 :
			output_rating(win,draw,lose)
			if FileLogging:
				for i in range(len(states)):
					f.write("["+str(i)+"] State = " + states[i] + "\n")

			if FileLogging:
				f.flush()


param = sys.argv

# args format
# 	HOMEPATH engine1 evaldir1 engine2 evaldir2 threads loop numa { rtime1 ... rtimeN }

# sample 
#   > c:\python27\python.exe \\WS2012_860C_YAN\yanehome\script\engine_invoker2.py \\WS2012_860C_YAN\yanehome\ YaneuraOuV350.exe Apery20160505 YaneuraOuV350.exe Apery20160505 8 1000 0  { 100 }


home = param[1]
if not (home.endswith('/') or home.endswith('\\')):
	home += '\\'

threads = int(param[6])
loop = int(param[7])
numa = param[8]

if param[9] != "{" :
	byoyomi_list = param[9]
else:
	byoyomi_list = []
	for i in range (10,len(param)-1):
		byoyomi_list.append(param[i])

# expand eval_dir

evaldirs = []
if not os.path.exists(home + param[5] + "/0") :
	evaldirs.append(param[5])
else:
	i = 0
	while os.path.exists(home + param[5] + "/" + str(i)):
		evaldirs.append(param[5] + "/" + str(i) )
		i += 1

hash = 16
book_moves = 24

# threads number for an each engine
engine_threads = 1

print "home           : " , home
print "byoyomi_list   : " , byoyomi_list
print "evaldirs       : " , evaldirs
print "hash size      : " , hash
print "book_moves     : " , book_moves
print "engine_threads : " , engine_threads

threads = threads / engine_threads

for evaldir in evaldirs:

	engines = ( param[2] , param[4] )
	engines_full = ( home + "exe\\" + engines[0] , home + "exe\\" + engines[1] )
	evals   = ( param[3] , evaldir )
	evals_full   = ( home + "eval\\" + param[3]  , home + "eval\\" + evaldir )

	for i in range(2):
		print "engine" + str(i+1) + " = " + engines[i] + " , eval = " + evals[i]

	for byoyomi in byoyomi_list:
		print "\nthreads = " + str(threads) + " , loop = " + str(loop) + " , numa = " + numa + " , byoyomi = " + byoyomi

		options = create_option(engines,engine_threads,evals_full,byoyomi,hash)

		for i in range(2):
			print "option " + str(i+1) + " = " + ' / '.join(options[i])

		sys.stdout.flush()

		vs_match(engines_full,options,threads,loop,numa)

		# output final result
		print "\nfinal result : "
		for i in range(2):
			print "engine" + str(i+1) + " = " + engines[i] + " , eval = " + evals[i]
		print "byoyomi = " + byoyomi + " , " ,
		output_rating(win,draw,lose)

