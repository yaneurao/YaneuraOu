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
def create_option(engines,engine_threads,evals,times,hash):

	options = []

	rtime = 0
	byoyomi = 0
	inc_time = 0
	total_time = 0

	for b in times.split("/"):
		t = int(b[1:])
		if b.startswith("r"):
			rtime = t
		elif b.startswith("b"):
			byoyomi = t
		elif b.startswith("i"):
			inc_time = t
		elif b.startswith("t"):
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
			option.append("setoption name Hash value " + str(hash))
			option.append("setoption name BookFile value no_book")
			if rtime:
				option.append("setoption name NetworkDelay value 0")
				option.append("setoption name NetworkDelay2 value 0")
			else:
				option.append("setoption name NetworkDelay value 550")
				option.append("setoption name NetworkDelay2 value 550")
#			option.append("setoption name EvalShare value false")
			option.append("setoption name EvalShare value true")
#			if i==0:
#				option.append("setoption name EvalShare value false")
#			else:
#				option.append("setoption name EvalShare value true")
		else:
			if rtime:
				option.append("go rtime " + str(rtime))
				print "Error! " + engines[i] + " doesn't support rtime "
			elif inc_time:
				option.append("go btime REST_TIME wtime REST_TIME inc " + str(inc_time))
			else:
				option.append("go btime REST_TIME wtime REST_TIME byoyomi " + str(byoyomi))

			option.append("setoption name Threads value " + str(engine_threads))
			option.append("setoption name EvalDir value " + evals[i])
			option.append("setoption name USI_Hash value " + str(hash))

		options.append(option)

	options.append([total_time,inc_time,byoyomi,rtime])

	return options

# play engine1 vs engine2
def vs_match(engines_full,options,threads,loop,numa,book_sfens):

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
	# times
	rest_times = []
	go_times = []

	for t in range(threads):
		sfens.append("")
		moves.append(0)
		for i in range(2):
			proc = subprocess.Popen(cmds[i], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE)
			pipe_non_blocking_set(proc.stdout.fileno())

			procs.append( proc )
			states.append("init")
			rest_times.append(0)
			go_times.append(0)


	# logging flag for console
#	Logging = True
	Logging = False

	# logging flag for file
#	FileLogging = True
	FileLogging = False

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

					# if (not random time)
					if options[2][3]==0:

						# elapsed time
						elapsed_time = int(math.ceil(time.time() - go_times[i])*1000)

						# rest_time += inc_time - elapsed_time
						r = rest_times[i] + options[2][1] - elapsed_time
						if r < 0:

							# if (rest_time + byoyomi < 0) then time_over
							r += options[2][2]
							if r < 0:
								elapsed_time2 = int((time.time() - go_times[i])*1000)
								r = rest_times[i] + options[2][1] + options[2][2] - elapsed_time2
								print "Error : TimeOver = " + engines[i & 1] \
									+ " overtime = " + str(-r)
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
# 	HOMEPATH engine1 evaldir1 engine2 evaldir2 threads loop numa { time1 ... timeN }

# sample 
#   > c:\python27\python.exe \\WS2012_860C_YAN\yanehome\script\engine_invoker2.py \\WS2012_860C_YAN\yanehome\ YaneuraOuV350.exe Apery20160505 YaneuraOuV350.exe Apery20160505 8 1000 0  { r100 }

# time1..timeN sample
#  r100    : random time 100
#  1000    : byoyomi time 1000
#  t300000 : total time 300000
#  i3000   : inc time 3000
#  t300000/i3000 : t300000 and i3000

home = param[1]
if not (home.endswith('/') or home.endswith('\\')):
	home += '\\'

threads = int(param[6])
loop = int(param[7])
numa = param[8]

if param[9] != "{" :
	play_time_list = [ param[9] ]
else:
	play_time_list = []
	for i in range (10,len(param)-1):
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

hash = 16
book_moves = 16

# threads number for an each engine
# engine_threads = 4
engine_threads = 1

print "home           : " , home
print "play_time_list : " , play_time_list
print "evaldirs       : " , evaldirs
print "hash size      : " , hash
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
		print "\nthreads = " + str(threads) + " , loop = " + str(loop) + " , numa = " + numa + " , play_time = " + play_time

		options = create_option(engines,engine_threads,evals_full,play_time,hash)

		for i in range(2):
			print "option " + str(i+1) + " = " + ' / '.join(options[i])
		print "time_setting(total_time,inc_time,byoyomi,rtime) = " + str(options[2])

		sys.stdout.flush()

		vs_match(engines_full,options,threads,loop,numa,book_sfens)

		# output final result
		print "\nfinal result : "
		for i in range(2):
			print "engine" + str(i+1) + " = " + engines[i] + " , eval = " + evals[i]
		print "play_time = " + play_time + " , " ,
		output_rating(win,draw,lose)

