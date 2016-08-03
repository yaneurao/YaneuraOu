import sys
import subprocess
import os.path
import math

def write_engine_file(num , engine_name , eval_dir , byoyomi):
	f = open("engine-config"+str(num)+".txt","w")
	f.write(engine_name+"\n")
#	f.write("go btime 0 wtime 0 byoyomi "+byoyomi+"\n")
	f.write("go rtime " + byoyomi + "\n")
	f.write("setoption name EvalDir value " + eval_dir+"\n")
	f.write("setoption name Hash value 16\n")
#	f.write("setoption name Hash value 128\n")
	f.write("setoption name Threads value 1\n")
	f.write("setoption name BookFile value no_book\n")
	f.write("setoption name NetworkDelay value 0\n")
	f.write("setoption name NetworkDelay2 value 0\n")

	f.close()

param = sys.argv

# args format
# 	HOMEPATH engine1 evaldir1 engine2 evaldir2 threads loop numa { rtime1 ... rtimeN }

# loop for invoker

home = param[1]
if not (home.endswith('/') or home.endswith('\\')):
	home += '\\'

threads = param[6]
loop = param[7]
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

print "home         : " , home
print "byoyomi_list : " , byoyomi_list
print "evaldirs     : " , evaldirs

for evaldir in evaldirs:

	print "engine1 = " + param[2] + " , eval = " + param[3]
	print "engine2 = " + param[4] + " , eval = " + evaldir

	for byoyomi in byoyomi_list:

		print "threads = " + threads + " , loop = " + loop + " , numa = " + numa + " , byoyomi = " + byoyomi
		write_engine_file(1,home + param[2],home + param[3],byoyomi)
		write_engine_file(2,home + param[4],home + evaldir,byoyomi)

		cmd = home + "\\exe\\local-game-serverV347.exe , booksfenfile " + home + "book/records1.sfen , threads " + threads + " , enginenuma " + numa + " , go btime " + loop + " , quit"

		# subprocess.call( cmd.strip().split(" "))
		print cmd
		p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		stdout_data, stderr_data = p.communicate()
		# print "finish: \n%s\n%s" % (stdout_data, stderr_data)

		win = lose = draw = 0
		for line in stdout_data.split('\n'):
			if line.startswith("win"):
				win += 1
			elif line.startswith("lose"):
				lose += 1
			elif line.startswith("draw"):
				draw += 1

			if "Error" in line:
				print line


		total = win + lose
		if total != 0 :
			win_rate = win / float(win+lose)
		else:
			win_rate = 0
		print "finish " + str(win) + " - " + str(draw) + " - " + str(lose) + "(" + str(round(win_rate*100,2)) + "% R" + str(round(-400*math.log(1/win_rate-1,10),2)) + ")\n"

