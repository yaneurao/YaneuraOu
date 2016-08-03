import sys
import math

# python calc_rating.py 60 40

param = sys.argv
win = float(param[1])
lose = float(param[2])
draw = 0

total = win + lose
if total != 0 :
	win_rate = win / float(win+lose)
else:
	win_rate = 0
print "finish " + str(win) + " - " + str(draw) + " - " + str(lose) + "(" + str(round(win_rate*100,2)) + "% R" + str(round(-400*math.log(1/win_rate-1,10),2)) + ")\n"

