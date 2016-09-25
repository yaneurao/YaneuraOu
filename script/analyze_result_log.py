# -*- coding: utf8 -*-
'''
やねうら王 2016 Lateのランダムパラメーター機能で探索パラメーターを変化させて
対局させたときのログを集計して一番勝率の高いものを選ぶための分析スクリプト。
'''
import os
import sys
import glob
import math
from collections import OrderedDict

params = OrderedDict()

def analyze_log(file_path):

#	print "file_path = " + file_path

	with open(file_path, 'rb') as fi:

		last_line = ""
		for line in fi.readlines():
			if ("draw" in line) or ("win" in line) or ("lose" in line):
				# "gameover win"の"win"の部分を取り出す
				result = line.split(' ')[1].strip()

				plist = last_line.split(",");
				# PARAM1:123,PARAM2:234,...のように並んでいる。
				for p in plist:
					q = p.split(":")
					# 末尾のカンマの可能性がある。
					if len(q) < 2:
						continue

					# keyがなければ作る。
					if not q[0] in params:
						params[q[0]] = {}
					if not q[1] in params[q[0]]:
						params[q[0]][q[1]] = {}
					if not result in params[q[0]][q[1]]:
						params[q[0]][q[1]][result] = 0

					params[q[0]][q[1]][result] += 1

			last_line = line

def rating(win,lose,draw):
	total = win + lose
	if total != 0 :
		win_rate = win / float(win+lose)
	else:
		win_rate = 0

	if win_rate == 0 or win_rate == 1:
		rating = ""
	else:
		rating = " R" + str(round(-400*math.log(1/win_rate-1,10),2))

	return str(win) + " - " + str(draw) + " - " + str(lose) + "(" + str(round(win_rate*100,2)) + "%" + rating + ")"


if __name__ == '__main__':

#	print os.path.join(sys.argv[1], '*', 'log')
	for file_path in sorted(glob.glob(os.path.join(sys.argv[1], '*.log'))):
		fig = analyze_log(file_path)
		sys.stdout.write(".")
	print

	t_win = t_lose = t_draw = 0
	first = True
	for key,param in params.items():
		if first:
			first = False
			for key2,result in param.items():
				if "win" in result:
					t_win += result["win"]
				if "lose" in result:
					t_lose += result["lose"]
				if "draw" in result:
					t_draw += result["draw"]
			total = t_win+t_lose+t_draw
			print "  total : " + rating(t_win,t_lose,t_draw)

		print key + ":"

		for key2,result in sorted(param.items(),key = lambda x:int(x[0])):
			win = lose = draw = 0
			if "win" in result:
				win = result["win"]
			if "lose" in result:
				lose = result["lose"]
			if "draw" in result:
				draw = result["draw"]
			print "  " + key2 + " : " + rating(win,lose,draw)

