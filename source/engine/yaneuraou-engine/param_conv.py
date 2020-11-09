
# これは、
#    yaneuraou-param.h から
# 1) yaneuraou-param-extern.h
# 2) yaneuraou-param-array.h
# 3) yaneuraou-param-string.h
# を自動生成するPythonのスクリプトです。

import re

pattern = "PARAM_DEFINE (.+?) = (.+);"

with open("yaneuraou-param.h",encoding='utf-8-sig') as f:
	lines = f.readlines()


# 変数がすべてextern intとして書かれているヘッダーファイル
with open("param/yaneuraou-param-extern.h",mode = "w",encoding='utf-8-sig') as f:
	for line in lines:
		# コメント行はそのまま出力
		if line.startswith("//"):
			f.write(line)

		# 空行もそのまま出力
		elif line == "":
			f.write()

		elif line.startswith("PARAM_DEFINE"):
			result = re.match(pattern,line)
			var_name = result.group(1)
			f.write("extern int " + var_name + ";" + "\n\n")


# 変数の名前が先頭に&がついてカンマ区切りになっているヘッダーファイル
with open("param/yaneuraou-param-array.h",mode = "w",encoding='utf-8-sig') as f:
	for line in lines:
		# コメント行はそのまま出力
		if line.startswith("//"):
			f.write(line)

		if line.startswith("PARAM_DEFINE"):
			result = re.match(pattern,line)
			var_name = result.group(1)
			f.write("&" + var_name + ","+"\n\n")


# 変数の名前が文字列(""で囲まれている)のカンマ区切りになっているヘッダーファイル
with open("param/yaneuraou-param-string.h",mode = "w",encoding='utf-8-sig') as f:
	for line in lines:
		# コメント行はそのまま出力
		if line.startswith("//"):
			f.write(line)

		if line.startswith("PARAM_DEFINE"):
			result = re.match(pattern,line)
			var_name = result.group(1)
			f.write('"' + var_name + '",'+"\n\n")


print("done.")
