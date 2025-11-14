#pragma once

#include "config.h"
#include "misc.h"
#include <vector>

namespace YaneuraOu {

#if 0
// デバッグ用に同じようにチェックポイントを通過しているかを調べる。
class Tracer /*alignas(64)*/
{
public:
	// データの型。これuint8_tとかuint16_tとか細かい型にすると
	// alignmentの関係で、バグが起きなくなる場合があるので注意。

	using T = uint32_t;


	Tracer(){
		// 事前に確保しておかないと、resizeでallocateしなおしになったらメモリ足りなくなる。
		// 8GB
		size = 8ULL*1024*1024*1024/sizeof(T);
		trace.reserve(size);
	}

	// 2度目以降のスタート
	void restart() { restarted = true; idx = 0; }

	// iでチェックポイントを通過
	void checkpoint(T data)
	{
		if (!restarted)
		{
			// the 1st time
			if (trace.size() == size)
			//{
			//	std::cout<< "checkpoint error! memory full!!" << std::endl;
			//	ASSERT(false);
			//}
				return;

			trace.emplace_back(data);
		} else {
			//  the second time and thereafter

			if (trace.size() <= idx)
				return;
			//{
			//	std::cout<< "checkpoint error! overrun!!" << std::endl;
			//	ASSERT(false);
			//}

			if (trace[idx] != data)
			{
				std::cout << "checkpoint error , first time = " << trace[idx]
					      << " , now = " << data << std::endl;
				ASSERT(false);
			}
		}
		++idx;
	}

private:
	std::vector<T> trace;
	size_t size;
	size_t idx = 0;
	bool restarted = false;
};
#endif

// ↑のメモリではなくファイルに書き込む版

// デバッグ用に同じようにチェックポイントを通過しているかを調べる。
class Tracer
{
public:
	const char* FILE_NAME = "C:\\Users\\yaneen\\Desktop\\yaneuraou-log.txt";

	Tracer(){
		fopen_s(&fp, FILE_NAME,"w");
	}

	// 2度目以降のスタート
	void restart() {
		fclose(fp);
		fopen_s(&fp, FILE_NAME,"r");

		restarted = true;
	}

	//
	// TRACER.checkpoint(__FILE__,__LINE__);
	// などと書いて埋めると良い。
	//
	void checkpoint(std::string data1,int data2)
	{
		checkpoint(data1 + " , " + std::to_string(data2));
	}

	void checkpoint(std::string data)
	{
		if (!restarted)
		{
			// the 1st time
			fprintf(fp, "%s\n", data.c_str());
		} else {
			//  the second time and thereafter

			char line[256];
			
			if (fgets(line,sizeof(line),fp) == nullptr)
			{
				std::cout<< "checkpoint error! overrun!!" << std::endl;
				ASSERT(false);
			}

			if (std::string(line)!=data+"\n")
			{
				std::cout << "checkpoint error , first time = " << line
					      << " , now = " << data << std::endl;

				ASSERT(false);
			}
		}
	}

	// iでチェックポイントを通過
	void checkpoint(int data)
	{
		if (!restarted)
		{
			// the 1st time
			fprintf(fp, "%d\n", data);
		} else {
			//  the second time and thereafter

			char line[256];
			
			if (fgets(line,sizeof(line),fp) == nullptr)
			{
				std::cout<< "checkpoint error! overrun!!" << std::endl;
				ASSERT(false);
			}

			if (atoi(line)!=data)
			{
				std::cout << "checkpoint error , first time = " << line
					      << " , now = " << data << std::endl;

				ASSERT(false);
			}
		}
	}

private:
	FILE* fp;
	bool restarted = false;
};

} // namespace YaneuraOu

//extern Tracer TRACER;
