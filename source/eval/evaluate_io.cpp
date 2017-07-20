#include "evaluate_io.h"

namespace EvalIO
{
	bool eval_convert(const EvalInfo& input, const EvalInfo& output, const std::vector<u16>* map)
	{
		// 特徴因子の型の数が異なるとそもそも変換できない。
		if (input.eval_info_array.size() != output.eval_info_array.size())
		{
			std::cout << "Error! input.eval_info_array.size() != output.eval_info_array.size() in eval_io()" << std::endl;
			return false;
		}

		int size = (int)input.eval_info_array.size();
		for (int i = 0; i < size; ++i)
		{
			auto in_ = input.eval_info_array[i];
			auto out_ = output.eval_info_array[i];

			// 特徴因子の型名が違うとそもそも変換は定義されていない。
			if (in_.feature != out_.feature)
			{
				std::cout << "Error! input.feature != output.feature in eval_io()" << std::endl;
				return false;
			}

			// メモリブロックのサイズの計算
			auto calc_block_size = [](u64 element_num , u64 element_size , EvalFeature feature , u64 sq_nb , u64 fe_end)
			{
				u64 block_size = element_size * element_num;
				// 型ごとにサイズが異なる
				switch (feature)
				{
				case KK  : block_size *= (sq_nb ) * (sq_nb)                       ; break;
				case KKP : block_size *= (sq_nb ) * (sq_nb)  * (fe_end)           ; break;
				case KPP : block_size *= (sq_nb ) * (fe_end) * (fe_end)           ; break;
				case PP  : block_size *= (fe_end) * (fe_end)                      ; break;
				case KKPP: block_size *= (sq_nb ) * (sq_nb)  * (fe_end) * (fe_end); break;
				case KPPP: block_size *= (sq_nb ) * (fe_end) * (fe_end) * (fe_end); break;
				default:
					ASSERT_LV1(false);
				}
				return block_size;
			};

			auto input_block_size = calc_block_size(in_.element_num, in_.element_size, in_.feature, input.sq_nb, input.fe_end);
			auto output_block_size = calc_block_size(out_.element_num, out_.element_size, out_.feature, output.sq_nb, output.fe_end);

			// 型変換の必要があるかどうかの判定
			// inとoutの型が全く同じでかつ変換mapが指定されていないのであれば、
			// 単にメモリコピー、ファイル入出力などで済む。このとき、オーバーヘッドは最も小さくて済む。
			bool convertless = 
					   in_.element_size == out_.element_size
					&& in_.element_num == out_.element_num
					&& input.sq_nb == output.sq_nb
					&& input.fe_end == output.fe_end
					&& map == nullptr;

			// 変換が不要のとき。
			if (convertless)
			{
				// memory to memory
				if (in_.file_or_memory.memory() && out_.file_or_memory.memory())
				{
					memcpy(out_.file_or_memory.ptr, in_.file_or_memory.ptr, input_block_size);
				}
				// file to memory
				else if (in_.file_or_memory.file() && out_.file_or_memory.memory())
				{
					std::ifstream ifs(in_.file_or_memory.filename, std::ios::binary);
					if (ifs) ifs.read(reinterpret_cast<char*>(out_.file_or_memory.ptr), input_block_size);
					else
					{
						// ToDo : read()自体に失敗したことも検出すべきなのだが、うまい書き方がよくわからない。

						std::cout << "info string read file error , file = " << in_.file_or_memory.filename << std::endl;
						return false;
					}
				}
				// memory to file
				else if (in_.file_or_memory.memory() && out_.file_or_memory.file())
				{
					std::ofstream ofs(out_.file_or_memory.filename, std::ios::binary);
					if (ofs) ofs.write(reinterpret_cast<char*>(in_.file_or_memory.ptr), output_block_size);
					else
					{
						std::cout << "info string write file error , file = " << out_.file_or_memory.filename << std::endl;
						return false;
					};
				}
				// file to file
				else if (in_.file_or_memory.file() && out_.file_or_memory.file())
				{
					std::vector<u8> buffer(input_block_size);
					std::ifstream ifs(in_.file_or_memory.filename, std::ios::binary);
					if (ifs) ifs.read(reinterpret_cast<char*>(&buffer[0]), input_block_size);
					else
					{
						std::cout << "info string read file error , file = " << in_.file_or_memory.filename << std::endl;
						return false;
					};
					std::ofstream ofs(out_.file_or_memory.filename, std::ios::binary);
					if (ofs) ofs.write(reinterpret_cast<char*>(&buffer[0]), output_block_size);
					else
					{
						std::cout << "info string write file error , file = " << out_.file_or_memory.filename << std::endl;
						return false;
					}
				}
			}
			else {
				// --- 変換が必要なとき。
				
				// 1) 出力先がファイルであるなら、まず変換をメモリ上で行なう必要があるので、
				//   出力のためのバッファをメモリ上に確保する。

				std::vector<u8> output_buffer;
				void* out_ptr = out_.file_or_memory.ptr;
				if (out_ptr == nullptr)
				{
					output_buffer.resize(output_block_size);
					out_ptr = (void*)&output_buffer[0];
				}

				// 2) 入力元がファイルであるなら、まずそのバッファをメモリ上に確保して、メモリに読み込む。

				std::vector<u8> input_buffer;
				void* in_ptr = in_.file_or_memory.ptr;
				if (in_ptr == nullptr)
				{
					input_buffer.resize(input_block_size);
					in_ptr = (void*)&input_buffer[0];

					std::ifstream ifs(in_.file_or_memory.filename, std::ios::binary);
					if (ifs) ifs.read(reinterpret_cast<char*>(in_ptr), input_block_size);
					else
					{
						std::cout << "info string read file error , file = " << in_.file_or_memory.filename << std::endl;
						return false;
					}
				}

				// 3) 変換する

				u64 input_feature_size = in_.element_size * in_.element_num;
				u64 output_feature_size = out_.element_size * out_.element_num;
				auto conv = [&](u8* src, u8* dst)
				{
					// in_.element_numとout_.element_numの数が異なることがあるのだが…。
					// とりあえずout_.element_numを基準に考える。
					for (u64 i = 0; i < out_.element_num; ++i)
					{
						s64 n;
						if (i < in_.element_num)
						{
							switch (in_.element_size)
							{
							case 1: n = *(s8*)src; break;
							case 2: n = *(s16*)src; break;
							case 4: n = *(s32*)src; break;
							case 8: n = *(s64*)src; break;
							default:
								UNREACHABLE;
							}
						}
						else {
							// out_.element_numのほうがin_.element_numより大きいので
							// 余る分は0でpaddingしておく。
							n = 0;
						}

						switch (out_.element_size)
						{
						case 1: *(s8* )dst = (s8 )n; break;
						case 2: *(s16*)dst = (s16)n; break;
						case 4: *(s32*)dst = (s32)n; break;
						case 8: *(s64*)dst = (s64)n; break;
						default:
							UNREACHABLE;
						}

						src += in_.element_size;
						dst += out_.element_size;
					}
				};
				switch (in_.feature)
				{
				case KK:
					for(u64 k1=0;k1 < output.sq_nb ; ++k1)
						for (u64 k2 = 0; k2 < output.sq_nb; ++k2)
						{
							u64 input_index = (k1)* input.sq_nb + (k2);
							u64 output_index = (k1)* output.sq_nb + (k2);
							conv((u8*)in_ptr + input_index * input_feature_size , (u8*)out_ptr + output_index * output_feature_size);
						}
					break;

				case KKP:
					for (u64 k1 = 0; k1 < output.sq_nb; ++k1)
						for (u64 k2 = 0; k2 < output.sq_nb; ++k2)
							for (u64 p1 = 0; p1< output.fe_end;++p1)
							{
								// mapが指定されていれば、input側のmap[p1]を参照する。
								u64 input_p1 = map == nullptr ? p1 : map->at(p1);
								u64 input_index  = ((k1)* input.sq_nb  + (k2)) * input.fe_end  + input_p1;
								u64 output_index = ((k1)* output.sq_nb + (k2)) * output.fe_end +       p1;
								conv((u8*)in_ptr + input_index * input_feature_size, (u8*)out_ptr + output_index * output_feature_size);
							}
					break;

				// --- ここ以下のコードはテストしていないので合ってるかどうかわからん…。

				case KPP:
					for (u64 k1 = 0; k1 < output.sq_nb; ++k1)
						for (u64 p1 = 0; p1 < output.fe_end; ++p1)
						{
							u64 input_p1 = map == nullptr ? p1 : map->at(p1);
							for (u64 p2 = 0; p2 < output.fe_end; ++p2)
							{
								u64 input_p2 = map == nullptr ? p2 : map->at(p2);
								u64 input_index  = ((k1)* input.fe_end  + (input_p1)) * input.fe_end  + input_p2;
								u64 output_index = ((k1)* output.fe_end + (      p1)) * output.fe_end +       p2;
								conv((u8*)in_ptr + input_index * input_feature_size, (u8*)out_ptr + output_index * output_feature_size);
							}
						}
					break;

				case PP:
					for (u64 p1 = 0; p1 < output.fe_end; ++p1)
					{
						u64 input_p1 = map == nullptr ? p1 : map->at(p1);
						for (u64 p2 = 0; p2 < output.fe_end; ++p2)
						{
							u64 input_p2 = map == nullptr ? p2 : map->at(p2);
							u64 input_index =  (input_p1) * input.fe_end  + input_p2;
							u64 output_index = (      p1) * output.fe_end + p2;
							conv((u8*)in_ptr + input_index * input_feature_size, (u8*)out_ptr + output_index * output_feature_size);
						}
					}
					break;

				case KKPP:
					for (u64 k1 = 0; k1 < output.sq_nb; ++k1)
						for (u64 k2 = 0; k2 < output.sq_nb; ++k2)
							for (u64 p1 = 0; p1 < output.fe_end; ++p1)
							{
								u64 input_p1 = map == nullptr ? p1 : map->at(p1);
								for (u64 p2 = 0; p2 < output.fe_end; ++p2)
								{
									u64 input_p2 = map == nullptr ? p2 : map->at(p2);
									u64 input_index  = ((k1*input.sq_nb  + k2) * input.fe_end  + (input_p1)) * input.fe_end  + input_p2;
									u64 output_index = ((k1*output.sq_nb + k2) * output.fe_end + (      p1)) * output.fe_end +       p2;
									conv((u8*)in_ptr + input_index * input_feature_size, (u8*)out_ptr + output_index * output_feature_size);
								}
							}
					break;

				case KPPP:
					for (u64 k1 = 0; k1 < output.sq_nb; ++k1)
						for (u64 p1 = 0; p1 < output.fe_end; ++p1)
						{
							u64 input_p1 = map == nullptr ? p1 : map->at(p1);
							for (u64 p2 = 0; p2 < output.fe_end; ++p2)
							{
								u64 input_p2 = map == nullptr ? p2 : map->at(p2);
								for (u64 p3 = 0; p3 < output.fe_end; ++p3)
								{
									u64 input_p3 = map == nullptr ? p3 : map->at(p3);
									u64 input_index  = (((k1)* input.fe_end  + (input_p1)) * input.fe_end  + input_p2)*input.fe_end  + input_p3;
									u64 output_index = (((k1)* output.fe_end + (      p1)) * output.fe_end +       p2)*output.fe_end +       p3;
									conv((u8*)in_ptr + input_index * input_feature_size, (u8*)out_ptr + output_index * output_feature_size);
								}
							}
						}
					break;
				}

				// 4) 出力先がファイルなら、出力バッファの内容を書き出す

				if (out_.file_or_memory.ptr == nullptr)
				{
					std::ofstream ofs(out_.file_or_memory.filename, std::ios::binary);
					if (ofs) ofs.write(reinterpret_cast<char*>(out_ptr), output_block_size);
					else
					{
						std::cout << "info string write file error , file = " << out_.file_or_memory.filename << std::endl;
						return false;
					}
				}
			}
		}

		return true;
	}
}
