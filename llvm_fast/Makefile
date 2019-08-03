COMPILER=clang++ -std=c++14 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fpermissive -DNDEBUG -stdlib=libstdc++ -Wno-unused-parameter -D_LINUX -DUNICODE -DNO_EXCEPTIONS -DUSE_AVX2 -mbmi -mbmi2 -mavx2 -march=corei7-avx -DUSE_MAKEFILE -DYANEURAOU_ENGINE_NNUE

main.bc: ../main.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
types.bc: ../types.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
bitboard.bc: ../bitboard.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
misc.bc: ../misc.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
movegen.bc: ../movegen.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
position.bc: ../position.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
usi.bc: ../usi.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
usi_option.bc: ../usi_option.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
thread.bc: ../thread.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
tt.bc: ../tt.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
movepick.bc: ../movepick.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
timeman.bc: ../timeman.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
apery_book.bc: ../extra/book/apery_book.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
book.bc: ../extra/book/book.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
makebook2019.bc: ../extra/book/makebook2019.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
bitop.bc: ../extra/bitop.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
long_effect.bc: ../extra/long_effect.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
mate1ply_with_effect.bc: ../extra/mate/mate1ply_with_effect.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
mate1ply_without_effect.bc: ../extra/mate/mate1ply_without_effect.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
mate_n_ply.bc: ../extra/mate/mate_n_ply.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
benchmark.bc: ../extra/benchmark.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
test_cmd.bc: ../extra/test_cmd.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
see.bc: ../extra/see.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
sfen_packer.bc: ../extra/sfen_packer.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
kif_convert_tools.bc: ../extra/kif_converter/kif_convert_tools.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
evaluate_bona_piece.bc: ../eval/evaluate_bona_piece.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
evaluate.bc: ../eval/evaluate.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
evaluate_io.bc: ../eval/evaluate_io.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
evaluate_mir_inv_tools.bc: ../eval/evaluate_mir_inv_tools.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
learner.bc: ../learn/learner.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
learning_tools.bc: ../learn/learning_tools.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
multi_think.bc: ../learn/multi_think.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
evaluate_nnue.bc: ../eval/nnue/evaluate_nnue.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
evaluate_nnue_learner.bc: ../eval/nnue/evaluate_nnue_learner.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
nnue_test_command.bc: ../eval/nnue/nnue_test_command.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
k.bc: ../eval/nnue/features/k.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
p.bc: ../eval/nnue/features/p.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
half_kp.bc: ../eval/nnue/features/half_kp.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
half_relative_kp.bc: ../eval/nnue/features/half_relative_kp.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@
yaneuraou-search.bc: ../engine/yaneuraou-engine/yaneuraou-search.cpp
	$(COMPILER) -emit-llvm -c $^ -o $@

big_one.bc: main.bc types.bc bitboard.bc misc.bc movegen.bc position.bc usi.bc usi_option.bc thread.bc tt.bc movepick.bc timeman.bc apery_book.bc book.bc makebook2019.bc bitop.bc long_effect.bc mate1ply_with_effect.bc mate1ply_without_effect.bc mate_n_ply.bc benchmark.bc test_cmd.bc see.bc sfen_packer.bc kif_convert_tools.bc evaluate_bona_piece.bc evaluate.bc evaluate_io.bc evaluate_mir_inv_tools.bc learner.bc learning_tools.bc multi_think.bc evaluate_nnue.bc evaluate_nnue_learner.bc nnue_test_command.bc k.bc p.bc half_kp.bc half_relative_kp.bc yaneuraou-search.bc
	llvm-link $^ -o $@

optimized.bc: big_one.bc
	opt $^ -internalize -internalize-public-api-list=main -globaldce -O3 -o $@

YaneuraOu-by-clang: optimized.bc
	clang++ -o $@ $^ -Wl,-s -lpthread -v -std=c++14 -fno-exceptions -fno-rtti -Wextra -Ofast -MMD -MP -fpermissive -DNDEBUG -stdlib=libstdc++ -Wno-unused-parameter -D_LINUX -DUNICODE -DNO_EXCEPTIONS -DUSE_AVX2 -mbmi -mbmi2 -mavx2 -march=corei7-avx -DUSE_MAKEFILE -DYANEURAOU_ENGINE_NNUE
	cp $@ ..

clean:
	find -not -name Makefile -type f -exec rm {} \;
