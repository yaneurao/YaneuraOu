# 評価関数ファイルを読み込んで、C++の文字列literalに変換する。[2023/12/13]
# ここで変換した.cppをやねうら王のプロジェクトに追加してビルドする。
# ビルドする時に、EVAL_EMBEDDINGをdefineしてビルドする。

def binary_to_cpp_literal(input_file:str, output_file:str):

    # バイナリファイルを読み込む
    with open(input_file, 'rb') as file:
        binary_data = file.read()

    # ファイルサイズを取得
    file_size:int = len(binary_data)

    # 16進数の文字列に変換
    hex_string = '\\x' + '\\x'.join(f'{byte:02x}' for byte in binary_data)

    with open(output_file, 'w') as file:
        file.write('#include <cstddef>\n')
        file.write('extern const char* gEmbeddedNNUEData;\n')
        file.write('extern const std::size_t gEmbeddedNNUESize;\n\n')
        file.write(f'const char* gEmbeddedNNUEData = "{hex_string}";\n')
        file.write(f'const std::size_t gEmbeddedNNUESize = {file_size};\n')

def binary_to_cpp_literal2(input_file:str, output_file:str):

    # ファイルを開き、ヘッダと宣言を書き込む
    with open(output_file, 'w') as outfile:

        outfile.write('#include <cstddef>\n')
        outfile.write('extern const char* gEmbeddedNNUEData;\n')
        outfile.write('extern const std::size_t gEmbeddedNNUESize;\n\n')

        outfile.write(f'const char* gEmbeddedNNUEData = "')

        # バイナリファイルを開いて、ストリームで読み込みながら処理
        file_size = 0
        with open(input_file, 'rb') as infile:
            while True:
                chunk = infile.read(1024)  # 1024バイトずつ読み込む
                if not chunk:
                    break
                # 16進数の文字列に変換して出力
                hex_string = '\\x' + '\\x'.join(f'{byte:02x}' for byte in chunk)
                outfile.write(hex_string)
                file_size += len(chunk)

        outfile.write('";\n')
        outfile.write(f'const std::size_t gEmbeddedNNUESize = {file_size};\n')

# 例：'nn.bin'を読み込み、'nnue_bin.cpp'に出力
binary_to_cpp_literal2('nn.bin', 'nnue_bin.cpp')
