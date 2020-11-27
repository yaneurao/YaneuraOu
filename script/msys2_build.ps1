Param(
  [String[]]$Compiler,
  [String[]]$Edition,
  [String[]]$Target,
  [String[]]$Cpu
)
Push-Location (Join-Path $PSScriptRoot ..\source);
# msys2_shell.cmd -msys2 -defterm -no-start -l -c 'pacboy -Syuu --needed --noconfirm --noprogressbar toolchain:m clang:m openblas:m base-devel: msys2-devel:';
$TGCPUS = @('ZEN3';'ZEN2';'ZEN1';'AVX512VNNI';'AVX512';'AVXVNNI';'AVX2';'SSE42';'SSE41';'SSSE3';'SSE2';'NO_SSE';'OTHER';);
$TGCOMPILERS = @('clang++';'g++';);
@(
  @{
    BUILDDIR = 'NNUE';
    EDITION = 'YANEURAOU_ENGINE_NNUE';
    BUILDNAME = 'YaneuraOu_NNUE';
    TARGET = @('evallearn';'normal';'tournament';'gensfen';);
  };
  @{
    BUILDDIR = 'NNUE_KPE9';
    EDITION = 'YANEURAOU_ENGINE_NNUE_HALFKPE9';
    BUILDNAME = 'YaneuraOu_NNUE_KPE9';
    TARGET = @('evallearn';'normal';'tournament';'gensfen';);
  };
  @{
    BUILDDIR = 'NNUE_KP256';
    EDITION = 'YANEURAOU_ENGINE_NNUE_KP256';
    BUILDNAME = 'YaneuraOu_NNUE_KP256';
    TARGET = @('evallearn';'normal';'tournament';'gensfen';);
  };
  @{
    BUILDDIR = 'KPPT';
    EDITION = 'YANEURAOU_ENGINE_KPPT';
    BUILDNAME = 'YaneuraOu_KPPT';
    TARGET = @('evallearn';'normal';'tournament';'gensfen';);
  };
  @{
    BUILDDIR = 'KPP_KKPT';
    EDITION = 'YANEURAOU_ENGINE_KPP_KKPT';
    BUILDNAME = 'YaneuraOu_KPP_KKPT';
    TARGET = @('evallearn';'normal';'tournament';'gensfen';);
  };
  @{
    BUILDDIR = 'KOMA';
    EDITION = 'YANEURAOU_ENGINE_MATERIAL';
    BUILDNAME = 'YaneuraOu_KOMA';
    TARGET = @('normal';'tournament';'gensfen';);
  };
  @{
    BUILDDIR = 'MATE';
    EDITION = 'MATE_ENGINE';
    BUILDNAME = 'tanuki_MATE';
    TARGET = @('normal';'tournament';'gensfen';);
  };
  @{
    BUILDDIR = 'USER';
    EDITION = 'USER_ENGINE';
    BUILDNAME = 'user';
    TARGET = @('normal';'tournament';'gensfen';);
  };
)|
Where-Object{
  $_Edition = $_.EDITION;
  (-not $Edition) -or ($Edition|Where-Object{$_Edition -like $_});
}|
ForEach-Object{
  $_Os = 'Windows_NT';
  $_Make = 'mingw32-make';
  $_Makefile = 'Makefile';
  $_Jobs = $env:NUMBER_OF_PROCESSORS;
  $_BuildDir = Join-Path '../build/windows/' $_.BUILDDIR;
  $_Edition = $_.EDITION;
  $_BuildName = $_.BUILDNAME;
  if(-not (Test-Path $_BuildDir)){
    New-Item $_BuildDir -ItemType Directory -Force;
  }
  $_.TARGET|Where-Object{ $_Target = $_; (-not $Target) -or ($Target|Where-Object{$_Target -like $_}); }|ForEach-Object{ $_Target = $_;
  $TGCOMPILERS|Where-Object{ $_Compiler = $_; (-not $Compiler) -or ($Compiler|Where-Object{$_Compiler -like $_}); }|ForEach-Object{ $_Compiler = $_;
  $TGCPUS|Where-Object{ $_Cpu = $_; ((-not $Cpu) -or ($Cpu|Where-Object{$_Cpu -like $_})) -and ($_Cpu -ne 'NO_SSE' -or $_Target -ne 'evallearn'); }|ForEach-Object{ $_Cpu = $_;
      Set-Item Env:MSYSTEM $(if ($_Cpu -ne 'NO_SSE') { 'MINGW64' } else { 'MINGW32' });
      msys2_shell.cmd -here -defterm -no-start $(if ($_Cpu -ne 'NO_SSE') { '-mingw64' } else { '-mingw32' }) -lc "$_Make -f $_Makefile clean YANEURAOU_EDITION=$_Edition";
      $log = $null;
      msys2_shell.cmd -here -defterm -no-start $(if ($_Cpu -ne 'NO_SSE') { '-mingw64' } else { '-mingw32' }) -lc "nice $_Make -f $_Makefile -j$_Jobs $_Target YANEURAOU_EDITION=$_Edition COMPILER=$_Compiler OS=$_Os TARGET_CPU=$_Cpu 2>&1"|Tee-Object -Variable log;
      $log|Out-File -Encoding utf8 -Force (Join-Path $_BuildDir "$_BuildName-$_Target-$_Compiler-$($_Cpu.ToLower()).log");
      Copy-Item YaneuraOu-by-gcc.exe (Join-Path $_BuildDir "$_BuildName-$_Target-$_Compiler-$($_Cpu.ToLower()).exe") -Force;
  }}};
  msys2_shell.cmd -here -defterm -no-start $(if ($_Cpu -ne 'NO_SSE') { '-mingw64' } else { '-mingw32' }) -lc "$_Make -f $_Makefile clean YANEURAOU_EDITION=$_Edition";
};
Pop-Location;
