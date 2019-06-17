Param(
  [String[]]$Compiler,
  [String[]]$Edition,
  [String[]]$Target
)
$loc = Get-Location;
Set-Location (Join-Path $PSScriptRoot ..\source);
# msys2_shell.cmd -msys2 -defterm -no-start -l -c 'pacboy -Syuu --needed --noconfirm --noprogressbar toolchain:m clang:m openblas:m base-devel: msys2-devel:';
@(
  @{
    COMPILER = 'clang++';
    TGCOMPILER = '-msys2-clang++';
  };
  @{
    COMPILER = 'g++';
    TGCOMPILER = '-msys2-g++';
  };
)|
Where-Object{
  $_Compiler = $_;
  (-not $Compiler) -or ($Compiler|Where-Object{$_Compiler.COMPILER -like $_});
}|
ForEach-Object{@(
  @{
    COMPILER = $_.COMPILER;
    BUILDDIR = 'kppt';
    EDITION = 'YANEURAOU_ENGINE_KPPT';
    TARGET = "YaneuraOu-kppt$($_.TGCOMPILER)";
    TGTAIL = @(
      'icelake';'cascadelake';'avx512';'avx2';'sse42';'sse2';'nosse';
      'tournament-icelake';'tournament-cascadelake';'tournament-avx512';'tournament-avx2';'tournament-sse42';
      'evallearn-icelake';'evallearn-cascadelake';'evallearn-avx512';'evallearn-avx2';'evallearn-sse42';
    );
  };
  @{
    COMPILER = $_.COMPILER;
    BUILDDIR = 'kpp-kkpt';
    EDITION = 'YANEURAOU_ENGINE_KPP_KKPT';
    TARGET = "YaneuraOu-kpp_kkpt$($_.TGCOMPILER)";
    TGTAIL = @(
      'icelake';'cascadelake';'avx512';'avx2';'sse42';'sse2';'nosse';
      'tournament-icelake';'tournament-cascadelake';'tournament-avx512';'tournament-avx2';'tournament-sse42';
      'evallearn-icelake';'evallearn-cascadelake';'evallearn-avx512';'evallearn-avx2';'evallearn-sse42';
    );
  };
  @{
    COMPILER = $_.COMPILER;
    BUILDDIR = 'material';
    EDITION = 'YANEURAOU_ENGINE_MATERIAL';
    TARGET = "YaneuraOu-material$($_.TGCOMPILER)";
    TGTAIL = @(
      'icelake';'cascadelake';'avx512';'avx2';'sse42';'sse2';'nosse';
      'tournament-icelake';'tournament-cascadelake';'tournament-avx512';'tournament-avx2';'tournament-sse42';
    );
  };
  @{
    COMPILER = $_.COMPILER;
    BUILDDIR = 'nnue-halfkp_256';
    EDITION = 'YANEURAOU_ENGINE_NNUE_HALFKP256';
    TARGET = "YaneuraOu-nnue-halfkp_256$($_.TGCOMPILER)";
    TGTAIL = @(
      'icelake';'cascadelake';'avx512';'avx2';'sse42';'sse2';'nosse';
      'tournament-icelake';'tournament-cascadelake';'tournament-avx512';'tournament-avx2';'tournament-sse42';
      'evallearn-icelake';'evallearn-cascadelake';'evallearn-avx512';'evallearn-avx2';'evallearn-sse42';
    );
  };
  @{
    COMPILER = $_.COMPILER;
    BUILDDIR = 'nnue_k_p_256';
    EDITION = 'YANEURAOU_ENGINE_NNUE_KP256';
    TARGET = "YaneuraOu-nnue-k_p_256$($_.TGCOMPILER)";
    TGTAIL = @(
      'icelake';'cascadelake';'avx512';'avx2';'sse42';'sse2';'nosse';
      'tournament-icelake';'tournament-cascadelake';'tournament-avx512';'tournament-avx2';'tournament-sse42';
      'evallearn-icelake';'evallearn-cascadelake';'evallearn-avx512';'evallearn-avx2';'evallearn-sse42';
    );
  };
  @{
    COMPILER = $_.COMPILER;
    BUILDDIR = 'mate';
    EDITION = 'MATE_ENGINE';
    TARGET = "YaneuraOu-mate$($_.TGCOMPILER)";
    TGTAIL = @(
      'icelake';'cascadelake';'avx512';'avx2';'sse42';'sse2';'nosse';
      'tournament-icelake';'tournament-cascadelake';'tournament-avx512';'tournament-avx2';'tournament-sse42';
    );
  };
)}|
Where-Object{
  $_Edition = $_.EDITION;
  (-not $Edition) -or ($Edition|Where-Object{$_Edition -like $_});
}|
ForEach-Object{
  $_OS = 'Windows_NT';
  $_MAKE = 'mingw32-make';
  $_MAKEFILE = 'Makefile';
  $_Jobs = $env:NUMBER_OF_PROCESSORS;
  $_Compiler = $_.COMPILER;
  $_BuildDir = Join-Path '../build/windows/' $_.BUILDDIR;
  $_Edition = $_.EDITION;
  $_Target = $_.Target;
  if(-not (Test-Path $_BuildDir)){
    New-Item $_BuildDir -ItemType Directory -Force;
  }
  $_.TGTAIL|
  Where-Object{
    $_TgTail = $_;
    (-not $Target) -or ($Target|Where-Object{$_TgTail -like $_});
  }|
  ForEach-Object{
    if ($_ -ne 'nosse') {
      Set-Item Env:MSYSTEM 'MINGW64';
    } else {
      Set-Item Env:MSYSTEM 'MINGW32';
    }
    msys2_shell.cmd -here -defterm -no-start -l -c "$_MAKE -f $_MAKEFILE clean YANEURAOU_EDITION=$_Edition";
    $log = $null;
    msys2_shell.cmd -here -defterm -no-start -l -c "nice $_MAKE -f $_MAKEFILE -j$_Jobs $_ YANEURAOU_EDITION=$_Edition COMPILER=$_Compiler OS=$_OS 2>&1"|Tee-Object -Variable log
    $log|Out-File -Encoding utf8 -Force (Join-Path $_BuildDir "$_Target-$_.log");
    Copy-Item YaneuraOu-by-gcc.exe (Join-Path $_BuildDir "$_Target-$_.exe") -Force;
  };
  msys2_shell.cmd -here -defterm -no-start -l -c "$_MAKE -f $_MAKEFILE clean YANEURAOU_EDITION=$_Edition";
};
Set-Location $loc;
