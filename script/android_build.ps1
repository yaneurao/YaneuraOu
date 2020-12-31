Param(
  [String[]]$Edition
)
Push-Location (Join-Path $PSScriptRoot ..);
@(
  @{
    EDITION = "YANEURAOU_ENGINE_NNUE";
    Dir = ".\build\android\NNUE";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_NNUE_HALFKPE9";
    Nnue = "HALFKPE9";
    Dir = ".\build\android\NNUE_HALFKPE9";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_NNUE_KP256";
    Nnue = "KP256";
    Dir = ".\build\android\NNUE_KP256";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_KPPT";
    Dir = ".\build\android\KPPT";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_KPP_KKPT";
    Dir = ".\build\android\KPP_KKPT";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL";
    Dir = ".\build\android\MaterialLv1";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=2";
    Dir = ".\build\android\MaterialLv2";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=3";
    Dir = ".\build\android\MaterialLv3";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=4";
    Dir = ".\build\android\MaterialLv4";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=5";
    Dir = ".\build\android\MaterialLv5";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=6";
    Dir = ".\build\android\MaterialLv6";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=7";
    Dir = ".\build\android\MaterialLv7";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=8";
    Dir = ".\build\android\MaterialLv8";
  };
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=9";
    Dir = ".\build\android\MaterialLv9";
  };
<#
  @{
    EDITION = "YANEURAOU_ENGINE_MATERIAL MATERIAL_LEVEL=10";
    Dir = ".\build\android\MaterialLv10";
  };
#>
  @{
    EDITION = "YANEURAOU_MATE_ENGINE";
    Dir = ".\build\android\YaneuraOu_MATE";
  };
  @{
    EDITION = "TANUKI_MATE_ENGINE";
    Dir = ".\build\android\tanuki_MATE";
  };
  @{
    EDITION = "USER_ENGINE";
    Dir = ".\build\android\USER";
  };
)|
Where-Object{
  $_Edition = $_.EDITION;
  (-not $Edition) -or ($Edition|Where-Object{$_Edition -like $_});
}|
ForEach-Object{

$_Edition = $_.EDITION;
$Dir = $_.Dir;
# 並列ジョブ数が多すぎると実行バイナリが生成されない模様
# とりあえずソースファイル数より少ない数と論理プロセッサ数の小さい方にする。F*ck.
$Jobs = [Math]::Min($env:NUMBER_OF_PROCESSORS, 30);

"`n# Build $_Edition to $Dir"|Out-Host;

if(-not (Test-Path $Dir)){
  "`n* Make Directory"|Out-Host;
  New-Item $Dir -ItemType Directory -Force;
}

"`n* Clean Build"|Out-Host;
ndk-build.cmd clean YANEURAOU_EDITION=$_Edition;

"`n* Build Binary"|Out-Host;
$log = $null;
ndk-build.cmd YANEURAOU_EDITION=$_Edition NNUE_EVAL_ARCH=$($_.Nnue) V=1 -j $Jobs|Tee-Object -Variable log;
$log|Out-File -Encoding utf8 -Force (Join-Path $Dir "build.log");

"`n* Copy Binary"|Out-Host;
Get-ChildItem .\libs -File -Recurse|
ForEach-Object{
  Copy-Item $_.PSPath -Destination $Dir -Force -PassThru;
};

"`n* Clean Build"|Out-Host;
ndk-build.cmd clean YANEURAOU_EDITION=$_Edition;

}

Pop-Location;
