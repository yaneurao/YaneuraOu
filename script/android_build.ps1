Set-Location (Join-Path $PSScriptRoot ..);
@(
  @{
    Target = "YANEURAOU_ENGINE_KPPT";
    Dir = ".\build\android\kppt";
  };
  @{
    Target = "YANEURAOU_ENGINE_KPP_KKPT";
    Dir = ".\build\android\kpp-kkpt";
  };
  @{
    Target = "YANEURAOU_ENGINE_NNUE_HALFKP256";
    Dir = ".\build\android\nnue_halfkp_256";
  };
  @{
    Target = "YANEURAOU_ENGINE_NNUE_KP256";
    Nnue = "KP256";
    Dir = ".\build\android\nnue_k_p_256";
  };
  @{
    Target = "YANEURAOU_ENGINE_MATERIAL";
    Dir = ".\build\android\material";
  };
  @{
    Target = "MATE_ENGINE";
    Dir = ".\build\android\mate";
  };
)|
ForEach-Object{

$Target = $_.Target;
$Dir = $_.Dir;
$Jobs = $env:NUMBER_OF_PROCESSORS;

"`n# Build $Target to $Dir"|Out-Host;

if(-not (Test-Path $Dir)){
  "`n* Make Directory"|Out-Host;
  New-Item $Dir -ItemType Directory -Force;
}

"`n* Clean Build"|Out-Host;
ndk-build.cmd clean ENGINE_TARGET=$Target;

"`n* Build Binary"|Out-Host;
$log = $null;
ndk-build.cmd ENGINE_TARGET=$Target NNUE_EVAL_ARCH=$($_.Nnue) -j $Jobs|Tee-Object -Variable log;
$log|Out-File -Encoding utf8 -Force (Join-Path $Dir "build.log");

"`n* Copy Binary"|Out-Host;
Get-ChildItem .\libs -File -Recurse|
ForEach-Object{
  Copy-Item $_.PSPath -Destination $Dir -Force -PassThru;
};

"`n* Clean Build"|Out-Host;
ndk-build.cmd clean ENGINE_TARGET=$Target;

}
