# สร้างโปรแกรม .exe แล้วบีบเป็น zip ให้แจก
# ใช้: .\build.ps1            -> dist\Cullo\ (ปิดโปรแกรมที่เปิดค้างอยู่ก่อน ไม่งั้นไฟล์ถูกล็อก)
#      .\build.ps1 -Out dist2 -> สร้างลงโฟลเดอร์อื่นได้ ถ้ายังเปิดโปรแกรมเวอร์ชันเก่าใช้งานอยู่
param([string]$Out = "dist")
Set-Location $PSScriptRoot

function Run($cmd) {
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "ล้มเหลว: $($cmd -join ' ')" }
}

Run @("python", "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "pyinstaller")

$work = if ($Out -eq "dist") { "build" } else { "build_$Out" }
Run @("python", "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--onedir",
      "--distpath", $Out, "--workpath", $work,
      "--name", "Cullo",
      "--add-data", "models;models",
      "--collect-all", "mediapipe",
      "--collect-all", "customtkinter",
      "--exclude-module", "torch", "--exclude-module", "torchvision", "--exclude-module", "tensorflow",
      "--exclude-module", "sklearn", "--exclude-module", "scipy", "--exclude-module", "matplotlib",
      "app.py")

# ตัวสแกนไวรัสอาจล็อกไฟล์ที่เพิ่งสร้างชั่วคราว จึงลองบีบ zip ซ้ำได้ 3 ครั้ง
$zip = "$Out\Cullo.zip"
for ($i = 1; $i -le 3; $i++) {
    try {
        Compress-Archive -Force -Path "$Out\Cullo" -DestinationPath $zip -ErrorAction Stop
        break
    } catch {
        if ($i -eq 3) { throw }
        Start-Sleep -Seconds 8
    }
}
Write-Host "เสร็จแล้ว: $Out\Cullo\Cullo.exe และ $zip"


