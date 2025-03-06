# 设置下载链接和目标路径
$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$downloadPath = "$env:TEMP\ffmpeg.zip"
$ffmpegPath = "C:\ffmpeg"

# 创建目标目录
New-Item -ItemType Directory -Force -Path $ffmpegPath

# 下载 FFmpeg
Write-Host "正在下载 FFmpeg..."
Invoke-WebRequest -Uri $ffmpegUrl -OutFile $downloadPath

# 解压文件
Write-Host "正在解压文件..."
Expand-Archive -Path $downloadPath -DestinationPath $ffmpegPath -Force

# 移动文件到正确位置
Get-ChildItem -Path "$ffmpegPath\ffmpeg-master-latest-win64-gpl" | Move-Item -Destination $ffmpegPath -Force
Remove-Item "$ffmpegPath\ffmpeg-master-latest-win64-gpl" -Force

# 添加到系统环境变量
$binPath = "$ffmpegPath\bin"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if ($currentPath -notlike "*$binPath*") {
    Write-Host "添加 FFmpeg 到系统环境变量..."
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$binPath", "Machine")
}

# 清理下载的文件
Remove-Item $downloadPath -Force

Write-Host "FFmpeg 安装完成！"
Write-Host "请重新打开命令提示符或 PowerShell 以使环境变量生效"
