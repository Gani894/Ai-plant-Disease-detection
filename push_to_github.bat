@echo off
cd /d "%~dp0"
set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Git.MinGit_Microsoft.Winget.Source_8wekyb3d8bbwe\cmd;%PATH%"
echo =======================================================
echo  Pushing to https://github.com/Gani894/Ai-plant-Disease-detection.git
echo =======================================================
git push -u origin main
pause
