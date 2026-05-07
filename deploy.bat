@echo off
cd /d "d:\Lập trình\OpenClaw"
git add .
git commit -m "Fix: Move tkinter import to GUI class for Render compatibility"
git push origin master
echo Deploy script completed
pause
