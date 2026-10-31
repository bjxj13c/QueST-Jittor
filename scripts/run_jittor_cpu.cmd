@echo off
setlocal
rem Use an existing compiler; do not download or install a system compiler/CUDA.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "cc_path=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
set "cc_flags= -DNOMINMAX"
"%~dp0..\.venv-jittor\Scripts\python.exe" "%~dp0jittor_cpu_entry.py" %*
exit /b %errorlevel%
