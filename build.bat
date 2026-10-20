@echo off
rem Build gemm_bench with nvcc + MSVC on Windows.  Usage: build.bat [sm_XX]
setlocal
set "ARCHFLAG=-arch=native"
if not "%~1"=="" set "ARCHFLAG=-arch=%~1"
cd /d "%~dp0"

where cl >nul 2>nul
if not errorlevel 1 goto build

rem cl.exe isn't on PATH: locate Visual Studio / Build Tools and load its x64 environment.
rem (Kept outside ( ) blocks: the ")" in "Program Files (x86)" would end the block early.)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto nomsvc
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR goto nomsvc
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul

:build
if not exist build mkdir build
nvcc -O3 -std=c++17 %ARCHFLAG% -lineinfo -Xptxas -v src\gemm_bench.cu -lcublas -o build\gemm_bench.exe
exit /b %errorlevel%

:nomsvc
echo MSVC not found. Install Visual Studio Build Tools with "Desktop development with C++".
exit /b 1
