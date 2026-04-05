@echo off
echo Building Meta OpenEnv C++ Project...
g++ tests/simulation.cpp env/NegotiationEnv.cpp opponent/Opponent.cpp -I. -std=c++17 -o test_sim.exe

if %ERRORLEVEL% EQU 0 (
    echo Build Successful! Running simulation...
    echo =======================================
    .\test_sim.exe
    echo =======================================
) else (
    echo Build Failed! Make sure g++ is installed and in your PATH.
)
pause
