@echo off
setlocal enabledelayedexpansion

@REM rem Set the path to the tests directory
@REM set "test_directory=.\tests"

@REM rem Loop through all Python files in the tests directory
@REM for %%i in ("%test_directory%\*.py") do (
@REM     echo Running pytest on "%%i"
@REM     python3 -m pytest --log-cli-level=WARNING --full-trace -rP "%%i"
@REM )

python3 -m pytest --log-cli-level=WARNING -rP .\tests\check_models_install.py || exit /b 1
python3 -m pytest --log-cli-level=WARNING -rP .\tests\check_models_run.py || exit /b 1
python3 -m pytest --log-cli-level=WARNING -rP .\tests\check_vpu.py || exit /b 1

endlocal
