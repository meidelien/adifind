@echo off
setlocal EnableExtensions
REM AdiFind Installation Script (Windows)
REM Usage: install.bat

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Failed to switch to the AdiFind repository root.
    exit /b 1
)

echo ========================================
echo  AdiFind Installation
echo ========================================
echo.

REM Check for conda
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: conda not found. Please install Miniconda or Anaconda first:
    echo   https://docs.anaconda.com/miniconda/
    goto :fail
)

REM Check for git (required by environment.yml for Detectron2)
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: git not found. Detectron2 is installed from a Git URL in environment.yml.
    echo   Install Git for Windows: https://git-scm.com/download/win
    goto :fail
)

echo Creating conda environment 'adifind'...
call conda env create -f environment.yml
if errorlevel 1 (
    echo.
    echo ERROR: Environment creation failed. See errors above.
    echo If the failure occurred while building Detectron2 on Windows:
    echo   1. Install Git for Windows.
    echo   2. Install Visual Studio 2022 Build Tools or Visual Studio 2022.
    echo   3. Select "Desktop development with C++".
    goto :fail
)

echo.
echo Running post-install validation from the repository root...
call conda run -n adifind python -c "import torch, detectron2, openslide; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available(), '| Detectron2', detectron2.__version__, '| OpenSlide', openslide.__library_version__)"
if errorlevel 1 (
    echo.
    echo ERROR: Base import validation failed.
    echo If this is an OpenSlide DLL issue on Windows:
    echo   1. Download OpenSlide from https://openslide.org/download/
    echo   2. Extract it to C:\OpenSlide
    echo   3. Set OPENSLIDE_PATH=C:\OpenSlide\bin
    echo   4. Rerun the validation commands shown below.
    goto :fail
)

call conda run -n adifind python -c "import openslide; slide = openslide.OpenSlide('example_data/K106942.svs'); print(slide.dimensions); slide.close()"
if errorlevel 1 (
    echo.
    echo ERROR: Bundled slide readability validation failed.
    echo Expected command:
    echo   conda run -n adifind python -c "import openslide; slide = openslide.OpenSlide('example_data/K106942.svs'); print^(slide.dimensions^); slide.close^(^)"
    goto :fail
)

call conda run -n adifind adifind --help >nul
if errorlevel 1 (
    echo.
    echo ERROR: Installed CLI validation failed: adifind --help
    goto :fail
)

call conda run -n adifind python code/main.py example_data/K106942.svs --tissue_guidance --dry_run
if errorlevel 1 (
    echo.
    echo ERROR: Repo-root source-tree validation failed.
    echo Expected command:
    echo   conda run -n adifind python code/main.py example_data/K106942.svs --tissue_guidance --dry_run
    goto :fail
)

echo.
echo ========================================
echo  Installation and Validation Complete!
echo ========================================
echo.
echo To get started:
echo   conda activate adifind
echo   adifind --help
echo.
echo Repo-root validation commands:
echo   conda run -n adifind adifind --help
echo   conda run -n adifind python code/main.py --help
echo   conda run -n adifind python -c "import openslide; slide = openslide.OpenSlide('example_data/K106942.svs'); print(slide.dimensions); slide.close()"
echo   conda run -n adifind adifind example_data/K106942.svs --tissue_guidance --dry_run
echo   conda run -n adifind python code/main.py example_data/K106942.svs --tissue_guidance --dry_run
echo.
echo Model downloads:
echo   Authenticated Hugging Face access downloads models automatically.
echo   Otherwise, use local canonical checkpoint files:
echo     adifind_adipocyte.pth
echo     adifind_tumor.pth
echo     adifind_tissue_guidance.pth
echo.
goto :success

:fail
popd >nul
exit /b 1

:success
popd >nul
endlocal
exit /b 0
