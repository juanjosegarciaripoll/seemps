@echo off
if "%1" == "all" (
    python -c "import exportnb; import glob; exportnb.export_notebooks(glob.glob('*.ipynb'),verbose=True); quit()"
)
if "%1" == "clean" (
    rmdir /S /Q mps
)
if "%1" == "docs" (
    cd docs
    make html
    cd ..
)
if "%1" == "cleanup" (
    for %%i in (*.ipynb); do jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "%%i"
)
