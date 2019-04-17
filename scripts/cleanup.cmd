set PYTHONIOENCODING=utf8
for %%i in (*.ipynb) do python ./scripts/ipynb_output_filter.py < "%%i" > foo.txt && move foo.txt "%%i"
