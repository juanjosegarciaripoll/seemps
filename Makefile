all:
	python -c "import exportnb; import glob; exportnb.export_notebooks(glob.glob('*.ipynb'),verbose=True); quit()"

clean:
	rm -rf seeq

cleanup:
	for in *.ipynb; do jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$i"; done

