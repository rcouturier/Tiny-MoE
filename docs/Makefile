# Build the paper: make        -> paper.pdf
#                  make figs   -> regenerate figures from logged numbers
#                  make clean  -> remove LaTeX aux files
paper.pdf: paper.tex references.bib $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

figs:
	python figures/make_figures.py

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out
