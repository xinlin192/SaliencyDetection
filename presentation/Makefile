# APPLICATION MAKEFILE
# Jimmy Lin <u5223173@anu.edu.au> 

#######################################################################
# DO NOT EDIT THIS MAKEFILE UNLESS YOU KNOW WHAT YOU ARE DOING. 
#######################################################################	

all: pdfGenerator cleanOthers

pdfGenerator:
	pdflatex ./presentation.tex 
	pdflatex ./presentation.tex

cleanOthers:
	rm -f ./presentation.aux ./presentation.log ./presentation.nav ./presentation.out ./presentation.snm
	rm -f ./presentation.toc 

clean: cleanOthers
