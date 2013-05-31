# COMP2550 final project MAKEFILE
# Jimmy Lin <u5223173@anu.edu.au> 
#
#######################################################################
# DO NOT EDIT THIS MAKEFILE UNLESS YOU KNOW WHAT YOU ARE DOING. 
#######################################################################	

DRWN_PATH := $(shell pwd)/../..
PROJ_PATH := $(shell pwd)

#######################################################################
# define project parameters here
#######################################################################

-include $(DRWN_PATH)/make.mk

#######################################################################
# add project source files here
#######################################################################

APP_SRC = trainModel.cpp  getFeatureMaps.cpp  getLabelledImages.cpp testModel.cpp

#######################################################################

APP_PROG_NAMES = $(APP_SRC:.cpp=)
APP_OBJ = $(APP_SRC:.cpp=.o)
PROJ_DEP = ${APP_OBJ:.o=.d}

.PHONY: clean
.PRECIOUS: $(APP_OBJ)

all: ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}

#
# applications
$(BIN_PATH)/%: %.o $(LIBDRWN)
	${CCC} $*.o -o $(@:.o=) $(LFLAGS)

# darwin libraries
$(LIBDRWN):
	@echo "** YOU NEED TO MAKE THE DARWIN LIBRARIES FIRST **"
	false

# dependencies and object files
%.o : %.cpp
	${CCC} -g -MM -MF $(subst .o,.d,$@) -MP -MT $@ $(CFLAGS) $<
	${CCC} -g ${CFLAGS} -c $< -o $@

# clear
clean:
	-rm $(APP_OBJ)
	-rm ${addprefix ${BIN_PATH}/,$(APP_PROG_NAMES)}
	-rm ${PROJ_DEP}
	-rm *~

-include ${PROJ_DEP}
