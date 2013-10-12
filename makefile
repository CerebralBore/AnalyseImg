# user dirs
SRC_DIR         = ./src/
OBJ_DIR         = ./obj/
DEP_DIR         = ./dep/
BIN_DIR         = ./

# bin name
BIN             = ./bin/AAI

# additionnal lib and includes dir
LIB_DIR         = ./lib/
INC_DIR         = ./include/

# compile commands
CPP      = g++
LD      = g++

# flags and libs
CPPFLAGS        = -I$(INC_DIR) -g -Wall -pipe 
CFLAGS          = -I$(INC_DIR) -g -Wall -pipe 
LDFLAGS = -L$(LIB_DIR) -lm `pkg-config --cflags --libs opencv`

SRCS_C          = $(wildcard $(SRC_DIR)*.cpp) 

#Liste des dépendances .cpp, ==> .d
DEPS    = $(SRCS_C:$(SRC_DIR)%.cpp=$(DEP_DIR)%.d)

#Liste des objets : .cpp ==> .o
OBJS    = $(SRCS_C:$(SRC_DIR)%.cpp=$(OBJ_DIR)%.o)

#Rules
all: $(BIN_DIR)/$(BIN)

#To executable
$(BIN_DIR)/$(BIN): $(OBJS)
	$(LD) $+ -o $@ $(LDFLAGS)
        
#To Objets
$(OBJ_DIR)%.o: $(SRC_DIR)%.cpp
	$(CPP) $(CFLAGS) -o $@ -c $<

#Gestion des dépendances
$(DEP_DIR)%.d: $(SRC_DIR)%.cpp
	$(CPP) $(CFLAGS) -MM -MD -o $@ $<

-include $(DEPS)

.PHONY: clean distclean

run: $(BIN_DIR)/$(BIN)
	$(BIN_DIR)/$(BIN)

gdb: $(BIN_DIR)/$(BIN)
	gdb $(BIN_DIR)/$(BIN)

valgrind: $(BIN_DIR)/$(BIN)
	valgrind $(BIN_DIR)/$(BIN)

clean:
	rm -f $(OBJ_DIR)*.o $(SRC_DIR)*~ $(DEP_DIR)*.d *~ $(BIN_DIR)/$(BIN)

distclean: clean
	rm -f $(BIN_DIR)/$(BIN) 
