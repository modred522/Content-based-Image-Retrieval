# Compiler
CC = g++

# Project directories
SRC_DIR = .
INC_DIR = ../include
BIN_DIR = ../bin
THIRD_PARTY = ../third_party

# --- Path Configuration ---
INCLUDES = -I$(INC_DIR) \
           -I/usr/local/include/opencv4 \
           -I/opt/homebrew/include/opencv4 \
           -I$(THIRD_PARTY)/imgui \
           -I$(THIRD_PARTY)/imgui/backends \
           -I/usr/local/include \
           -I/opt/homebrew/include

# Library paths
LIB_DIRS = -L/usr/local/lib -L/opt/homebrew/lib

# Libraries to link
LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
GUI_LIBS = -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo

# Flags
CFLAGS = -std=c++17 -Wall $(INCLUDES)
LDFLAGS = $(LIB_DIRS) $(LIBS)
GUI_LDFLAGS = $(LIB_DIRS) $(LIBS) $(GUI_LIBS)

# Targets
TARGETS = cbir_build cbir_query cbir_gui

# ImGui sources
IMGUI_SRC = $(THIRD_PARTY)/imgui/imgui.cpp \
            $(THIRD_PARTY)/imgui/imgui_demo.cpp \
            $(THIRD_PARTY)/imgui/imgui_draw.cpp \
            $(THIRD_PARTY)/imgui/imgui_tables.cpp \
            $(THIRD_PARTY)/imgui/imgui_widgets.cpp \
            $(THIRD_PARTY)/imgui/backends/imgui_impl_glfw.cpp \
            $(THIRD_PARTY)/imgui/backends/imgui_impl_opengl3.cpp

IMGUI_OBJ = imgui.o imgui_demo.o imgui_draw.o imgui_tables.o imgui_widgets.o imgui_impl_glfw.o imgui_impl_opengl3.o

all: $(TARGETS)

# CBIR Build Tool
cbir_build: cbir_build.o feature.o distance.o cbir.o
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$@ $(LDFLAGS)

# CBIR Query Tool
cbir_query: cbir_query.o feature.o distance.o cbir.o
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$@ $(LDFLAGS)

# CBIR GUI Tool (with ImGui)
cbir_gui: cbir_gui.o feature.o distance.o cbir.o $(IMGUI_OBJ)
	$(CC) $(CFLAGS) $^ -o $(BIN_DIR)/$@ $(GUI_LDFLAGS)

# Generic compilation
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# ImGui compilation
imgui.o: $(THIRD_PARTY)/imgui/imgui.cpp
	$(CC) $(CFLAGS) -c $< -o $@

imgui_demo.o: $(THIRD_PARTY)/imgui/imgui_demo.cpp
	$(CC) $(CFLAGS) -c $< -o $@

imgui_draw.o: $(THIRD_PARTY)/imgui/imgui_draw.cpp
	$(CC) $(CFLAGS) -c $< -o $@

imgui_tables.o: $(THIRD_PARTY)/imgui/imgui_tables.cpp
	$(CC) $(CFLAGS) -c $< -o $@

imgui_widgets.o: $(THIRD_PARTY)/imgui/imgui_widgets.cpp
	$(CC) $(CFLAGS) -c $< -o $@

imgui_impl_glfw.o: $(THIRD_PARTY)/imgui/backends/imgui_impl_glfw.cpp
	$(CC) $(CFLAGS) -c $< -o $@

imgui_impl_opengl3.o: $(THIRD_PARTY)/imgui/backends/imgui_impl_opengl3.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *~ $(BIN_DIR)/cbir_build $(BIN_DIR)/cbir_query $(BIN_DIR)/cbir_gui

.PHONY: all clean
