# Project 2: Content-Based Image Retrieval (CBIR)

## Author
- **Name:** Borui Chen

## Development Environment
- **Operating System:** macOS
- **IDE:** VS Code with C++ extension
- **Compiler:** g++ (C++17)
- **OpenCV Version:** 4.x

## Project Structure
```
Project2/
├── include/
│   ├── feature.h       # Feature extraction interface
│   ├── distance.h      # Distance metric functions
│   └── cbir.h          # CBIR system main interface
├── src/
│   ├── feature.cpp     # Feature extraction implementations (7 tasks)
│   ├── distance.cpp    # Distance metric implementations
│   ├── cbir.cpp        # CBIR system core logic
│   ├── cbir_build.cpp  # Database building program
│   ├── cbir_query.cpp  # Query program
│   ├── cbir_gui.cpp    # GUI application (extension)
│   └── Makefile
├── third_party/        # Third-party libraries (not included in submission)
│   └── imgui/          # ImGui library for GUI (needs to be downloaded separately)
├── bin/                # Compiled executables
├── data/olympus/       # Image database
└── report.md           # Project report with results
```

**Note:** The `third_party/imgui` directory is not included in the submission. To build the GUI (`cbir_gui`), you need to download ImGui from https://github.com/ocornut/imgui and place it in `third_party/imgui/`. The command-line tools (`cbir_build` and `cbir_query`) do not require ImGui and can be built without it.

## Build Instructions

### Prerequisites
- OpenCV 4.x installed
- C++17 compatible compiler
- GLFW and OpenGL (for GUI only)
- ImGui library (for GUI only, see note below)

### Third-Party Library Note
The GUI extension (`cbir_gui`) requires the **ImGui** library. It is NOT included in this submission. To build the GUI:
1. Download ImGui from https://github.com/ocornut/imgui
2. Extract to `Project2/third_party/imgui/`
3. The directory structure should be:
   ```
   third_party/imgui/
   ├── imgui.cpp
   ├── imgui.h
   ├── imgui_demo.cpp
   ├── imgui_draw.cpp
   ├── imgui_tables.cpp
   ├── imgui_widgets.cpp
   └── backends/
       ├── imgui_impl_glfw.cpp
       ├── imgui_impl_glfw.h
       ├── imgui_impl_opengl3.cpp
       └── imgui_impl_opengl3.h
   ```

The command-line tools (`cbir_build` and `cbir_query`) do NOT require ImGui and can be built without it.

### Compilation

#### Build All (including GUI):
```bash
cd src
make
```

#### Build Command-Line Tools Only (without ImGui):
```bash
cd src
make cbir_build cbir_query
```

This will build:
- `../bin/cbir_build` - Build feature database
- `../bin/cbir_query` - Query similar images
- `../bin/cbir_gui` - Interactive GUI (extension, requires ImGui)

## Running the Executables

### 1. Build Feature Database
```bash
./bin/cbir_build -d <image_directory> -f <feature_type> -o <output.csv> [-c <dnn_csv>]
```

**Feature Types:**
- `baseline` - Task 1: 7x7 center square (147 dims)
- `histogram` - Task 2: Color histogram (4096 dims)
- `multi_histogram` - Task 3: Multi-region histogram (1024 dims)
- `texture_color` - Task 4: Texture + Color (520 dims)
- `dnn_embedding` - Task 5: ResNet18 embeddings (512 dims)
- `custom` - Task 7: Blue Sky Detector (30 dims)

**Examples:**
```bash
# Task 1: Baseline
./bin/cbir_build -d data/olympus -f baseline -o features_baseline.csv

# Task 2: Histogram
./bin/cbir_build -d data/olympus -f histogram -o features_histogram.csv

# Task 5: DNN (requires CSV file with embeddings)
./bin/cbir_build -d data/olympus -f dnn_embedding -c resnet18_features.csv -o features_dnn.csv

# Task 7: Blue Sky Detector
./bin/cbir_build -d data/olympus -f custom -o features_bluesky.csv
```

### 2. Query Similar Images
```bash
./bin/cbir_query -t <target_image> -f <feature_type> -i <features.csv> -n <num_results> [-c <dnn_csv>]
```

**Examples:**
```bash
# Task 1: Query with baseline
./bin/cbir_query -t data/olympus/pic.1016.jpg -f baseline -i features_baseline.csv -n 4

# Task 2: Query with histogram
./bin/cbir_query -t data/olympus/pic.0164.jpg -f histogram -i features_histogram.csv -n 3

# Task 7: Query with blue sky detector
./bin/cbir_query -t data/olympus/pic.0001.jpg -f custom -i features_bluesky.csv -n 5
```

### 3. GUI Application (Extension)
```bash
./bin/cbir_gui -d <image_directory> [-c <dnn_csv>]
```

**Example:**
```bash
./bin/cbir_gui -d data/olympus
```

**GUI Features:**
- Browse and select image directory
- Select feature type from dropdown (with descriptions)
- Build database with progress indication
- Load target image via file browser
- Perform queries and view results with thumbnails
- Save/load feature databases

## Testing the Tasks

### Task 1: Baseline Matching
```bash
./bin/cbir_build -d data/olympus -f baseline -o features_baseline.csv
./bin/cbir_query -t data/olympus/pic.1016.jpg -f baseline -i features_baseline.csv -n 4
```
Expected top matches: pic.0986.jpg, pic.0641.jpg, pic.0547.jpg

### Task 2: Histogram Matching
```bash
./bin/cbir_build -d data/olympus -f histogram -o features_histogram.csv
./bin/cbir_query -t data/olympus/pic.0164.jpg -f histogram -i features_histogram.csv -n 3
```

### Task 3: Multi-histogram Matching
```bash
./bin/cbir_build -d data/olympus -f multi_histogram -o features_multi.csv
./bin/cbir_query -t data/olympus/pic.0274.jpg -f multi_histogram -i features_multi.csv -n 4
```
Expected top matches: pic.0273.jpg, pic.1031.jpg, pic.0409.jpg

### Task 4: Texture and Color
```bash
./bin/cbir_build -d data/olympus -f texture_color -o features_texture.csv
./bin/cbir_query -t data/olympus/pic.0535.jpg -f texture_color -i features_texture.csv -n 3
```

### Task 5: DNN Embeddings
```bash
./bin/cbir_build -d data/olympus -f dnn_embedding -c resnet18_features.csv -o features_dnn.csv
./bin/cbir_query -t data/olympus/pic.0893.jpg -f dnn_embedding -i features_dnn.csv -n 3
```

### Task 7: Custom Blue Sky Detector
```bash
./bin/cbir_build -d data/olympus -f custom -o features_bluesky.csv
./bin/cbir_query -t data/olympus/pic.0001.jpg -f custom -i features_bluesky.csv -n 5
```

## Extensions Implemented

### 1. Graphical User Interface (GUI)
An interactive GUI built with ImGui + OpenGL + GLFW providing:
- Visual file/directory browsing
- Feature type selection with descriptions
- Real-time progress indication
- Thumbnail image display for results

## Notes

- The `custom` feature type (Task 7) was originally designed as a sunset detector but was changed to a blue sky detector because the database contains more blue sky images.
- DNN embeddings (Task 5) require a pre-computed CSV file with ResNet18 features.
- Feature databases can be pre-built and saved to CSV files for faster querying.

## Video Links
N/A

## Time Travel Days Used
0
