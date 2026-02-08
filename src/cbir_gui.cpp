/*
  Name: Borui Chen
  Date: 2026-02-03
  Purpose: GUI application for CBIR system using ImGui + OpenGL + GLFW
  Usage: ./cbir_gui -d <image_dir> [-c <dnn_csv>]
*/

#include "cbir.h"
#include "feature.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>

// Texture structure for OpenGL
typedef struct {
    GLuint texID;
    int width, height;
} Texture;

// Load OpenCV image to OpenGL texture
Texture loadTextureFromMat(const cv::Mat& image) {
    Texture tex = {0, 0, 0};
    if (image.empty()) return tex;

    cv::Mat rgbImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGBA);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, rgbImage, cv::COLOR_GRAY2RGBA);
    } else {
        rgbImage = image.clone();
    }

    tex.width = rgbImage.cols;
    tex.height = rgbImage.rows;

    glGenTextures(1, &tex.texID);
    glBindTexture(GL_TEXTURE_2D, tex.texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgbImage.data);
    glBindTexture(GL_TEXTURE_2D, 0);

    return tex;
}

void deleteTexture(Texture& tex) {
    if (tex.texID != 0) {
        glDeleteTextures(1, &tex.texID);
        tex.texID = 0;
    }
}

// File dialog helper functions using native OS dialogs
#ifdef __APPLE__
#include <array>
#include <memory>

// Execute command and return output
std::string execCommand(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    return result;
}

// Open file dialog for selecting image files
std::string openImageFileDialog() {
    const char* script = R"(
osascript -e '
tell application "System Events"
    activate
    set imageExtensions to {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "ppm"}
    set selectedFile to choose file with prompt "Select an image file" of type imageExtensions
    return POSIX path of selectedFile
end tell'
)";
    return execCommand(script);
}

// Open file dialog for selecting CSV files
std::string openCSVFileDialog() {
    const char* script = R"(
osascript -e '
tell application "System Events"
    activate
    set selectedFile to choose file with prompt "Select a CSV file" of type {"csv"}
    return POSIX path of selectedFile
end tell'
)";
    return execCommand(script);
}

// Open directory dialog
std::string openDirectoryDialog() {
    const char* script = R"(
osascript -e '
tell application "System Events"
    activate
    set selectedFolder to choose folder with prompt "Select image directory"
    return POSIX path of selectedFolder
end tell'
)";
    return execCommand(script);
}
#else
// Linux/Windows placeholders - use simple file browser or return empty
std::string openImageFileDialog() { return ""; }
std::string openCSVFileDialog() { return ""; }
std::string openDirectoryDialog() { return ""; }
#endif

// CBIR GUI App state
class CBIRGUIApp {
public:
    CBIRSystem cbir;
    std::string imageDir;
    std::string dnnCsvPath;

    // Feature types
    const char* featureTypes[6] = {"baseline", "histogram", "multi_histogram", "texture_color", "dnn_embedding", "custom"};
    int currentFeatureType = 0;
    int databaseFeatureType = -1;  // Track which feature type was used to build the database
    bool databaseBuilt = false;

    // Target image
    std::string targetImagePath;
    cv::Mat targetImage;
    Texture targetTexture;

    // Query results
    std::vector<MatchResult> results;
    std::vector<Texture> resultTextures;
    std::vector<cv::Mat> resultImages;
    int numResults = 5;
    bool hasResults = false;

    // Status message
    std::string statusMessage;
    float statusTimer = 0.0f;

    // Progress
    bool isBuilding = false;
    float buildProgress = 0.0f;

    CBIRGUIApp() {}

    ~CBIRGUIApp() {
        cleanupTextures();
    }

    void setStatus(const std::string& msg) {
        statusMessage = msg;
        statusTimer = 3.0f;
    }

    void cleanupTextures() {
        deleteTexture(targetTexture);
        for (auto& tex : resultTextures) {
            deleteTexture(tex);
        }
        resultTextures.clear();
        resultImages.clear();
    }

    void loadTargetImage(const std::string& path) {
        targetImage = cv::imread(path);
        if (!targetImage.empty()) {
            targetImagePath = path;
            deleteTexture(targetTexture);
            targetTexture = loadTextureFromMat(targetImage);
            setStatus("Loaded target image: " + path);
        } else {
            setStatus("Failed to load image: " + path);
        }
    }

    void buildDatabase() {
        if (imageDir.empty()) {
            setStatus("Error: Image directory not set");
            return;
        }

        isBuilding = true;
        buildProgress = 0.0f;

        // Clear previous results when building new database
        cleanupResultTextures();

        FeatureType ft = static_cast<FeatureType>(currentFeatureType);

        if (ft == FeatureType::DNN_EMBEDDING && dnnCsvPath.empty()) {
            setStatus("Error: DNN CSV path required for dnn_embedding");
            isBuilding = false;
            return;
        }

        if (!dnnCsvPath.empty()) {
            cbir.setDNNCsvPath(dnnCsvPath);
        }

        setStatus("Building database...");
        int count = cbir.buildDatabase(imageDir, ft);

        if (count > 0) {
            databaseBuilt = true;
            databaseFeatureType = currentFeatureType;  // Record the feature type used
            setStatus("Database built with " + std::to_string(count) + " images");
        } else {
            setStatus("Error: Failed to build database");
        }

        isBuilding = false;
    }

    void performQuery() {
        if (!databaseBuilt) {
            setStatus("Error: Database not built");
            return;
        }

        // Check if feature type has changed since database was built
        if (databaseFeatureType != currentFeatureType) {
            setStatus("Error: Feature type changed. Please rebuild database.");
            return;
        }

        if (targetImagePath.empty()) {
            setStatus("Error: No target image selected");
            return;
        }

        setStatus("Querying...");

        // Clear previous results
        cleanupResultTextures();

        results = cbir.query(targetImagePath, numResults + 1); // +1 to include query itself

        // Remove query image itself from results
        results.erase(std::remove_if(results.begin(), results.end(),
            [this](const MatchResult& r) {
                return r.imagePath.find(targetImagePath) != std::string::npos ||
                       targetImagePath.find(r.imagePath) != std::string::npos;
            }), results.end());

        // Limit to numResults
        if (results.size() > static_cast<size_t>(numResults)) {
            results.resize(numResults);
        }

        // Load result images

        for (const auto& result : results) {
            // Handle path correctly - avoid duplicate directory paths
            std::string fullPath = result.imagePath;

            // Check if result.imagePath contains the directory separator (has path component)
            size_t lastSlash = result.imagePath.find_last_of("/\\");
            if (lastSlash == std::string::npos) {
                // Just a filename, prepend imageDir
                fullPath = imageDir + "/" + result.imagePath;
            }
            // else: result.imagePath already contains path, use as-is

            cv::Mat img = cv::imread(fullPath);
            if (!img.empty()) {
                resultImages.push_back(img);
                resultTextures.push_back(loadTextureFromMat(img));
            }
        }

        hasResults = !results.empty();
        setStatus("Query completed. Found " + std::to_string(results.size()) + " matches");
    }

    void saveFeatures(const std::string& filename) {
        if (!databaseBuilt) {
            setStatus("Error: Database not built");
            return;
        }

        if (cbir.saveFeatures(filename) == 0) {
            setStatus("Features saved to " + filename);
        } else {
            setStatus("Error: Failed to save features");
        }
    }

    void loadFeatures(const std::string& filename) {
        // Clear previous results before loading new features
        cleanupResultTextures();

        if (cbir.loadFeatures(filename) > 0) {
            databaseBuilt = true;
            // Sync the feature type from loaded database
            FeatureType loadedType = cbir.getFeatureType();
            switch (loadedType) {
                case FeatureType::BASELINE: currentFeatureType = 0; break;
                case FeatureType::HISTOGRAM: currentFeatureType = 1; break;
                case FeatureType::MULTI_HISTOGRAM: currentFeatureType = 2; break;
                case FeatureType::TEXTURE_COLOR: currentFeatureType = 3; break;
                case FeatureType::DNN_EMBEDDING: currentFeatureType = 4; break;
                case FeatureType::CUSTOM: currentFeatureType = 5; break;
            }
            databaseFeatureType = currentFeatureType;
            setStatus("Features loaded from " + filename);
        } else {
            setStatus("Error: Failed to load features");
        }
    }

private:
    void cleanupResultTextures() {
        for (auto& tex : resultTextures) {
            deleteTexture(tex);
        }
        resultTextures.clear();
        resultImages.clear();
        results.clear();
        hasResults = false;
    }
};

// Global app instance
CBIRGUIApp g_app;

// Helper to render image with ImGui
void renderImage(Texture& tex, float maxWidth, float maxHeight) {
    if (tex.texID == 0) return;

    float aspect = (float)tex.width / (float)tex.height;
    float width = maxWidth;
    float height = width / aspect;

    if (height > maxHeight) {
        height = maxHeight;
        width = height * aspect;
    }

    ImGui::Image((ImTextureID)(intptr_t)tex.texID, ImVec2(width, height));
}

int main(int argc, char* argv[]) {
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            g_app.imageDir = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            g_app.dnnCsvPath = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " -d <image_dir> [-c <dnn_csv>]" << std::endl;
            return 0;
        }
    }

    // Setup GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // OpenGL version
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(1400, 900, "CBIR - Content-Based Image Retrieval", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Main window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(1400, 900));
        ImGui::Begin("CBIR System", nullptr,
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

        // Left panel - Controls
        ImGui::BeginChild("Controls", ImVec2(350, 800), true);

        ImGui::Text("CBIR System");
        ImGui::Separator();

        // Image directory
        static char dirBuf[256] = "";
        strncpy(dirBuf, g_app.imageDir.c_str(), sizeof(dirBuf) - 1);
        ImGui::Text("Image Directory:");
        ImGui::InputText("##dir", dirBuf, sizeof(dirBuf));
        g_app.imageDir = dirBuf;
        ImGui::SameLine();
        if (ImGui::Button("Browse##dir", ImVec2(60, 20))) {
            std::string selectedDir = openDirectoryDialog();
            if (!selectedDir.empty()) {
                g_app.imageDir = selectedDir;
                strncpy(dirBuf, selectedDir.c_str(), sizeof(dirBuf) - 1);
            }
        }

        // Feature type selection
        ImGui::Text("Feature Type:");
        ImGui::Combo("##feature", &g_app.currentFeatureType, g_app.featureTypes, IM_ARRAYSIZE(g_app.featureTypes));

        // Show description for selected feature type
        ImGui::Spacing();
        switch (g_app.currentFeatureType) {
            case 0:
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Task 1: 7x7 Center Square");
                ImGui::TextWrapped("Extracts 147-dimensional feature from image center. Uses SSD distance.");
                break;
            case 1:
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Task 2: Color Histogram");
                ImGui::TextWrapped("3D RGB histogram with 16 bins per channel (4096 dims). Uses histogram intersection.");
                break;
            case 2:
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Task 3: Multi-Histogram");
                ImGui::TextWrapped("Two histograms from top/bottom halves. Good for spatial color distribution.");
                break;
            case 3:
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Task 4: Texture + Color");
                ImGui::TextWrapped("Combines color histogram with gradient magnitude histogram.");
                break;
            case 4:
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Task 5: DNN Embedding");
                ImGui::TextWrapped("512-dimensional ResNet18 features from CSV file. Uses cosine distance.");
                break;
            case 5:
                ImGui::TextColored(ImVec4(0.2f, 0.6f, 1.0f, 1.0f), "Task 7: Blue Sky Detector");
                ImGui::TextWrapped("Custom feature for blue sky detection. 30 dims: blue color histogram (16), spatial distribution (8), brightness (4), sky position (2).");
                break;
        }
        ImGui::Spacing();

        // DNN CSV path (only for dnn_embedding)
        if (g_app.currentFeatureType == 4) { // dnn_embedding
            static char dnnBuf[256] = "";
            strncpy(dnnBuf, g_app.dnnCsvPath.c_str(), sizeof(dnnBuf) - 1);
            ImGui::Text("DNN CSV Path:");
            ImGui::InputText("##dnn", dnnBuf, sizeof(dnnBuf));
            g_app.dnnCsvPath = dnnBuf;
            ImGui::SameLine();
            if (ImGui::Button("Browse##dnn", ImVec2(60, 20))) {
                std::string selectedFile = openCSVFileDialog();
                if (!selectedFile.empty()) {
                    g_app.dnnCsvPath = selectedFile;
                    strncpy(dnnBuf, selectedFile.c_str(), sizeof(dnnBuf) - 1);
                }
            }
        }

        ImGui::Separator();

        // Build database button
        if (ImGui::Button("Build Database", ImVec2(150, 30))) {
            g_app.buildDatabase();
        }

        if (g_app.isBuilding) {
            ImGui::SameLine();
            ImGui::ProgressBar(g_app.buildProgress, ImVec2(150, 0), "Building...");
        }

        ImGui::Text("Database Status: %s", g_app.databaseBuilt ? "Ready" : "Not Built");
        if (g_app.databaseBuilt) {
            ImGui::Text("Images: %zu", g_app.cbir.getDatabaseSize());
        }

        ImGui::Separator();

        // Load/Save features
        static char saveBuf[256] = "features.csv";
        ImGui::Text("Feature File:");
        ImGui::InputText("##save", saveBuf, sizeof(saveBuf));

        if (ImGui::Button("Save Features", ImVec2(120, 25))) {
            g_app.saveFeatures(saveBuf);
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Features", ImVec2(120, 25))) {
            g_app.loadFeatures(saveBuf);
        }

        ImGui::Separator();

        // Target image selection
        static char targetBuf[256] = "";
        ImGui::Text("Target Image Path:");
        ImGui::InputText("##target", targetBuf, sizeof(targetBuf));
        ImGui::SameLine();
        if (ImGui::Button("Browse##target", ImVec2(60, 20))) {
            std::string selectedFile = openImageFileDialog();
            if (!selectedFile.empty()) {
                strncpy(targetBuf, selectedFile.c_str(), sizeof(targetBuf) - 1);
            }
        }

        if (ImGui::Button("Load Target Image", ImVec2(150, 25))) {
            g_app.loadTargetImage(targetBuf);
        }

        ImGui::Separator();

        // Query settings
        ImGui::Text("Number of Results:");
        ImGui::SliderInt("##num", &g_app.numResults, 1, 10);

        if (ImGui::Button("Perform Query", ImVec2(150, 40))) {
            g_app.performQuery();
        }

        ImGui::Separator();

        // Status message
        if (!g_app.statusMessage.empty()) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Status: %s", g_app.statusMessage.c_str());
            g_app.statusTimer -= ImGui::GetIO().DeltaTime;
            if (g_app.statusTimer <= 0) {
                g_app.statusMessage.clear();
            }
        }

        ImGui::EndChild();

        ImGui::SameLine();

        // Right panel - Display
        ImGui::BeginChild("Display", ImVec2(1000, 800), true);

        // Target image
        ImGui::Text("Target Image:");
        if (g_app.targetTexture.texID != 0) {
            renderImage(g_app.targetTexture, 300, 200);
            ImGui::Text("Path: %s", g_app.targetImagePath.c_str());
        } else {
            ImGui::Text("No target image loaded");
        }

        ImGui::Separator();

        // Query results
        ImGui::Text("Query Results:");

        if (g_app.hasResults) {
            for (size_t i = 0; i < g_app.results.size() && i < g_app.resultTextures.size(); i++) {
                ImGui::BeginGroup();

                ImGui::Text("%zu. %s", i + 1, g_app.results[i].imagePath.c_str());
                ImGui::Text("   Distance: %.4f", g_app.results[i].distance);

                if (g_app.resultTextures[i].texID != 0) {
                    renderImage(g_app.resultTextures[i], 180, 120);
                }

                ImGui::EndGroup();

                if ((i + 1) % 4 != 0 && i < g_app.results.size() - 1) {
                    ImGui::SameLine();
                }
            }
        } else {
            ImGui::Text("No query results");
        }

        ImGui::EndChild();

        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
