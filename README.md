# Netryx: Open Source AI Geolocation Engine

Netryx is a powerful, locally-hosted geolocation tool that uses state-of-the-art computer vision to identify the exact coordinates of a street-level image. It replicates the core pipeline of high-end geolocation SaaS platforms but runs entirely on your local hardware.

## Demos

Check out Netryx in action:
- **Missile Strike Geolocation (Qatar)**: [Watch on YouTube](https://youtu.be/Y_eC5VPypPU)
- **Paris Protests Geolocation**: [Watch on YouTube](https://youtu.be/DV8vsoa5sLU)
- **Random Picture Geolocation (Paris)**: [Watch on YouTube](https://youtu.be/N5Cx7j6qA7I)

## How it Works

Netryx uses a three-stage refinement pipeline to achieve high accuracy:

1.  **Global Ranking (CosPlace)**: The system first uses CosPlace (Global Place Recognition) to search its entire local index. It generates a digital fingerprint of your query image and finds the top 1000 candidates that look similar from a global perspective.
2.  **Local Geometric Verification (DISK + LightGlue)**: For the top candidates, the system downloads high-resolution street view panoramas and uses DISK (feature extraction) and LightGlue (deep matching) to find exact point-to-point correspondences.
3.  **Refinement and Multi-FOV Crops**: To handle zoom mismatches, the system extracts features at multiple Fields of View (FOV-20, FOV, FOV+20). For difficult images (nighttime, blur, or low texture), the system can utilize **LoFTR** (Detector-Free Local Feature Matching) which performs dense matching without relying on specific keypoints.

## Getting Started

### Hardware Requirements
*   **Operating System**: macOS (M1/M2/M3), Linux, or Windows.
*   **GPU**: 8GB+ VRAM recommended. 
    *   **Mac**: Uses MPS (Metal Performance Shaders).
    *   **Windows/Linux**: Uses CUDA for NVIDIA cards.
*   **Storage**: 10GB-50GB of free space depending on the size of the areas you plan to index.
*   **Internet**: A stable broadband connection is required for downloading street view tiles during the creation of an index.

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/sparkyniner/Netryx-OpenSource-Next-Gen-Street-Level-Geolocation.git
    cd llmgeo
    ```
2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install kornia  # For advanced image transformations
    ```

## Usage Guide

### 1. Creating a Local Index (Create Mode)
Before you can search, you must index an area. This process downloads street view data and extracts global fingerprints.
*   **Setup**: Enter the Latitude/Longitude of the center point and a Search Radius (start with 5km).
*   **Start**: Select **Create** mode and click **Create Index**.
*   **Performance**: On an M2 Max, indexing a 5km radius takes approximately 4-8 hours. The data is saved to `cosplace_parts/` and then compacted into a searchable format in `index/`.

### 2. Geolocating an Image (Search Mode)
*   **Load Image**: Click **Browse Query Image** to select your photo.
*   **AI Coarse**: If you don't know the city, select **AI Coarse**. This uses the Gemini AI model to analyze visual clues like signs, driving side, architecture, and vegetation to provide 4 initial guesses.
*   **Manual**: If you know the city, enter the coordinates manually to center the search.
*   **Run**: Click **Run Search**. The tool will filter the index, download potential candidates, and perform deep matching.

## AI Coarse: How it Works
The AI Coarse feature is designed for "blind" geolocation.
1.  **Visual Extraction**: The Gemini model performs a forensic analysis of signs (language), car plates, building styles, and flora.
2.  **Geopolitical Context**: It identifies road markings and infrastructure types that are specific to certain regions or countries.
3.  **Search Seed**: It returns the top 4 locations. Netryx then creates a search grid around each of these points to find the exact street.

## Frequently Asked Questions (FAQ)

### Does accuracy reduce if the search radius increases?
No. Accuracy does not decrease as you increase the search radius. This is because **CosPlace** is a global descriptor system; it ranks the most similar images from the entire index, whether that index is 1km or 50km. Furthermore, search time also doesn't increase significantly because the "Detective" (LightGlue) only ever analyzes the top-ranked candidates (usually the top 500-1000). While you can increase the number of candidates for even higher accuracy the basic search remains stable regardless of the radius size.

### Why is it taking so long to build an index?
The program has to download high-resolution panoramas, stitch them together, and run them through a deep learning model to capture their "fingerprint." This is a compute-heavy process that depends on both your internet speed and your GPU power.

### Can I index an entire country?
It is not recommended. The storage requirements and crawling time would be immense. It is best to index specific cities or regions (10km - 20km sections) where you expect to perform searches.

### My search found 0 candidates. What happened?
This usually means your coordinates are set to an area that hasn't been indexed yet. Make sure you have run **Create Mode** for the specific area you are searching in.

### Does this cost money to run?
Netryx itself is open source and free. However, if you use the AI Coarse feature, it requires a Gemini API key. Depending on your usage, you may fall under the free tier or the paid tier of the Google AI Studio.

### The matching results look wrong. How can I improve accuracy?
Ensure your query image is relatively clear. If the result is slightly off try enabling **Ultra Mode** (if available) or increasing the **Grid Resolution** when creating your index.

## Project Structure
*   `test_super.py`: Main application and GUI.
*   `build_index.py`: High-performance index builder for large datasets (uses disk-memmap).
*   `cosplace_utils.py`: Logic for global ranking and embeddings.
*   `shared_utils.py`: Coordinate math and equirectangular projection logic.
*   `index/`: The compacted database used for fast searching.

---
Built by Sairaj Balaji. I would love to connect: [LinkedIn](https://www.linkedin.com/in/sairaj-balaji-7295b2246/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Legal Disclaimer and Responsible Use

Netryx is built strictly for legal OSINT (Open Source Intelligence), research, and educational purposes. 
- **User Responsibility**: The developers of Netryx are not responsible for any misuse, nefarious activities, or legal violations committed using this tool. 
- **Terms of Service**: Users must ensure compliance with the Google Maps/Street View Terms of Service and local privacy laws when crawling or indexing data.
- **Ethical Use**: This tool is designed to assist in investigative journalism, human rights monitoring, and geographic research. Do not use it for stalking, harassment, or any unauthorized surveillance.

Disclaimer: This tool is intended for research and educational purposes. Always ensure your use of the Google Street View API complies with their Terms of Service.
