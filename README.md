# OmniMind AI - Multimodal Emotion Engine

A state-of-the-art Deep Learning application that goes beyond simple sentiment analysis. OmniMind AI detects detailed human emotions (Joy, Anger, Sadness, Surprise, etc.) from both **text reviews** and **facial expressions**.

![Dashboard Screenshot](dashboard_ui_screenshot.png)

## Key Features

### 1. Advanced NLP Emotion Recognition
*   **Model**: Fine-tuned **RoBERTa** (Transformer) model (`j-hartmann/emotion-english-distilroberta-base`).
*   **Capabilities**: Detects subtle emotions including:
    *   😊 Joy / Happiness
    *   😡 Anger
    *   😢 Sadness
    *   😨 Fear
    *   😯 Surprise
    *   😐 Neutral
*   **Spotify Integration**: Automatically generates a curated playlist link based on the detected mood.

### 2. Facial Expression Analysis
*   **Model**: **DeepFace** (hybrid VGG-Face / Google FaceNet).
*   **Capabilities**:
    *   Real-time Face Detection.
    *   Emotion classification from uploaded images.
    *   **Live Webcam** feed analysis.
    *   Draws bounding boxes and labels directly on the visual feed.

### 3. Modern Tech Stack
*   **Backend**: Flask (Python), TensorFlow, PyTorch, Transformers.
*   **Frontend**: HTML5, CSS3 Glassmorphism UI, JavaScript (Vanilla).
*   **Computer Vision**: OpenCV, DeepFace.

## Installation

1.  **Clone the Repository**
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application**:
    ```bash
    python app.py
    ```
    *Note: First run will download necessary AI models (~500MB).*

## Usage

*   **Text Analysis**: Type any sentence to see its emotional breakdown and confidence score.
*   **Vision Analysis**: Switch tabs to "Face & Emotion", grant camera permissions, and capture your expression in real-time.
*   **Music Recommendation**: Click the "Listen on Spotify" button to hear tracks that match your current vibe.
