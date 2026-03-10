# AI Enabled Analysis of Myofascial Sling Exercises

This project is an AI-enabled rehabilitation framework designed to assess, track, and improve lumbar flexibility, posture, and balance using Myofascial Sling Exercises.

It utilizes **YOLOv8-Pose** for real-time skeletal keypoint detection and **Flask** for the web interface.

## Prerequisites

- Python 3.8+
- Webcam

## Installation

1.  **Navigate to the project directory:**
    ```bash
    cd "d:\HMS\lumbar_ai_analysis"
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```

2.  **Open your browser:**
    Go to `http://127.0.0.1:5000`

## Features

-   **Real-time Pose Estimation**: Uses YOLOv8 to detect joints.
-   **Flexibility Analysis**: Measures Hip/Spine angles to guide forward bending.
-   **Balance Stability**: Tracks Center of Gravity (CoG) sway.
-   **Motion Efficiency**: Analyzes movement smoothness.
-   **Interactive Dashboard**: Live video feed with instant feedback.
