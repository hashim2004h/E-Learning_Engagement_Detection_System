# E-Learning Engagement Detection System

This project is a basic prototype for an E-Learning Engagement Detection System using a laptop camera. It leverages OpenCV for real-time video capture, face and eye detection, and overlays a dummy engagement score on the video feed.

## Project Structure
- **main.py:** The application entry point that handles video capture and display.
- **detector.py:** Contains functions to detect faces and eyes using Haar cascades.
- **models.py:** Includes a dummy engagement prediction function (to be replaced with real machine learning models).
- **utils.py:** Utility functions, such as overlaying text on video frames.
- **config.py:** Configuration parameters for the application.
- **requirements.txt:** Lists the project dependencies.
- **README.md:** This documentation file.
- **data/** (Optional): Directory to store training data or sample images.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
