# IR Image Classification Streamlit App

## Overview

The Streamlit app provides a user-friendly web interface for the IR Image Classification System. It allows users to upload their own images or select from examples, configure classification parameters, and view results in an interactive format.

## Features

- **Image Selection**: Upload custom images or choose from examples
- **Parameter Configuration**: Configure all mission parameters through an intuitive UI
- **Results Visualization**: View classification results with confidence scores and object identification
- **Export Options**: Save results to JSON files for further analysis

## Running the App

1. Install the required dependencies:

   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Usage Guide

### Basic Configuration

The app provides three configuration tabs:

1. **Basic**: Essential parameters for classification

   - Database path
   - Model path (optional)
   - Configuration preset
   - Confidence and similarity thresholds
   - Maximum results

2. **Advanced**: Fine-tuning parameters

   - Ranking and confidence strategies
   - Validation mode
   - Processing options (GPU, caching, etc.)
   - Debug mode

3. **Output**: Result formatting and saving options
   - Output format (table, detailed, military, JSON)
   - File saving options
   - Mission parameters
   - Metadata inclusion

### Running a Classification

1. Select an image source (upload or examples)
2. Configure parameters as needed
3. Click "Run Classification"
4. View results in the main panel

### Interpreting Results

The results display includes:

- Mission information (ID, processing time, model version)
- Table of identified objects with confidence scores
- Bar chart visualization of confidence scores
- Detailed metadata (optional)

## Troubleshooting

- If the app fails to start, ensure all dependencies are installed
- If classification fails, check the error message for details
- For GPU acceleration issues, verify CUDA is properly installed
- For database errors, ensure the vector database path is correct
