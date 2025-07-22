# Streamlit App Documentation

The Streamlit application provides a user-friendly web interface for interacting with the IR image classification system. It allows users to upload IR images, perform similarity searches, visualize results, and explore the database of classified objects.

## Overview

The Streamlit app (`app.py` in the root directory) serves as the main user interface for the IR image classification system. It provides a range of features for both operational use and system development/debugging.

## Features

### Image Upload and Query

- Upload IR images from local files
- Capture images from camera (if available)
- URL-based image loading
- Batch processing of multiple images
- Support for different image formats

### Similarity Search

- Perform similarity searches on uploaded images
- Adjust search parameters (k, confidence threshold, search mode)
- Apply filters by object category, class, or attributes
- View and compare multiple search results

### Result Visualization

- Display similarity results with confidence scores
- Highlight matching regions in images
- Side-by-side comparison of query and result images
- Heatmap visualization of feature importance
- Confidence level indicators

### Database Exploration

- Browse the database of classified objects
- Filter and search by object class, category, or attributes
- View statistics and distribution of object classes
- Explore embeddings in a reduced-dimension space (t-SNE/UMAP)

### System Monitoring

- View system performance metrics
- Monitor query processing times
- Track cache hit rates
- View embedding extraction statistics
- Database health and status indicators

### Development Tools

- Model performance evaluation
- Confidence calibration tools
- A/B testing of different configurations
- Error analysis and debugging views
- Batch validation of results

## Implementation

The Streamlit app is implemented using the following components:

### Main Application Structure

- Session state management for maintaining application state
- Sidebar for navigation and configuration options
- Tabs for organizing different functionality
- Responsive layout for different screen sizes

### Key Components

- Image upload and preprocessing component
- Query configuration and execution component
- Results display and visualization component
- Database exploration component
- System monitoring dashboard
- Settings and configuration management

### Integration with Backend

- Direct integration with the QueryProcessor
- Access to the vector database for exploration
- Visualization of embeddings and similarity metrics
- Performance monitoring and statistics

## Usage

To run the Streamlit app:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501` by default.

## Configuration

The app can be configured through:

1. Command-line arguments when starting the app
2. Configuration files loaded at startup
3. UI settings within the app itself

Key configuration options include:

- Database path and collection name
- Model selection and parameters
- UI customization options
- Performance and caching settings
- Development mode toggles

## Development and Extension

The Streamlit app is designed to be extensible. New features can be added by:

1. Creating new functions for specific functionality
2. Adding new pages or tabs to the interface
3. Extending the sidebar with additional options
4. Creating new visualization components

The app follows a modular design pattern, making it easy to add or modify functionality without affecting the core application structure.
