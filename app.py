#!/usr/bin/env python3
"""
IR Image Classification System - Streamlit Web Interface

This application provides a user-friendly web interface for the IR Image Classification System,
allowing users to upload images or select from examples, configure parameters, and view results.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go                
from plotly.subplots import make_subplots


import streamlit as st
import pandas as pd
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import necessary modules from the project
try:
    from src.models.object_classes import ObjectClass, ObjectCategory, OBJECT_REGISTRY
    from src.query import ValidationMode, CachePolicy, RankingStrategy, ConfidenceStrategy
except ImportError:
    # If imports fail, display a more user-friendly error message
    import streamlit as st
    st.error("""
    Failed to import required modules. Please make sure you're running the app from the project root directory.
    
    Try running:
    ```
    pip install -e .
    streamlit run app.py
    ```
    """)

# Set page configuration
st.set_page_config(
    page_title="IR Image Classification System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define constants
EXAMPLES_DIR = "examples"
DATABASE_DIR = "data/vector_db"
RESULTS_DIR = "results"
DEFAULT_MODEL = None  # Default model path or None to use system default


def get_example_images() -> List[str]:
    """Get list of example images from the examples directory."""
    examples_path = Path(EXAMPLES_DIR)
    if not examples_path.exists():
        return []
    
    return [str(f.relative_to(project_root)) for f in examples_path.glob("*.png")]


def run_mission(params: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Run the mission script with the provided parameters.
    
    Args:
        params: Dictionary of parameters to pass to the script
        
    Returns:
        Tuple of (success, output_text, results_dict)
    """
    # Check if the script exists
    script_path = Path("scripts/run_mission.py")
    if not script_path.exists():
        return False, f"Error: Script not found at {script_path.absolute()}", None
        
    # Build command
    cmd = [sys.executable, str(script_path)]
    
    # Parameter name mapping from underscore to hyphen
    param_mapping = {
        "confidence_threshold": "confidence-threshold",
        "similarity_threshold": "similarity-threshold", 
        "max_results": "max-results",
        "confidence_strategy": "confidence-strategy",
        "validation_mode": "validation-mode",
        "max_query_time": "max-query-time",
        "disable_gpu": "disable-gpu",
        "disable_cache": "disable-cache",
        "enable_diversity": "enable-diversity",
        "save_metadata": "save-metadata",
        "mission_id": "mission-id"
    }
    
    # Add parameters
    for key, value in params.items():
        if value is None or value == "":
            continue
            
        # Map parameter name if needed
        param_name = param_mapping.get(key, key)
        
        # Handle boolean flags
        if isinstance(value, bool):
            if value:  # Only add flag if True
                cmd.append(f"--{param_name}")
        else:
            cmd.append(f"--{param_name}")
            cmd.append(str(value))
    
    # Add format as JSON for parsing
    cmd.extend(["--format", "json"])
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse JSON output
        output_text = result.stdout
        try:
            # Try to find the JSON in the output using a more robust approach
            lines = output_text.strip().split('\n')
            
            # Look for the start of JSON (line that starts with '{')
            json_start_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    json_start_idx = i
                    break
            
            if json_start_idx is not None:
                # Take all lines from the JSON start to the end
                json_lines = lines[json_start_idx:]
                json_text = '\n'.join(json_lines)
                
                # Try to parse the JSON
                results_dict = json.loads(json_text)
                return True, output_text, results_dict
            else:
                # If no JSON found, return None
                return True, output_text, None
                
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try a different approach
            try:
                # Look for JSON between the last occurrence of '{' and '}'
                start_idx = output_text.rfind('{')
                end_idx = output_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_text = output_text[start_idx:end_idx]
                    results_dict = json.loads(json_text)
                    return True, output_text, results_dict
                else:
                    return True, output_text, None
            except json.JSONDecodeError:
                return True, output_text, None
            
    except subprocess.CalledProcessError as e:
        return False, e.stderr, None
    except Exception as e:
        return False, str(e), None


def display_results(results_dict: Dict[str, Any], image_path: str):
    """Display the results in a user-friendly format."""
    st.subheader("Classification Results")
    
    # Display image and basic info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            img = Image.open(image_path)
            st.image(img, caption="Query Image", use_container_width=True)
        except Exception as e:
            st.error(f"Could not display image: {e}")
    
    with col2:
        st.write("**Mission Information**")
        st.write(f"Mission ID: {results_dict.get('mission_id', 'N/A')}")
        st.write(f"Processing Time: {results_dict.get('processing_time', 0):.3f} seconds")
        st.write(f"Model Version: {results_dict.get('model_version', 'Unknown')}")
        
        if 'operator' in results_dict:
            st.write(f"Operator: {results_dict['operator']}")
        if 'classification' in results_dict:
            st.write(f"Classification: {results_dict['classification']}")
    
    # Display results table
    st.write("**Identified Objects**")
    
    results = results_dict.get('results', [])
    if results:
        # Create DataFrame for results
        df_data = []
        for res in results:
            df_data.append({
                "Rank": res.get('rank', 0),
                "Object Class": res.get('object_class', 'Unknown'),
                "Confidence": f"{res.get('confidence', 0):.3f}",
                "Similarity": f"{res.get('similarity_score', 0):.3f}",
                "Confidence Level": res.get('confidence_level', 'Unknown'),
                "Image ID": res.get('image_id', 'N/A')
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Create a bar chart for confidence scores
        st.subheader("Confidence and Similarity Scores Analysis")
        chart_data = pd.DataFrame({
            'Object Class': [r.get('object_class', 'Unknown') for r in results],
            'Confidence': [r.get('confidence', 0) for r in results],
            'Similarity': [r.get('similarity_score', 0) for r in results]
        })
        
        # Create tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🔍 Log Scale (Enhanced)", "📊 Linear Scale", "📈 Difference Analysis"])
        
        with viz_tab1:
            st.write("**Log scale visualization to emphasize small differences between scores**")
            # Use log scale to emphasize small differences with Plotly
            try:

                
                fig = make_subplots(
                    rows=1, cols=1,
                    subplot_titles=["Confidence and Similarity Scores (Log₁₀ Scale)"]
                )
                
                # Prepare data for log scale
                epsilon = 1e-10
                confidence_vals = np.array(chart_data['Confidence'].values, dtype=float)
                similarity_vals = np.array(chart_data['Similarity'].values, dtype=float)
                object_classes = [cls[:15] + '...' if len(cls) > 15 else cls for cls in chart_data['Object Class']]
                
                # Add traces
                fig.add_trace(go.Bar(
                    name='Confidence (log)',
                    x=object_classes,
                    y=np.log10(confidence_vals + epsilon),
                    text=[f'{val:.4f}' for val in confidence_vals],
                    textposition='auto',
                    marker_color='rgba(31, 119, 180, 0.8)',
                    hovertemplate='<b>%{x}</b><br>Confidence: %{text}<br>Log Value: %{y:.3f}<extra></extra>'
                ))
                
                fig.add_trace(go.Bar(
                    name='Similarity (log)',
                    x=object_classes,
                    y=np.log10(similarity_vals + epsilon),
                    text=[f'{val:.4f}' for val in similarity_vals],
                    textposition='auto',
                    marker_color='rgba(255, 127, 14, 0.8)',
                    hovertemplate='<b>%{x}</b><br>Similarity: %{text}<br>Log Value: %{y:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Confidence and Similarity Scores (Log₁₀ Scale)",
                    xaxis_title="Object Class",
                    yaxis_title="Score (Log₁₀ Scale)",
                    barmode='group',
                    height=500,
                    showlegend=True,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation help
                st.info("💡 **Log Scale Interpretation**: Small differences in the original scores (0.001-0.01) become more visible as larger differences in the log scale. This helps identify subtle performance variations between results.")
                
            except ImportError:
                st.warning("Plotly not available. Using matplotlib fallback.")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                x = np.arange(len(chart_data))
                width = 0.35
                
                # Convert to log scale
                epsilon = 1e-10
                confidence_log = np.log10(chart_data['Confidence'] + epsilon)
                similarity_log = np.log10(chart_data['Similarity'] + epsilon)
                
                bars1 = ax.bar(x - width/2, confidence_log, width, label='Confidence (log)', alpha=0.8, color='#1f77b4')
                bars2 = ax.bar(x + width/2, similarity_log, width, label='Similarity (log)', alpha=0.8, color='#ff7f0e')
                
                ax.set_xlabel('Object Class')
                ax.set_ylabel('Score (Log₁₀ Scale)')
                ax.set_title('Confidence and Similarity Scores (Log₁₀ Scale)')
                ax.set_xticks(x)
                ax.set_xticklabels([cls[:20] + '...' if len(cls) > 20 else cls for cls in chart_data['Object Class']], 
                                  rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with viz_tab2:
            st.write("**Traditional linear scale visualization**")
            st.bar_chart(chart_data.set_index('Object Class'))
        
        with viz_tab3:
            st.write("**Analysis of differences between confidence and similarity scores**")
            # Difference analysis
            chart_data_copy = chart_data.copy()
            chart_data_copy['Absolute_Diff'] = chart_data_copy['Confidence'] - chart_data_copy['Similarity']
            chart_data_copy['Relative_Diff_Pct'] = (chart_data_copy['Confidence'] - chart_data_copy['Similarity']) / chart_data_copy['Similarity'] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Absolute Differences (Confidence - Similarity)**")
                diff_chart = chart_data_copy[['Object Class', 'Absolute_Diff']].set_index('Object Class')
                st.bar_chart(diff_chart)
            
            with col2:
                st.write("**Relative Differences (%)**")
                rel_diff_chart = chart_data_copy[['Object Class', 'Relative_Diff_Pct']].set_index('Object Class')
                st.bar_chart(rel_diff_chart)
            
            # Show detailed statistics
            st.write("**📊 Statistical Summary:**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean Confidence', 'Mean Similarity', 'Mean Difference', 'Max Difference', 'Min Difference', 'Std Difference'],
                'Value': [
                    f"{chart_data_copy['Confidence'].mean():.6f}",
                    f"{chart_data_copy['Similarity'].mean():.6f}",
                    f"{chart_data_copy['Absolute_Diff'].mean():.6f}",
                    f"{chart_data_copy['Absolute_Diff'].max():.6f}",
                    f"{chart_data_copy['Absolute_Diff'].min():.6f}",
                    f"{chart_data_copy['Absolute_Diff'].std():.6f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        # Display detailed metadata if available
        if st.checkbox("Show Detailed Metadata"):
            st.json(results_dict)
    else:
        st.info("No results found matching the specified criteria.")


def main():
    """Main application function."""
    st.title("🎯 IR Image Classification System")
    st.write("Advanced AI-powered object identification for military intelligence")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Image selection
    st.sidebar.subheader("Image Selection")
    image_source = st.sidebar.radio(
        "Select image source:",
        ["Upload Image", "Choose from Examples"]
    )
    
    image_path = None
    
    if image_source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload an image file",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
        )
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_path = Path("temp_upload.png")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            image_path = str(temp_path)
            
            # Display the uploaded image
            st.sidebar.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    else:
        example_images = get_example_images()
        if not example_images:
            st.sidebar.error("No example images found in the examples directory.")
        else:
            selected_example = st.sidebar.selectbox(
                "Select an example image:",
                example_images,
                format_func=lambda x: Path(x).name
            )
            
            if selected_example:
                image_path = selected_example
                
                # Display the selected example image
                try:
                    img = Image.open(selected_example)
                    st.sidebar.image(img, caption=Path(selected_example).name, use_container_width=True)
                except Exception as e:
                    st.sidebar.error(f"Could not load image: {e}")
    
    # Configuration tabs
    tab1, tab2, tab3 = st.sidebar.tabs(["Basic", "Advanced", "Output"])
    
    with tab1:
        # Basic configuration
        database = st.text_input("Database Path", DATABASE_DIR)
        model = st.text_input("Model Path (optional)", value="")
        
        preset = st.selectbox(
            "Configuration Preset",
            ["", "military", "development", "production", "testing"],
            format_func=lambda x: x.capitalize() if x else "Custom"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        max_results = st.number_input(
            "Maximum Results",
            min_value=1,
            max_value=50,
            value=5
        )
    
    with tab2:
        # Advanced configuration
        st.subheader("Strategies")
        strategy = st.selectbox(
            "Ranking Strategy",
            ["similarity_only", "confidence_weighted", "hybrid_score", "military_priority"],
            index=2  # Default to hybrid_score
        )
        
        confidence_strategy = st.selectbox(
            "Confidence Strategy",
            ["similarity_based", "statistical", "ensemble", "military_calibrated"],
            index=2  # Default to ensemble 
        )
        
        st.subheader("Processing Options")
        validation_mode = st.selectbox(
            "Validation Mode",
            ["relaxed", "strict", "disabled"],
            index=0  # Default to relaxed
        )
        
        max_query_time = st.number_input(
            "Max Query Time (seconds)",
            min_value=0.1,
            max_value=30.0,
            value=2.0,
            step=0.1
        )
        
        disable_gpu = st.checkbox("Disable GPU Acceleration")
        disable_cache = st.checkbox("Disable Result Caching")
        enable_diversity = st.checkbox("Enable Diversity Filtering")
        debug = st.checkbox("Enable Debug Mode")
    
    with tab3:
        # Output options
        output_format = st.selectbox(
            "Output Format",
            ["table", "detailed", "military", "json"],
            index=0  # Default to table
        )
        
        save_output = st.checkbox("Save Results to File")
        
        output_file = None
        if save_output:
            output_dir = Path(RESULTS_DIR)
            output_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_filename = f"results_{timestamp}.json"
            
            output_filename = st.text_input("Output Filename", default_filename)
            output_file = str(output_dir / output_filename)
        
        st.subheader("Mission Parameters")
        mission_id = st.text_input("Mission ID (optional)")
        operator = st.text_input("Operator/Analyst Name (optional)")
        classification = st.selectbox(
            "Classification Level",
            ["UNCLASSIFIED", "RESTRICTED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"],
            index=0  # Default to UNCLASSIFIED
        )
        save_metadata = st.checkbox("Include Detailed Metadata")
    
    # Main content area
    if not image_path:
        st.info("Please select or upload an image to begin.")
        return
    
    # Run button
    if st.button("🚀 Run Classification", type="primary"):
        with st.spinner("Processing image..."):
            # Build parameters dictionary
            params = {
                "image": image_path,
                "database": database,
                "preset": preset if preset else None,
                "confidence_threshold": confidence_threshold,
                "similarity_threshold": similarity_threshold,
                "max_results": max_results,
                "strategy": strategy,
                "confidence_strategy": confidence_strategy,
                "validation_mode": validation_mode,
                "max_query_time": max_query_time,
                "disable_gpu": disable_gpu,
                "disable_cache": disable_cache,
                "enable_diversity": enable_diversity,
                "debug": debug,
                "format": output_format,
                "output": output_file,
                "mission_id": mission_id if mission_id else None,
                "operator": operator if operator else None,
                "classification": classification,
                "save_metadata": save_metadata
            }
            
            # Add model if specified
            if model:
                params["model"] = model
            
            # Run the mission
            success, output_text, results_dict = run_mission(params)
        
            
            if success and results_dict:
                display_results(results_dict, image_path)
                
                # Clean up temporary file if it was an upload
                if image_source == "Upload Image" and Path("temp_upload.png").exists():
                    try:
                        Path("temp_upload.png").unlink()
                    except:
                        pass
            else:
                st.error("Error running classification:")
                st.code(output_text)


if __name__ == "__main__":
    main()