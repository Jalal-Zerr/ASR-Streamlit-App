import streamlit as st
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
import glob
import gdown
import requests
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Modify import to handle potential errors
try:
    from advanced_speech_recognition import AdvancedSpeechRecognitionModel
except ImportError as e:
    st.error(f"Error importing model: {e}")
    st.error(traceback.format_exc())
    sys.exit(1)

# Enhanced page configuration
st.set_page_config(
    page_title="Speech Recognition Digit Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def display_project_metadata() -> None:
    st.markdown("""
    <style>
    .project-metadata-card {
        background: linear-gradient(135deg, #2E3A59 0%, #c3cfe2 100%);
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        padding: 25px;
        display: flex;
        align-items: center;
        gap: 20px;
        transition: all 0.3s ease;
        max-width: 1000px;
        margin: 10px auto;
    }
    .project-metadata-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    .project-logo {
        width: 300px; /* Taille réduite de la largeur */
        max-width: 100px; /* Limite de largeur maximale */
        height: auto; /* Pour préserver le ratio d'aspect */
        border-radius: 12px;
        border: 3px solid #2c3e50;
        transition: transform 0.3s ease;
    }
    .project-logo:hover {
        transform: scale(1.05);
    }
    .project-details {
        flex-grow: 1;
    }
    .project-details h2 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .project-details .metadata-list {
        list-style-type: none;
        padding: 0;
    }
    .project-details .metadata-list li {
        margin-bottom: 10px;
        color: #34495e;
        display: flex;
        align-items: center;
        font-size: 1rem;
    }
    .project-details .metadata-list li .icon {
        margin-right: 15px;
        color: #3498db;
        font-size: 1.2rem;
        min-width: 30px;
        text-align: center;
    }
    .project-details .metadata-list li .label {
        font-weight: bold;
        min-width: 120px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="project-metadata-card">
        <img src="https://www.fsdm.usmba.ac.ma/Template/useful/loading/img/logo.png" alt="Project Logo" class="project-logo">
        <div class="project-details">
            <h2>🎙️ Speech Recognition Project</h2>
            <ul class="metadata-list">
                <li><span class="icon">🏫</span><span class="label">Institution:</span> Université Sidi Mohamed Ben Abdellah</li>
                <li><span class="icon">📚</span><span class="label">Faculty:</span> Sciences Dhar El Mehraz - Fès</li>
                <li><span class="icon">🎓</span><span class="label">Program:</span> Master BDSI</li>
                <li><span class="icon">👨‍🏫</span><span class="label">Supervisor:</span> Prof. Hassan SATORI</li>
                <li><span class="icon">👨‍💻</span><span class="label">Developer:</span> Jalal Zerroudi</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True )


def validate_project_metadata() -> None:
    """
    Detailed project metadata validation with expandable section
    """
    detailed_metadata = {
        "🔬 Research Domain": "Automatic Speech Recognition",
        "💻 Technologies": "TensorFlow, Python, Streamlit",
        "📅 Project Duration": "2023-2024",
        "🌐 Scope": "Advanced Machine Learning Application"
    }

    with st.expander("🔍 Extended Project Details"):
        for icon, detail in detailed_metadata.items():
            st.markdown(f"{icon} **{detail.split(':')[0]}:** {detail.split(':')[1] if ':' in detail else detail}")



class SpeechRecognitionApp:
    def __init__(self):
        """
        Initialize the application with session state management
        """
        # Initialize session state
        if 'model' not in st.session_state:
            st.session_state.model = None
            st.session_state.model_loaded = False

    def run(self):
        """
        Main application runner
        """
        display_project_metadata()
        # Sidebar navigation
        page = st.sidebar.radio(
            "Select Page",
            ["Train Model", "Make Prediction", "Model Information"]
        )

        # Page routing
        if page == "Train Model":
            self.train_model_page()
        elif page == "Make Prediction":
            self.predict_page()
        elif page == "Model Information":
            self.model_info_page()

    def train_model_page(self):
        """
        Model training interface
        """
        st.header("🚀 Model Training")

        # Training configuration
        st.subheader("Training Parameters")

        # Dataset path
        dataset_path = st.text_input(
            "Dataset Path",
            placeholder="Enter full path to dataset directory"
        )

        # Training configuration columns
        col1, col2 = st.columns(2)

        with col1:
            epochs = st.slider(
                "Number of Epochs",
                min_value=10,
                max_value=100,
                value=50
            )
            test_size = st.slider(
                "Test Set Proportion",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05
            )

        with col2:
            batch_size = st.slider(
                "Batch Size",
                min_value=8,
                max_value=64,
                value=32
            )
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-2,
                value=1e-3,
                format="%.4f"
            )

        # Train button
        if st.button("Train Model"):
            if not os.path.exists(dataset_path):
                st.error("Invalid dataset path!")
                return

            try:
                # Create model instance
                with st.spinner("Training in progress..."):
                    # Assuming AdvancedSpeechRecognitionModel can accept these parameters
                    model = AdvancedSpeechRecognitionModel()

                    # Update configuration dynamically
                    model.config.DATASET['base_path'] = dataset_path
                    model.config.TRAINING['epochs'] = epochs
                    model.config.TRAINING['test_size'] = test_size
                    model.config.TRAINING['batch_size'] = batch_size
                    model.config.TRAINING['learning_rate'] = learning_rate

                    # Train the model
                    history = model.train()

                    # Save the model
                    model.save_model("test_gpu_v01")

                    # Store in session state
                    st.session_state.model = model
                    st.session_state.model_loaded = True

                # Plot training history
                self._plot_training_history(history)

                st.success("Model trained successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"Training failed: {e}")
                st.error(traceback.format_exc())

    def model_info_page(self):
        """
        Comprehensive Model Information Page with Enhanced Visualization and Error Handling
        """
        st.header("📊 Model Insights")

        # Model loading options
        upload_method = st.radio(
            "Choose Model Upload Method",
            ["Upload from Local", "Download from Google Drive"]
        )

        model_path = None

        if upload_method == "Upload from Local":
            # Local file upload
            uploaded_model = st.file_uploader(
                "Upload Pre-trained Model File", 
                type=['h5', 'pth', 'keras', 'model'],
                help="Upload a pre-trained model file"
            )

            if uploaded_model is not None:
                # Ensure models directory exists
                os.makedirs("models", exist_ok=True)
                
                # Save the uploaded file
                model_path = os.path.join("models", uploaded_model.name)
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                st.success(f"Model {uploaded_model.name} uploaded successfully!")

        else:
            # Google Drive upload method
            share_method = st.radio(
                "Choose How to Share the File",
                ["Direct Google Drive Share Link", "Google Drive File ID"]
            )

            if share_method == "Direct Google Drive Share Link":
                drive_link = st.text_input(
                    "Google Drive Shareable Link use this link : https://drive.google.com/file/d/1zPrsdalDVOjgGr0rGPBXq4HpoeYa9h2E/view?usp=sharing", 
                    placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
                )

                if drive_link and st.button("Download from Drive"):
                    try:
                        file_id = drive_link.split('/d/')[1].split('/')[0]
                        download_url = f"https://drive.google.com/uc?id={file_id}"
                        
                        with st.spinner("Downloading model..."):
                            os.makedirs("models", exist_ok=True)
                            model_path = os.path.join("models", "downloaded_model.h5")
                            gdown.download(download_url, model_path, quiet=False)
                            
                            st.success("Model downloaded successfully!")

                    except Exception as e:
                        st.error(f"Error processing link: {e}")

        # If a model path is available, load and display model information
        if model_path and os.path.exists(model_path):
            try:
                # Attempt to load the model with multiple methods
                loaded_model = None
                model_type = "Unknown"
                
                # Try Keras/TensorFlow model loading
                try:
                    loaded_model = tf.keras.models.load_model(model_path)
                    st.success("Model loaded successfully using TensorFlow/Keras")
                    model_type = "Keras/TensorFlow"
                except Exception as keras_error:
                    st.warning(f"Keras loading failed: {keras_error}")
                    
                    # Try PyTorch model loading
                    try:
                        import torch
                        loaded_model = torch.load(model_path)
                        st.success("Model loaded successfully using PyTorch")
                        model_type = "PyTorch"
                    except Exception as pytorch_error:
                        st.error(f"Failed to load model: {pytorch_error}")
                        return

                # Model Basic Information
                st.subheader("Model Overview")
                st.write(f"**Model Type:** {model_type}")
                st.write(f"**Model Path:** {model_path}")
                
                # Visualization Section
                if model_type == "Keras/TensorFlow":
                    # Layer Type Distribution
                    layer_types = [type(layer).__name__ for layer in loaded_model.layers]
                    layer_type_counts = {}
                    for layer_type in layer_types:
                        layer_type_counts[layer_type] = layer_type_counts.get(layer_type, 0) + 1

                    # Pie Chart of Layer Types
                    st.subheader("Layer Type Distribution")
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(layer_type_counts.keys()), 
                        values=list(layer_type_counts.values()),
                        hole=.3
                    )])
                    fig_pie.update_layout(title="Layer Type Composition")
                    st.plotly_chart(fig_pie)

                    # Layer Parameter Visualization
                    st.subheader("Layer Parameter Analysis")
                    
                    # Compute layer parameters
                    layer_params = [
                        {
                            "Name": layer.name, 
                            "Type": type(layer).__name__, 
                            "Params": layer.count_params()
                        } 
                        for layer in loaded_model.layers if layer.count_params() > 0
                    ]

                    # Bar chart of layer parameters
                    fig_bar = px.bar(
                        x=[layer['Name'] for layer in layer_params],
                        y=[layer['Params'] for layer in layer_params],
                        labels={'x': 'Layer Name', 'y': 'Number of Parameters'},
                        title="Layer-wise Parameter Count",
                        color=[layer['Type'] for layer in layer_params]
                    )
                    st.plotly_chart(fig_bar)

                    # Network Architecture Visualization
                    st.subheader("Network Architecture Flow")
                    
                    # Create network graph
                    G = nx.DiGraph()
                    for i, layer in enumerate(loaded_model.layers):
                        G.add_node(layer.name, type=type(layer).__name__)
                        if i > 0:
                            G.add_edge(loaded_model.layers[i-1].name, layer.name)

                    # Convert to Plotly graph
                    edge_trace = go.Scatter(
                        x=[],
                        y=[],
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines')

                    node_trace = go.Scatter(
                        x=[],
                        y=[],
                        text=[],
                        mode='markers+text',
                        hoverinfo='text',
                        marker=dict(
                            showscale=True,
                            colorscale='Viridis',
                            size=10,
                            colorbar=dict(
                                thickness=15,
                                title='Node Connections',
                                xanchor='left',
                                titleside='right'
                            )
                        )
                    )

                    # Position nodes
                    pos = nx.spring_layout(G)
                    for node in G.nodes():
                        x, y = pos[node]
                        node_trace['x'] += tuple([x])
                        node_trace['y'] += tuple([y])
                        node_trace['text'] += tuple([node])

                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_trace['x'] += tuple([x0, x1, None])
                        edge_trace['y'] += tuple([y0, y1, None])

                    fig_network = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title='Model Architecture Network',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

                    st.plotly_chart(fig_network)

                    # Layer Details
                    st.subheader("Layer Details")
                    layers_info = []
                    for layer in loaded_model.layers:
                        try:
                            layer_info = {
                                "Name": layer.name,
                                "Type": type(layer).__name__,
                                "Trainable": layer.trainable
                            }
                            
                            # Robust input shape handling
                            try:
                                input_shape = layer.get_input_shape_at(0)
                                layer_info["Input Shape"] = str(input_shape)
                            except Exception:
                                layer_info["Input Shape"] = "Multiple/Complex Inputs"
                            
                            # Robust output shape handling
                            try:
                                output_shape = layer.output_shape
                                layer_info["Output Shape"] = str(output_shape)
                            except Exception:
                                layer_info["Output Shape"] = "Multiple/Complex Outputs"
                            
                            layers_info.append(layer_info)
                        except Exception as layer_error:
                            st.warning(f"Could not fully process layer {layer.name}: {layer_error}")
                    
                    st.dataframe(layers_info)

                    # Additional model configuration
                    st.subheader("Model Configuration")
                    st.write(f"Total Layers: {len(loaded_model.layers)}")
                    st.write(f"Trainable Parameters: {loaded_model.count_params():,}")

                # Handling PyTorch models
                elif model_type == "PyTorch":
                    st.write("PyTorch Model Details:")
                    st.code(str(loaded_model))
                    
                    try:
                        layers_info = [
                            {
                                "Name": name, 
                                "Type": str(module.__class__.__name__),
                            } 
                            for name, module in loaded_model.named_children()
                        ]
                        st.dataframe(layers_info)
                    except Exception as pytorch_layer_error:
                        st.warning(f"Could not extract layer details: {pytorch_layer_error}")

            except Exception as e:
                st.error(f"Unexpected error loading model: {e}")
                st.error(traceback.format_exc())
        else:
            st.info("Please upload or download a model to view its details.")

    def predict_page(self):
        """
        Prediction interface with model loading and persistent audio recording
        """
        st.header("🔮 Digit Prediction")

        # Check if model is already loaded in session state
        if 'loaded_model_path' not in st.session_state:
            st.session_state.loaded_model_path = None
            st.session_state.model_instance = None

        # Model upload method selection
        upload_method = st.radio(
            "Choose Model Upload Method",
            ["Upload from Local", "Download from Google Drive"]
        )

        # Local file upload
        if upload_method == "Upload from Local":
            uploaded_model = st.file_uploader(
                "Upload Pre-trained Model (.h5 file)", 
                type=['h5'],
                help="Upload a pre-trained Keras model file (.h5)"
            )

            if uploaded_model is not None:
                os.makedirs("models", exist_ok=True)
                model_path = os.path.join("models", uploaded_model.name)
                
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                # Store in session state
                st.session_state.loaded_model_path = model_path
                st.success(f"Model {uploaded_model.name} uploaded successfully!")

        # Google Drive upload method
        else:
            share_method = st.radio(
                "Choose How to Share the File , use this line : https://drive.google.com/file/d/1zPrsdalDVOjgGr0rGPBXq4HpoeYa9h2E/view?usp=sharing",
                ["Direct Google Drive Share Link", "Google Drive File ID"]
            )

            if share_method == "Direct Google Drive Share Link":
                drive_link = st.text_input(
                    "Google Drive Shareable Link", 
                    placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
                )

                if drive_link and st.button("Download from Drive"):
                    try:
                        file_id = drive_link.split('/d/')[1].split('/')[0]
                        download_url = f"https://drive.google.com/uc?id={file_id}"
                        
                        with st.spinner("Downloading model..."):
                            os.makedirs("models", exist_ok=True)
                            model_path = os.path.join("models", "downloaded_model.h5")
                            gdown.download(download_url, model_path, quiet=False)
                            
                            # Store in session state
                            st.session_state.loaded_model_path = model_path
                            st.success("Model downloaded successfully!")

                    except Exception as e:
                        st.error(f"Error processing link: {e}")

            else:  # Google Drive File ID method
                file_id = st.text_input(
                    "Enter Google Drive File ID", 
                    placeholder="1234abcd_your_file_id_here"
                )

                if st.button("Download from Drive") and file_id:
                    try:
                        with st.spinner("Downloading model..."):
                            download_url = f"https://drive.google.com/uc?id={file_id}"
                            
                            os.makedirs("models", exist_ok=True)
                            model_path = os.path.join("models", "downloaded_model.h5")
                            gdown.download(download_url, model_path, quiet=False)
                            
                            # Store in session state
                            st.session_state.loaded_model_path = model_path
                            st.success("Model downloaded successfully!")

                    except Exception as e:
                        st.error(f"Error downloading file: {e}")

        # Proceed only if model is in session state
        if st.session_state.loaded_model_path and os.path.exists(st.session_state.loaded_model_path):
            # Audio input method selection
            input_method = st.radio(
                "Choose Audio Input Method",
                ["Upload WAV File", "Record Audio"]
            )

            # Persistent audio file path
            TEMP_AUDIO_PATH = "temp_audio.wav"

            if input_method == "Upload WAV File":
                uploaded_audio = st.file_uploader(
                    "Upload WAV File",
                    type=['wav'],
                    help="Upload a .wav file with a spoken digit"
                )
                
                if uploaded_audio is not None:
                    with open(TEMP_AUDIO_PATH, "wb") as f:
                        f.write(uploaded_audio.getbuffer())
                    
                    st.audio(TEMP_AUDIO_PATH, format="audio/wav")

            else:
                import sounddevice as sd
                import soundfile as sf

                duration = 3  # seconds
                sample_rate = 44100  # standard sample rate
                channels = 1  # mono recording

                if st.button("Start Recording (3 seconds)"):
                    st.info("Recording... Speak a digit clearly")
                    
                    recording = sd.rec(
                        int(duration * sample_rate), 
                        samplerate=sample_rate, 
                        channels=channels
                    )
                    sd.wait()  # Wait until recording is finished
                    
                    sf.write(TEMP_AUDIO_PATH, recording, sample_rate)
                    
                    st.success("Recording completed!")
                    st.audio(TEMP_AUDIO_PATH, format="audio/wav")

            # Prediction button (always available if audio exists)
            if os.path.exists(TEMP_AUDIO_PATH):
                if st.button("Predict Digit"):
                    try:
                        # Lazy load model instance only when prediction is needed
                        if st.session_state.model_instance is None:
                            model_instance = AdvancedSpeechRecognitionModel()
                            model_instance.load_model(st.session_state.loaded_model_path)
                            st.session_state.model_instance = model_instance
                        
                        # Extract features
                        features = st.session_state.model_instance.extract_features(TEMP_AUDIO_PATH)

                        # Reshape features
                        features = features.reshape((1, *features.shape, 1))

                        # Make prediction
                        prediction = st.session_state.model_instance.model.predict(features)
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)

                        # Display results
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Predicted Digit", predicted_class)

                        with col2:
                            st.metric("Confidence", f"{confidence*100:.2f}%")

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.error(traceback.format_exc())
            else:
                st.warning("Please upload or record an audio file first.")
        else:
            st.info("Please upload or download a model to continue.")

    def _plot_training_history(self, history):
        """
        Visualize training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        st.pyplot(fig)

def main():
    app = SpeechRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()