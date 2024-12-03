import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import Config

class AdvancedSpeechRecognitionModel:
    def __init__(self, config=Config):
        """
        Enhanced initialization with centralized configuration
        """
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None

        # Validate and create necessary paths
        config.validate_paths()

        # GPU Configuration
        self._configure_gpu()

    def _configure_gpu(self):
        """
        Advanced GPU configuration with compatibility checks.
        """
        if self.config.PERFORMANCE['use_gpu']:
            physical_devices = tf.config.list_physical_devices('GPU')

            if physical_devices:
                try:
                    # Enable memory growth
                    if self.config.PERFORMANCE['memory_growth']:
                        for device in physical_devices:
                            tf.config.experimental.set_memory_growth(device, True)

                    # Mixed precision training compatibility check
                    if self.config.PERFORMANCE['mixed_precision']:
                        # Check compute capability
                        gpu_info = tf.config.experimental.get_device_details(physical_devices[0])
                        compute_capability = gpu_info.get("compute_capability", (0, 0))
                        if compute_capability[0] >= 7:
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            print(f"✅ Mixed precision enabled: GPU compute capability {compute_capability}")
                        else:
                            print(f"⚠️ Mixed precision disabled: GPU compute capability {compute_capability} is insufficient.")

                    print(f"✅ GPU Optimization Enabled: {len(physical_devices)} GPU(s)")

                except Exception as e:
                    print(f"❌ GPU Configuration Error: {e}")
            else:
                print("❗ No GPU detected. Falling back to CPU.")

    def extract_features(self, audio_path):
        """
        Advanced feature extraction with resizing to match model input shape.
        """
        try:
            # Load audio
            audio, sample_rate = sf.read(audio_path)

            # Convert to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Apply augmentations
            augmented_audio = self._apply_audio_augmentations(audio)

            # Generate spectrogram
            spectrogram = librosa.stft(
                augmented_audio,
                n_fft=2048,
                hop_length=512
            )
            spectrogram_db = librosa.amplitude_to_db(abs(spectrogram), ref=np.max)

            # Resize frequency bins to match the model's expected input
            resized_spectrogram = librosa.util.fix_length(
                spectrogram_db, size=128, axis=0  # Resize frequency bins to 128
            )

            # Dynamic padding for time frames
            max_pad_len = self.config.MODEL['max_pad_len']
            if resized_spectrogram.shape[1] > max_pad_len:
                resized_spectrogram = resized_spectrogram[:, :max_pad_len]
            else:
                pad_width = max_pad_len - resized_spectrogram.shape[1]
                resized_spectrogram = np.pad(
                    resized_spectrogram,
                    ((0, 0), (0, pad_width)),
                    mode='constant'
                )

            # Add a channel dimension
            return np.expand_dims(resized_spectrogram, axis=-1)

        except Exception as e:
            print(f"Feature extraction error for {audio_path}: {e}")
            return None

    def _apply_audio_augmentations(self, audio):
        """
        Advanced audio augmentation techniques
        """
        # Noise injection
        noise = np.random.normal(0, 0.005 * np.max(audio), audio.shape)
        audio_with_noise = audio + noise

        # Slight time stretching
        stretch_factor = np.random.uniform(0.9, 1.1)
        stretched_audio = librosa.effects.time_stretch(audio_with_noise, rate=stretch_factor)

        return stretched_audio

    def build_model(self):
        """
        Advanced CNN architecture for spoken digit recognition
        """
        input_shape = (self.config.MODEL['max_pad_len'], self.config.MODEL['max_pad_len'], 1)

        model = models.Sequential([
            layers.Input(shape=input_shape),

            # Enhanced First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Improved Separable Convolution Block
            layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Third Convolutional Block with Increased Depth
            layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),

            # Global Features with Additional Processing
            layers.GlobalAveragePooling2D(),

            # Enhanced Dense Layers
            layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(128, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Output Layer
            layers.Dense(10, activation='softmax')
        ])

        return model

    def train(self):
        """
        Enhanced training method with comprehensive callbacks and analysis
        """
        # Load and preprocess dataset
        X, y = self._load_dataset()

        # Log dataset statistics
        self._log_dataset_stats(X, y)

        # Reshape data if needed
        if X.ndim == 5:
            X = X.squeeze(axis=-1)

        # Convert labels to categorical
        y = tf.keras.utils.to_categorical(y, num_classes=10)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TRAINING['test_size'],
            stratify=np.argmax(y, axis=1),
            random_state=42
        )

        # Build model
        self.model = self.build_model()

        # Compile model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.TRAINING['learning_rate']
        )

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create log directory
        log_dir = os.path.join(
            self.config.LOGGING['log_dir'],
            f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        os.makedirs(log_dir, exist_ok=True)

        # Comprehensive Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                min_delta=0.001
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-5,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'best_model/best_model_{epoch:02d}_{val_accuracy:.3f}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            callbacks.CSVLogger(
                filename=os.path.join(log_dir, 'training_log.csv'),
                separator=',',
                append=False
            )
        ]

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.config.TRAINING['epochs'],
            batch_size=self.config.TRAINING['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )

        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        # Prediction and Analysis
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Classification Report
        from sklearn.metrics import classification_report, confusion_matrix

        print("\nDetailed Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes,
                                    target_names=[str(i) for i in range(10)]))

        # Confusion Matrix Visualization
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[str(i) for i in range(10)],
                    yticklabels=[str(i) for i in range(10)])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'))
        plt.close()

        return self.history

    def _log_dataset_stats(self, X, y):
        """
        Log comprehensive dataset statistics
        """
        class_counts = np.bincount(y)
        total_samples = len(y)

        print("\n🔍 Dataset Statistics:")
        print(f"Total Samples: {total_samples}")
        print("-" * 30)

        for i, count in enumerate(class_counts):
            percentage = count / total_samples * 100
            print(f"Digit {i}: {count} samples ({percentage:.2f}%)")

        print("-" * 30)
        print(f"Input Shape: {X.shape}")

    def _load_dataset(self):
        """
        Load and preprocess the audio dataset for training
        """
        X = []
        y = []

        # Print the full base path to help debug
        print(f"Looking for dataset in: {self.config.DATASET['base_path']}")

        # Check if base dataset directory exists
        if not os.path.exists(self.config.DATASET['base_path']):
            raise FileNotFoundError(f"Dataset base directory not found: {self.config.DATASET['base_path']}")

        # Iterate through dataset subdirectories
        for label, subdir in enumerate(self.config.DATASET['subdirs']):
            subdir_path = os.path.join(self.config.DATASET['base_path'], subdir)

            # Check if directory exists
            if not os.path.exists(subdir_path):
                print(f"Warning: Directory {subdir_path} not found")
                continue

            # Iterate through audio files in the subdirectory
            audio_files = [f for f in os.listdir(subdir_path) if f.endswith(self.config.DATASET['file_extension'])]

            if not audio_files:
                print(f"No audio files found in {subdir_path}")
                continue

            for filename in audio_files:
                file_path = os.path.join(subdir_path, filename)

                try:
                    # Extract features
                    features = self.extract_features(file_path)

                    if features is not None:
                        # Reshape features for model input
                        features = features.reshape((*features.shape, 1))
                        X.append(features)
                        y.append(label)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Provide more detailed error message
        if len(X) == 0:
            raise ValueError(f"""
            No audio files could be processed in the dataset.
            Please ensure:
            1. Dataset directory exists: {self.config.DATASET['base_path']}
            2. Subdirectories exist: {self.config.DATASET['subdirs']}
            3. .wav files are present and valid in these subdirectories
            4. Feature extraction is working correctly
            """)

        print(f"Loaded dataset: {len(X)} samples across {len(np.unique(y))} classes")

        return X, y

    def save_model(self, save_path=None, model_name="digit_recognition_model"):
        """
        Save trained model with comprehensive metadata
        """
        if save_path is None:
            save_path = self.config.MODEL['save_path']

        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, model_name+'.h5')

        self.model.save(model_path)
        print(f"✅ Model saved successfully at {model_path}")

    def load_model(self, model_path):
        """
        Load a pre-trained model from an H5 file
        """
        self.model = tf.keras.models.load_model(model_path)