#config.py
import os

class Config:
    """Centralized configuration management for the Speech Recognition Project"""
    # Base Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = BASE_DIR

    # Dataset Configuration
    DATASET = {
        'base_path': os.path.join(BASE_DIR, 'dataset'),
        'subdirs': ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'],
        'file_extension': '.wav'
    }

    # Model Configuration
    MODEL = {
        'name': 'AdvancedSpeechRecognition',
        'save_path': os.path.join(BASE_DIR, 'models'),
        'max_pad_len': 128,
        'input_shape': (None, None, 1)
    }

    # Performance Configuration
    PERFORMANCE = {
        'use_gpu': True,
        'mixed_precision': True,
        'memory_growth': True
    }

    # Training Hyperparameters
    TRAINING = {
        'test_size': 0.2,
        'validation_split': 0.2,
        'epochs': 100,
        'batch_size': 8,
        'early_stopping_patience': 20,
        'learning_rate': 0.001
    }

    # Logging and Visualization
    LOGGING = {
        'verbose': True,
        'log_dir': os.path.join(BASE_DIR, 'logs'),
        'tensorboard_logs': True
    }

    @classmethod
    def validate_paths(cls):
        """Validates and creates necessary directories"""
        paths_to_create = [
            cls.DATASET['base_path'],
            cls.MODEL['save_path'],
            cls.LOGGING['log_dir']
        ]

        for path in paths_to_create:
            os.makedirs(path, exist_ok=True)

    @classmethod
    def validate_dataset(cls):
        """
        Validates the dataset directory and its contents
        """
        # Check if base dataset directory exists
        if not os.path.exists(cls.DATASET['base_path']):
            raise FileNotFoundError(f"Dataset directory not found: {cls.DATASET['base_path']}")

        # Check subdirectories
        for subdir in cls.DATASET['subdirs']:
            subdir_path = os.path.join(cls.DATASET['base_path'], subdir)
            if not os.path.exists(subdir_path):
                raise FileNotFoundError(f"Subdirectory not found: {subdir_path}")

            # Check for audio files
            audio_files = [f for f in os.listdir(subdir_path) if f.endswith(cls.DATASET['file_extension'])]
            if not audio_files:
                print(f"Warning: No audio files found in {subdir_path}")