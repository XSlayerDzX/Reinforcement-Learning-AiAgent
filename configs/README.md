# Configuration Files Documentation

This directory contains all configuration files (YAML format) for the Clash Royale RL Agent project.

## Configuration Files

### 1. **paths.yaml** 📁
Defines all file and directory paths used throughout the project.

**Contains:**
- Project root paths
- Data paths (raw, processed, images, temp screens)
- Model paths (checkpoints, pretrained, ONNX)
- Source code paths
- Experiment and log paths
- Output paths

**Usage:**
```python
import yaml

with open('configs/paths.yaml', 'r') as f:
    config = yaml.safe_load(f)
    data_root = config['data']['root']
    models_root = config['models']['root']
```

---

### 2. **data.yaml** 📊
Defines all data-related settings and preprocessing parameters.

**Contains:**
- Dataset configuration (name, version, description)
- Data splitting ratios (train/val/test)
- Image processing settings (size, channels, format)
- Frame sampling parameters
- Data augmentation settings (rotation, brightness, contrast, zoom)
- Card detection parameters
- Preprocessing normalization
- DataLoader configuration
- Cache settings

**Key Sections:**
- **augmentation**: Enable/disable various image transformations
- **dataloader**: Batch size, workers, shuffling options
- **normalize**: ImageNet normalization defaults

---

### 3. **training.yaml** 🚀
Defines all training-related settings and hyperparameters.

**Contains:**
- Training phases (imitation learning and reinforcement learning)
- Model architecture specifications
- Optimizer settings (Adam/SGD/AdamW)
- Learning rate scheduler configuration
- Loss functions (imitation, RL, regularization)
- Training hyperparameters (epochs, batch size, gradient clipping)
- Device settings (CUDA/CPU/MPS)
- Checkpoint and saving configuration
- Validation settings
- Logging and monitoring (TensorBoard, Weights & Biases)
- Debug options

**Key Sections:**
- **phases**: Configure imitation and reinforcement learning separately
- **optimizer**: Learning rate and weight decay settings
- **scheduler**: Learning rate scheduling strategy
- **checkpoint**: Model saving and recovery

---

### 4. **model.yaml** 🧠
Defines all model architectures and their parameters.

**Contains:**
- Vision models (backbone, card detector, state predictor, action predictor)
- Reinforcement learning models (policy network, value network)
- Model initialization methods
- Quantization and optimization settings
- Model export formats (ONNX, TorchScript, TensorFlow)
- Evaluation metrics for different tasks

**Key Sections:**
- **vision_models**: CNN architectures for vision tasks
- **rl_models**: Actor-critic network configurations
- **export**: Model deployment options
- **evaluation**: Metrics for monitoring performance

---

## Project Structure

```
clash-royale-rl-agent/
├── configs/                    # 📁 Configuration files (this directory)
│   ├── paths.yaml             # File and directory paths
│   ├── data.yaml              # Data processing and augmentation
│   ├── training.yaml          # Training hyperparameters
│   ├── model.yaml             # Model architectures
│   └── README.md              # This file
├── data/                       # 📊 Data directory
│   ├── raw/                   # Raw game recordings
│   ├── processed/             # Preprocessed data
│   └── images/                # Game screenshots
├── src/                        # 💻 Source code
│   ├── control/               # Game control (ADB, clicking)
│   ├── cr_agent/              # Agent logic
│   │   └── data_pipepline/   # Data processing
│   ├── vision/                # Computer vision models
│   └── training/              # Training scripts
├── models/                     # 🧠 Pre-trained and trained models
│   ├── checkpoints/           # Training checkpoints
│   ├── pretrained/            # Pre-trained weights
│   └── onnx/                  # ONNX model exports
├── experiments/               # 🧪 Notebooks and experiment logs
└── output/                    # 📤 Output (predictions, reports)
```

---

## How to Use Configuration Files

### Basic Usage

```python
import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_dir="./configs"):
        self.config_dir = Path(config_dir)
        self.config = {}
        
    def load_all(self):
        """Load all configuration files"""
        for yaml_file in self.config_dir.glob("*.yaml"):
            with open(yaml_file, 'r') as f:
                self.config[yaml_file.stem] = yaml.safe_load(f)
        return self.config
    
    def load_specific(self, config_name):
        """Load a specific configuration file"""
        with open(self.config_dir / f"{config_name}.yaml", 'r') as f:
            return yaml.safe_load(f)

# Usage
loader = ConfigLoader()
all_configs = loader.load_all()
training_config = loader.load_specific('training')
```

### Accessing Configuration Values

```python
# Load data configuration
with open('configs/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

# Access nested values
image_width = data_config['image']['width']
batch_size = data_config['dataloader']['batch_size']
augmentation_enabled = data_config['augmentation']['enabled']
```

---

## Best Practices

1. **Before Training**: Review and adjust hyperparameters in `training.yaml`
2. **Data Preparation**: Configure data paths in `paths.yaml` and settings in `data.yaml`
3. **Model Selection**: Choose model architecture in `model.yaml`
4. **Reproducibility**: Keep random seed consistent in `data.yaml`
5. **Versioning**: Comment changes with timestamps and reasons

---

## Configuration Modification Guide

### For Different Experiments

**Lightweight Training** (quick experiments):
```yaml
# In training.yaml
training:
  epochs: 10
  batch_size: 16
scheduler:
  type: "linear"
```

**Production Training** (full training):
```yaml
# In training.yaml
training:
  epochs: 100
  batch_size: 64
scheduler:
  type: "cosine"
optimizer:
  weight_decay: 1e-04
```

**Debug Mode**:
```yaml
# In training.yaml
debug:
  enabled: true
  save_predictions: true
  save_gradients: true
logging:
  level: "DEBUG"
```

---

## Integration with Python Code

See `src/training/Train.py` and other modules for examples of loading and using these configurations.

---

## Support

For issues or questions about configurations, check the main README.md or contact the development team.

