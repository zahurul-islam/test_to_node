# Text-to-Node

This tool converts text prompts to sequences of operation nodes, leveraging advanced language models for workflow analysis.

## Overview

Text-to-Node is an AI-powered system that analyzes natural language descriptions of workflows and converts them into structured sequences of operational nodes. The system uses fine-tuned language models to understand and decompose complex process descriptions into their fundamental steps.

## Hardware Requirements

- NVIDIA GeForce RTX 4090
- Intel(R) Core(TM) i9-14900F
- 64 GB RAM

## Prerequisites

Due to hardware constraints, we use a quantized version of the original model:

- Base Model: `download 'codegemma-7b.Q4_K_M.gguf' from 'bartowski/codegemma-7b-GGUF` (download from Huggingface)
- Place the model in the `model` folder before running

## Installation
1. Clone this repository
2. Install requirements:
```bash
pip install -r requirements.txt
```

## Docker
Build the Docker image:
```bash
docker build -t incari-ml .
```

Run the container:
```bash
docker run -p 8000:8000 incari-ml
```

## Usage
### Training
Run the training script:
```bash
python src/train.py
```

### Inference
Start the API server:
```bash
python src/api/main.py
```

### Testing
Run unit tests:
```bash
python src/test.py
```

## File Structure
```
.
├── checkpoints/          # Training checkpoints
├── data/                 # Training data
│   └── training_examples.json
├── models/               # Pretrained models
│   ├ 
│   └── codegemma-7b.Q4_K_M.gguf
├── src/                  # Source code
│   ├── api/              # API implementation
│   │   └── main.py
│   ├── main.py           # Main entry point
│   ├── preprocess.py     # Data preprocessing
│   ├── test_api.py       # API tests
│   ├── test.py           # Unit tests
│   └── train.py          # Training script
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Models
- codegemma-7b.Q4_K_M.gguf

## Training
Training checkpoints are saved in the `checkpoints/` directory with timestamps.

## API
The API is implemented using FastAPI and can be started with:
```bash
python src/api/main.py
```

## Model Details

- Base Model: `bartowski/codegemma-7b-GGUF`
- Training Split: 85% training, 13% validation, 2% testing
- Fine-tuned on approximately 1050 examples
- Optimized for RTX 4090 GPU

## License

This project is licensed under the GPL-3.0 license - see the LICENSE file for details.


## Acknowledgments

- HuggingFace for providing the base model

---

For more information or support, please open an issue in the repository.
