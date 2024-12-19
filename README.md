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

- Base Model: `bartowski/codegemma-7b-GGUF` (download from Huggingface)
- Place the model in the `model` folder before running

## Project Structure

```
text_to_node/
├── data/
│   └── training_data.json
├── src/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── data_processor.py
│   ├── model.py
│   ├── config.py
│   └── train.py
	
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   
│   
├── model/
│   └── [place model files here]
├── Dockerfile
├── main.py
└── requirements.txt
```

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/text-to-node.git
cd text-to-node
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training and Evaluation

1. Run the main training script:
```bash
python src/main.py
```

2. Start the API server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t text-to-node .
```

2. Run the container:
```bash
docker run -p 8000:8000 text-to-node
```

## API Usage

The API endpoint accepts POST requests with text descriptions and returns the corresponding node sequences.

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "When user clicks submit button, validate form data and show confirmation"}'
```

Example response:
```json
{
    "nodes": ["OnClick", "Branch", "Show"]
}
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Model Details

- Base Model: `bartowski/codegemma-7b-GGUF`
- Training Split: 85% training, 13% validation, 2% testing
- Fine-tuned on approximately 1050 examples
- Optimized for RTX 4090 GPU

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- HuggingFace for providing the base model

---

For more information or support, please open an issue in the repository.