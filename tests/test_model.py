import pytest
from src.model import NodeSequenceModel
from src.config import MODEL_NAME, TEST_SPLIT
from src.data_processor import load_and_split_data, prepare_datasets

def test_model_prediction():
    model = NodeSequenceModel(MODEL_NAME)
    
    # Test prediction
    prompt = "Display a list of products and allow users to add items to their cart"
    nodes = model.predict(prompt)
    
    assert isinstance(nodes, list)
    assert len(nodes) > 0
    assert all(isinstance(node, str) for node in nodes)

def test_model_save_load(tmp_path):
   
    model = NodeSequenceModel(MODEL_NAME)
    model.save(str(tmp_path))
       
    loaded_model = NodeSequenceModel.load(str(tmp_path))
    
    # Test prediction with loaded model
    prompt = "Display a list of products"
    original_prediction = model.predict(prompt)
    loaded_prediction = loaded_model.predict(prompt)
    
    assert original_prediction == loaded_prediction
