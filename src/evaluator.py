from typing import List, Dict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeSequenceEvaluator:
    @staticmethod
    def calculate_metrics(predicted_nodes: List[List[str]], actual_nodes: List[List[str]]) -> Dict:
        """Calculate evaluation metrics for node sequence predictions."""

        all_nodes = list(set([node for nodes in actual_nodes for node in nodes]))
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Convert to binary matrices
        y_true = np.zeros((len(actual_nodes), len(all_nodes)))
        y_pred = np.zeros((len(predicted_nodes), len(all_nodes)))
        
        for i, nodes in enumerate(actual_nodes):
            for node in nodes:
                if node in node_to_idx:
                    y_true[i, node_to_idx[node]] = 1
                    
        for i, nodes in enumerate(predicted_nodes):
            for node in nodes:
                if node in node_to_idx:
                    y_pred[i, node_to_idx[node]] = 1
        
        # Calculating metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Calculate exact match
        exact_matches = sum(
            1 for pred, actual in zip(predicted_nodes, actual_nodes)
            if set(pred) == set(actual)
        )
        exact_match_ratio = exact_matches / len(actual_nodes)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'exact_match_ratio': exact_match_ratio
        }
        
        return metrics
