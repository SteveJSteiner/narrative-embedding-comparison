"""
Simple comparison of baseline vs KB-augmented text segmentation.
"""

from typing import List, Dict, Any
import numpy as np


class SimpleSegmentationComparator:
    """Minimal segmentation comparison focused on core idea."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def compare_methods(self, text: str, baseline_segments: List, kb_segments: List) -> Dict[str, Any]:
        """
        Compare baseline vs KB-augmented segmentation.
        
        Args:
            text: Original text
            baseline_segments: List of baseline text segments
            kb_segments: List of KB-augmented segments
            
        Returns:
            Simple comparison metrics
        """
        return {
            "baseline": {
                "count": len(baseline_segments),
                "avg_length": np.mean([len(s.text) for s in baseline_segments]) if baseline_segments else 0,
                "coherence": self._coherence(baseline_segments)
            },
            "kb_augmented": {
                "count": len(kb_segments),
                "avg_length": np.mean([len(s.text) for s in kb_segments]) if kb_segments else 0,
                "coherence": self._coherence(kb_segments)
            },
            "improvement": self._coherence(kb_segments) - self._coherence(baseline_segments)
        }
    
    def _coherence(self, segments: List) -> float:
        """Calculate coherence = avg similarity between adjacent segments."""
        if len(segments) < 2:
            return 1.0
        
        try:
            texts = [seg.text for seg in segments]
            embeddings = self.embedding_model.embed_texts(texts)
            
            similarities = []
            for i in range(len(embeddings) - 1):
                # Cosine similarity
                dot_product = np.dot(embeddings[i], embeddings[i+1])
                norm_product = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
                sim = dot_product / norm_product if norm_product > 0 else 0
                similarities.append(sim)
            
            return float(np.mean(similarities))
            
        except Exception as e:
            print(f"Error calculating coherence: {e}")
            return 0.0
