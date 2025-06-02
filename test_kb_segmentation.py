"""
Simple test script to run KB-augmented segmentation comparison.
"""

import openai
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List
import re
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Simple segment class
@dataclass
class SimpleSegment:
    text: str
    start_pos: int = 0
    end_pos: int = 0

# Simple KB extractor
class SimpleKBExtractor:
    def __init__(self):
        self.client = openai.OpenAI()
    
    def extract_facts(self, text: str) -> List[str]:
        """Extract simple facts from text."""
        prompt = f"""Extract key facts from this text as: SUBJECT | PREDICATE | OBJECT

Text: {text}

Facts:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            facts = []
            content = response.choices[0].message.content
            if content:
                for line in content.strip().split('\n'):
                    if '|' in line:
                        facts.append(line.strip())
            return facts
        except Exception as e:
            print(f"Error extracting facts: {e}")
            return []

# Simple embedding model wrapper
class SimpleEmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_texts(self, texts: List[str]):
        return self.model.encode(texts)

# Simple segmenter
def simple_sentence_segmentation(text: str) -> List[SimpleSegment]:
    """Split text into sentences."""
    sentences = re.split(r'[.!?]+', text)
    segments = []
    
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) > 10:  # Skip very short segments
            segments.append(SimpleSegment(text=sent))
    
    return segments

def augment_segments_with_kb(segments: List[SimpleSegment], kb_extractor: SimpleKBExtractor) -> List[SimpleSegment]:
    """Add KB facts to segments."""
    augmented = []
    
    for seg in segments:
        facts = kb_extractor.extract_facts(seg.text)
        if facts:
            # Append facts to text
            fact_text = "; ".join(facts)
            augmented_text = f"{seg.text}\n[KB: {fact_text}]"
        else:
            augmented_text = seg.text
        
        augmented.append(SimpleSegment(text=augmented_text))
    
    return augmented

def test_kb_segmentation(text_file: str = None):
    """Test the KB-augmented segmentation."""
    
    # Load text from file or use default sample
    if text_file:
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            print(f"Loaded text from: {text_file}")
        except FileNotFoundError:
            print(f"Error: File '{text_file}' not found.")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Default sample narrative text
        text = """
        John was angry about the argument. He stormed out of the room and slammed the door.
        Mary felt terrible about what had happened. She had never seen him so upset before.
        The old house creaked in the wind. Sarah wondered if they would ever reconcile.
        Later that evening, John returned with flowers. He apologized for his outburst.
        """
        print("Using default sample text")
    
    print("\nTesting KB-Augmented Segmentation")
    print("=" * 50)
    print(f"Text preview (first 200 chars):\n{text[:200]}{'...' if len(text) > 200 else ''}\n")
    
    # Initialize components
    kb_extractor = SimpleKBExtractor()
    embedding_model = SimpleEmbeddingModel()
    
    # Baseline segmentation
    baseline_segments = simple_sentence_segmentation(text)
    print(f"Baseline segments ({len(baseline_segments)}):")
    for i, seg in enumerate(baseline_segments):
        print(f"  {i+1}: {seg.text}")
    print()
    
    # KB-augmented segmentation
    kb_segments = augment_segments_with_kb(baseline_segments, kb_extractor)
    print(f"KB-augmented segments ({len(kb_segments)}):")
    for i, seg in enumerate(kb_segments):
        print(f"  {i+1}: {seg.text}")
    print()
    
    # Compare using the segmentation comparator
    from segmentation_comparison import SimpleSegmentationComparator
    
    comparator = SimpleSegmentationComparator(embedding_model)
    results = comparator.compare_methods(text, baseline_segments, kb_segments)
    
    print("Comparison Results:")
    print("=" * 30)
    print(f"Baseline coherence: {results['baseline']['coherence']:.3f}")
    print(f"KB-augmented coherence: {results['kb_augmented']['coherence']:.3f}")
    print(f"Improvement: {results['improvement']:.3f}")
    
    if results['improvement'] > 0:
        print("✓ KB augmentation improves coherence!")
    else:
        print("✗ KB augmentation doesn't help coherence")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KB-augmented narrative segmentation")
    parser.add_argument("--text-file", "-f", type=str, help="Path to text file to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    test_kb_segmentation(args.text_file)
