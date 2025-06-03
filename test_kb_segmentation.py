"""
Simple test script to run KB-augmented segmentation comparison.
"""

import openai
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
                max_tokens=2000
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

# LLM-based segmentation using KB context
def llm_kb_segmentation(text: str, kb_facts: List[str], client) -> List[SimpleSegment]:
    """Let LLM segment text using KB facts as context."""
    facts_context = "; ".join(kb_facts) if kb_facts else "No facts extracted"
    
    prompt = f"""Segment this text into meaningful narrative units using the knowledge facts as context.

Text:
{text}

Knowledge Facts: {facts_context}

Instructions: Insert [BREAK] markers where you think segments should be split. Consider character moments, scene changes, and thematic shifts based on the knowledge facts.

Segmented text:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=12000
        )
        
        segmented_text = response.choices[0].message.content
        if not segmented_text:
            return simple_sentence_segmentation(text)
            
        # Split on [BREAK] markers
        segments = []
        parts = segmented_text.split('[BREAK]')
        
        for part in parts:
            part = part.strip()
            if len(part) > 10:  # Skip very short segments
                segments.append(SimpleSegment(text=part))
        
        return segments if segments else simple_sentence_segmentation(text)
        
    except Exception as e:
        print(f"Error in LLM segmentation: {e}")
        return simple_sentence_segmentation(text)

# LLM-only segmentation (no KB context)
def llm_only_segmentation(text: str, client) -> List[SimpleSegment]:
    """Let LLM segment text WITHOUT KB context."""
    prompt = f"""Segment this text into meaningful narrative units.

Text:
{text}

Instructions: Insert [BREAK] markers where you think segments should be split. Consider character moments, scene changes, and thematic shifts.

Segmented text:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=8000
        )
        
        segmented_text = response.choices[0].message.content
        if not segmented_text:
            return simple_sentence_segmentation(text)
            
        # Split on [BREAK] markers
        segments = []
        parts = segmented_text.split('[BREAK]')
        
        for part in parts:
            part = part.strip()
            if len(part) > 10:  # Skip very short segments
                segments.append(SimpleSegment(text=part))
        
        return segments if segments else simple_sentence_segmentation(text)
        
    except Exception as e:
        print(f"Error in LLM-only segmentation: {e}")
        return simple_sentence_segmentation(text)

def test_kb_segmentation(text_file: str):
    """Test the KB-augmented segmentation."""
    
    # Load text from file
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
    
    print("\nTesting KB-Augmented Segmentation")
    print("=" * 50)
    print(f"Text preview (first 200 chars):\n{text[:200]}{'...' if len(text) > 200 else ''}\n")
      # Initialize components
    kb_extractor = SimpleKBExtractor()
      # Baseline segmentation
    baseline_segments = simple_sentence_segmentation(text)
    print(f"Baseline segments ({len(baseline_segments)}):")
    for i, seg in enumerate(baseline_segments):
        print(f"  {i+1}: {seg.text[:60]}{'...' if len(seg.text) > 60 else ''}")
    print()
    
    # Extract KB facts from full text
    print("Extracting KB facts...")
    all_facts = kb_extractor.extract_facts(text)
    print(f"Extracted {len(all_facts)} facts")
      # LLM-only segmentation (no KB)
    print("Creating LLM-only segments...")
    llm_only_segments = llm_only_segmentation(text, kb_extractor.client)
    print(f"LLM-only segments ({len(llm_only_segments)}):")
    for i, seg in enumerate(llm_only_segments):
        print(f"  {i+1}: {seg.text[:60]}{'...' if len(seg.text) > 60 else ''}")
    print()
      # LLM-based KB-informed segmentation
    print("Creating LLM KB-informed segments...")
    llm_kb_segments = llm_kb_segmentation(text, all_facts, kb_extractor.client)
    print(f"LLM KB segments ({len(llm_kb_segments)}):")
    for i, seg in enumerate(llm_kb_segments):
        print(f"  {i+1}: {seg.text[:60]}{'...' if len(seg.text) > 60 else ''}")
    print()
    
    print("Segmentation Complete!")
    print("=" * 50)
    print(f"Baseline segments: {len(baseline_segments)}")
    print(f"LLM-only segments: {len(llm_only_segments)}")
    print(f"LLM+KB segments: {len(llm_kb_segments)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test KB-augmented narrative segmentation")
    parser.add_argument("--text-file", "-f", type=str, required=True, help="Path to text file to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    test_kb_segmentation(args.text_file)
