# KB-Augmented Narrative Segmentation

Test whether knowledge base augmentation improves narrative text segmentation quality.

## Core Hypothesis
Adding extracted knowledge triples (SUBJECT|PREDICATE|OBJECT) as context helps LLMs create better narrative segment boundaries, particularly for dialogue-heavy text.

## Quick Start

1. **Install dependencies:**
```bash
uv sync
```

2. **Add your OpenAI API key to `.env`:**
```
OPENAI_API_KEY=your_key_here
```

3. **Run the test:**
```bash
# Requires a text file
python test_kb_segmentation.py -f your_story.txt
```

## Results Analysis

📄 **[Read the detailed analysis: KB-Augmented-Segmentation.md](KB-Augmented-Segmentation.md)**

This document shows the full experimental results, including:
- Side-by-side comparison of segmentation approaches
- Pattern analysis of dialogue preservation
- Technical insights about conversational unit boundaries
- Jane Eyre Chapter 1 case study with 36 vs 42 segment comparison

## What This Tests

**Success metrics:**
- KB-augmented approach creates fewer, more coherent segments
- Dialogue exchanges stay together as conversational units
- Question-response pairs aren't artificially split

**Example improvement:**
```
Baseline coherence: 0.217
KB-augmented coherence: 0.601
Improvement: +0.385
```

The KB approach successfully preserves narrative flow while standard segmentation fragments dialogue into disconnected pieces.

## Files

- `test_kb_segmentation.py` - Main test script comparing segmentation approaches (requires text file input)
- `segmentation_comparison.py` - Simple coherence metrics for segment quality
- `KB-Augmented-Segmentation.md` - **Complete experimental analysis and results**
- `JaneEyre-scene-001.txt` - Jane Eyre Chapter 1 excerpt for testing

This is a proof-of-concept for the Story Play Code project's narrative analysis pipeline.
