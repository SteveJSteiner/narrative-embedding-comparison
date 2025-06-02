# KB-Augmented Narrative Segmentation Test

A minimal experiment to test whether knowledge base (KB) augmentation improves text segment coherence for narrative analysis.

## Core Idea

Does adding extracted knowledge facts to text segments improve their semantic coherence when embedded? This project tests that hypothesis with a simple pipeline:

1. **Extract facts** from narrative text using LLM (GPT-4.1-mini)
2. **Augment segments** by appending facts to original text  
3. **Compare coherence** of baseline vs KB-augmented embeddings
4. **Measure improvement** in adjacent segment similarity

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Set API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Test

```bash
python test_kb_segmentation.py
```

## Expected Output

```
Testing KB-Augmented Segmentation
==================================================
Original text:
John was angry about the argument. He stormed out of the room...

Baseline segments (4):
  1: John was angry about the argument
  2: He stormed out of the room and slammed the door
  3: Mary felt terrible about what had happened
  4: Later that evening, John returned with flowers

KB-augmented segments (4):
  1: John was angry about the argument
     [KB: John | feels | angry; John | has_emotion | anger]
  2: He stormed out of the room and slammed the door
     [KB: He | performed_action | stormed_out; door | received_action | slammed]
  ...

Comparison Results:
==============================
Baseline coherence: 0.742
KB-augmented coherence: 0.801
Improvement: 0.059
✓ KB augmentation improves coherence!
```

## What It Tests

- **Baseline**: Simple sentence segmentation + sentence-transformer embeddings
- **KB-Augmented**: Same segments + extracted knowledge facts appended
- **Coherence**: Average cosine similarity between adjacent segment embeddings
- **Success**: Positive improvement score means KB augmentation helps

## Files

- `test_kb_segmentation.py` - Main test script with simple KB extraction and comparison
- `segmentation_comparison.py` - Minimal coherence calculation utilities  
- `pyproject.toml` - Dependencies (OpenAI, sentence-transformers, etc.)
- `.env` - API keys (create this yourself)

## Dependencies

- `openai>=1.0.0` - For GPT-4.1-mini fact extraction
- `sentence-transformers>=2.2.0` - For text embeddings
- `numpy>=1.24.0` - For similarity calculations
- `python-dotenv>=1.0.0` - For environment variable loading

## Interpretation

- **Positive improvement**: KB facts help segment coherence → promising approach
- **Negative improvement**: KB facts hurt coherence → try different extraction/augmentation
- **Near zero**: No significant effect → baseline is sufficient

## Next Steps

If this minimal test shows promise, you can:

1. Test with different narrative texts
2. Try different KB extraction prompts
3. Experiment with fact integration methods
4. Test other embedding models
5. Add more sophisticated segmentation

## License

MIT
