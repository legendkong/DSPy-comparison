# Comparison for DSPy vs non-DSPy(Using LangChain prompts)

Install dependencies.

```python
pip install -r requirements.txt
```

Create the Chroma DB.

```python
python create_database.py
```

Comparison between using DSPy vs non-DSPy(LangChain prompts):
1. DSPy
```python
python dspy_test.py "How does Alice meet the Mad Hatter?"
```

2. non-DSPy (LangChain prompts)
```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work.
