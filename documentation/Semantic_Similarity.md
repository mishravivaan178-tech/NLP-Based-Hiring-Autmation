# Semantic Similarity Analysis

## Objective

Measure contextual similarity between resumes and job descriptions beyond keyword matching.

## Methodology

Model Used:

SentenceTransformer (all-MiniLM-L6-v2)

Process:

1. Generate embeddings for resume text
2. Generate embeddings for JD text
3. Compute cosine similarity

Generated Feature:

- semantic_similarity_score

Score Range:

0 → Completely Different

1 → Highly Similar

## Results

- Total Records: 500
- Mean Similarity Score: 0.60
- Minimum Score: 0.36
- Maximum Score: 0.77

Validation by Match Label:

- Match = 0.669
- Partial Match = 0.612
- No Match = 0.534

Correlation with Match Score:

- Semantic Similarity = 0.622
- Skill Match = 0.581
- Experience Match = -0.027

Observation:

Semantic Similarity emerged as the strongest feature developed so far.

## Challenges

- Large unstructured text
- Compressed similarity range

## Limitations

- Entire documents compared
- Section-level comparison not implemented

## Outcome

Successfully generated:

- semantic_similarity_score