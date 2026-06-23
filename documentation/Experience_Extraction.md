# Experience Extraction and Matching Module

## Objective

Extract years of experience from resumes and job descriptions and generate an experience matching score.

## Methodology

### Resume Experience Extraction

Regex patterns were used to identify years of experience.

Example:

"CA/MBA with 5 years in Valuation"

Output:

resume_experience = 5

### Job Description Experience Extraction

Example:

"Minimum 8 years experience required"

Output:

jd_experience = 8

### Experience Match Score

Formula:

Experience Match Score = min(resume_experience / jd_experience, 1)

Example:

Resume = 5 years
JD = 8 years

Score = 0.62

Generated Features:

- resume_experience
- jd_experience
- experience_match_score

## Results

- Resume Extraction Coverage: 100%
- JD Extraction Coverage: 100%
- Average Resume Experience: 6.86 years
- Average JD Requirement: 4.78 years
- Average Match Score: 0.88

## Challenges

- Multiple experience formats
- Overqualified candidates

## Limitations

- Only total experience considered
- Domain-specific experience not considered
- Seniority level not considered

## Outcome

Successfully generated:

- resume_experience
- jd_experience
- experience_match_score