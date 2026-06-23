# Skill Extraction and Matching Module

## Objective

The objective of this phase was to preprocess resume and job description text and extract relevant skills using NLP techniques.

## Methodology

### Data Preprocessing

The following preprocessing steps were performed:

- Removed duplicate records
- Removed emails
- Removed phone numbers
- Removed URLs and LinkedIn links
- Removed special characters
- Converted text to lowercase
- Standardized formatting

### Skill Database Construction

A skill repository containing 160+ skills was created across domains such as:

- Programming
- Data Science
- Cloud Computing
- Finance
- HR
- Marketing
- Product Management
- Cyber Security
- Sales

### Skill Normalization

Examples:

- Chartered Accountant → CA
- Customer Relationship Management → CRM
- Financial Planning and Analysis → FP&A

### Skill Extraction

Regex-based pattern matching was used to identify skills from resume and job description text.

Example:

Resume Skills:
CFA, Valuation, Equity Research

Extracted:
['cfa', 'valuation', 'equity research']

Generated Features:

- resume_skills
- jd_skills

### Skill Matching

Generated Features:

- matched_skills
- missing_skills
- matched_skill_count
- missing_skill_count

### Skill Match Score

Formula:

Skill Match Score = Matched Skills / Total JD Skills

Example:

Matched Skills = 2
JD Skills = 4

Score = 0.50

Generated Feature:

- skill_match_score

## Results

- Dataset Size: 500
- Resume Skill Coverage: 98.6%
- JD Skill Coverage: 99.6%
- Average Skill Match Score: 0.19
- Maximum Score: 1.00

## Challenges

- Limited skill repository
- False skill detection
- Skill naming variations

## Limitations

- Depends on predefined skill dictionary
- Cannot detect unseen skills
- Relies on direct skill overlap

## Outcome

Successfully generated:

- resume_skills
- jd_skills
- matched_skills
- missing_skills
- matched_skill_count
- missing_skill_count
- skill_match_score