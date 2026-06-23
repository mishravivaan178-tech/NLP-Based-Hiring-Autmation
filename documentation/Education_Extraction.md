# Education Extraction Module

## Objective

Extract educational qualifications from resumes and convert them into structured categories.

## Methodology

Regex-based extraction was used.

Examples:

- B.Tech → Bachelor
- MBA → Master
- CA → Professional
- PhD → Doctorate

Generated Features:

- resume_education
- resume_education_level

## Results

- Total Records: 500
- Successfully Extracted: 453
- Unknown Records: 47
- Coverage: 90.6%

Education Level Distribution:

- Master: 221
- Bachelor: 208
- Professional: 11
- Diploma: 11
- Doctorate: 2
- Unknown: 47

## Challenges

- Multiple degree formats
- Missing education requirements in JD

## Limitations

- No education matching score
- Education requirements unavailable in JD

## Outcome

Successfully generated:

- resume_education
- resume_education_level