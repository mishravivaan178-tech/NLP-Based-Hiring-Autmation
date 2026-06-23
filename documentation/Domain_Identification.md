# Domain Identification Module

## Objective

Identify the professional domain of resumes and job descriptions and determine whether they belong to the same domain.

## Methodology

Domains were assigned using extracted skills.

Examples:

Finance:
- CFA
- Valuation
- Treasury

HR:
- Recruitment
- Payroll
- Talent Acquisition

Data Science:
- Python
- Machine Learning
- NLP

Generated Features:

- resume_domain
- jd_domain
- domain_match

Where:

1 = Same Domain

0 = Different Domain

## Results

Resume Domain Distribution:

- Finance = 137
- DevOps = 99
- Marketing = 58
- HR = 57

Domain Match Distribution:

- Match = 188
- No Match = 312

Validation:

Average Match Score:

- Domain Match = 0.703
- Domain Mismatch = 0.425

Observation:

Candidates belonging to the same domain achieved significantly higher match scores.

## Challenges

- Domain overlap
- Unknown domains

## Limitations

- Single domain assignment
- Depends on extracted skills

## Outcome

Successfully generated:

- resume_domain
- jd_domain
- domain_match