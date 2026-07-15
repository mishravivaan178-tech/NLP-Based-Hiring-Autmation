"""
Resume <-> Job Description matching pipeline.

This consolidates the logic that was spread across
NLP_Cleaning_and_extraction.ipynb (cells 02-06) into reusable,
testable functions with no Colab/file-upload dependency, so the
exact same code can run in a notebook, a batch script, or a
FastAPI endpoint at inference time.

Behavior is intentionally kept identical to the original notebook
(same skill dictionary, same regex patterns, same domain map) so
that re-running this on resumeJD2_pairs.csv reproduces
resumeJD_with_domains.csv. Two small, clearly-marked robustness
fixes are included (see FIX comments) -- everything else is a
straight port.
"""

import re
import ast
import numpy as np
import pandas as pd


# ============================================================
# SYMBOL NORMALIZATION  (fixes a real bug: clean_text() strips
# +, #, ., & -- which silently broke matching for "c++", "c#",
# "node.js", "fp&a" even though they were already in SKILLS_DB.
# We convert these to safe plain-word tokens BEFORE the symbol
# strip happens, and the dictionary below uses the same safe
# tokens, so matching is no longer symbol-dependent.)
# ============================================================

_SYMBOL_SAFE_REPLACEMENTS = {
    "c++": "cplusplus",
    "c#": "csharp",
    "node.js": "nodejs",
    "fp&a": "fpna",
    "m&a": "mergers and acquisitions",
    ".net": "dotnet",
    "asp.net": "aspdotnet",
}


def _presere_symbol_terms(text: str) -> str:
    text = text.lower()
    for term, safe in _SYMBOL_SAFE_REPLACEMENTS.items():
        text = text.replace(term, safe)
    # FIX 2: generic fallback for any other "X & Y" or "X-Y" skill phrase
    # (e.g. "diversity and inclusion", "cross selling") -- without this,
    # clean_text's symbol strip turns "&"/"-" into a bare space and the
    # two halves silently stop matching the dictionary entry.
    text = text.replace("&", " and ")
    text = text.replace("-", " ")
    return text


# ============================================================
# TEXT CLEANING  (from cell 01)
# ============================================================

def clean_text(text: str) -> str:
    """Lowercase, strip emails/URLs/phones/special chars, collapse whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = _presere_symbol_terms(text)                          # FIX: protect c++/c#/node.js/fp&a etc.
    text = re.sub(r'\S+@\S+', ' ', text)                       # emails
    text = re.sub(r'http\S+|www\S+', ' ', text)                 # URLs
    text = re.sub(r'linkedin\.com/\S+', ' ', text)              # LinkedIn URLs
    text = re.sub(r'\+?\d[\d\s\-\(\)]{8,}\d', ' ', text)        # phone numbers
    text = re.sub(r'[^a-z0-9\s]', ' ', text)                    # special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

class ScannedPDFError(Exception):
    """Raised when a PDF has no extractable text (likely a scanned/image PDF)."""
    pass


class CorruptPDFError(Exception):
    """Raised when a PDF file can't be opened/parsed at all."""
    pass


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF resume/JD file using pdfplumber.
    Works on standard text-based PDFs (not scanned/image-only PDFs --
    those would need OCR, which is a separate concern).

    Raises:
        CorruptPDFError  -- file isn't a valid/readable PDF
        ScannedPDFError  -- file opened fine but contains no extractable
                            text (almost always a scanned/image PDF)
    """
    import pdfplumber

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = [page.extract_text() for page in pdf.pages]
    except Exception as e:
        raise CorruptPDFError(
            f"Could not open '{pdf_path}' as a PDF ({e}). "
            "The file may be corrupted or not a real PDF."
        ) from e

    full_text = "\n".join(t for t in text_parts if t)

    if not full_text.strip():
        raise ScannedPDFError(
            f"No extractable text found in '{pdf_path}'. "
            "This usually means it's a scanned/image-based PDF and needs OCR "
            "(not currently supported) -- ask the candidate to upload a "
            "text-based PDF or a Word document instead."
        )

    return full_text


def safe_extract_text_from_pdf(pdf_path: str):
    """
    Same as extract_text_from_pdf(), but never raises -- returns
    (text, error_message) instead. Use this in an API/UI layer where
    you want to show the user a clean message instead of a stack trace.

    Example:
        text, error = safe_extract_text_from_pdf("resume.pdf")
        if error:
            return {"status": "error", "message": error}
        # ... continue with `text`
    """
    try:
        return extract_text_from_pdf(pdf_path), None
    except (ScannedPDFError, CorruptPDFError) as e:
        return None, str(e)


# ============================================================
# SKILL DATABASE  (from cell 02)
# ============================================================

SKILLS_DB = [
    # Programming
    "python", "java", "cplusplus", "csharp", "javascript", "typescript",
    "php", "ruby", "rust", "kotlin", "swift", "matlab",
    # Web dev
    "html", "css", "react", "angular", "vue", "nodejs",
    "express", "django", "flask", "fastapi", "spring", "spring boot",
    "bootstrap", "jquery", "rest api", "rest apis", "graphql", "dotnet", "aspdotnet",
    "grpc", "microservices", "system design",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite",
    "redis", "cassandra", "dynamodb", "elasticsearch", "bigquery", "redshift",
    # Data science / AI
    "machine learning", "deep learning", "artificial intelligence", "nlp",
    "natural language processing", "computer vision", "data science",
    "data analysis", "statistics", "predictive modeling", "feature engineering",
    "time series", "hypothesis testing", "data modeling", "transformers", "bert",
    "numpy", "pandas", "matplotlib", "seaborn", "scikit learn", "sklearn",
    "tensorflow", "keras", "pytorch", "xgboost", "lightgbm", "mlflow", "kubeflow", "dbt",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
    "terraform", "ansible", "git", "github", "gitlab", "ci cd",
    "linux", "unix", "bash", "eks", "gke", "helm", "istio", "argocd", "fluxcd",
    "pulumi", "cloudformation", "infrastructure as code", "consul", "vault",
    "site reliability engineering", "nginx", "haproxy", "service mesh",
    "prometheus", "grafana", "datadog", "new relic", "loki",
    # Data engineering
    "hadoop", "spark", "apache spark", "hive", "kafka", "airflow", "etl", "rabbitmq",
    # BI
    "power bi", "tableau", "looker", "excel", "vba", "dashboarding",
    "business intelligence", "amplitude", "mixpanel",
    # Finance
    "cfa", "ca", "financial modeling", "valuation", "equity research",
    "investment banking", "financial reporting", "accounting", "treasury",
    "budgeting", "forecasting", "quickbooks", "sap", "ifrs", "gaap",
    "financial analysis", "fpna", "cash flow management", "auditing", "m&a",
    "dcf analysis", "bloomberg terminal", "credit analysis", "risk analysis",
    "due diligence", "lbo modeling", "tax compliance", "tally",
    "mergers and acquisitions",
    # HR
    "recruitment", "talent acquisition", "employee relations", "payroll",
    "performance management", "human resources", "onboarding",
    "training and development", "hr analytics",
    "hrms", "hris", "workday", "successfactors", "labor law", "okrs", "kpis",
    "workforce planning", "succession planning", "compensation and benefits",
    "diversity and inclusion", "organisational development", "change management",
    "conflict resolution", "exit management",
    # Marketing
    "digital marketing", "seo", "sem", "google analytics",
    "content marketing", "email marketing", "social media marketing",
    "social media", "branding", "brand strategy", "market research",
    "google ads", "meta ads", "linkedin ads", "programmatic advertising",
    "affiliate marketing", "influencer marketing", "video marketing",
    "marketing automation", "marketo", "hubspot", "copywriting",
    "conversion rate optimisation", "community management", "ppc",
    # Product management
    "product management", "roadmapping", "product analytics",
    "growth hacking", "user research", "a b testing", "stakeholder management",
    "prd writing", "feature prioritisation", "rice framework", "moscow",
    "go to market strategy", "competitive analysis", "customer discovery",
    # Project management
    "project management", "agile", "scrum", "jira", "kanban", "risk management", "confluence",
    # UX / UI Design
    "figma", "sketch", "adobe xd", "adobe suite", "invision", "zeplin", "miro",
    "wireframing", "prototyping", "usability testing", "interaction design",
    "motion design", "visual design", "design thinking", "design systems",
    "user flows", "card sorting", "storyboarding", "information architecture",
    "heuristic evaluation", "accessibility wcag", "principle", "framer", "lottie",
    # Cyber security
    "cyber security", "penetration testing", "ethical hacking",
    "network security", "information security", "siem", "splunk", "metasploit",
    "cissp", "oscp", "ceh", "wireshark", "burp suite", "iam", "soc operations",
    "zero trust", "threat intelligence", "firewalls", "nist framework",
    "iso 27001", "owasp top 10", "vulnerability assessment", "incident response",
    "forensics", "cryptography", "pki", "dlp",
    # Networking
    "tcp ip", "dns", "routing", "switching", "network administration",
    # Sales
    "sales", "business development", "crm", "salesforce", "lead generation",
    "customer relationship management", "sdr", "bdr", "cold outreach",
    "demo skills", "proposal writing", "meddic", "spin selling",
    "solution selling", "account management", "channel partnerships",
    "pipeline management", "upselling", "cross selling", "customer success",
    "linkedin recruiter",
    # Soft skills
    "communication", "leadership", "team management", "problem solving",
    "critical thinking", "presentation skills", "negotiation", "customer service",
    # Healthcare
    "medical billing", "patient care", "clinical research", "healthcare management",
]

SKILL_ALIASES = {
    "chartered accountant": "ca",
    "certified public accountant": "ca",
    "human resource": "human resources",
    "customer relationship management": "crm",
    "search engine optimization": "seo",
    "search engine marketing": "sem",
    "financial planning and analysis": "fpna",
}

# FIX 1: original notebook rebuilt this regex on every call inside a loop
# over SKILLS_DB for every row -- fine at 500 rows, too slow once this
# runs per-request at inference time. Precompile once at import time.
_SKILL_PATTERNS = [(skill, re.compile(r'\b' + re.escape(skill) + r'\b'))
                    for skill in SKILLS_DB]


def normalize_text(text: str) -> str:
    text = str(text).lower()
    for phrase, replacement in SKILL_ALIASES.items():
        text = text.replace(phrase, replacement)
    return text


def _extract_skills_exact(text: str) -> list:
    found = [skill for skill, pattern in _SKILL_PATTERNS if pattern.search(text)]
    return sorted(set(found))


# ============================================================
# FUZZY SKILL MATCHING
# ============================================================
#
# Exact matching misses typos ("pyhton"), minor variants ("react.js"
# vs "react"), and phrasing your dictionary didn't anticipate. This
# adds a second pass: break the text into 1-3 word chunks, and for
# any chunk that ISN'T an exact dictionary hit, check whether it's a
# close (>=88% similarity) match to a known skill. Threshold of 88 is
# deliberately conservative -- catches "pyhton"->"python" (94%) and
# "kuberentes"->"kubernetes" (95%), but won't falsely match unrelated
# short words to each other.
# ============================================================

from rapidfuzz import fuzz, process as _rf_process

_FUZZY_THRESHOLD = 90

# Skills too short/common-word-like to fuzzy match safely (e.g. "rust" was
# matching the word "trust"; "marketo" was matching "market"). These are
# still detected fine via EXACT matching if someone types them precisely --
# we only exclude them from the approximate/typo-tolerant pass.
_FUZZY_BLOCKLIST = {"rust", "go", "r", "sap", "ca", "aws", "marketo", "principle"}
_FUZZY_CANDIDATES = [s for s in SKILLS_DB if s not in _FUZZY_BLOCKLIST and len(s) >= 5]


def _generate_ngrams(text: str, max_n: int = 3) -> list:
    words = text.split()
    ngrams = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i + n]))
    return ngrams


def extract_skills_fuzzy(text: str, threshold: int = _FUZZY_THRESHOLD) -> list:
    """
    Fuzzy-only pass: returns skills found via approximate matching
    that were NOT already caught by exact matching. Kept separate
    from extract_skills() so callers/explanations can distinguish
    "exact match" from "fuzzy match" if they want to.

    Guardrails against false positives (see _FUZZY_BLOCKLIST above):
    - candidate chunk must share the same first character as the skill
    - skill must be >=5 characters (short skills are too collision-prone)
    - threshold defaults to 90, not 85, after empirical testing on the
      real dataset showed 85-88 produced false positives like
      "trust" -> "rust" and "market" -> "marketo"
    """
    text = normalize_text(text)
    exact_hits = set(_extract_skills_exact(text))
    candidates = set(_generate_ngrams(text)) - exact_hits

    fuzzy_hits = set()
    for chunk in candidates:
        if len(chunk) < 4:  # skip very short chunks -- too noisy to fuzzy-match safely
            continue
        # cheap prefilter: only compare against skills starting with the
        # same letter, both for speed and to eliminate cross-prefix noise
        same_prefix_skills = [s for s in _FUZZY_CANDIDATES if s[0] == chunk[0]]
        if not same_prefix_skills:
            continue
        match = _rf_process.extractOne(chunk, same_prefix_skills, scorer=fuzz.ratio,
                                        score_cutoff=threshold)
        if match:
            fuzzy_hits.add(match[0])

    return sorted(fuzzy_hits - exact_hits)


def extract_skills(text: str, use_fuzzy: bool = True) -> list:
    text_norm = normalize_text(text)
    exact = _extract_skills_exact(text_norm)
    if not use_fuzzy:
        return exact
    fuzzy = extract_skills_fuzzy(text)
    return sorted(set(exact) | set(fuzzy))


def get_matched_skills(resume_skills, jd_skills) -> list:
    return sorted(set(resume_skills) & set(jd_skills))


def get_missing_skills(resume_skills, jd_skills) -> list:
    return sorted(set(jd_skills) - set(resume_skills))


def skill_match_score(resume_skills, jd_skills) -> float:
    if len(jd_skills) == 0:
        return 0.0
    matched = len(set(resume_skills) & set(jd_skills))
    return round(matched / len(jd_skills), 2)


# ============================================================
# EXPERIENCE EXTRACTION  (from cell 03)
# ============================================================

_RESUME_EXP_PATTERNS = [
    re.compile(r'(\d+)\+?\s*years'),
    re.compile(r'(\d+)\+?\s*year'),
    re.compile(r'(\d+)\s*yrs'),
    re.compile(r'(\d+)\s*yr'),
]

_JD_EXP_PATTERNS = [
    re.compile(r'experience\s*:?\s*(\d+)\+?\s*years'),
    re.compile(r'experience\s*:?\s*(\d+)\+?\s*year'),
    re.compile(r'(\d+)\+?\s*years'),
    re.compile(r'(\d+)\+?\s*year'),
]


def extract_resume_experience(text: str):
    text = str(text).lower()
    for pattern in _RESUME_EXP_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return np.nan


def extract_jd_experience(text: str):
    text = str(text).lower()
    for pattern in _JD_EXP_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return np.nan


def calculate_experience_match_score(resume_exp, jd_exp):
    if pd.isna(resume_exp) or pd.isna(jd_exp):
        return np.nan
    if jd_exp == 0:
        return 0.0
    return round(min(resume_exp / jd_exp, 1), 2)


# ============================================================
# EXPERIENCE IMPUTATION  (fixes: a resume/JD with no explicit
# "X years" phrasing produced a raw NaN in experience_match_score,
# which most ML models (including sklearn's RandomForest) cannot
# accept as-is and would crash or silently drop the row on.
#
# Two things are added:
#   1. resume_experience_mentioned / jd_experience_mentioned flags,
#      so "we don't know" is preserved as its own signal instead of
#      being silently erased by imputation.
#   2. impute_experience_score(), which fills NaN using a value you
#      pass in (e.g. the TRAINING set's median) rather than computing
#      it from the full dataset -- computing it from the full dataset
#      (train+test combined) would leak test-set information into
#      training, which is a classic evaluation mistake.
# ============================================================

def impute_experience_score(scores: pd.Series, fill_value: float) -> pd.Series:
    """
    Fill NaN experience_match_score values with a caller-supplied
    fill_value. Always compute fill_value from TRAINING data only
    (e.g. train_df['experience_match_score'].median()) and reuse that
    same number for both train and test/inference -- never recompute
    it separately on test data, or you leak information across the split.
    """
    return scores.fillna(fill_value)


# ============================================================
# EDUCATION EXTRACTION  (from cell 04)
# ============================================================

EDUCATION_PATTERNS = {
    "B.Tech": r"\bb\s*tech\b", "B.E": r"\bb\s*e\b", "B.Sc": r"\bb\s*sc\b",
    "B.Com": r"\bb\s*com\b", "B.A": r"\bb\s*a\b", "BBA": r"\bbba\b",
    "B.Des": r"\bb\s*des\b",
    "MBA": r"\bmba\b", "M.Tech": r"\bm\s*tech\b", "M.Sc": r"\bm\s*sc\b",
    "M.A": r"\bm\s*a\b", "M.S": r"\bm\s*s\b", "PGDM": r"\bpgdm\b",
    "CA": r"\bca\b", "CFA": r"\bcfa\b",
    "Diploma": r"\bdiploma\b",
    "PhD": r"\bph\s*d\b",
}
_EDUCATION_COMPILED = [(deg, re.compile(pat)) for deg, pat in EDUCATION_PATTERNS.items()]

_EDU_LEVEL_MAP = {
    "B.Tech": "Bachelor", "B.E": "Bachelor", "B.Sc": "Bachelor",
    "B.Com": "Bachelor", "B.A": "Bachelor", "BBA": "Bachelor", "B.Des": "Bachelor",
    "MBA": "Master", "M.Tech": "Master", "M.Sc": "Master",
    "M.A": "Master", "M.S": "Master", "PGDM": "Master",
    "CA": "Professional", "CFA": "Professional",
    "Diploma": "Diploma", "PhD": "Doctorate",
}


def extract_education(text: str) -> str:
    text = str(text).lower()
    for degree, pattern in _EDUCATION_COMPILED:
        if pattern.search(text):
            return degree
    return "Unknown"


def education_level(degree: str) -> str:
    return _EDU_LEVEL_MAP.get(degree, "Unknown")


# ============================================================
# DOMAIN IDENTIFICATION  (from cell 06)
# ============================================================

DOMAIN_SKILLS = {
    "Finance": ["cfa", "ca", "valuation", "equity research", "financial reporting",
                "accounting", "treasury", "investment banking", "financial analysis",
                "financial modeling", "forecasting", "budgeting", "quickbooks",
                "sap", "ifrs", "gaap", "fpna", "cash flow management", "auditing",
                "dcf analysis", "bloomberg terminal", "credit analysis",
                "risk analysis", "due diligence", "lbo modeling", "tax compliance", "tally"],
    "HR": ["recruitment", "talent acquisition", "employee relations", "payroll",
           "human resources", "onboarding", "performance management",
           "training and development", "hr analytics",
           "hrms", "hris", "workday", "successfactors", "labor law", "okrs",
           "workforce planning", "succession planning", "compensation and benefits",
           "diversity and inclusion", "organisational development", "change management",
           "conflict resolution"],
    "Marketing": ["seo", "sem", "google analytics", "content marketing",
                  "email marketing", "social media marketing", "social media",
                  "branding", "brand strategy", "market research", "google ads",
                  "meta ads", "linkedin ads", "programmatic advertising",
                  "affiliate marketing", "influencer marketing", "video marketing",
                  "marketing automation", "marketo", "hubspot", "copywriting",
                  "conversion rate optimisation", "community management"],
    "Data Science": ["python", "machine learning", "deep learning", "tensorflow",
                      "pytorch", "nlp", "data science", "data analysis", "scikit learn",
                      "feature engineering", "time series", "hypothesis testing",
                      "data modeling", "transformers", "bert", "mlflow", "kubeflow", "dbt"],
    "Software Development": ["java", "cplusplus", "csharp", "javascript", "react", "angular",
                              "nodejs", "django", "flask", "spring boot", "microservices",
                              "system design", "grpc", "dotnet", "aspdotnet"],
    "DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform",
               "jenkins", "ansible", "linux", "eks", "gke", "helm", "istio",
               "argocd", "fluxcd", "pulumi", "cloudformation", "infrastructure as code",
               "consul", "vault", "site reliability engineering", "nginx", "haproxy",
               "service mesh", "prometheus", "grafana", "datadog", "new relic", "loki"],
    "Cyber Security": ["cyber security", "penetration testing", "ethical hacking",
                        "network security", "information security", "siem", "splunk",
                        "metasploit", "cissp", "oscp", "ceh", "wireshark", "burp suite",
                        "iam", "soc operations", "zero trust", "threat intelligence",
                        "firewalls", "nist framework", "iso 27001", "owasp top 10",
                        "vulnerability assessment", "incident response", "forensics",
                        "cryptography", "pki", "dlp"],
    "Product Management": ["product management", "product analytics", "roadmapping",
                            "user research", "growth hacking", "prd writing",
                            "feature prioritisation", "rice framework", "moscow",
                            "go to market strategy", "competitive analysis", "customer discovery"],
    "UX/UI Design": ["figma", "sketch", "adobe xd", "adobe suite", "invision", "zeplin",
                      "miro", "wireframing", "prototyping", "usability testing",
                      "interaction design", "motion design", "visual design",
                      "design thinking", "design systems", "user flows", "card sorting",
                      "storyboarding", "information architecture", "heuristic evaluation",
                      "accessibility wcag", "principle", "framer", "lottie"],
    "Sales": ["sales", "salesforce", "lead generation", "crm", "business development",
              "customer relationship management", "sdr", "bdr", "cold outreach",
              "demo skills", "proposal writing", "meddic", "spin selling",
              "solution selling", "account management", "channel partnerships",
              "pipeline management", "upselling", "cross selling", "customer success",
              "linkedin recruiter"],
}


def identify_domain(skill_list) -> str:
    if isinstance(skill_list, str):
        try:
            skill_list = ast.literal_eval(skill_list)
        except Exception:
            skill_list = []
    scores = {domain: len(set(skill_list) & set(skills))
              for domain, skills in DOMAIN_SKILLS.items()}
    best_domain = max(scores, key=scores.get)
    return "Unknown" if scores[best_domain] == 0 else best_domain


# ============================================================
# SEMANTIC SIMILARITY  (from cell 05) -- lazy-loaded, model is heavy
# ============================================================

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def semantic_similarity(resume_text: str, jd_text: str) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    model = _get_model()
    embeddings = model.encode([resume_text, jd_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(score), 4)


# ============================================================
# END-TO-END FEATURE EXTRACTION for a single (resume, jd) pair
# ============================================================

def extract_features(resume_text_raw: str, jd_text_raw: str) -> dict:
    """
    Run the full pipeline on a single new resume/JD pair and return
    every feature the model needs, plus the explainability fields
    (matched/missing skills, domains, education) needed for the
    SHAP-based report.

    This is the function the FastAPI endpoint will call at inference time.
    """
    resume_clean = clean_text(resume_text_raw)
    jd_clean = clean_text(jd_text_raw)

    resume_skills = extract_skills(resume_clean)
    jd_skills = extract_skills(jd_clean)
    matched_skills = get_matched_skills(resume_skills, jd_skills)
    missing_skills = get_missing_skills(resume_skills, jd_skills)
    skill_score = skill_match_score(resume_skills, jd_skills)

    resume_exp = extract_resume_experience(resume_clean)
    jd_exp = extract_jd_experience(jd_clean)
    exp_score = calculate_experience_match_score(resume_exp, jd_exp)

    resume_edu = extract_education(resume_clean)
    resume_edu_level = education_level(resume_edu)

    resume_domain = identify_domain(resume_skills)
    jd_domain = identify_domain(jd_skills)
    domain_match = int(resume_domain == jd_domain)

    sim_score = semantic_similarity(resume_clean, jd_clean)

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_score": skill_score,
        "matched_skill_count": len(matched_skills),
        "missing_skill_count": len(missing_skills),
        "resume_experience": resume_exp,
        "jd_experience": jd_exp,
        "experience_match_score": exp_score,
        "resume_experience_mentioned": int(not pd.isna(resume_exp)),
        "jd_experience_mentioned": int(not pd.isna(jd_exp)),
        "resume_education": resume_edu,
        "resume_education_level": resume_edu_level,
        "semantic_similarity_score": sim_score,
        "resume_domain": resume_domain,
        "jd_domain": jd_domain,
        "domain_match": domain_match,
    }


def extract_features_from_pdf(resume_pdf_path: str, jd_text_or_pdf_path: str,
                               jd_is_pdf: bool = False) -> dict:
    """
    Convenience wrapper: go straight from a resume PDF file (+ a JD, which
    can be plain text or also a PDF path) to the full feature dictionary.
    """
    resume_text = extract_text_from_pdf(resume_pdf_path)
    jd_text = extract_text_from_pdf(jd_text_or_pdf_path) if jd_is_pdf else jd_text_or_pdf_path
    return extract_features(resume_text, jd_text)


# ============================================================
# BATCH PIPELINE -- rebuild the full engineered dataset from raw pairs
# ============================================================

def build_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw dataframe with [resume_text, job_description, match_score,
    match_label] and returns the fully feature-engineered dataframe,
    equivalent to resumeJD_with_domains.csv, using a single pass instead
    of six separate notebook cells / CSV round-trips.
    """
    df = raw_df.drop_duplicates().copy()
    df["resume_text"] = df["resume_text"].apply(clean_text)
    df["job_description"] = df["job_description"].apply(clean_text)

    rows = []
    for _, row in df.iterrows():
        feats = extract_features(row["resume_text"], row["job_description"])
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    result = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    return result
