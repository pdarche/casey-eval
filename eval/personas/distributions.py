"""
Target distributions for persona generation based on ODL client population.

These distributions should be updated with actual ODL client demographic data
to ensure synthetic test coverage matches the real client population.
"""

from eval.personas.models import (
    Language,
    LegalIssue,
    Gender,
    EmploymentStatus,
    HousingStatus,
    EnglishFluency,
    CommunicationStyle,
    TrustLevel,
)


# =============================================================================
# LANGUAGE DISTRIBUTION
# Based on ODL client language preferences
# =============================================================================

LANGUAGE_DISTRIBUTION = {
    Language.ENGLISH: 0.70,      # 70% - Primary language
    Language.SPANISH: 0.20,      # 20% - Second largest group
    Language.CHINESE: 0.05,      # 5%  - Significant SF population
    Language.FRENCH: 0.02,       # 2%  - Other languages combined
    Language.THAI: 0.01,         # 1%
    Language.KOREAN: 0.01,       # 1%
    Language.JAPANESE: 0.01,     # 1%
}


# =============================================================================
# LEGAL ISSUE DISTRIBUTION
# TODO: Update with actual ODL case type volumes
# =============================================================================

LEGAL_ISSUE_DISTRIBUTION = {
    LegalIssue.HOUSING: 0.25,           # Housing issues (high volume)
    LegalIssue.UD_HOUSING: 0.20,        # Unlawful detainer (eviction)
    LegalIssue.FAMILY: 0.15,            # Family law
    LegalIssue.EMPLOYMENT: 0.12,        # Employment issues
    LegalIssue.IMMIGRATION: 0.10,       # Immigration
    LegalIssue.CONSUMER: 0.08,          # Consumer protection
    LegalIssue.ELDER_ESTATE: 0.05,      # Elder/Estate
    LegalIssue.CIVIL_LITIGATION: 0.03,  # Civil litigation
    LegalIssue.BRIEF_SERVICES: 0.02,    # Brief services
}


# =============================================================================
# DEMOGRAPHIC DISTRIBUTIONS
# TODO: Update with actual ODL client demographics
# =============================================================================

GENDER_DISTRIBUTION = {
    Gender.FEMALE: 0.52,
    Gender.MALE: 0.45,
    Gender.NONBINARY: 0.02,
    Gender.TRANS_FEMALE: 0.005,
    Gender.TRANS_MALE: 0.005,
    Gender.NOT_LISTED: 0.00,
    Gender.DECLINE: 0.00,
}

# Ethnicity distribution for San Francisco service area
# These are placeholder estimates - update with actual data
ETHNICITY_DISTRIBUTION = {
    "Latino-Mexican": 0.18,
    "Latino-Central American": 0.08,
    "Latino-South American": 0.04,
    "Asian-Chinese": 0.15,
    "Asian-Filipino": 0.05,
    "Asian-Vietnamese": 0.03,
    "Asian-Korean": 0.02,
    "Asian-Japanese": 0.02,
    "Asian-Indian": 0.02,
    "Asian-Other": 0.03,
    "Black-African American": 0.10,
    "Black-African": 0.02,
    "Black-Caribbean, Central American": 0.02,
    "White-European": 0.15,
    "White-Other": 0.03,
    "Pacific Islander-Native Hawaiian": 0.01,
    "Pacific Islander-Samoan": 0.01,
    "Pacific Islander-Other": 0.01,
    "Indigenous-American Indian": 0.01,
    "Middle Eastern/West Asian": 0.02,
}

EMPLOYMENT_DISTRIBUTION = {
    EmploymentStatus.EMPLOYED_FULL: 0.35,
    EmploymentStatus.EMPLOYED_PART: 0.20,
    EmploymentStatus.UNEMPLOYED_LOOKING: 0.25,
    EmploymentStatus.UNEMPLOYED_NOT_LOOKING: 0.10,
    EmploymentStatus.RETIRED: 0.10,
}

HOUSING_DISTRIBUTION = {
    HousingStatus.RENTS: 0.60,
    HousingStatus.LIVING_WITH_OTHERS: 0.15,
    HousingStatus.UNSTABLE: 0.10,
    HousingStatus.SHELTER: 0.08,
    HousingStatus.OWNS: 0.07,
}

ENGLISH_FLUENCY_DISTRIBUTION = {
    EnglishFluency.NATIVE: 0.45,
    EnglishFluency.VERY_FLUENT: 0.15,
    EnglishFluency.FLUENT: 0.15,
    EnglishFluency.SOMEWHAT_FLUENT: 0.15,
    EnglishFluency.NOT_FLUENT: 0.10,
}

EDUCATION_DISTRIBUTION = {
    "Less than high school": 0.15,
    "High school diploma or equivalency (GED)": 0.30,
    "Some college": 0.25,
    "Associate degree/Vocational degree": 0.10,
    "Bachelor's degree": 0.12,
    "Master's degree": 0.05,
    "Doctorate Professional (MD, JD, DDS, etc.)": 0.01,
    "Other": 0.02,
}


# =============================================================================
# BEHAVIORAL TRAIT DISTRIBUTIONS
# These control how synthetic clients behave in conversations
# =============================================================================

COMMUNICATION_STYLE_DISTRIBUTION = {
    CommunicationStyle.DIRECT: 0.40,      # Clear, straightforward
    CommunicationStyle.BRIEF: 0.25,       # Short responses
    CommunicationStyle.VERBOSE: 0.15,     # Long, detailed
    CommunicationStyle.ANXIOUS: 0.15,     # Stressed, worried
    CommunicationStyle.CONFUSED: 0.05,    # Needs clarification
}

TRUST_LEVEL_DISTRIBUTION = {
    TrustLevel.TRUSTING: 0.30,
    TrustLevel.NEUTRAL: 0.50,
    TrustLevel.GUARDED: 0.15,
    TrustLevel.SKEPTICAL: 0.05,
}


# =============================================================================
# AGE DISTRIBUTION
# =============================================================================

# Age ranges with their probabilities
AGE_DISTRIBUTION = {
    (18, 25): 0.15,   # Young adults
    (26, 35): 0.25,   # Early career
    (36, 45): 0.20,   # Mid career
    (46, 55): 0.18,   # Late career
    (56, 65): 0.12,   # Pre-retirement
    (66, 80): 0.10,   # Seniors
}


# =============================================================================
# INCOME DISTRIBUTION
# Based on 200% federal poverty line eligibility
# =============================================================================

# Monthly income ranges (based on household size 1-4)
INCOME_DISTRIBUTION = {
    (0, 500): 0.15,         # Very low income
    (500, 1000): 0.20,      # Low income
    (1000, 2000): 0.30,     # Moderate low income
    (2000, 3000): 0.25,     # Near poverty line
    (3000, 4500): 0.10,     # At poverty line limit
}


# =============================================================================
# HOUSEHOLD SIZE DISTRIBUTION
# =============================================================================

HOUSEHOLD_SIZE_DISTRIBUTION = {
    1: 0.35,  # Single person
    2: 0.25,  # Couple or single parent with one child
    3: 0.20,  # Small family
    4: 0.12,  # Medium family
    5: 0.05,  # Larger family
    6: 0.02,  # Large family
    7: 0.01,  # Very large family
}


# =============================================================================
# EDGE CASE INJECTION RATES
# How often to inject special scenarios into random personas
# =============================================================================

EDGE_CASE_INJECTION_RATES = {
    "discloses_dv": 0.05,              # 5% mention domestic violence
    "discloses_crisis": 0.02,          # 2% express crisis/distress
    "gives_impossible_answers": 0.03,  # 3% give impossible answers
    "mentions_multiple_issues": 0.08,  # 8% have multiple issues
    "attempts_out_of_scope": 0.05,     # 5% ask about out-of-scope things
    "is_returning_client": 0.10,       # 10% are returning clients
}


def validate_distribution(dist: dict) -> bool:
    """Validate that a distribution sums to 1.0 (within floating point tolerance)."""
    total = sum(dist.values())
    return abs(total - 1.0) < 0.01


def get_all_distributions() -> dict:
    """Get all distributions for validation."""
    return {
        "language": LANGUAGE_DISTRIBUTION,
        "legal_issue": LEGAL_ISSUE_DISTRIBUTION,
        "gender": GENDER_DISTRIBUTION,
        "ethnicity": ETHNICITY_DISTRIBUTION,
        "employment": EMPLOYMENT_DISTRIBUTION,
        "housing": HOUSING_DISTRIBUTION,
        "english_fluency": ENGLISH_FLUENCY_DISTRIBUTION,
        "education": EDUCATION_DISTRIBUTION,
        "communication_style": COMMUNICATION_STYLE_DISTRIBUTION,
        "trust_level": TRUST_LEVEL_DISTRIBUTION,
        "age": AGE_DISTRIBUTION,
        "income": INCOME_DISTRIBUTION,
        "household_size": HOUSEHOLD_SIZE_DISTRIBUTION,
    }


def validate_all_distributions() -> dict[str, bool]:
    """Validate all distributions sum to 1.0."""
    return {
        name: validate_distribution(dist)
        for name, dist in get_all_distributions().items()
    }
