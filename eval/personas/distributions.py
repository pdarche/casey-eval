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

# Distributions updated from Salesforce Demographic Summary Report
# Total Records: 8,097 | Generated: March 19, 2026


# =============================================================================
# LANGUAGE DISTRIBUTION
# Source: Salesforce Primary Language field (6,839 valid responses)
# =============================================================================

LANGUAGE_DISTRIBUTION = {
    Language.ENGLISH: 0.666,      # 66.6%
    Language.SPANISH: 0.251,      # 25.1%
    Language.CANTONESE: 0.019,    # 1.9%
    Language.MANDARIN: 0.010,     # 1.0%
    Language.ARABIC: 0.010,       # 1.0%
    Language.FILIPINO: 0.007,     # 0.7%
    Language.RUSSIAN: 0.006,      # 0.6%
    Language.VIETNAMESE: 0.004,   # 0.4%
    Language.BURMESE: 0.001,      # 0.1%
    Language.OTHER: 0.026,        # 2.6%
}


# =============================================================================
# LEGAL ISSUE DISTRIBUTION
# Source: Salesforce Legal Issue field (1,110 valid responses, keyword-based)
# Note: Categories may overlap; normalized to sum to 1.0
# =============================================================================

LEGAL_ISSUE_DISTRIBUTION = {
    LegalIssue.HOUSING: 0.29,           # 24.4% Housing/Tenant Rights
    LegalIssue.UD_HOUSING: 0.06,        # 5.1% Eviction/Unlawful Detainer
    LegalIssue.FAMILY: 0.13,            # 10.9% Family Law/Custody
    LegalIssue.EMPLOYMENT: 0.06,        # 4.7% Employment/Wage
    LegalIssue.IMMIGRATION: 0.07,       # 6.0% Immigration
    LegalIssue.CONSUMER: 0.05,          # 4.5% Debt/Collections
    LegalIssue.ELDER_ESTATE: 0.04,      # 3.4% Personal Injury (nearest match)
    LegalIssue.CIVIL_LITIGATION: 0.01,  # 0.8% Criminal/Record
    LegalIssue.BRIEF_SERVICES: 0.29,    # 28.2% Public Benefits + Restraining Order/DV
}


# =============================================================================
# DEMOGRAPHIC DISTRIBUTIONS
# Source: Salesforce demographic fields
# =============================================================================

# Gender (2,492 valid responses — 69.2% missing, use with caution)
GENDER_DISTRIBUTION = {
    Gender.FEMALE: 0.587,
    Gender.MALE: 0.376,
    Gender.NONBINARY: 0.006,
    Gender.TRANS_FEMALE: 0.004,
    Gender.TRANS_MALE: 0.001,
    Gender.NOT_LISTED: 0.026,
    Gender.DECLINE: 0.00,
}

# Ethnicity distribution
# Source: Salesforce Race/Ethnicity field (6,521 valid responses, multi-select)
# Note: Multi-select field — percentages reflect individual selections, not unique records
ETHNICITY_DISTRIBUTION = {
    "Latino-Mexican": 0.110,
    "Latino-Central American": 0.124,
    "Latino-South American": 0.043,
    "Latino-Caribbean": 0.012,
    "Latino-Other": 0.054,
    "Asian-Chinese": 0.051,
    "Asian-Filipino": 0.036,
    "Asian-Vietnamese": 0.011,
    "Asian-Korean": 0.009,
    "Asian-Japanese": 0.007,
    "Asian-Indian": 0.015,
    "Asian-Cambodian": 0.004,
    "Asian-Central": 0.004,
    "Asian-Other": 0.019,
    "Black-African American": 0.152,
    "Black-African": 0.041,
    "Black-Caribbean, Central American, South American or Mexican": 0.012,
    "Black-Other": 0.020,
    "White-European": 0.118,
    "White-Other": 0.080,
    "Pacific Islander-Samoan": 0.007,
    "Pacific Islander-Native Hawaiian": 0.002,
    "Pacific Islander-Chamorro": 0.001,
    "Pacific Islander-Other": 0.005,
    "Indigenous-American Indian/Native American": 0.022,
    "Indigenous-Indigenous from Mexico, Caribbean, Central/South America": 0.009,
    "Indigenous-Other Indigenous": 0.005,
    "Middle Eastern/North African-North African": 0.013,
    "Middle Eastern/North African-West Asian": 0.006,
    "Middle Eastern/North African-Other": 0.008,
}

# Employment (6,499 valid responses)
EMPLOYMENT_DISTRIBUTION = {
    EmploymentStatus.EMPLOYED_FULL: 0.240,
    EmploymentStatus.EMPLOYED_PART: 0.171,
    EmploymentStatus.UNEMPLOYED_LOOKING: 0.282,
    EmploymentStatus.UNEMPLOYED_NOT_LOOKING: 0.056,
    EmploymentStatus.DISABLED: 0.170,
    EmploymentStatus.RETIRED: 0.081,
}

HOUSING_DISTRIBUTION = {
    HousingStatus.RENTS: 0.60,
    HousingStatus.LIVING_WITH_OTHERS: 0.15,
    HousingStatus.UNSTABLE: 0.10,
    HousingStatus.SHELTER: 0.08,
    HousingStatus.OWNS: 0.07,
}

# English Proficiency (8,090 valid responses)
ENGLISH_FLUENCY_DISTRIBUTION = {
    EnglishFluency.NATIVE: 0.552,
    EnglishFluency.VERY_FLUENT: 0.136,
    EnglishFluency.FLUENT: 0.122,
    EnglishFluency.SOMEWHAT_FLUENT: 0.063,
    EnglishFluency.NOT_FLUENT: 0.127,
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
# Source: Salesforce Age field (7,682 valid responses, median 41, mean 42.4)
AGE_DISTRIBUTION = {
    (18, 24): 0.05,    # 5.0% (Under 18 excluded — 6.3% in data are likely data entry errors)
    (25, 34): 0.212,   # 21.2%
    (35, 44): 0.244,   # 24.4%
    (45, 54): 0.169,   # 16.9%
    (55, 64): 0.131,   # 13.1%
    (65, 74): 0.088,   # 8.8%
    (75, 85): 0.041,   # 4.1%
    (16, 17): 0.065,   # 6.5% Under 18 (minors with legal issues)
}


# =============================================================================
# INCOME DISTRIBUTION
# Based on 200% federal poverty line eligibility
# =============================================================================

# Monthly income ranges
# Source: Salesforce Total Monthly Income (6,983 valid responses, median $1,550, mean $2,564)
INCOME_DISTRIBUTION = {
    (0, 499): 0.240,         # 24.0%
    (500, 999): 0.084,       # 8.4%
    (1000, 1499): 0.145,     # 14.5%
    (1500, 1999): 0.091,     # 9.1%
    (2000, 2999): 0.137,     # 13.7%
    (3000, 3999): 0.101,     # 10.1%
    (4000, 4999): 0.067,     # 6.7%
    (5000, 7499): 0.071,     # 7.1%
    (7500, 9999): 0.024,     # 2.4%
    (10000, 15000): 0.041,   # 4.1% ($10,000+)
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
