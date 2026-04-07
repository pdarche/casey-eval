"""
Configuration for persona generation.

Allows pinning dimensions, custom distributions, and guaranteed scenario slots.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

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
from eval.personas.distributions import (
    LANGUAGE_DISTRIBUTION,
    LEGAL_ISSUE_DISTRIBUTION,
    GENDER_DISTRIBUTION,
    ETHNICITY_DISTRIBUTION,
    EMPLOYMENT_DISTRIBUTION,
    HOUSING_DISTRIBUTION,
    ENGLISH_FLUENCY_DISTRIBUTION,
    EDUCATION_DISTRIBUTION,
    COMMUNICATION_STYLE_DISTRIBUTION,
    TRUST_LEVEL_DISTRIBUTION,
)


# Scenario keys that map to Persona boolean flags
VALID_SCENARIO_KEYS = {
    "discloses_dv",
    "discloses_crisis",
    "gives_impossible_answers",
    "mentions_multiple_issues",
    "attempts_out_of_scope",
    "is_returning_client",
}

# Maps dimension names to their default distributions and enum types
_DIMENSION_DEFAULTS = {
    "language": (LANGUAGE_DISTRIBUTION, Language),
    "legal_issue": (LEGAL_ISSUE_DISTRIBUTION, LegalIssue),
    "gender": (GENDER_DISTRIBUTION, Gender),
    "ethnicity": (ETHNICITY_DISTRIBUTION, None),  # string keys, no enum
    "employment": (EMPLOYMENT_DISTRIBUTION, EmploymentStatus),
    "housing": (HOUSING_DISTRIBUTION, HousingStatus),
    "english_fluency": (ENGLISH_FLUENCY_DISTRIBUTION, EnglishFluency),
    "education": (EDUCATION_DISTRIBUTION, None),  # string keys
    "communication_style": (COMMUNICATION_STYLE_DISTRIBUTION, CommunicationStyle),
    "trust_level": (TRUST_LEVEL_DISTRIBUTION, TrustLevel),
}


def _resolve_enum_value(value: str, enum_cls):
    """Resolve a string to an enum value, trying name, value, and case-insensitive match."""
    if enum_cls is None:
        return value

    # Direct value match
    for member in enum_cls:
        if member.value == value:
            return member

    # Name match (e.g., "ENGLISH" -> Language.ENGLISH)
    try:
        return enum_cls[value.upper().replace(" ", "_")]
    except (KeyError, AttributeError):
        pass

    # Case-insensitive value match
    value_lower = value.lower()
    for member in enum_cls:
        if member.value.lower() == value_lower:
            return member

    # Common aliases
    aliases = {
        "spanish": "Español",
        "español": "Español",
        "cantonese": "粵語",
        "chinese": "粵語",
        "mandarin": "普通话",
        "arabic": "العربية",
        "filipino": "Filipino",
        "russian": "Русский",
        "vietnamese": "Tiếng Việt",
        "burmese": "ဗမာစာ",
        "other": "Other",
    }
    alias_value = aliases.get(value_lower)
    if alias_value:
        for member in enum_cls:
            if member.value == alias_value:
                return member

    raise ValueError(f"Cannot resolve '{value}' to {enum_cls.__name__}")


def _resolve_distribution(override, enum_cls) -> dict:
    """Resolve an override value to a full distribution dict.

    Args:
        override: str (pin), dict (custom distribution), or None (use default)
        enum_cls: The enum class for this dimension (None for string-keyed dimensions)
    """
    if override is None:
        return None

    if isinstance(override, str):
        # Pin to single value
        resolved = _resolve_enum_value(override, enum_cls)
        return {resolved: 1.0}

    if isinstance(override, dict):
        # Custom distribution - resolve keys
        resolved = {}
        for key, weight in override.items():
            resolved_key = _resolve_enum_value(key, enum_cls)
            resolved[resolved_key] = weight
        # Normalize weights
        total = sum(resolved.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            resolved = {k: v / total for k, v in resolved.items()}
        return resolved

    raise ValueError(f"Invalid override type: {type(override)}")


# Type alias for dimension overrides
DimensionOverride = Optional[Union[str, dict]]


@dataclass
class PersonaGenerationConfig:
    """Configuration for persona generation.

    Each dimension field accepts:
    - None: use ODL default distribution
    - str: pin to a single value (e.g., "Español")
    - dict[str, float]: custom distribution (e.g., {"Español": 0.7, "English": 0.3})

    scenarios: guaranteed slot counts per scenario key.
    """

    language: DimensionOverride = None
    legal_issue: DimensionOverride = None
    gender: DimensionOverride = None
    ethnicity: DimensionOverride = None
    employment: DimensionOverride = None
    housing: DimensionOverride = None
    english_fluency: DimensionOverride = None
    education: DimensionOverride = None
    communication_style: DimensionOverride = None
    trust_level: DimensionOverride = None

    scenarios: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        # Validate scenario keys
        for key in self.scenarios:
            if key not in VALID_SCENARIO_KEYS:
                raise ValueError(
                    f"Invalid scenario key '{key}'. Valid keys: {VALID_SCENARIO_KEYS}"
                )

    def get_distribution(self, dimension: str) -> dict:
        """Get the effective distribution for a dimension.

        Returns the override if set, otherwise the ODL default.
        """
        if dimension not in _DIMENSION_DEFAULTS:
            raise ValueError(f"Unknown dimension '{dimension}'")

        default_dist, enum_cls = _DIMENSION_DEFAULTS[dimension]
        override = getattr(self, dimension, None)

        resolved = _resolve_distribution(override, enum_cls)
        if resolved is not None:
            return resolved

        return dict(default_dist)

    def get_scenario_slots(self) -> dict[str, int]:
        """Return guaranteed scenario slot counts."""
        return {k: v for k, v in self.scenarios.items() if v > 0}

    @classmethod
    def from_dict(cls, data: dict) -> "PersonaGenerationConfig":
        """Create config from a JSON-compatible dict.

        Accepts keys matching field names. Enum values can be strings
        in any case or their display values.
        """
        if not data:
            return cls.default()

        kwargs = {}

        # Dimension overrides
        for dim_name in _DIMENSION_DEFAULTS:
            if dim_name in data:
                kwargs[dim_name] = data[dim_name]

        # Scenarios
        if "scenarios" in data:
            scenarios = {}
            for key, count in data["scenarios"].items():
                if key not in VALID_SCENARIO_KEYS:
                    raise ValueError(f"Invalid scenario key '{key}'")
                scenarios[key] = int(count)
            kwargs["scenarios"] = scenarios

        return cls(**kwargs)

    @classmethod
    def default(cls) -> "PersonaGenerationConfig":
        """ODL population defaults, no guaranteed scenarios."""
        return cls()

    @classmethod
    def all_spanish(cls) -> "PersonaGenerationConfig":
        """Pin language to Spanish."""
        return cls(language="Español")

    @classmethod
    def safety_focus(cls) -> "PersonaGenerationConfig":
        """Default distribution with guaranteed slots for all safety scenarios."""
        return cls(
            scenarios={
                "discloses_dv": 3,
                "discloses_crisis": 2,
                "gives_impossible_answers": 2,
                "mentions_multiple_issues": 2,
                "attempts_out_of_scope": 2,
                "is_returning_client": 2,
            }
        )

    @classmethod
    def language_coverage(cls) -> "PersonaGenerationConfig":
        """Ensure minimum representation per language by using a flatter distribution."""
        return cls(
            language={
                "English": 0.25,
                "Español": 0.20,
                "粵語": 0.10,
                "普通话": 0.10,
                "العربية": 0.10,
                "Filipino": 0.08,
                "Русский": 0.07,
                "Tiếng Việt": 0.05,
                "ဗမာစာ": 0.03,
                "Other": 0.02,
            }
        )
