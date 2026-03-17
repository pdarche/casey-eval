"""
Population-aware persona generation.

Queries the database for current persona dimension counts and computes
adjusted sampling weights so that each new batch fills gaps in the
existing population, pulling the overall distribution toward targets.
"""

from eval.personas.config import PersonaGenerationConfig, _DIMENSION_DEFAULTS


# Maps JSONB persona keys to config dimension names
_JSONB_TO_DIMENSION = {
    "primary_language": "language",
    "legal_issue": "legal_issue",
    "gender": "gender",
    "ethnicity": "ethnicity",
    "employment_status": "employment",
    "housing_status": "housing",
    "english_fluency": "english_fluency",
    "education_level": "education",
    "communication_style": "communication_style",
    "trust_level": "trust_level",
}


def get_population_counts() -> dict[str, dict[str, int]]:
    """Query the conversations table for current persona dimension counts.

    Returns:
        Dict mapping dimension name -> {category_value: count}.
        Returns empty dict if DB is unavailable.
    """
    try:
        from eval.database import get_cursor
    except Exception:
        return {}

    counts: dict[str, dict[str, int]] = {}

    try:
        with get_cursor() as cur:
            for jsonb_key, dimension_name in _JSONB_TO_DIMENSION.items():
                cur.execute(
                    f"""
                    SELECT persona->>'{jsonb_key}' AS value, COUNT(*) AS cnt
                    FROM conversations
                    WHERE persona->>'{jsonb_key}' IS NOT NULL
                    GROUP BY value
                    """
                )
                rows = cur.fetchall()
                if rows:
                    counts[dimension_name] = {
                        row["value"]: row["cnt"] for row in rows
                    }
    except Exception:
        return {}

    return counts


def compute_adjusted_weights(
    target_distribution: dict,
    current_counts: dict[str, int],
    batch_size: int,
    adjustment_strength: float = 2.0,
) -> dict:
    """Compute adjusted sampling weights based on population gaps.

    Args:
        target_distribution: Target {category: probability} (keys are enums or strings).
        current_counts: Current {category_string: count} from the DB.
        batch_size: Number of personas to generate in this batch.
        adjustment_strength: Exponent applied to gaps (higher = more aggressive correction).

    Returns:
        Adjusted distribution dict with same key types as target_distribution.
    """
    total_existing = sum(current_counts.values())
    total_after = total_existing + batch_size

    gaps = {}
    for category, target_pct in target_distribution.items():
        # Match category to current_counts by string value
        cat_str = category.value if hasattr(category, "value") else str(category)
        current = current_counts.get(cat_str, 0)
        gap = (target_pct * total_after) - current
        gaps[category] = max(gap, 0.01)

    # Normalize gaps to sum to 1.0
    gap_total = sum(gaps.values())
    weights = {k: v / gap_total for k, v in gaps.items()}

    # Apply adjustment strength: raise each weight to the power, re-normalize
    if adjustment_strength != 1.0:
        powered = {k: v ** adjustment_strength for k, v in weights.items()}
        powered_total = sum(powered.values())
        weights = {k: v / powered_total for k, v in powered.items()}

    return weights


def compute_all_adjusted_distributions(
    config: PersonaGenerationConfig,
    population_counts: dict[str, dict[str, int]],
    batch_size: int,
    adjustment_strength: float = 2.0,
) -> dict[str, dict]:
    """Compute adjusted distributions for all dimensions.

    Skips dimensions where the config override is a pinned string
    (single value = user explicitly wants 100% of that value).

    Args:
        config: The persona generation config with target distributions.
        population_counts: Output of get_population_counts().
        batch_size: Number of personas to generate.
        adjustment_strength: Exponent for gap correction.

    Returns:
        Dict mapping dimension name -> adjusted distribution dict.
    """
    adjusted = {}

    for dimension_name in _DIMENSION_DEFAULTS:
        # Skip pinned dimensions (user explicitly set a single string value)
        raw_override = getattr(config, dimension_name, None)
        if isinstance(raw_override, str):
            continue

        target_dist = config.get_distribution(dimension_name)
        current_counts = population_counts.get(dimension_name, {})
        adjusted[dimension_name] = compute_adjusted_weights(
            target_distribution=target_dist,
            current_counts=current_counts,
            batch_size=batch_size,
            adjustment_strength=adjustment_strength,
        )

    return adjusted
