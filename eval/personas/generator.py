"""
Persona generator for creating realistic synthetic ODL clients.

Generates personas based on target distributions to ensure test coverage
matches the actual client population.
"""

import random
from typing import Optional
from string import Template

from eval.personas.models import (
    Persona,
    Language,
    EnglishFluency,
    LegalIssue,
    Gender,
    EmploymentStatus,
    HousingStatus,
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
    AGE_DISTRIBUTION,
    INCOME_DISTRIBUTION,
    HOUSEHOLD_SIZE_DISTRIBUTION,
    EDGE_CASE_INJECTION_RATES,
)


# Prompt for LLM-based persona detail generation
PERSONA_DETAIL_PROMPT = Template("""Generate realistic details for a synthetic legal aid client persona.

Requirements:
- Language: ${language}
- Legal issue type: ${legal_issue}
- Gender: ${gender}
- Ethnicity: ${ethnicity}
- Age: ${age}
- Employment: ${employment}

Generate a JSON object with:
1. "name": A culturally appropriate full name for this person
2. "issue_details": A 2-3 sentence description of their specific legal situation (be specific with dates, names, amounts)
3. "pronouns": Appropriate pronouns based on gender
4. "email": A realistic-looking email address
5. "phone": A phone number in format +1415XXXXXXX
6. "address": An object with "MailingStreet", "MailingCity" (San Francisco), "MailingState" (CA), "MailingPostalCode" (a real SF zip)

Make the details realistic and consistent with someone seeking free legal aid in San Francisco.
The issue_details should be specific enough to drive a realistic intake conversation.

Return only valid JSON, no other text.""")


class PersonaGenerator:
    """
    Generates synthetic personas for evaluation testing.

    Can generate random personas based on distributions or use
    an LLM to add realistic details.
    """

    def __init__(self, llm_client=None, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            llm_client: Optional OpenAI/Anthropic client for generating details
            seed: Random seed for reproducibility
        """
        self.llm_client = llm_client
        if seed is not None:
            random.seed(seed)

        # Track generated personas to balance distributions
        self.generated_counts = {
            "language": {lang: 0 for lang in Language},
            "legal_issue": {issue: 0 for issue in LegalIssue},
            "gender": {g: 0 for g in Gender},
        }
        self.total_generated = 0

    def _sample_from_distribution(self, distribution: dict) -> any:
        """Sample a value from a probability distribution."""
        items = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(items, weights=weights, k=1)[0]

    def _sample_age(self) -> int:
        """Sample an age from the age distribution."""
        age_range = self._sample_from_distribution(AGE_DISTRIBUTION)
        return random.randint(age_range[0], age_range[1])

    def _sample_income(self) -> float:
        """Sample a monthly income from the income distribution."""
        income_range = self._sample_from_distribution(INCOME_DISTRIBUTION)
        return random.uniform(income_range[0], income_range[1])

    def _get_pronouns_for_gender(self, gender: Gender) -> str:
        """Get appropriate pronouns for a gender."""
        pronoun_map = {
            Gender.FEMALE: "She/her/hers",
            Gender.MALE: "He/him/his",
            Gender.NONBINARY: "They/them/theirs",
            Gender.TRANS_FEMALE: "She/her/hers",
            Gender.TRANS_MALE: "He/him/his",
            Gender.NOT_LISTED: "They/them/theirs",
            Gender.DECLINE: "They/them/theirs",
        }
        return pronoun_map.get(gender, "They/them/theirs")

    def _generate_placeholder_name(self, language: Language, gender: Gender) -> str:
        """Generate a placeholder name based on language and gender."""
        # Simple name pools by language/gender for non-LLM generation
        names = {
            Language.ENGLISH: {
                Gender.FEMALE: ["Sarah Johnson", "Emily Davis", "Jessica Brown"],
                Gender.MALE: ["Michael Smith", "James Wilson", "Robert Taylor"],
            },
            Language.SPANISH: {
                Gender.FEMALE: ["Maria Garcia", "Ana Rodriguez", "Carmen Lopez"],
                Gender.MALE: ["Jose Martinez", "Carlos Hernandez", "Miguel Sanchez"],
            },
            Language.CHINESE: {
                Gender.FEMALE: ["李美华", "王小红", "张丽"],
                Gender.MALE: ["李明", "王强", "张伟"],
            },
            Language.KOREAN: {
                Gender.FEMALE: ["김민지", "이서연", "박지은"],
                Gender.MALE: ["김민수", "이준호", "박성민"],
            },
            Language.FRENCH: {
                Gender.FEMALE: ["Marie Dubois", "Sophie Martin", "Claire Bernard"],
                Gender.MALE: ["Jean Dupont", "Pierre Martin", "Michel Bernard"],
            },
            Language.THAI: {
                Gender.FEMALE: ["สมศรี จันทร์ดี", "นิตยา สุขใจ", "วิไล ดีงาม"],
                Gender.MALE: ["สมชาย ดีใจ", "วิชัย สุขสันต์", "ประเสริฐ มั่นคง"],
            },
            Language.JAPANESE: {
                Gender.FEMALE: ["田中花子", "山田美咲", "佐藤愛"],
                Gender.MALE: ["田中太郎", "山田健一", "佐藤大輔"],
            },
        }

        # Default to neutral names
        default_names = ["Alex Morgan", "Jordan Lee", "Casey Kim"]

        lang_names = names.get(language, {})
        gender_names = lang_names.get(gender, default_names)

        return random.choice(gender_names)

    def _generate_placeholder_issue_details(self, legal_issue: LegalIssue) -> str:
        """Generate placeholder issue details based on issue type."""
        details = {
            LegalIssue.HOUSING: "Landlord is not making necessary repairs and is threatening eviction. The heating has been broken for 2 months.",
            LegalIssue.UD_HOUSING: "Received a 3-day notice to pay or quit. Behind on rent due to job loss. Court date is in 2 weeks.",
            LegalIssue.FAMILY: "Going through divorce proceedings. Need help with custody arrangement for 2 children.",
            LegalIssue.EMPLOYMENT: "Terminated from job without proper notice. Employer owes back wages for overtime work.",
            LegalIssue.IMMIGRATION: "Work visa expires in 3 months. Need help understanding options for renewal or status change.",
            LegalIssue.CONSUMER: "Paid $1,500 for services that were never delivered. Company is not responding to refund requests.",
            LegalIssue.ELDER_ESTATE: "Need help setting up power of attorney and updating will. Want to ensure assets go to children.",
            LegalIssue.CIVIL_LITIGATION: "Neighbor damaged property during construction. Need help recovering repair costs.",
            LegalIssue.BRIEF_SERVICES: "Need a legal letter drafted to resolve a dispute with a contractor.",
        }
        return details.get(legal_issue, "Need legal assistance with a pressing matter.")

    def _should_inject_edge_case(self, case_type: str) -> bool:
        """Determine if an edge case should be injected based on rates."""
        rate = EDGE_CASE_INJECTION_RATES.get(case_type, 0)
        return random.random() < rate

    def generate_random_persona(
        self,
        language: Optional[Language] = None,
        legal_issue: Optional[LegalIssue] = None,
        include_edge_cases: bool = True,
    ) -> Persona:
        """
        Generate a random persona based on distributions.

        Args:
            language: Force a specific language (or sample from distribution)
            legal_issue: Force a specific legal issue (or sample from distribution)
            include_edge_cases: Whether to potentially inject edge case behaviors

        Returns:
            A generated Persona
        """
        # Sample or use provided values
        lang = language or self._sample_from_distribution(LANGUAGE_DISTRIBUTION)
        issue = legal_issue or self._sample_from_distribution(LEGAL_ISSUE_DISTRIBUTION)
        gender = self._sample_from_distribution(GENDER_DISTRIBUTION)
        ethnicity = self._sample_from_distribution(ETHNICITY_DISTRIBUTION)
        employment = self._sample_from_distribution(EMPLOYMENT_DISTRIBUTION)
        housing = self._sample_from_distribution(HOUSING_DISTRIBUTION)
        english_fluency = self._sample_from_distribution(ENGLISH_FLUENCY_DISTRIBUTION)
        education = self._sample_from_distribution(EDUCATION_DISTRIBUTION)
        comm_style = self._sample_from_distribution(COMMUNICATION_STYLE_DISTRIBUTION)
        trust = self._sample_from_distribution(TRUST_LEVEL_DISTRIBUTION)
        household_size = self._sample_from_distribution(HOUSEHOLD_SIZE_DISTRIBUTION)

        age = self._sample_age()
        income = self._sample_income()

        # Adjust English fluency based on language
        if lang != Language.ENGLISH and english_fluency in [EnglishFluency.NATIVE, EnglishFluency.VERY_FLUENT]:
            english_fluency = self._sample_from_distribution({
                EnglishFluency.FLUENT: 0.3,
                EnglishFluency.SOMEWHAT_FLUENT: 0.4,
                EnglishFluency.NOT_FLUENT: 0.3,
            })

        # Determine number of minors (based on household size)
        num_minors = 0
        if household_size > 1:
            max_minors = min(household_size - 1, 5)
            num_minors = random.randint(0, max_minors)

        # Generate basic details
        name = self._generate_placeholder_name(lang, gender)
        pronouns = self._get_pronouns_for_gender(gender)
        issue_details = self._generate_placeholder_issue_details(issue)

        # Determine edge case flags
        discloses_dv = include_edge_cases and self._should_inject_edge_case("discloses_dv")
        discloses_crisis = include_edge_cases and self._should_inject_edge_case("discloses_crisis")
        gives_impossible = include_edge_cases and self._should_inject_edge_case("gives_impossible_answers")
        multi_issues = include_edge_cases and self._should_inject_edge_case("mentions_multiple_issues")
        out_of_scope = include_edge_cases and self._should_inject_edge_case("attempts_out_of_scope")
        returning = include_edge_cases and self._should_inject_edge_case("is_returning_client")

        # Determine urgency (housing/UD cases more likely urgent)
        is_urgent = random.random() < 0.3 if issue in [LegalIssue.HOUSING, LegalIssue.UD_HOUSING] else random.random() < 0.1

        persona = Persona(
            name=name,
            age=age,
            gender=gender,
            pronouns=pronouns,
            ethnicity=ethnicity,
            primary_language=lang,
            english_fluency=english_fluency,
            education_level=education,
            legal_issue=issue,
            issue_details=issue_details,
            issue_severity="urgent" if is_urgent else "standard",
            housing_status=housing,
            employment_status=employment,
            household_size=household_size,
            num_minors=num_minors,
            monthly_income=round(income, 2),
            communication_style=comm_style,
            trust_level=trust,
            discloses_dv=discloses_dv,
            discloses_crisis=discloses_crisis,
            gives_impossible_answers=gives_impossible,
            mentions_multiple_issues=multi_issues,
            attempts_out_of_scope=out_of_scope,
            is_returning_client=returning,
        )

        # Update tracking
        self.generated_counts["language"][lang] += 1
        self.generated_counts["legal_issue"][issue] += 1
        self.generated_counts["gender"][gender] += 1
        self.total_generated += 1

        return persona

    def generate_batch(
        self,
        count: int,
        include_edge_cases: bool = True,
    ) -> list[Persona]:
        """
        Generate a batch of personas.

        Args:
            count: Number of personas to generate
            include_edge_cases: Whether to include edge case behaviors

        Returns:
            List of generated Personas
        """
        return [
            self.generate_random_persona(include_edge_cases=include_edge_cases)
            for _ in range(count)
        ]

    def generate_stratified_batch(
        self,
        count: int,
        by_language: bool = True,
        by_legal_issue: bool = True,
    ) -> list[Persona]:
        """
        Generate personas ensuring coverage of key dimensions.

        Args:
            count: Approximate number of personas to generate
            by_language: Ensure all languages are covered
            by_legal_issue: Ensure all legal issues are covered

        Returns:
            List of generated Personas
        """
        personas = []

        # First, ensure minimum coverage
        if by_language:
            for lang in Language:
                personas.append(self.generate_random_persona(language=lang))

        if by_legal_issue:
            for issue in LegalIssue:
                personas.append(self.generate_random_persona(legal_issue=issue))

        # Fill remaining with random personas
        remaining = max(0, count - len(personas))
        personas.extend(self.generate_batch(remaining))

        return personas

    def get_distribution_stats(self) -> dict:
        """Get statistics on generated persona distributions."""
        if self.total_generated == 0:
            return {"total": 0, "distributions": {}}

        stats = {"total": self.total_generated, "distributions": {}}

        for dimension, counts in self.generated_counts.items():
            stats["distributions"][dimension] = {
                str(key): {
                    "count": value,
                    "percentage": round(value / self.total_generated * 100, 1),
                }
                for key, value in counts.items()
            }

        return stats
