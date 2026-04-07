"""
Persona generator for creating realistic synthetic ODL clients.

Generates personas based on configurable distributions to ensure test coverage
matches the desired population profile. Supports guaranteed scenario slots
and LLM-generated unique details.
"""

import json
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
    AGE_DISTRIBUTION,
    INCOME_DISTRIBUTION,
    HOUSEHOLD_SIZE_DISTRIBUTION,
)
from eval.personas.config import PersonaGenerationConfig


# Prompt for LLM-based persona detail generation
PERSONA_DETAIL_PROMPT = Template("""Generate realistic details for a synthetic legal aid client persona.

Requirements:
- Language: ${language}
- Legal issue type: ${legal_issue}
- Gender: ${gender}
- Ethnicity: ${ethnicity}
- Age: ${age}
- Employment: ${employment}
${scenario_context}
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

    Uses a PersonaGenerationConfig to control distributions, guaranteed
    scenario slots, and LLM-generated details.
    """

    def __init__(
        self,
        config: Optional[PersonaGenerationConfig] = None,
        llm_client=None,
        seed: Optional[int] = None,
        adjusted_distributions: Optional[dict] = None,
    ):
        """
        Initialize the generator.

        Args:
            config: Generation configuration (defaults to ODL population)
            llm_client: Optional OpenAI client for generating unique details
            seed: Random seed for reproducibility
            adjusted_distributions: Population-adjusted weights per dimension
        """
        self.config = config or PersonaGenerationConfig.default()
        self.llm_client = llm_client
        self.adjusted_distributions = adjusted_distributions or {}
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

    def _get_effective_distribution(self, dimension: str) -> dict:
        """Get distribution, preferring adjusted weights if available."""
        if dimension in self.adjusted_distributions:
            return self.adjusted_distributions[dimension]
        return self.config.get_distribution(dimension)

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
        names = {
            Language.ENGLISH: {
                Gender.FEMALE: ["Sarah Johnson", "Emily Davis", "Jessica Brown"],
                Gender.MALE: ["Michael Smith", "James Wilson", "Robert Taylor"],
            },
            Language.SPANISH: {
                Gender.FEMALE: ["Maria Garcia", "Ana Rodriguez", "Carmen Lopez"],
                Gender.MALE: ["Jose Martinez", "Carlos Hernandez", "Miguel Sanchez"],
            },
            Language.CANTONESE: {
                Gender.FEMALE: ["李美华", "王小红", "张丽"],
                Gender.MALE: ["李明", "王强", "张伟"],
            },
            Language.MANDARIN: {
                Gender.FEMALE: ["陈晓燕", "刘芳", "赵静"],
                Gender.MALE: ["陈伟", "刘洋", "赵磊"],
            },
            Language.ARABIC: {
                Gender.FEMALE: ["Fatima Hassan", "Amira Said", "Nour Ibrahim"],
                Gender.MALE: ["Ahmed Hassan", "Omar Said", "Youssef Ibrahim"],
            },
            Language.FILIPINO: {
                Gender.FEMALE: ["Maria Santos", "Ana Reyes", "Rose Cruz"],
                Gender.MALE: ["Juan Santos", "Jose Reyes", "Mark Cruz"],
            },
            Language.RUSSIAN: {
                Gender.FEMALE: ["Анна Иванова", "Мария Петрова", "Елена Сидорова"],
                Gender.MALE: ["Иван Иванов", "Дмитрий Петров", "Алексей Сидоров"],
            },
            Language.VIETNAMESE: {
                Gender.FEMALE: ["Nguyen Thi Mai", "Tran Thi Lan", "Le Thi Hoa"],
                Gender.MALE: ["Nguyen Van Minh", "Tran Van Duc", "Le Van Hung"],
            },
            Language.BURMESE: {
                Gender.FEMALE: ["Aye Aye Win", "Khin Mar Oo", "Su Su Lwin"],
                Gender.MALE: ["Aung Kyaw", "Min Thu", "Zaw Win"],
            },
        }

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

    def _get_scenario_context(self, scenario_flags: dict) -> str:
        """Build scenario context for the LLM detail prompt."""
        lines = []
        if scenario_flags.get("discloses_dv"):
            lines.append("- This person is experiencing domestic violence and will disclose this during intake")
        if scenario_flags.get("discloses_crisis"):
            lines.append("- This person is in emotional distress and may express hopelessness")
        if scenario_flags.get("gives_impossible_answers"):
            lines.append("- This person may initially give confused or incorrect answers")
        if scenario_flags.get("mentions_multiple_issues"):
            lines.append("- This person has multiple legal problems they want to discuss")
        if scenario_flags.get("attempts_out_of_scope"):
            lines.append("- This person will also ask about matters ODL does not handle (e.g., criminal defense)")
        if scenario_flags.get("is_returning_client"):
            lines.append("- This person has used ODL services before, about 2 years ago")
        if lines:
            return "Scenario context:\n" + "\n".join(lines) + "\n\n"
        return ""

    def _generate_details_with_llm(self, persona: Persona, scenario_flags: dict) -> Persona:
        """Use LLM to generate unique name, issue details, contact info.

        Falls back to template approach if no llm_client.
        """
        if not self.llm_client:
            return persona

        scenario_context = self._get_scenario_context(scenario_flags)

        prompt = PERSONA_DETAIL_PROMPT.substitute(
            language=persona.primary_language.value,
            legal_issue=persona.legal_issue.value,
            gender=persona.gender.value,
            ethnicity=persona.ethnicity,
            age=persona.age,
            employment=persona.employment_status.value,
            scenario_context=scenario_context,
        )

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                response_format={"type": "json_object"},
            )
            details = json.loads(response.choices[0].message.content)

            if "name" in details:
                persona.name = details["name"]
            if "issue_details" in details:
                persona.issue_details = details["issue_details"]
            if "pronouns" in details:
                persona.pronouns = details["pronouns"]
            if "email" in details:
                persona.email = details["email"]
            if "phone" in details:
                persona.phone = details["phone"]
            if "address" in details:
                persona.address = details["address"]
        except Exception:
            # Silently fall back to template details
            pass

        return persona

    def generate_random_persona(
        self,
        language: Optional[Language] = None,
        legal_issue: Optional[LegalIssue] = None,
        scenario_flags: Optional[dict] = None,
    ) -> Persona:
        """
        Generate a random persona based on config distributions.

        Args:
            language: Force a specific language (overrides config)
            legal_issue: Force a specific legal issue (overrides config)
            scenario_flags: Forced scenario flags (for guaranteed slots).
                            If None, no scenario flags are set.

        Returns:
            A generated Persona
        """
        # Sample from config-driven distributions
        lang = language or self._sample_from_distribution(self._get_effective_distribution("language"))
        issue = legal_issue or self._sample_from_distribution(self._get_effective_distribution("legal_issue"))
        gender = self._sample_from_distribution(self._get_effective_distribution("gender"))
        ethnicity = self._sample_from_distribution(self._get_effective_distribution("ethnicity"))
        employment = self._sample_from_distribution(self._get_effective_distribution("employment"))
        housing = self._sample_from_distribution(self._get_effective_distribution("housing"))
        english_fluency = self._sample_from_distribution(self._get_effective_distribution("english_fluency"))
        education = self._sample_from_distribution(self._get_effective_distribution("education"))
        comm_style = self._sample_from_distribution(self._get_effective_distribution("communication_style"))
        trust = self._sample_from_distribution(self._get_effective_distribution("trust_level"))
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

        # Generate basic details (may be overridden by LLM)
        name = self._generate_placeholder_name(lang, gender)
        pronouns = self._get_pronouns_for_gender(gender)
        issue_details = self._generate_placeholder_issue_details(issue)

        # Apply scenario flags (only from guaranteed slots, no probabilistic injection)
        flags = scenario_flags or {}

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
            discloses_dv=flags.get("discloses_dv", False),
            discloses_crisis=flags.get("discloses_crisis", False),
            gives_impossible_answers=flags.get("gives_impossible_answers", False),
            mentions_multiple_issues=flags.get("mentions_multiple_issues", False),
            attempts_out_of_scope=flags.get("attempts_out_of_scope", False),
            is_returning_client=flags.get("is_returning_client", False),
        )

        # Update tracking
        self.generated_counts["language"][lang] += 1
        self.generated_counts["legal_issue"][issue] += 1
        self.generated_counts["gender"][gender] += 1
        self.total_generated += 1

        return persona

    def generate_batch(self, count: int) -> list[Persona]:
        """
        Generate a batch of personas with guaranteed scenario slots.

        1. Generate personas for each guaranteed scenario slot
        2. Generate remaining clean personas (no forced scenarios)
        3. Enrich all personas with LLM-generated details (in parallel)
        4. Shuffle together

        Args:
            count: Total number of personas to generate

        Returns:
            List of generated Personas
        """
        import concurrent.futures

        personas_with_flags = []
        slots = self.config.get_scenario_slots()

        # Generate guaranteed scenario personas
        for scenario_key, slot_count in slots.items():
            for _ in range(slot_count):
                flags = {scenario_key: True}
                persona = self.generate_random_persona(scenario_flags=flags)
                personas_with_flags.append((persona, flags))

        # Generate remaining clean personas
        remaining = max(0, count - len(personas_with_flags))
        for _ in range(remaining):
            persona = self.generate_random_persona()
            personas_with_flags.append((persona, {}))

        # Enrich all personas with LLM-generated details in parallel
        if self.llm_client:
            def enrich(item):
                persona, flags = item
                return self._generate_details_with_llm(persona, flags)

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                personas = list(executor.map(enrich, personas_with_flags))
        else:
            personas = [p for p, _ in personas_with_flags]

        # Shuffle so scenario personas aren't all at the front
        random.shuffle(personas)

        return personas

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

        if by_language:
            for lang in Language:
                personas.append(self.generate_random_persona(language=lang))

        if by_legal_issue:
            for issue in LegalIssue:
                personas.append(self.generate_random_persona(legal_issue=issue))

        remaining = max(0, count - len(personas))
        personas.extend([self.generate_random_persona() for _ in range(remaining)])

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
