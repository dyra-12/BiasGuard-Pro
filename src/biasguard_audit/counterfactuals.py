import re
from typing import List, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class GrammarPolisher:
    def __init__(self):
        print("🔧 Loading grammar polisher...")
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("✅ Grammar polisher loaded!")

    def polish_phrase(self, text: str) -> str:
        """Apply a few lightweight handwritten regex repairs to a suggestion.

        This function performs a small set of deterministic substitutions to
        improve readability of automatically generated counterfactuals. It is
        intentionally small and rule-based to avoid heavy model costs during
        runtime.

        Args:
            text: Candidate suggestion string.

        Returns:
            Polished suggestion string.
        """

        issues = {
            r"\b(\w+) skills skills\b": r"\1 skills",
            r"\b(\w+) abilities abilities\b": r"\1 abilities",
            r"\bdemonstrate well-suited\b": "are well-suited",
            r"\bpeople may be healthcare\b": "people may pursue healthcare roles",
            r"\bat technical roles\b": "for technical roles",
            r"\bin technical\b": "in technical fields",
        }

        for pattern, replacement in issues.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text


class CounterfactualGenerator:
    def __init__(self):
        print("🚀 Initializing Counterfactual Generator...")
        self.grammar_polisher = GrammarPolisher()

        self.templates = {
            "gender_role": [
                "People with {skill} often excel in {field} roles.",
                "Those who demonstrate {skill} are well-suited for {field} careers.",
                "Individuals with strong {skill} can succeed in {field} positions.",
            ],
            "gender_trait": [
                "Developing {trait} is valuable for professional growth.",
                "{trait} contributes to success in various professional roles.",
                "Professional effectiveness often involves {trait}.",
            ],
            "gender_comparison": [
                "People frequently develop expertise in {field} through dedicated effort.",
                "Success in {field} comes from learning and practical experience.",
                "Many individuals build strong capabilities in {field} roles.",
            ],
            "behavior_description": [
                "The {role} demonstrated {trait} during the {context}.",
                "During the {context}, the {role} showed {trait}.",
            ],
        }

        self.replacements = {
            "women": "people",
            "men": "people",
            "woman": "person",
            "man": "person",
            "she": "they",
            "he": "they",
            "female": "",
            "male": "",
            "naturally": "",
            "inherently": "",
            "should": "may",
            "must": "can",
            "better": "well-suited",
            "best": "highly qualified",
            "too": "",
            "compassionate": "interpersonal skills",
            "caring": "supportive abilities",
            "emotional": "emotional intelligence",
            "aggressive": "assertiveness",
            "nurturing": "supportive qualities",
            "decisive": "strategic decision-making",
            "nurse": "healthcare",
            "nurses": "healthcare",
            "nursing": "healthcare",
            "engineer": "technical",
            "engineers": "technical",
            "engineering": "technical",
            "secretary": "administrative",
            "teacher": "educational",
            "leadership": "leadership",
            "business": "business",
            "needs": "can benefit from",
            "need": "can benefit from",
            "was": "demonstrated",
            "is": "demonstrates",
            "are": "demonstrate",
            "very": "",
            "less": "appropriate",
            "more": "enhanced",
        }

        self.generic_fallbacks = [
            "Professional success often involves developing relevant skills and capabilities.",
            "People can build strong professional abilities through dedicated learning.",
            "Career growth benefits from continuous skill development and adaptation.",
        ]

    def _extract_components(self, text: str, shap_words: List[str]) -> dict:
        """Extract semantic components (field, skill, trait, role, context).

        The heuristic extracts simple components from the input text and the
        top SHAP words to be used when filling generation templates. This is a
        lightweight heuristic (not an NLP pipeline) intended to produce
        reasonable placeholders for template rendering.
        """

        text_lower = text.lower()
        components = {
            "field": "professional",
            "skill": "relevant skills",
            "trait": "important abilities",
            "role": "professional",
            "context": "professional context",
        }

        fields = ["nurse", "engineer", "secretary", "teacher", "leadership", "business"]
        for field in fields:
            if field in text_lower:
                components["field"] = self.replacements.get(field, field)
                break

        roles = {
            "secretary": "administrative professional",
            "candidate": "professional",
        }
        for role, mapped_role in roles.items():
            if role in text_lower:
                components["role"] = mapped_role
                break

        if "today" in text_lower:
            components["context"] = "workday"
        elif "presentation" in text_lower:
            components["context"] = "presentation"
        elif "meeting" in text_lower:
            components["context"] = "meeting"

        traits = {
            "compassionate": "interpersonal skills",
            "emotional": "emotional intelligence",
            "aggressive": "assertiveness",
            "decisive": "strategic decision-making",
        }

        for word in shap_words:
            clean_word = re.sub(r"[^\w]", "", word.lower())
            if clean_word in traits:
                components["skill"] = traits[clean_word]
                components["trait"] = traits[clean_word]
                break

        if components["skill"] == "relevant skills" and shap_words:
            components["skill"] = self.replacements.get(
                shap_words[0].lower(), shap_words[0]
            )
            components["trait"] = components["skill"]

        return components

    def _apply_smart_replacements(self, text: str, shap_words: List[str]) -> str:
        """Replace known gendered or biased tokens with neutral alternatives.

        This routine performs token-level replacements using the `replacements`
        mapping and preserves capitalization where reasonable. If no
        replacement was applied, an empty string is returned.
        """

        words = text.split()
        new_words = []
        changed = False

        for word in words:
            clean_word = re.sub(r"[^\w]", "", word.lower())

            if clean_word in self.replacements:
                replacement = self.replacements[clean_word]
                if replacement:
                    if word[0].isupper():
                        replacement = replacement.capitalize()
                    new_words.append(replacement)
                    changed = True
                else:
                    continue
            else:
                new_words.append(word)

        if not changed:
            return ""

        result = " ".join(new_words)
        result = re.sub(r"\bthey needs\b", "they need", result, flags=re.IGNORECASE)
        result = re.sub(r"\bcan benefit from to be\b", "can benefit from being", result)
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def _clean_grammar(self, text: str) -> str:
        """Normalize spacing and sentence punctuation/casing for suggestions."""

        text = re.sub(r"\s+", " ", text).strip()
        if not text.endswith((".", "!", "?")):
            text = text + "."
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    def generate_counterfactuals(
        self,
        text: str,
        shap_results: List[Tuple[str, float]],
        num_alternatives: int = 3,
    ) -> List[str]:
        """Generate a short list of candidate counterfactual suggestions.

        The method combines template-based suggestions, smart token
        replacements, and generic fallbacks, then polishes the output with a
        small grammar pass. It favors diversity and will return up to
        `num_alternatives` distinct suggestions.

        Args:
            text: Original input text to transform.
            shap_results: List of (word, score) tuples from an explainer.
            num_alternatives: Desired number of suggestions to return.

        Returns:
            List of polished suggestion strings.
        """

        shap_words = [word for word, score in shap_results[:3]]
        components = self._extract_components(text, shap_words)

        suggestions = []
        text_lower = text.lower()

        # Template-based generation
        template_categories = []

        if any(
            phrase in text_lower for phrase in ["should be", "would be", "needs to be"]
        ):
            template_categories.append("gender_role")
        if any(
            phrase in text_lower
            for phrase in ["naturally better", "inherently better", "are better"]
        ):
            template_categories.append("gender_comparison")
        if any(word in text_lower for word in ["today", "during", "presentation"]):
            template_categories.append("behavior_description")
        if any(
            word in text_lower for word in ["emotional", "aggressive", "compassionate"]
        ):
            template_categories.append("gender_trait")

        if not template_categories:
            template_categories = ["gender_role", "gender_trait", "gender_comparison"]

        used_templates = set()
        for category in template_categories:
            for template in self.templates[category][:2]:
                if template not in used_templates:
                    try:
                        suggestion = template.format(
                            skill=components["skill"],
                            field=components["field"],
                            trait=components["trait"],
                            role=components["role"],
                            context=components["context"],
                        )
                        suggestion = self._clean_grammar(suggestion)
                        if suggestion not in suggestions:
                            suggestions.append(suggestion)
                            used_templates.add(template)
                    except:
                        continue

        # Smart replacements
        replaced = self._apply_smart_replacements(text, shap_words)
        if replaced and replaced not in suggestions:
            suggestions.append(replaced)

        # Ensure we have enough suggestions
        while len(suggestions) < num_alternatives:
            if len(suggestions) < num_alternatives:
                gender_removed = re.sub(
                    r"\b(women|men|woman|man|she|he|female|male)\b",
                    "people",
                    text,
                    flags=re.IGNORECASE,
                )
                gender_removed = self._clean_grammar(gender_removed)
                if gender_removed != text and gender_removed not in suggestions:
                    suggestions.append(gender_removed)

            if len(suggestions) < num_alternatives:
                for fallback in self.generic_fallbacks:
                    if fallback not in suggestions:
                        suggestions.append(fallback)
                        break

        # Polish all suggestions
        polished_suggestions = []
        for s in suggestions[:num_alternatives]:
            polished = self.grammar_polisher.polish_phrase(s)
            polished = self._clean_grammar(polished)
            if polished and polished not in polished_suggestions:
                polished_suggestions.append(polished)

        return polished_suggestions[:num_alternatives]
