"""Strict M2.4 evaluation-set and exact-contamination schemas."""

from __future__ import annotations

import json
import unicodedata
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from tinyllm.data.tokenization_schema import ChatTemplateIdentity, TokenizerIdentity
from tinyllm.schemas.base import StrictSchema

EvaluationCategory = Literal[
    "config",
    "json",
    "linux",
    "logs",
    "python",
    "refusal",
    "short_code",
]
EvaluationLanguage = Literal["en", "zh"]
MatchKind = Literal["full_sequence", "prompt_prefix"]


def _freeze_json_array(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _validate_canonical_text(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("evaluation text cannot be blank")
    if value != value.strip():
        raise ValueError("evaluation text must not have outer whitespace")
    if "\r" in value or unicodedata.normalize("NFC", value) != value:
        raise ValueError("evaluation text must use NFC and LF line endings")
    if any(unicodedata.category(char) == "Cc" and char not in {"\n", "\t"} for char in value):
        raise ValueError("evaluation text contains a forbidden control character")
    return value


def _validate_sorted_unique(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if any(not value for value in values) or tuple(sorted(set(values))) != values:
        raise ValueError(f"{field_name} must be non-empty, unique, and sorted")
    return values


class EvaluationPromptMessage(StrictSchema):
    """One canonical non-assistant message in a frozen evaluation Prompt."""

    role: Literal["system", "user"]
    content: str = Field(min_length=1, max_length=131_072)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Refuse silent normalization of public evaluation text."""

        return _validate_canonical_text(value)


class AuthoredProvenance(StrictSchema):
    """Redistribution declaration for TinyLLM-authored evaluation content."""

    origin: Literal["tinyllm-authored"]
    license: Literal["Apache-2.0"]
    redistribution_allowed: Literal[True]
    source_note: str = Field(min_length=1, max_length=256)


class ExactMatchScorer(StrictSchema):
    """Normalized exact-string scoring contract."""

    kind: Literal["exact_match"]
    accepted_answers: tuple[str, ...] = Field(min_length=1)
    case_sensitive: bool
    strip_outer_whitespace: Literal[True]

    @field_validator("accepted_answers", mode="before")
    @classmethod
    def freeze_answers(cls, value: object) -> object:
        """Convert the natural JSON array representation to an immutable tuple."""

        return _freeze_json_array(value)

    @field_validator("accepted_answers")
    @classmethod
    def validate_answers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require canonical unique answers in stable order."""

        canonical = tuple(_validate_canonical_text(item) for item in value)
        return _validate_sorted_unique(canonical, field_name="accepted answers")


class MultipleChoiceScorer(StrictSchema):
    """Frozen option-order and answer-index scoring contract."""

    kind: Literal["multiple_choice"]
    choices: tuple[str, ...] = Field(min_length=2, max_length=8)
    answer_index: int = Field(ge=0)

    @field_validator("choices", mode="before")
    @classmethod
    def freeze_choices(cls, value: object) -> object:
        """Convert JSON choices to an immutable tuple without reordering."""

        return _freeze_json_array(value)

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty or duplicate choices without reordering them."""

        canonical = tuple(_validate_canonical_text(item) for item in value)
        if len(set(canonical)) != len(canonical):
            raise ValueError("multiple-choice options must be unique")
        return canonical

    @model_validator(mode="after")
    def validate_answer_index(self) -> MultipleChoiceScorer:
        """Keep the answer index inside the frozen option list."""

        if self.answer_index >= len(self.choices):
            raise ValueError("multiple-choice answer index is outside choices")
        return self


class JsonObjectScorer(StrictSchema):
    """Canonical JSON-object scoring contract."""

    kind: Literal["json_object"]
    expected_json: str = Field(min_length=2)
    required_keys: tuple[str, ...]

    @field_validator("required_keys", mode="before")
    @classmethod
    def freeze_required_keys(cls, value: object) -> object:
        """Convert JSON key arrays to immutable tuples."""

        return _freeze_json_array(value)

    @field_validator("expected_json")
    @classmethod
    def validate_expected_json(cls, value: str) -> str:
        """Require a canonical JSON object rather than an ambiguous Python mapping."""

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("expected JSON is invalid") from exc
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON must be an object")
        canonical = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if value != canonical:
            raise ValueError("expected JSON must use canonical encoding")
        return value

    @field_validator("required_keys")
    @classmethod
    def validate_required_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require stable unique JSON keys."""

        return _validate_sorted_unique(value, field_name="required JSON keys")

    @model_validator(mode="after")
    def validate_keys_exist(self) -> JsonObjectScorer:
        """Every required key must exist in the canonical reference object."""

        parsed = json.loads(self.expected_json)
        if any(key not in parsed for key in self.required_keys):
            raise ValueError("required JSON key is absent from expected object")
        return self


class RequiredTermsScorer(StrictSchema):
    """Deterministic required/forbidden-term scoring contract."""

    kind: Literal["required_terms"]
    required_terms: tuple[str, ...] = Field(min_length=1)
    forbidden_terms: tuple[str, ...] = ()
    case_sensitive: bool

    @field_validator("required_terms", "forbidden_terms", mode="before")
    @classmethod
    def freeze_terms(cls, value: object) -> object:
        """Convert JSON term arrays to immutable tuples."""

        return _freeze_json_array(value)

    @field_validator("required_terms", "forbidden_terms")
    @classmethod
    def validate_terms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require canonical stable term sets."""

        canonical = tuple(_validate_canonical_text(item) for item in value)
        return _validate_sorted_unique(canonical, field_name="scoring terms")

    @model_validator(mode="after")
    def validate_disjoint_terms(self) -> RequiredTermsScorer:
        """A term cannot be simultaneously required and forbidden."""

        if set(self.required_terms) & set(self.forbidden_terms):
            raise ValueError("required and forbidden scoring terms must be disjoint")
        return self


class HumanRubricScorer(StrictSchema):
    """Explicit human-judgment rubric whose item-level rationale must be retained."""

    kind: Literal["human_rubric"]
    criteria: tuple[str, ...] = Field(min_length=1)
    pass_threshold: int = Field(gt=0)
    retain_judgment_rationale: Literal[True]

    @field_validator("criteria", mode="before")
    @classmethod
    def freeze_criteria(cls, value: object) -> object:
        """Convert JSON criteria arrays to immutable tuples."""

        return _freeze_json_array(value)

    @field_validator("criteria")
    @classmethod
    def validate_criteria(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Freeze unique canonical criteria."""

        canonical = tuple(_validate_canonical_text(item) for item in value)
        return _validate_sorted_unique(canonical, field_name="rubric criteria")

    @model_validator(mode="after")
    def validate_threshold(self) -> HumanRubricScorer:
        """The passing threshold cannot exceed available binary criteria."""

        if self.pass_threshold > len(self.criteria):
            raise ValueError("human-rubric threshold exceeds criteria count")
        return self


ScorerSpec = Annotated[
    ExactMatchScorer
    | MultipleChoiceScorer
    | JsonObjectScorer
    | RequiredTermsScorer
    | HumanRubricScorer,
    Field(discriminator="kind"),
]


class EvaluationItem(StrictSchema):
    """One authored, licensed, objectively scorable domain evaluation item."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^domain-(config|json|linux|logs|python|refusal|short-code)-[0-9]{3}$")
    language: EvaluationLanguage
    category: EvaluationCategory
    prompt_messages: tuple[EvaluationPromptMessage, ...] = Field(min_length=1, max_length=2)
    reference_answer: str = Field(min_length=1, max_length=131_072)
    scorer: ScorerSpec
    provenance: AuthoredProvenance
    tags: tuple[str, ...] = ()

    @field_validator("prompt_messages", "tags", mode="before")
    @classmethod
    def freeze_item_sequences(cls, value: object) -> object:
        """Convert persisted JSON arrays to immutable item fields."""

        return _freeze_json_array(value)

    @field_validator("reference_answer")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        """Require the same canonical text boundary as Prompt messages."""

        return _validate_canonical_text(value)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Keep public tags deterministic."""

        return _validate_sorted_unique(value, field_name="evaluation tags")

    @model_validator(mode="after")
    def validate_item_contract(self) -> EvaluationItem:
        """Bind ID/category, Prompt roles, and Canonical Reference to the scorer."""

        category_id = self.category.replace("_", "-")
        if not self.id.startswith(f"domain-{category_id}-"):
            raise ValueError("evaluation item ID must match category")
        roles = tuple(message.role for message in self.prompt_messages)
        if roles not in {("user",), ("system", "user")}:
            raise ValueError("evaluation Prompt must be user or system/user")
        if isinstance(self.scorer, ExactMatchScorer):
            if self.reference_answer not in self.scorer.accepted_answers:
                raise ValueError("reference answer must be accepted by exact-match scorer")
        elif isinstance(self.scorer, MultipleChoiceScorer):
            if self.reference_answer != self.scorer.choices[self.scorer.answer_index]:
                raise ValueError("reference answer must equal the correct multiple-choice option")
        elif (
            isinstance(self.scorer, JsonObjectScorer)
            and self.reference_answer != self.scorer.expected_json
        ):
            raise ValueError("reference answer must equal canonical expected JSON")
        return self


class LanguageCounts(StrictSchema):
    """Frozen English/Chinese item counts."""

    en: int = Field(ge=0)
    zh: int = Field(ge=0)


class CategoryCounts(StrictSchema):
    """Frozen domain-category item counts."""

    config: int = Field(ge=0)
    json_items: int = Field(ge=0)
    linux: int = Field(ge=0)
    logs: int = Field(ge=0)
    python: int = Field(ge=0)
    refusal: int = Field(ge=0)
    short_code: int = Field(ge=0)


class DecodingConfig(StrictSchema):
    """Frozen deterministic generation settings for the later Baseline."""

    do_sample: Literal[False]
    temperature: float = Field(ge=0.0, le=0.0)
    top_p: float = Field(ge=1.0, le=1.0)
    max_new_tokens: int = Field(gt=0, le=4096)
    seed: int = Field(ge=0, le=2**32 - 1)


class ContaminationPolicy(StrictSchema):
    """Exact-only M2 contamination policy."""

    split: Literal["train"]
    full_sequence: Literal[True]
    prompt_prefix: Literal[True]
    near_dedup: Literal[False]
    fingerprint_algorithm: Literal["token-sequence-sha256-v1"]


class EvaluationBuildConfig(StrictSchema):
    """Complete deterministic evaluation-set build configuration."""

    schema_version: Literal["1.0"] = "1.0"
    suite_name: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    version_prefix: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}-v1$")
    expected_items: int = Field(gt=0)
    language_counts: LanguageCounts
    category_counts: CategoryCounts
    tokenizer: TokenizerIdentity
    template: ChatTemplateIdentity
    max_sequence_length: int = Field(gt=1)
    decoding: DecodingConfig
    contamination: ContaminationPolicy

    @model_validator(mode="after")
    def validate_counts(self) -> EvaluationBuildConfig:
        """Expected total must match both independent frozen partitions."""

        if self.version_prefix != f"{self.suite_name}-v1":
            raise ValueError("evaluation version prefix must match suite name")
        if sum(self.language_counts.to_dict().values()) != self.expected_items:
            raise ValueError("evaluation language counts do not match expected total")
        if sum(self.category_counts.to_dict().values()) != self.expected_items:
            raise ValueError("evaluation category counts do not match expected total")
        return self


class EvaluationSetManifest(StrictSchema):
    """Timestamp-free content identity for a frozen evaluation set."""

    schema_version: Literal["1.0"] = "1.0"
    suite_name: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    suite_version: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}-v1-[0-9a-f]{8}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    items_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: int = Field(gt=0)
    language_counts: LanguageCounts
    category_counts: CategoryCounts
    scorer_counts: dict[str, int]
    tokenizer: TokenizerIdentity
    template: ChatTemplateIdentity
    max_sequence_length: int = Field(gt=1)
    decoding: DecodingConfig
    contamination: ContaminationPolicy

    @field_validator("scorer_counts")
    @classmethod
    def validate_scorer_counts(cls, value: dict[str, int]) -> dict[str, int]:
        """Require stable positive sparse scorer counts."""

        if list(value) != sorted(value) or any(
            not key or count <= 0 for key, count in value.items()
        ):
            raise ValueError("scorer counts must use sorted keys and positive values")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> EvaluationSetManifest:
        """Bind version suffix and aggregate counts to the content identity."""

        if self.suite_version != f"{self.suite_name}-v1-{self.content_sha256[:8]}":
            raise ValueError("evaluation version does not match content hash")
        if sum(self.language_counts.to_dict().values()) != self.item_count:
            raise ValueError("manifest language counts do not match item count")
        if sum(self.category_counts.to_dict().values()) != self.item_count:
            raise ValueError("manifest category counts do not match item count")
        if sum(self.scorer_counts.values()) != self.item_count:
            raise ValueError("manifest scorer counts do not match item count")
        return self


class ContaminationMatch(StrictSchema):
    """Content-free evidence that one evaluation fingerprint matched Train."""

    evaluation_item_id: str = Field(
        pattern=r"^domain-(config|json|linux|logs|python|refusal|short-code)-[0-9]{3}$"
    )
    match_kind: MatchKind
    fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_sample_id_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ContaminationReport(StrictSchema):
    """Path-free stable result of exact Train/evaluation fingerprint comparison."""

    schema_version: Literal["1.0"] = "1.0"
    status: Literal["clean", "contaminated"]
    fingerprint_algorithm: Literal["token-sequence-sha256-v1"]
    near_dedup: Literal["not_evaluated"]
    evaluation_suite_version: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}-v1-[0-9a-f]{8}$")
    evaluation_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_version: str = Field(pattern=r"^m2-sft-v1-[0-9a-f]{8}$")
    dataset_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checked_evaluation_items: int = Field(gt=0)
    checked_training_samples: int = Field(ge=0)
    contaminated_items: int = Field(ge=0)
    full_sequence_matches: int = Field(ge=0)
    prompt_prefix_matches: int = Field(ge=0)
    matches: tuple[ContaminationMatch, ...]

    @field_validator("matches", mode="before")
    @classmethod
    def freeze_matches(cls, value: object) -> object:
        """Convert persisted match arrays to immutable report evidence."""

        return _freeze_json_array(value)

    @field_validator("matches")
    @classmethod
    def validate_match_order(
        cls, value: tuple[ContaminationMatch, ...]
    ) -> tuple[ContaminationMatch, ...]:
        """Make reports reproducible and reject duplicate evidence rows."""

        keys = tuple(
            (
                item.evaluation_item_id,
                item.match_kind,
                item.training_sample_id_sha256,
            )
            for item in value
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("contamination matches must be unique and sorted")
        return value

    @model_validator(mode="after")
    def validate_summary(self) -> ContaminationReport:
        """Bind status and counts to the content-free match rows."""

        full = sum(item.match_kind == "full_sequence" for item in self.matches)
        prompt = sum(item.match_kind == "prompt_prefix" for item in self.matches)
        contaminated = len({item.evaluation_item_id for item in self.matches})
        if (full, prompt, contaminated) != (
            self.full_sequence_matches,
            self.prompt_prefix_matches,
            self.contaminated_items,
        ):
            raise ValueError("contamination summary counts do not match evidence")
        if self.contaminated_items > self.checked_evaluation_items:
            raise ValueError("contaminated count exceeds checked evaluation items")
        expected_status = "contaminated" if self.matches else "clean"
        if self.status != expected_status:
            raise ValueError("contamination status does not match evidence")
        return self
