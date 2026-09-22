from typing import Final, Literal

from pydantic import Field

from .base import GuardrailConfigModel

XECGUARD_DEFAULT_POLICY_OPTIONS: Final = [
    "Default_Policy_SystemPromptEnforcement",
    "Default_Policy_GeneralPromptAttackProtection",
    "Default_Policy_ContentBiasProtection",
    "Default_Policy_HarmfulContentProtection",
    "Default_Policy_SkillsProtection",
    "Default_Policy_PIISensitiveDataProtection",
]


class XecGuardConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description=(
            "Service Token for XecGuard (prefix 'xgs_'). "
            "If not provided, the XECGUARD_API_KEY environment "
            "variable is used."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "XecGuard API base URL. "
            "Defaults to https://api-xecguard.cycraft.ai. "
            "Falls back to the XECGUARD_API_BASE env var."
        ),
    )
    xecguard_model: str | None = Field(
        default=None,
        description=("XecGuard scanning model identifier. Defaults to 'xecguard_v2'."),
    )
    policy_names: list[str] | None = Field(
        default=None,
        description=(
            "XecGuard policies to apply on each scan. Select one or more "
            "of the built-in default policies; if none are selected, "
            "the guardrail defaults to System Prompt Enforcement + "
            "Harmful Content Protection."
        ),
        json_schema_extra={
            "ui_type": "multiselect",
            "options": list(XECGUARD_DEFAULT_POLICY_OPTIONS),
        },
    )
    block_on_error: bool | None = Field(
        default=None,
        description=(
            "Whether to block requests when the XecGuard API is "
            "unreachable. Defaults to true (fail-closed). "
            "Falls back to the XECGUARD_BLOCK_ON_ERROR env var."
        ),
    )
    grounding_strictness: Literal["BALANCED", "STRICT"] | None = Field(
        default=None,
        description=(
            "Strictness level for XecGuard context-grounding "
            "validation. 'BALANCED' (default) treats INCOMPLETE "
            "answers as SAFE; 'STRICT' flags them as UNSAFE. "
            "Grounding only runs in post_call when "
            "`metadata.xecguard_grounding_documents` is provided."
        ),
    )
    stream_mode: Literal["window", "buffer", "audit"] | None = Field(
        default=None,
        description=(
            "How streamed responses are scanned. 'window' (default) "
            "scans a rolling window of deltas and can cut the stream "
            "mid-flight. 'buffer' collects the whole response before "
            "scanning once (no token-by-token feel, zero leak). "
            "'audit' scans on the same cadence as 'window' but never "
            "blocks."
        ),
    )
    stream_chunk: int | None = Field(
        default=None,
        description=("Number of new deltas that trigger a window scan in 'window'/'audit' mode. Defaults to 40."),
    )
    stream_context: int | None = Field(
        default=None,
        description=(
            "Number of prior deltas kept as overlap context for each window scan in "
            "'window'/'audit' mode. Defaults to 10."
        ),
    )
    stream_first: bool | None = Field(
        default=None,
        description=(
            "In 'window'/'audit' mode, whether each delta is released to the client "
            "before it has been covered by a scan. Defaults to true."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "XecGuard"
