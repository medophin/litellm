from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .xecguard import XecGuardGuardrail

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
):
    import litellm

    _cb: Final = XecGuardGuardrail(
        api_base=litellm_params.api_base,
        api_key=litellm_params.api_key,
        xecguard_model=litellm_params.xecguard_model,
        policy_names=litellm_params.policy_names,
        custom_policy_names=litellm_params.custom_policy_names,
        custom_policy_min_chars=litellm_params.custom_policy_min_chars,
        apply_to_aliases=litellm_params.apply_to_aliases,
        except_aliases=litellm_params.except_aliases,
        send_meta=litellm_params.send_meta,
        meta_data_fields=litellm_params.meta_data_fields,
        custom_header_meta_fields=litellm_params.custom_header_meta_fields,
        meta_identity_format=litellm_params.meta_identity_format,
        block_on_error=litellm_params.block_on_error,
        grounding_strictness=litellm_params.grounding_strictness,
        stream_mode=litellm_params.stream_mode or "window",
        stream_chunk=litellm_params.stream_chunk or 1200,
        stream_context=litellm_params.stream_context or 120,
        stream_first=litellm_params.stream_first if litellm_params.stream_first is not None else False,
        rationale_poll_seconds=litellm_params.rationale_poll_seconds,
        guardrail_name=guardrail.get(
            "guardrail_name",
            "",
        ),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )
    litellm.logging_callback_manager.add_litellm_callback(
        _cb,
    )

    return _cb


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.XECGUARD.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.XECGUARD.value: XecGuardGuardrail,
}
