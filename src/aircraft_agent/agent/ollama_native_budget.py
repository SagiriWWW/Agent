"""Explicit native-Jinja policy, closed pending independent calibration.

The fixed runtime exposes native render/tokenize observations. Those observations
are not an independent local renderer, so this version never admits generation.
A profile edit cannot enable it. The original Go policies remain unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from aircraft_agent.agent import ollama_budget as go_budget
from aircraft_agent.agent.input_policy import InputPolicyError

VERSION = 'tool-view-v1-ollama-native-budget-session21-v1'
TEMPLATE_SHA256 = '40c21f34cf67d8c760ef72f8ad3ae5afad514299d4b06e91dd9a8d705af7b541'


def assess_request(request: dict, *, profile: dict | None = None) -> dict:
    """Validate the opt-in identity/resources, then refuse before any I/O to a model.

    Calibration status is deliberately not a trusted profile-controlled switch.
    Enabling admission requires a separately reviewed implementation/protocol.
    """
    try:
        go_budget.check_scope(request, model=go_budget.CANDIDATE_MODEL)
        size = len(json.dumps(request, ensure_ascii=False).encode())
        if size > go_budget.MAX_REQUEST_RESOURCE_BYTES:
            raise InputPolicyError('input_request_resource_exceeded', {
                'bytes': size, 'limit_bytes': go_budget.MAX_REQUEST_RESOURCE_BYTES})
        if profile is None:
            profile = json.loads(Path(os.environ['AGENT_MVP_OLLAMA_BUDGET_PROFILE']).read_text())
        expected = {
            'policy_version': VERSION,
            'model_name': go_budget.CANDIDATE_MODEL,
            'model_digest': go_budget.CANDIDATE_DIGEST,
            'template_sha256': TEMPLATE_SHA256,
            'template_kind': 'gguf-native-jinja',
        }
        if not isinstance(profile, dict) or any(profile.get(k) != v for k, v in expected.items()):
            raise InputPolicyError('input_count_identity_mismatch', {'reason': 'native profile identity'})
        raise InputPolicyError('input_count_uncalibrated', {
            'policy_version': VERSION,
            'reason': 'independent native full-render/token calibration not established',
            'generation_allowed': False,
            'exact_input_tokens': None,
        })
    except InputPolicyError:
        raise
    except Exception as exc:
        raise InputPolicyError('input_count_unknown', {'reason': type(exc).__name__}) from exc


def observe_tokens(prompt: str, *, runner_endpoint: str, transport) -> dict:
    """Observe an already identity-verified runner; never returns an admission.

    This utility does not establish identity or independent rendering by itself.
    Callers must preserve the full service render and runtime identity separately.
    """
    if not isinstance(prompt, str):
        raise InputPolicyError('input_count_unsupported', {'reason': 'non-text native prompt'})
    tokens = transport('/tokenize', {'content': prompt}, endpoint=runner_endpoint)['tokens']
    generation = transport('/tokenize', {'content': prompt, 'add_special': True,
                            'parse_special': True}, endpoint=runner_endpoint)['tokens']
    if (not isinstance(tokens, list) or not all(type(t) is int and t >= 0 for t in tokens)
            or not isinstance(generation, list)
            or not all(type(t) is int and t >= 0 for t in generation) or tokens != generation):
        raise InputPolicyError('input_count_unknown', {'reason': 'native special-token path mismatch'})
    return {'status': 'input_count_uncalibrated', 'generation_allowed': False,
            'observed_input_tokens': len(tokens), 'token_ids': tokens,
            'render_sha256': go_budget.sha(prompt.encode()),
            'scope': 'same-runner observation; not independent calibration or generation input attestation'}
