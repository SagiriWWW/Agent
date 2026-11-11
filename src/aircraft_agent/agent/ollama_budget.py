"""Opt-in, identity-bound Ollama v0.33.3 text budget. No tokenizer surrogate.

Local rendering uses the upstream Go packages; each request must byte-match the
running server's non-truncating render before the same runner tokenizes it.
Internal debug/tokenize interfaces are intentionally limited to this experiment.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request
from typing import Any
from aircraft_agent.agent.input_policy import CONTEXT, OUTPUT_RESERVE, REQUEST_SETTINGS, InputPolicyError, validate_pairs
from aircraft_agent.common.utils import hash_json

VERSION = 'tool-view-v1-ollama-render-budget-v1'
COMPARISON_VERSION = 'tool-view-v1-ollama-render-budget-session20-v1'
CANDIDATE_MODEL = 'qwen3:4b-instruct-2507-q4_K_M'
CANDIDATE_DIGEST = '0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0'
VIEW_RESOURCE_VERSION = 'view-resource-64k-v1'
MAX_VIEW_RESOURCE_BYTES = 65536
MAX_REQUEST_RESOURCE_BYTES = 4 * 1024 * 1024
DIGEST = '359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7'
UPSTREAM = 'http://127.0.0.1:11435'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def post(path: str, payload: Any = None, *, endpoint: str = UPSTREAM) -> Any:
    req = urllib.request.Request(endpoint + path, data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def check_scope(request: dict, *, model: str = 'qwen3:4b') -> None:
    if not isinstance(request, dict) or not isinstance(request.get('messages'), list):
        raise InputPolicyError('input_count_unknown', {'reason': 'invalid request/messages'})
    allowed = {'model', 'messages', 'tools', 'stream', *REQUEST_SETTINGS}
    if model not in ('qwen3:4b', CANDIDATE_MODEL) or set(request) - allowed or request.get('model') != model or any(request.get(k) != v for k,v in REQUEST_SETTINGS.items()):
        raise InputPolicyError('input_count_unsupported', {'reason': 'request identity/settings/fields outside frozen scope'})
    messages = request['messages']
    for i,m in enumerate(messages):
        if not isinstance(m, dict) or set(m)-{'role','content','tool_calls','tool_call_id'} or not isinstance(m.get('content'), (str,type(None))):
            raise InputPolicyError('input_count_unsupported', {'reason': 'unknown or multimodal message'})
        if m.get('role') not in ('system','user','assistant','tool') or (i>1 and m.get('role') in ('system','user')):
            raise InputPolicyError('input_count_unsupported', {'reason': 'only initial system/user followed by assistant/tool supported'})
        if i and m.get('role') == messages[i-1].get('role') == 'assistant':
            raise InputPolicyError('input_count_unsupported', {'reason': 'consecutive assistant merge would lose call metadata'})
        for c in m.get('tool_calls',[]):
            if not isinstance(c,dict) or set(c)-{'id','type','function'} or c.get('type')!='function' or not isinstance(c.get('function'),dict):
                raise InputPolicyError('input_count_unsupported', {'reason':'unknown call shape'})
            f=c['function']
            if set(f)!={'name','arguments'} or not isinstance(f['name'],str) or not isinstance(f['arguments'],str):
                raise InputPolicyError('input_count_unsupported', {'reason':'unknown function shape'})
            try:
                if not isinstance(json.loads(f['arguments']),dict): raise ValueError()
            except (ValueError,TypeError):
                raise InputPolicyError('input_count_unknown', {'reason':'invalid function arguments'}) from None
    validate_pairs(messages)
    from aircraft_agent.agent.model_providers import to_chat_completions_tools
    from aircraft_agent.tools.catalog import OPENAI_TOOL_SCHEMAS
    if request.get('tools') != to_chat_completions_tools(OPENAI_TOOL_SCHEMAS):
        raise InputPolicyError('input_count_unsupported', {'reason':'tool schema identity mismatch'})


def check_identity(profile: dict, *, model: str = 'qwen3:4b', digest: str = DIGEST) -> tuple[dict, str]:
    if (model, digest) not in (('qwen3:4b', DIGEST), (CANDIDATE_MODEL, CANDIDATE_DIGEST)):
        raise InputPolicyError('input_count_identity_mismatch', {'reason':'unregistered counting identity'})
    if post('/api/version') != {'version':'0.33.3'}:
        raise InputPolicyError('input_count_identity_mismatch', {'reason':'Ollama version'})
    tags=post('/api/tags')['models']
    if not any(m['name']==model and m['digest']==digest for m in tags):
        raise InputPolicyError('input_count_identity_mismatch', {'reason':'model digest'})
    show=post('/api/show',{'model':model,'verbose':True})
    if sha(show['template'].encode()) != profile['template_sha256'] or hash_json(show['model_info']) != profile['model_info_hash'] or show.get('parameters') != profile['parameters']:
        raise InputPolicyError('input_count_identity_mismatch', {'reason':'template/tokenizer/parameters'})
    # Debug rendering loads the model if needed, without generating a token.
    post('/api/chat',{'model':model,'messages':[], 'stream':False,'keep_alive':'30m'})
    models=post('/api/ps')['models']
    if len(models)!=1 or models[0]['digest']!=digest or models[0]['context_length']!=CONTEXT:
        raise InputPolicyError('input_count_identity_mismatch', {'reason':'actual loaded context/model'})
    processes=subprocess.check_output(['ps','-ax','-o','pid=,ppid=,command='],text=True)
    matches=[]
    for line in processes.splitlines():
        fields=line.strip().split(None,2)
        if len(fields)<3: continue
        pid,ppid,cmd=fields
        if int(ppid)!=profile['service_pid'] or not cmd.startswith(profile['runner_binary']+' --model '): continue
        if '--model '+profile['model_blob']+' ' not in cmd or ' -c 32768 ' not in cmd or ' -np 1 ' not in cmd: continue
        port=re.search(r' --port (\d+) ',cmd)
        if port: matches.append((int(pid),port.group(1)))
    if len(matches)!=1 or sha(Path(profile['runner_binary']).read_bytes())!=profile['runner_sha256']:
        raise InputPolicyError('input_count_identity_mismatch', {'reason':'runner identity'})
    return show, 'http://127.0.0.1:'+matches[0][1]


def decide(count: int, *, context: int = CONTEXT) -> str:
    if type(count) is not int or count<0 or context!=CONTEXT:
        raise InputPolicyError('input_count_unknown', {'reason':'invalid count/context'})
    return 'calibrated_admit' if count + OUTPUT_RESERVE <= context else 'input_budget_exceeded'


def assess_request(request: dict, *, profile: dict | None = None, artifact_dir: Path | None = None,
                   policy_version: str = VERSION) -> dict:
    from aircraft_agent.agent import ollama_native_budget
    if policy_version == ollama_native_budget.VERSION:
        return ollama_native_budget.assess_request(request, profile=profile)
    try:
        if policy_version not in (VERSION, COMPARISON_VERSION):
            raise InputPolicyError('input_count_identity_mismatch', {'reason':'unknown counting policy'})
        model = request.get('model') if policy_version == COMPARISON_VERSION and isinstance(request, dict) else 'qwen3:4b'
        check_scope(request, model=model)
        wire=json.dumps(request,ensure_ascii=False).encode()
        if len(wire)>MAX_REQUEST_RESOURCE_BYTES:
            raise InputPolicyError('input_request_resource_exceeded', {'bytes':len(wire),'limit_bytes':MAX_REQUEST_RESOURCE_BYTES})
        if profile is None:
            profile=json.loads(Path(os.environ['AGENT_MVP_OLLAMA_BUDGET_PROFILE']).read_text())
        digest = CANDIDATE_DIGEST if model == CANDIDATE_MODEL else DIGEST
        if (profile.get('policy_version') != policy_version or profile.get('model_digest') != digest
                or (policy_version == COMPARISON_VERSION and profile.get('model_name') != model)):
            raise InputPolicyError('input_count_identity_mismatch', {'reason':'profile version/model'})
        helper=Path(profile['renderer_binary'])
        if sha(helper.read_bytes())!=profile['renderer_sha256']:
            raise InputPolicyError('input_count_identity_mismatch', {'reason':'renderer binary'})
        show,runner=check_identity(profile, model=model, digest=digest)
        local=json.loads(subprocess.run([str(helper)],input=json.dumps({'request':request,'template':show['template']},ensure_ascii=False),text=True,capture_output=True,check=True,timeout=30).stdout)
        prompt=local['rendered']
        native=local['native_request']
        service=post('/api/chat',native)
        actual=service['_debug_info']['rendered_template']
        if actual!=prompt or service['_debug_info'].get('image_count', 0)!=0:
            raise InputPolicyError('input_count_render_mismatch', {'local_sha256':sha(prompt.encode()),'server_sha256':sha(actual.encode())})
        # Match both the chatPrompt admission tokenizer and completion tokenizer.
        tokens=post('/tokenize',{'content':prompt},endpoint=runner)['tokens']
        completion_tokens=post('/tokenize',{'content':prompt,'add_special':True},endpoint=runner)['tokens']
        if tokens!=completion_tokens or not all(type(t) is int for t in tokens):
            raise InputPolicyError('input_count_unknown', {'reason':'special-token path mismatch'})
        status=decide(len(tokens))
        # Normal service rendering is checked too, but never used as the full count.
        native['truncate']=True
        normal=post('/api/chat',native)['_debug_info']['rendered_template']
        if status=='calibrated_admit' and normal!=prompt:
            raise InputPolicyError('input_count_render_mismatch', {'reason':'normal service would truncate'})
        result={'policy_version':policy_version,'status':status,'request_hash':hash_json(request),'request_resource_bytes':len(wire),
                'view_resource_version':VIEW_RESOURCE_VERSION,'view_resource_limit_bytes':MAX_VIEW_RESOURCE_BYTES,
                'context_tokens':CONTEXT,'output_reserve_tokens':OUTPUT_RESERVE,'exact_input_tokens':len(tokens),
                'total_budget_tokens':len(tokens)+OUTPUT_RESERVE,'local_render_sha256':sha(prompt.encode()),
                'server_render_sha256':sha(actual.encode()),'rendered_bytes':len(prompt.encode()),'token_ids_sha256':hash_json(tokens),
                'normal_render_matches_full':normal==prompt,'local_service_render_difference_bytes':0,
                'count_method':'upstream v0.33.3 Go conversion/template byte-matched to non-truncating service render; actual identified runner tokenizer, both special-token paths',
                'scope':'frozen text-only schema and one initial user; exact preflight count, not per-generation input attestation; no empirical universal bound',
                'history_removed':False,'profile_hash':hash_json(profile),'runner_endpoint':runner}
        if artifact_dir is None and os.environ.get('AGENT_MVP_BUDGET_ARTIFACT_DIR'):
            artifact_dir=Path(os.environ['AGENT_MVP_BUDGET_ARTIFACT_DIR'])/hash_json(request)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True,exist_ok=False)
            for name,data in [('request.json',request),('local.json',local),('server.json',service),('tokens.json',tokens),('assessment.json',result)]:
                (artifact_dir/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        return result
    except InputPolicyError:
        raise
    except Exception as exc:
        raise InputPolicyError('input_count_unknown', {'reason':type(exc).__name__,'detail':str(exc)[:1000]}) from exc
