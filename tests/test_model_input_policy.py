from copy import deepcopy
import json
import pytest
from aircraft_agent.agent.input_policy import (VERSION, OUTPUT_RESERVE, InputPolicyError, assess_request, build_tool_view, sizes, validate_pairs)
from aircraft_agent.common.utils import hash_json
from aircraft_agent.tools.catalog import TOOL_NAME_ALIASES


def request():
    return {'model':'qwen3:4b','messages':[{'role':'system','content':'规则不可删除'}, {'role':'user','content':'任务'}], 'tools':[], 'max_tokens':OUTPUT_RESERVE}


def test_profile_projection_preserves_statistics_ids_and_source():
    result={'dataset_id':'real-id','profile':{'rows':123,'columns':1,'column_names':['x'], 'sample_rows':[{'x':42}], 'column_profiles':[{'name':'x','numeric':{'min':1,'max':42}, 'missing_rate':0,'sample_values':[42],'top_values':[42]}]},'summary':'ok'}
    original=deepcopy(result)
    text,audit=build_tool_view('profile_dataset',result); view=json.loads(text)
    assert result==original and view['result']['dataset_id']=='real-id'
    assert view['result']['profile']['column_profiles']==[{'name':'x','numeric':{'min':1,'max':42},'missing_rate':0}]
    assert audit['original_output_hash']==hash_json(result) and audit['view_hash']==hash_json(view)
    assert view['omissions'][0]['original_count']==1 and view['omissions'][0]['returned_count']==0
    assert view['omissions'][1]['returned_count']==1
    assert 'does not mean' in view['source_scope']


def test_mapping_equality_checked_and_decisions_preserved():
    row={'field':'actual','status':'ambiguous','reason':'requires confirmation','confidence':.2,'tokens':['x']}
    result={'fields':[row], 'mappings':[{'status':'missing','rule_field':'r','selected_field':None}], 'json_path':'artifact.json'}
    result['rule_fields']=deepcopy(result['mappings'])
    view=json.loads(build_tool_view('review_schema_mapping',result)[0])
    assert view['result']['fields'][0]['reason']==row['reason']
    assert view['result']['mappings']==result['mappings'] and 'rule_fields' not in view['result']
    assert view['omissions'][-1]['equivalent_to']=='$.mappings'
    result['rule_fields']=[{'different':True}]
    assert json.loads(build_tool_view('review_schema_mapping',result)[0])['result']['rule_fields']==result['rule_fields']


def test_errors_unknown_fields_and_aliases():
    error={'error':'dataset_id is invalid','summary':'失败','details':{'required':'none'}}
    assert json.loads(build_tool_view('unknown_tool',error)[0])['result']==error
    for alias,canonical in TOOL_NAME_ALIASES.items():
        text,audit=build_tool_view(alias,{'summary':'ok','future_critical_field':['id']})
        assert audit['canonical_tool']==canonical and json.loads(text)['result']['future_critical_field']==['id']


def test_oversized_necessary_result_fails_without_cutting_string():
    with pytest.raises(InputPolicyError,match='tool_view_budget_exceeded'):
        build_tool_view('validate_analysis',{'error':'错误'*20000})


def test_budget_exact_boundary_in_admission_units_and_no_mutation():
    req=request(); original=deepcopy(req); measured=assess_request(req)
    total=measured['total_budget_units']
    assert assess_request(req,context=total)['status']=='conservative_admit'
    assert assess_request(req,context=total-1)['status']=='input_budget_exceeded'
    assert req==original and measured['exact_input_tokens'] is None
    assert measured['request_size']['bytes']>measured['request_size']['characters']
    req['tools']=[{'description':'x'*40000}]
    assert assess_request(req)['status']=='input_budget_exceeded'


def paired():
    return request()['messages']+[{'role':'assistant','content':'reasoning', 'tool_calls':[{'id':'a'}, {'id':'b'}]}, {'role':'tool','tool_call_id':'b','content':'{}'}, {'role':'tool','tool_call_id':'a','content':'{}'}]


def test_multicall_pairs_reordered_outputs_valid():
    validate_pairs(paired())


@pytest.mark.parametrize('change',['missing','orphan','duplicate','interrupted','reused','prefix'])
def test_bad_pairs_rejected(change):
    messages=paired()
    if change=='missing': messages.pop()
    elif change=='orphan': messages[-1]['tool_call_id']='z'
    elif change=='duplicate': messages[-1]['tool_call_id']='b'
    elif change=='interrupted': messages.insert(4,{'role':'user','content':'x'})
    elif change=='reused': messages += [{'role':'assistant','tool_calls':[{'id':'a'}]}, {'role':'tool','tool_call_id':'a'}]
    elif change=='prefix': messages.pop(0)
    with pytest.raises(InputPolicyError,match='pairing_invalid'): validate_pairs(messages)


def test_adapter_rejects_before_network():
    from aircraft_agent.agent.model_providers import ChatCompletionsAdapter
    from types import SimpleNamespace
    calls=[]
    adapter=ChatCompletionsAdapter(api_key='x',model='qwen3:4b',base_url=None,client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:calls.append(kw)))))
    def guard(req):
        verdict=assess_request(req)
        if verdict['status']!='conservative_admit': raise InputPolicyError(verdict['status'],verdict)
    adapter.request_guard=guard
    with pytest.raises(InputPolicyError,match='budget_exceeded'): adapter.start(system='x'*40000,user='task',tools=[])
    assert calls==[]
