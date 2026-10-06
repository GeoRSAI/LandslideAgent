import json

import pytest

from scripts.llm_service import ChatRequest, _build_agent_runner
from tests.test_agent_runtime_regressions import _call, _step, _traces
from tests.test_agent_loop_guards import _FakeRegistry, _make_agent


def test_backend_exception_invalidates_live_derived_evidence(monkeypatch, tmp_path):
    class BrokenRegistry(_FakeRegistry):
        def call_tool(self, name, args):
            self.calls.append(name)
            raise RuntimeError('seg backend disconnected')
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path, flags={'loop_hygiene': False})
    agent.registry = BrokenRegistry()
    outputs = {'seg.run': {'area_ratio': .4}, 'seg.refine': {'regions': []},
               'fuse.decision': {'has_landslide': True}, 'report.write': {'report_path': 'old.json'}}
    result = _step(agent, [_call('seg.run')], outputs)
    assert result['outputs']['seg.run'].get('error')
    assert not set(result['outputs']) & {'seg.refine', 'fuse.decision', 'report.write'}
    assert _traces(result)[0]['execution_state'] == 'failed'


def test_invalid_backend_result_invalidates_live_derived_evidence(monkeypatch, tmp_path):
    agent = _make_agent(lambda *_: {}, monkeypatch, tmp_path, flags={'loop_hygiene': False})
    agent.registry.call_tool = lambda *_: None
    result = _step(agent, [_call('seg.run')], {'seg.run': {'area_ratio': .4}, 'seg.refine': {'regions': []}})
    assert result['outputs']['seg.run'].get('error')
    assert 'seg.refine' not in result['outputs']


def test_trace_failure_cannot_resurrect_successful_message_history(tmp_path):
    img = tmp_path / 'img.png'
    img.write_bytes(b'image')
    req = ChatRequest(messages=[], agent_mode='agent', agent_trace=[
        {'tool': 'seg.run', 'execution_state': 'completed', 'input': {}, 'output': {'area_ratio': .4}},
        {'tool': 'seg.run', 'execution_state': 'failed', 'input': {}, 'output': {'error': 'offline'}},
    ])
    agent = _build_agent_runner(req=req, thresholds_path='configs/thresholds.json', image_path=str(img), report_write_required=False)
    agent.model_fn = lambda *_: (_ for _ in ()).throw(RuntimeError('stop'))
    history = [{'role': 'tool', 'name': name, 'content': json.dumps(output)} for name, output in [
        ('seg.run', {'area_ratio': .4}), ('seg.refine', {'regions': []}), ('fuse.decision', {'has_landslide': True})]]
    last = list(agent.stream(history))[-1]
    assert not set(last['outputs']) & {'seg.run', 'seg.refine', 'fuse.decision'}


@pytest.mark.parametrize('calls', ['bad', [None], [{'function': 'bad'}], [{'function': {'name': []}}]])
def test_malformed_model_calls_end_in_inspectable_fallback(monkeypatch, tmp_path, calls):
    agent = _make_agent(lambda *_: {'message': {'role': 'assistant', 'tool_calls': calls}}, monkeypatch, tmp_path)
    last = list(agent.stream([{'role': 'user', 'content': 'analyze'}]))[-1]
    assert last['type'] == 'fallback'
    assert 'model' in last['reason']
    assert last['turns_used'] == 1
    assert not agent.registry.calls


def test_geo_trace_from_old_coordinates_is_not_current_evidence(tmp_path):
    img = tmp_path / 'img.png'
    img.write_bytes(b'image')
    req = ChatRequest(messages=[], agent_mode='agent', latitude=30, longitude=104, nearby_radius=500, agent_trace=[
        {'tool': 'geo.nearby', 'execution_state': 'completed', 'input': {'lat': 29.6, 'lon': 103, 'radius': 300},
         'output': {'count': 1, 'features': [{'type': 'road'}]}},
        {'tool': 'fuse.decision', 'execution_state': 'completed', 'input': {}, 'output': {'has_landslide': True}},
    ])
    agent = _build_agent_runner(req=req, thresholds_path='configs/thresholds.json', image_path=str(img), report_write_required=False)
    assert 'geo.nearby' not in agent._initial_outputs
    assert 'fuse.decision' not in agent._initial_outputs


def test_report_does_not_verify_old_fusion_after_new_segmentation():
    from src.pipelines.structured_report import build_structured_report, VERIFIED
    from tests.test_structured_report import full_trace, call
    trace = full_trace() + [call('seg.run', {'area_ratio': .01, 'mask_path': 'new.png'})]
    report = build_structured_report(trace)
    parts = report['fields']['presence']['parts']
    assert not any(p['status'] == VERIFIED and 'fuse.decision#' in p['source'] for p in parts)


def test_report_does_not_resurrect_description_after_backend_failure():
    from src.pipelines.structured_report import build_structured_report, UNAVAILABLE
    from tests.test_structured_report import full_trace, call
    trace = full_trace() + [call('vlm.describe', {'error': 'offline'}, state='failed')]
    assert build_structured_report(trace)['fields']['morphology']['status'] == UNAVAILABLE


def test_refusal_preserves_current_report_evidence():
    from src.pipelines.structured_report import build_structured_report, VERIFIED
    from tests.test_structured_report import full_trace, call
    trace = full_trace() + [call('vlm.describe', {'error': 'wrong image'}, state='refused')]
    assert build_structured_report(trace)['fields']['morphology']['status'] == VERIFIED


def test_foreign_image_trace_cannot_seed_unscoped_downstream_results(tmp_path):
    img = tmp_path / 'current.png'
    img.write_bytes(b'image')
    req = ChatRequest(messages=[], agent_mode='agent', agent_trace=[
        {'tool': 'tiff.info', 'execution_state': 'completed', 'input': {}, 'output': {'image_path': '/old.png'}},
        {'tool': 'seg.run', 'execution_state': 'completed', 'input': {}, 'output': {'area_ratio': .4}},
        {'tool': 'fuse.decision', 'execution_state': 'completed', 'input': {}, 'output': {'has_landslide': True}},
    ])
    agent = _build_agent_runner(req=req, thresholds_path='configs/thresholds.json', image_path=str(img), report_write_required=False)
    assert agent._initial_outputs == {}


def test_changed_geo_trace_cannot_be_verified_by_report(tmp_path):
    from scripts.llm_service import ChatMessage, _initial_agent_trace
    from src.pipelines.structured_report import build_structured_report, VERIFIED
    from tests.test_structured_report import full_trace
    img = tmp_path / 'current.png'
    img.write_bytes(b'image')
    trace = full_trace()
    trace[-2]['input'] = {'lat': 29.6, 'lon': 103, 'radius': 300}
    req = ChatRequest(messages=[ChatMessage(role='user', content=[{'type': 'image', 'image_path': str(img)}])],
                      agent_mode='agent', latitude=30, longitude=104, nearby_radius=500, agent_trace=trace)
    report = build_structured_report(_initial_agent_trace(req, []), latitude=30, longitude=104)
    assert not any(p['status'] == VERIFIED and 'geo.nearby#' in p['source'] for p in report['fields']['impact']['parts'])


def test_refused_foreign_reference_does_not_invalidate_resume_ledger(tmp_path):
    img = tmp_path / 'current.png'
    img.write_bytes(b'image')
    req = ChatRequest(messages=[], agent_mode='agent', latitude=30, longitude=104, nearby_radius=500, agent_trace=[
        {'tool': 'tiff.info', 'execution_state': 'completed', 'input': {'image_path': str(img)},
         'output': {'image_path': str(img), 'width': 512, 'height': 512}},
        {'tool': 'tiff.info', 'execution_state': 'refused', 'input': {'image_path': '/other.png'}, 'output': {'error': 'wrong image'}},
        {'tool': 'geo.nearby', 'execution_state': 'completed', 'input': {'lat': 30, 'lon': 104, 'radius': 500},
         'output': {'count': 0, 'features': []}},
        {'tool': 'geo.nearby', 'execution_state': 'refused', 'input': {'lat': 29, 'lon': 103, 'radius': 300}, 'output': {'error': 'wrong coordinates'}},
    ])
    agent = _build_agent_runner(req=req, thresholds_path='configs/thresholds.json', image_path=str(img), report_write_required=False)
    assert set(agent._initial_outputs) == {'tiff.info', 'geo.nearby'}


@pytest.mark.parametrize('metadata,state', [({'verification': {'status': 'refused'}}, 'refused'), ({'not_executed': True}, 'deferred')])
def test_legacy_history_preserves_unexecuted_state(metadata, state):
    from scripts.llm_service import _history_tool_trace
    from src.pipelines.structured_report import build_structured_report, VERIFIED
    from tests.test_structured_report import DESC
    history = [
        {'role': 'tool', 'name': 'vlm.describe', 'content': json.dumps(DESC)},
        {'role': 'tool', 'name': 'vlm.describe', 'content': json.dumps({'error': 'not executed', **metadata})},
    ]
    trace = _history_tool_trace(history)
    assert trace[-1]['execution_state'] == state
    assert build_structured_report(trace)['fields']['morphology']['status'] == VERIFIED
