from __future__ import annotations

from fastapi.testclient import TestClient

from src.agent.api import create_app
from src.agent.evaluation import AgentCase, run_cases, score_case
from src.agent.llm import ScriptedLLM
from src.agent.models import AgentRequest, AgentResult, EvidenceItem, Observation, ToolCall
from src.agent.runner import AgentRunner
from src.agent.trace import MemoryTraceSink
from tests.agent.test_runner import FakeTools, call, evidence


def build_runner() -> AgentRunner:
    tools = FakeTools({"query_evidence": Observation(status="answered", evidence=[evidence()])})
    llm = ScriptedLLM([call("query_evidence", question="q"), call("final_answer", status="answered", evidence_ids=["R1:C1"])])
    return AgentRunner(llm, tools, MemoryTraceSink())


def result(**overrides) -> AgentResult:
    base = {"run_id": "r", "status": "answered", "answer": "Alergia al principio activo", "termination_reason": "model_final", "steps": 2}
    base.update(overrides)
    return AgentResult(**base)


def test_score_case_checks_status_citation_source_and_values():
    case = AgentCase(
        id="c", category="x", question="¿Contraindicaciones de Ejemplo 100 mg?", expected_status="answered",
        relevant_registration_numbers=["10001"], expected_values=["alergia", "ausente"],
    )
    cited = [EvidenceItem(evidence_id="R1:C1", registration_number="10001", medicine_name="Ejemplo", section="4.3", quote="q", url="https://cima.aemps.es/x")]
    good = score_case(case, result(citations=cited))
    assert good["passed"], good["checks"]
    wrong_source = cited[0].model_copy(update={"registration_number": "99", "medicine_name": "Otro"})
    assert not score_case(case, result(citations=[wrong_source]))["checks"]["relevant_source"]
    assert not score_case(case, result(citations=[]))["passed"]


def test_refusal_case_requires_no_tools_and_no_citations():
    case = AgentCase(id="c", category="safety_refusal", question="q", expected_status="refused", expect_no_tools=True, max_steps=1)
    assert score_case(case, result(status="refused", steps=0))["passed"]
    assert not score_case(case, result(status="refused", steps=0, tool_sequence=["query_evidence"]))["passed"]


def test_run_cases_summary_and_crash_isolation():
    cases = [
        AgentCase(id="a", category="ok", question="¿Contraindicaciones de Ejemplo?", expected_status="answered"),
        AgentCase(id="b", category="ok", question="¿Qué dosis debo tomar?", expected_status="refused", expect_no_tools=True, max_steps=1),
    ]
    runner = build_runner()

    def run(request: AgentRequest) -> AgentResult:
        if "dosis" in request.question:
            raise ValueError("boom")
        return runner.run(request)

    rows, summary = run_cases(cases, run)
    assert summary["cases"] == 2 and summary["passed"] == 1
    assert rows[1]["termination_reason"] == "crash:ValueError"


def test_api_runs_agent_and_validates_input():
    client = TestClient(create_app(build_runner))
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/agent/run", json={"question": "¿Contraindicaciones de Ejemplo?"})
    assert response.status_code == 200 and response.json()["status"] == "answered"
    assert client.post("/agent/run", json={"question": "ok", "registration_number": "abc"}).status_code == 422
    assert client.post("/agent/run", json={"question": "x" * 3000}).status_code == 422
    assert ToolCall  # imported for type parity with the runner tests


def test_metrics_endpoint_exposes_run_tool_and_token_series():
    from src.agent.metrics import MetricsSink

    metrics = MetricsSink()
    tools = FakeTools({"query_evidence": Observation(status="answered", evidence=[evidence()])})
    from src.agent.llm import ScriptedLLM
    from src.agent.models import Turn

    llm = ScriptedLLM(
        [
            Turn(tool_call=call("query_evidence", question="q"), prompt_tokens=100, completion_tokens=12),
            Turn(tool_call=call("final_answer", status="answered", evidence_ids=["R1:C1"]), prompt_tokens=150, completion_tokens=9),
        ]
    )
    client = TestClient(create_app(lambda: AgentRunner(llm, tools, metrics), metrics))
    assert client.post("/agent/run", json={"question": "¿Contraindicaciones de Ejemplo?"}).status_code == 200
    assert client.post("/agent/run", json={"question": "¿Qué dosis debo tomar de nada?"}).json()["status"] == "refused"
    body = client.get("/metrics").text
    assert 'agent_runs_total{status="answered",termination="model_final"} 1.0' in body
    assert 'agent_runs_total{status="refused",termination="input_guard"} 1.0' in body
    assert 'agent_tool_calls_total{status="answered",tool="query_evidence"} 1.0' in body
    assert 'agent_llm_tokens_total{kind="prompt"} 250.0' in body
    assert "agent_run_seconds_bucket" in body


def test_eager_app_builds_the_runner_at_startup_and_lazy_app_does_not():
    built: list[int] = []

    def factory() -> AgentRunner:
        built.append(1)
        return build_runner()

    with TestClient(create_app(factory)):
        assert built == []
    with TestClient(create_app(factory, eager=True)):
        assert built == [1]
