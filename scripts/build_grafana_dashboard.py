"""Generate deploy/grafana/dashboards/pharma-assistant.json (keeps the JSON valid and the panels aligned)."""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}


def panel(pid, title, x, y, exprs, unit="short", kind="timeseries", w=12, h=8):
    return {
        "id": pid,
        "title": title,
        "type": kind,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [
            {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": chr(65 + i)}
            for i, (expr, legend) in enumerate(exprs)
        ],
    }


def quantile(q, metric, by=""):
    group = f"sum by (le{', ' + by if by else ''})"
    return f"histogram_quantile({q}, {group} (rate({metric}_bucket[5m])))"


panels = [
    panel(1, "Agent runs per minute by status", 0, 0, [("sum by (status) (rate(agent_runs_total[1m])) * 60", "{{status}}")]),
    panel(2, "Run latency p50 / p95", 12, 0, [(quantile(0.5, "agent_run_seconds"), "p50"), (quantile(0.95, "agent_run_seconds"), "p95")], unit="s"),
    panel(3, "Average steps per run", 0, 8, [("rate(agent_run_steps_sum[5m]) / rate(agent_run_steps_count[5m])", "steps")]),
    panel(4, "Tool latency p95", 12, 8, [(quantile(0.95, "agent_tool_seconds", "tool"), "{{tool}}")], unit="s"),
    panel(5, "Model decision latency p95", 0, 16, [(quantile(0.95, "agent_llm_seconds"), "p95")], unit="s"),
    panel(6, "Agent tokens per second", 12, 16, [("sum by (kind) (rate(agent_llm_tokens_total[1m]))", "{{kind}}")]),
    panel(7, "llama-server speed (tokens/s)", 0, 24, [("llamacpp:predicted_tokens_seconds", "generation"), ("llamacpp:prompt_tokens_seconds", "prompt processing")]),
    panel(8, "llama-server queue", 12, 24, [("llamacpp:requests_processing", "processing"), ("llamacpp:requests_deferred", "deferred")]),
    panel(9, "Terminations in the last hour", 0, 32, [("sum by (termination) (increase(agent_runs_total[1h]))", "{{termination}}")], kind="bargauge", w=12),
    panel(10, "Tool errors per minute", 12, 32, [("sum by (tool) (rate(agent_tool_calls_total{status=\"error\"}[5m])) * 60", "{{tool}}"), ("rate(agent_llm_errors_total[5m]) * 60", "llm")]),
]

dashboard = {
    "uid": "pharma-assistant",
    "title": "Pharma assistant: agent and LLM serving",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"},
    "panels": panels,
}
out = Path(__file__).resolve().parents[1] / "deploy" / "grafana" / "dashboards" / "pharma-assistant.json"
out.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
print(out)
