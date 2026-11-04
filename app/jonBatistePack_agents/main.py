"""Jon Batiste pack: AgentCore Runtime entrypoint.

One runtime, six agents. The payload names the agent with an "agent" key.
An unknown or missing key is an error. There is no default agent.


Payload shape:
    {"agent": "arturo", "mode": "sync" | "start" | "status", "input": {...agent-specific fields...}}

Modes (added 2026-09-23, see jobs.py):
    sync    default; run to completion and return the result in the reply. Laptop runs, tests,
            small folders. Subject to the 15-minute synchronous cap on AgentCore.
    start   list inputs in the foreground, run the rest in the background, return in seconds.
            Result lands in S3 (JB_RESULTS_BUCKET). For anything that may run long.
    status  read the run's status.json from S3 and return it; when the state is done the
            reply names the response keys to fetch.
Agents that support start/status expose prepare() and work(); the others are sync only.

Return shape (sync):
    {"agent": "arturo", "output": {...}, "usage": {...}}
Return shape (start / status): the status document, see jobs.py.
"""

from bedrock_agentcore.runtime import BedrockAgentCoreApp

import jobs
from agents import arturo

app = BedrockAgentCoreApp()

# Each module exposes run(payload_input: dict) -> dict with keys "output" and "usage".
AGENTS = {
    "arturo": arturo.run,
}

# Agents that can run in the background: (prepare, work). See jobs.py.
BACKGROUND = {
    "arturo": (arturo.prepare, arturo.work),
}

MODES = ("sync", "start", "status")


@app.entrypoint
def invoke(payload):
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    name = payload.get("agent")
    if name is None:
        raise ValueError("payload is missing the 'agent' key")
    if name not in AGENTS:
        raise ValueError(f"unknown agent '{name}'; valid values: {sorted(AGENTS)}")
    if "input" not in payload:
        raise ValueError("payload is missing the 'input' key")
    if not isinstance(payload["input"], dict):
        raise ValueError("'input' must be a JSON object")
    mode = payload.get("mode", "sync")
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; valid values: {list(MODES)}")
    inp = payload["input"]

    if mode == "sync":
        result = AGENTS[name](inp)
        return {"agent": name, "output": result["output"], "usage": result["usage"]}

    if name not in BACKGROUND:
        raise ValueError(f"agent '{name}' does not support mode '{mode}'")
    for k in ("client_id", "run_id"):
        if not inp.get(k):
            raise ValueError(f"mode '{mode}' requires input.{k}")
    if mode == "start":
        prepare, work = BACKGROUND[name]
        return jobs.start(app, name, inp, prepare, work)
    return jobs.status(name, inp)


if __name__ == "__main__":
    app.run()
