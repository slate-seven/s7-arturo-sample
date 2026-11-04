"""Graph access token for the Jon Batiste pack from AgentCore Identity.

Runtime only. AgentCore Identity builds a client assertion for the S7 Arturo app
registration, KMS signs it (the private key never leaves KMS), Entra verifies it
against the uploaded certificate and returns a Graph token scoped to the folders
that app has been granted. No secret is stored anywhere in this path.

Local runs never come through here. On a laptop the workload access token is
absent, on_runtime() is False, and the caller keeps its existing token source.
"""

from __future__ import annotations

import os

# Private Key JWT provider: KMS alias/s7-arturo-entra signs the client assertion, Entra holds the
# matching certificate (expires 2028-09-20; scripts/make_entra_cert.py rebuilds it). The federated
# method (AWS_IAM_ID_TOKEN_JWT) was tried first and retired 9/20: AgentCore sends no client_id on
# its M2M token request and Entra requires one.
PROVIDER_NAME = os.environ.get("JB_GRAPH_PROVIDER", "s7-arturo-graph-pkjwt")
# Optional second provider: an app registration in the client's own Entra tenant
# (2026-10-05). The CDK sets JB_CLIENT_GRAPH_PROVIDER from
# s7-targets.json clientGraphProvider for the target; absent means no client tenant.
CLIENT_PROVIDER_NAME = os.environ.get("JB_CLIENT_GRAPH_PROVIDER") or None
GRAPH_SCOPES = ["https://graph.microsoft.com/.default"]


def on_runtime() -> bool:
    """True when AgentCore Runtime has placed a workload access token in context."""
    try:
        from bedrock_agentcore.runtime import BedrockAgentCoreContext
    except ImportError:
        return False
    return BedrockAgentCoreContext.get_workload_access_token() is not None


def _fetch_token(provider_name: str) -> str:
    from bedrock_agentcore.identity.auth import requires_access_token

    @requires_access_token(provider_name=provider_name, scopes=GRAPH_SCOPES, auth_flow="M2M")
    def _inner(*, access_token: str) -> str:
        return access_token

    return _inner()


def cached_provider(provider_name: str = PROVIDER_NAME):
    """One token per run. Graph tokens outlive a run by a wide margin, so a second
    fetch inside the same run would only add latency and log noise. One cache per
    provider: the S7 app and a client-tenant app return different tokens."""
    cache: dict[str, str] = {}

    def provider() -> str:
        if "token" not in cache:
            cache["token"] = _fetch_token(provider_name)
        return cache["token"]

    return provider
