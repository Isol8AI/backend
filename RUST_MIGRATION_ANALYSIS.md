# Rust Migration Analysis

## Current System Overview

| Component | Files | LOC | Complexity |
|-----------|-------|-----|------------|
| Core infra (auth, config, db) | 3 | ~400 | Low |
| Enclave integration | 8 | ~2,250 | High |
| Services (business logic) | 12 | ~3,800 | High |
| SQLAlchemy models | 12 | ~1,825 | Medium |
| API routers (92 endpoints) | 10 | ~4,100 | High |
| Pydantic schemas | 8 | ~975 | Low |
| Crypto primitives | 2 | ~300 | High |
| Standalone enclave code | 15 | ~5,150 | High |
| Tests | 75 | ~15,600 | Medium |
| **Total** | **~145** | **~34,000+** | |

## Recommendation: Do NOT Rewrite the Entire Backend

### Why a Full Rewrite Is Not Worth It

1. **~34K LOC to rewrite** — Rust equivalent would be ~50-60K lines due to explicit error handling, lifetimes, and type annotations.
2. **92 API endpoints** to re-implement in Axum/Actix-web with all schemas, middleware, and SSE streams.
3. **Ecosystem gaps**: No official Clerk Rust SDK, less mature Stripe crate, younger AWS SDK, no SQLAlchemy equivalent.
4. **~15,600 lines of tests** to rebuild without pytest-asyncio, factory-boy, respx, or schemathesis equivalents.
5. **Security risk**: Rewriting crypto code (vsock, HKDF, X25519, AES-256-GCM, KMS envelope) introduces regression risk.
6. **Marginal performance gain**: The bottleneck is Bedrock API calls (I/O-bound), not Python CPU overhead.

### What to Do Instead

**Replace the Node.js OpenClaw bridge with the Rust OpenClaw binary, keep the Python backend.**

- Affects ~500 lines of Python (`agent_bridge.py` + `agent_runner.py`)
- Compiles Rust OpenClaw to a binary that runs inside the Nitro Enclave
- Gets the performance benefit where it matters (inside the constrained enclave)

### Incremental Migration Path (If Desired)

1. **Now**: Swap Node.js OpenClaw for Rust OpenClaw binary inside the enclave
2. **Later**: Consider rewriting `bedrock_server.py` (1,515 lines) in Rust — the enclave server benefits most from Rust's memory safety and performance in a constrained environment
3. **Keep in Python**: The FastAPI API layer, where ecosystem advantages (Clerk, Stripe, SQLAlchemy, rapid iteration) outweigh Rust's benefits

### Ecosystem Mapping (For Reference)

| Python | Rust Equivalent | Maturity |
|--------|----------------|----------|
| FastAPI | Axum / Actix-web | Mature |
| SQLAlchemy | SQLx / Diesel | Good (no ORM richness) |
| Pydantic | serde + validator | Mature |
| Clerk SDK | Manual JWKS | No official SDK |
| Stripe SDK | async-stripe | Less mature |
| boto3 | aws-sdk-rust | Usable, younger |
| httpx | reqwest | Mature |
| python-jose | jsonwebtoken | Mature |
| pytest | cargo test | Mature (different paradigm) |
| factory-boy | No equivalent | Must build custom |

### Conclusion

A full migration is a multi-month effort that rewrites a working, well-tested, production system for marginal performance gains on I/O-bound code. The targeted approach — Rust OpenClaw inside the enclave, Python everything else — delivers ~80% of the benefit at ~5% of the cost.
