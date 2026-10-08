# EthosoftLib

EthosoftLib is an extensible, multi-project Python SDK for Ethosoft technologies.

This repository is intentionally **not** a Mercan-only wrapper. Providers live in independent namespaces and are registered lazily, so integrations can be added without making their native dependencies mandatory for every user.

Initial provider: `ethosoftlib.mercan` (MercanRuntime native C ABI). See the upcoming package files and `docs/ARCHITECTURE.md` for details.
