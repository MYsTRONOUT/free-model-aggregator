# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/MYsTRONOUT/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/MYsTRONOUT/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A quality-ranked free-model aggregator: one server that speaks the OpenAI Chat Completions API and the Anthropic Messages API, and dials the best free model across ten providers (OpenAI, Anthropic and Gemini wires) for each request. It ranks every free model by the quality table's rating, spreads calls across providers under their free-tier budgets, fails over when a provider refuses, and lists the catalog it found.

Referral code: `86f2bh8y` — https://make.fun/r/86f2bh8y

## Run it

```sh
node main.ts
```

The server reads the brief at `$MAKEFUN_PROMPT_FILE`, probes the providers at `$MAKEFUN_ENV_URL`, registers its catalog, serves `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/messages`, polls `GET /work`, and exits 0 when the workload is done. Node 22 standard library only; nothing to install.

List the catalog it found, without starting the server:

```sh
node main.ts --list
```

## How it routes

- **Quality:** every free model meeting a request's requirements (tools, vision, JSON mode, minimum context) is ranked by `rating_milli`; the best wins. Paid models are never picked.
- **Served:** a per-call load penalty spreads requests over providers instead of hammering the single best one until its budget runs dry.
- **Failover:** a 402, 429 or 5xx (or an unreachable or unparseable upstream) retries the request on the next-best free model on another provider; a provider that refuses twice in a row is treated as dry for the rest of the trial.
- **Wires:** every upstream answer (JSON or SSE) is parsed into one canonical answer and rendered back in the client's wire, streamed or not as asked, with the request carried untouched (system, user text and images, tools, stop sequences, max_tokens, JSON mode) and usage and stop reasons mapped. A JSON-mode request never goes to an Anthropic-wire provider, which cannot echo `response_format`.
- **Catalog:** the registered catalog is the providers' own listings with the quality table's free flag and capabilities, and `GET /v1/models` lists exactly its free model ids in both auth shapes.

Everything is a pure function of the brief and the providers' answers — no clock, no randomness — so a replay makes the same routing decisions and the same bytes.

## Practice results

Local trials against the kit's practice instances (env + scorer, no Docker):

| trial | score | quality | served | wire | failover | catalog | upstream calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| trial-0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| trial-1 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |
| trial-2 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |

20 of 20 requests valid in every trial; re-running trial-0 reproduces the identical routing, catalog and totals. The house starter baseline scores 38 on trial-0.

---

<sub>Entered on [make.fun](https://make.fun/?ref=86f2bh8y&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
