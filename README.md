# VRTX

**A distributed GPU cloud with its own LLM inference engine.**

VRTX is three systems that work as one:

| Part | What it is | Real-world equivalent |
|---|---|---|
| **VRTX Engine** | Serves open-source LLMs (Qwen, Llama, Mistral) as fast as possible on a GPU | vLLM, NVIDIA TensorRT-LLM, SGLang |
| **VRTX Cloud** | Deploys, schedules, routes and autoscales models and apps across machines | AWS Bedrock, Heroku, Vercel, Kubernetes |
| **VRTX Ops** | An AI that reads crash logs and metrics and explains what broke, running on VRTX Engine itself | Datadog, PagerDuty, AWS DevOps Guru |

Owner: Jacob Bhatt ([@JacobBhatt12](https://github.com/JacobBhatt12))
Started: October 2026
Target: ~9 months
Budget: $0

---

## Table of contents

1. [The pitch](#1-the-pitch)
2. [What we are using](#2-what-we-are-using-tech-stack)
3. [System wireframe](#3-system-wireframe)
4. [Request lifecycle](#4-request-lifecycle)
5. [VRTX Engine in depth](#5-vrtx-engine-in-depth)
6. [VRTX Cloud in depth](#6-vrtx-cloud-in-depth)
7. [VRTX Ops in depth](#7-vrtx-ops-in-depth)
8. [API design](#8-api-design)
9. [Dashboard wireframes](#9-dashboard-wireframes)
10. [CLI wireframe](#10-cli-wireframe)
11. [Repo structure](#11-repo-structure)
12. [Build plan, month by month](#12-build-plan-month-by-month)
13. [Free compute plan](#13-free-compute-plan)
14. [Benchmarking plan](#14-benchmarking-plan)
15. [The math cheat sheet](#15-the-math-cheat-sheet)
16. [Risks and how we handle them](#16-risks-and-how-we-handle-them)
17. [What "done" looks like](#17-what-done-looks-like)
18. [Getting started today](#18-getting-started-today)
19. [Reading list](#19-reading-list)

---

## 1. The pitch

Every AI company has the same expensive problem: GPUs cost a fortune and most of that power gets wasted. Generating a single token means reading every weight of the model from GPU memory, so a GPU serving one user sits mostly idle. The companies that win (NVIDIA, AWS, OpenAI, Together AI) are the ones that squeeze the most users onto each GPU and scale them across a fleet without anything falling over.

VRTX attacks that problem from the bottom of the stack to the top:

- **The bottom:** custom GPU kernels, a paged memory system for the KV cache, continuous batching, quantization and speculative decoding.
- **The middle:** a fault-tolerant control plane (Raft), a bin-packing scheduler, and a queueing-theory autoscaler.
- **The top:** an OpenAI-compatible API, a web dashboard, a CLI, and an AI ops agent that uses the platform's own model to debug the platform.

**Resume line when done:**

> Built VRTX, a distributed GPU cloud with a custom LLM inference engine (Triton kernels, paged KV cache, continuous batching, speculative decoding) serving X tokens/sec across Y concurrent users, with a Raft-replicated control plane and an Erlang-C autoscaler.

---

## 2. What we are using (tech stack)

Everything here is free and open source.

### Core languages

| Tool | Used for |
|---|---|
| **Python 3.11+** | Engine, gateway, control plane, ops (one language keeps the project moving) |
| **Triton** | Custom GPU kernels (OpenAI's GPU language, written in Python) |
| **TypeScript** | Dashboard |
| **Go** (stretch) | Optional rewrite of the worker agent or Raft for speed and range |

### AI and GPU

| Tool | Used for |
|---|---|
| **PyTorch** | Tensor math, running the model layers |
| **Hugging Face `transformers` + `safetensors`** | Downloading and loading model weights; also the "naive baseline" we benchmark against |
| **Open-weight models** | Qwen 2.5 (0.5B to 7B), Llama 3, Mistral 7B, Gemma. Free to download from Hugging Face |
| **vLLM** | Not part of VRTX. Only used as the "pro" benchmark to compare against |

### Backend and APIs

| Tool | Used for |
|---|---|
| **FastAPI** | REST API for the gateway and the control plane |
| **Uvicorn** | ASGI server that runs FastAPI |
| **Pydantic** | Request and response validation |
| **Server-Sent Events + WebSockets** | Streaming tokens to clients, live logs to the dashboard |
| **httpx** | Async HTTP between services |
| **gRPC** (stretch) | Faster internal calls between control plane and workers |

### Cloud and infrastructure

| Tool | Used for |
|---|---|
| **Docker** | Running user apps and model replicas in containers |
| **Linux namespaces and cgroups** | Isolation and resource limits (what Docker uses under the hood) |
| **Firecracker** (stretch) | MicroVMs, what AWS Lambda uses |
| **SQLite** | Each control plane node's state machine (driven by the Raft log) |
| **Prometheus text format** | Metrics endpoint on every service |

### Frontend

| Tool | Used for |
|---|---|
| **Next.js + React** | VRTX dashboard |
| **Tailwind CSS** | Styling |
| **Recharts** | Live charts (tokens/sec, latency, GPU memory) |

### Testing and tooling

| Tool | Used for |
|---|---|
| **pytest** | Unit and integration tests |
| **Locust or k6** | Load testing (simulate hundreds of users) |
| **GitHub Actions** | CI: run tests on every push |
| **Typer** | The `vrtx` command-line tool |

---

## 3. System wireframe

```
                         +-------------------------------+
                         |        USERS AND APPS         |
                         |  chat apps, curl, OpenAI SDK, |
                         |  vrtx CLI, web dashboard      |
                         +---------------+---------------+
                                         |
                                  HTTPS  |  REST + SSE + WebSocket
                                         v
+--------------------------------------------------------------------------+
|                              VRTX GATEWAY                                |
|                               (FastAPI)                                  |
|                                                                          |
|   +-------------+   +----------------+   +------------------------+     |
|   | Auth        |   | Rate limiter   |   | Router                 |     |
|   | API keys    |-->| token bucket   |-->| consistent hashing on  |     |
|   |             |   | per key        |   | prompt prefix          |     |
|   +-------------+   +----------------+   +-----------+------------+     |
|                                                       |                  |
|   +-----------------------+                           |                  |
|   | Usage metering        |                           |                  |
|   | tokens per key        |                           |                  |
|   +-----------------------+                           |                  |
+-------------------------------------------------------+------------------+
                     ^                                  |
       routing table |                                  | forwards request
                     |                                  v
+--------------------+--------------+    +-------------------------------------+
|        CONTROL PLANE (x3)         |    |          GPU WORKER NODES           |
|                                   |    |                                     |
|  +-----------------------------+  |    |  +-------------------------------+  |
|  | Raft consensus              |  |    |  | Worker agent                  |  |
|  | leader election, log repl.  |  |    |  | starts/stops containers,      |  |
|  +-----------------------------+  |    |  | reports health + GPU memory   |  |
|  | API server (FastAPI)        |  |<---+  +-------------------------------+  |
|  | deployments, keys, nodes    |  |    |                                     |
|  +-----------------------------+  |    |  +---------------+ +-------------+  |
|  | Scheduler                   |  |--->|  | VRTX ENGINE   | | APP         |  |
|  | bin packing onto GPUs       |  |    |  | replica       | | CONTAINERS  |  |
|  +-----------------------------+  |    |  |               | |             |  |
|  | Autoscaler                  |  |    |  | scheduler     | | user code   |  |
|  | Erlang C + forecasting      |  |    |  | paged KV      | | from        |  |
|  +-----------------------------+  |    |  | Triton kernels| | git push    |  |
|  | Builder                     |  |    |  | spec decoding | |             |  |
|  | git push -> container image |  |    |  +-------+-------+ +-------------+  |
|  +-----------------------------+  |    |          |                          |
+-----------------+-----------------+    +----------+--------------------------+
                  |                                 |
                  | metrics + logs                  | metrics + logs
                  v                                 v
+--------------------------------------------------------------------------+
|                                VRTX OPS                                  |
|                                                                          |
|   +----------------+   +--------------------+   +---------------------+  |
|   | Collector      |   | Anomaly detection  |   | AI debugger         |  |
|   | logs, metrics, |-->| z-scores, log      |-->| sends logs to VRTX  |  |
|   | traces         |   | clustering         |   | Engine, gets fix    |  |
|   +----------------+   +--------------------+   +----------+----------+  |
+------------------------------------------------------------+-------------+
                                                             |
                                                             v
                                              +-----------------------------+
                                              |  DASHBOARD (Next.js)        |
                                              |  incidents, charts, deploys |
                                              +-----------------------------+
```

**Key idea:** the AI debugger in VRTX Ops calls VRTX Engine to explain failures. The platform debugs itself with its own model.

---

## 4. Request lifecycle

What happens when someone sends one chat message:

```
Client          Gateway             Control plane        Worker / Engine
  |                |                      |                     |
  |-- POST ------->|                      |                     |
  |  /v1/chat/     | 1. check API key     |                     |
  |  completions   | 2. token bucket      |                     |
  |                | 3. hash prompt       |                     |
  |                |    prefix, look up   |                     |
  |                |    routing table  <--| (pushed on change)  |
  |                |                      |                     |
  |                |-- forward ---------------------------------->|
  |                |                      |   4. tokenize        |
  |                |                      |   5. join next batch |
  |                |                      |   6. prefill prompt  |
  |                |                      |      (reuse cached   |
  |                |                      |       prefix blocks) |
  |                |                      |   7. decode loop:    |
  |<== token ======|<=========================  one token per    |
  |<== token ======|<=========================  forward pass,     |
  |<== token ======|<=========================  streamed via SSE  |
  |                |                      |                     |
  |<== [DONE] =====|<========================= 8. free KV blocks |
  |                | 9. meter tokens      |                     |
  |                |                      |<-- 10. metrics -----|
  |                |                      |  autoscaler decides |
  |                |                      |  if more replicas   |
  |                |                      |  are needed         |
```

---

## 5. VRTX Engine in depth

### 5.1 The core insight: decoding is memory-bound

To generate one token, the GPU must read every weight in the model. So for one user:

```
max tokens/sec  ~=  memory bandwidth / model size in bytes
```

Example: Llama 3 8B in fp16 is ~16 GB. An RTX 4090 has ~1 TB/s of bandwidth. Ceiling: ~60 tokens/sec for a single user, no matter how fast the math units are. A T4 (320 GB/s) running Qwen 2.5 0.5B (~1 GB in fp16) has a ceiling around 300 tokens/sec.

If we serve **B** users in the same forward pass, we read the weights once and do B times the math. Arithmetic intensity (FLOPs per byte read) grows roughly with B. Under the **roofline model**, the GPU only becomes compute-bound once:

```
B  >  peak FLOPs / memory bandwidth      (the "ridge point")
```

That is in the low hundreds for most GPUs. **The whole engine is a machine for making B as large as possible without running out of memory.**

### 5.2 Engine architecture

```
                    +--------------------------------------------+
  requests -------> |              REQUEST QUEUE                 |
                    +----------------------+---------------------+
                                           |
                                           v
                    +--------------------------------------------+
                    |                SCHEDULER                   |
                    |  every iteration:                          |
                    |   - remove finished sequences              |
                    |   - admit waiting sequences if KV fits     |
                    |   - split prefill into chunks              |
                    |   - preempt (swap or recompute) if full    |
                    +----------+--------------------+------------+
                               |                    |
                               v                    v
               +---------------------+   +-------------------------+
               |  KV CACHE MANAGER   |   |      MODEL RUNNER       |
               |  block allocator    |   |  embeddings             |
               |  block tables       |<->|  N x [RMSNorm, attention|
               |  prefix hash cache  |   |       (Triton), MLP]    |
               |  copy-on-write      |   |  LM head -> logits      |
               +---------------------+   +------------+------------+
                                                      |
                                                      v
                                         +-------------------------+
                                         |        SAMPLER          |
                                         |  temperature, top-k,    |
                                         |  top-p, spec decoding   |
                                         +------------+------------+
                                                      |
                                                      v
                                              stream tokens out
```

### 5.3 Continuous batching

Naive batching waits for every request in a batch to finish before starting new ones, so short replies sit idle waiting on long ones. VRTX does **iteration-level scheduling**: after every forward pass, finished sequences leave and waiting ones join.

Two phases with very different costs:

- **Prefill** (process the whole prompt at once): compute-heavy. Cost ~ `2 x params x prompt_length` FLOPs, plus attention that grows with `n^2`.
- **Decode** (one token at a time): memory-heavy.

Long prompts get **chunked** (for example 512 tokens per step) so one giant prompt can't freeze everyone else's stream.

### 5.4 KV cache memory math

Every token in context stores a key and a value vector in every layer:

```
KV bytes per token = 2 x layers x kv_heads x head_dim x bytes_per_value
```

| Model | Layers | KV heads | Head dim | fp16 KV per token |
|---|---|---|---|---|
| Qwen 2.5 0.5B | 24 | 2 | 64 | 12 KB |
| Llama 3 8B | 32 | 8 | 128 | 128 KB |

An 8K-token conversation on Llama 3 8B uses ~1 GB of KV cache. **That is what limits batch size.**

### 5.5 Paged attention

Instead of reserving max-context memory for every request up front, split the KV cache into fixed-size **blocks** (for example 16 tokens each), just like an operating system splits RAM into pages.

```
Sequence A block table:  [ 7 ][ 2 ][ 9 ]        Physical GPU blocks
Sequence B block table:  [ 7 ][ 4 ]             +---+---+---+---+---+---+---+---+---+---+
                           ^                    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
                           |                    +---+---+---+---+---+---+---+---+---+---+
              shared prefix block (refcount 2)        A       B           A,B     A
```

- Waste per sequence drops to **less than one block**.
- **Prefix caching:** hash each full block of token IDs. If 500 users share a system prompt, its blocks are stored once with a reference count.
- **Copy-on-write:** when a shared block needs to change, copy it first.
- **Eviction:** when memory is full, either swap blocks to CPU RAM or drop and recompute later. Pick whichever is cheaper:

```
swap cost      ~= bytes / PCIe bandwidth   (x2: out and back in)
recompute cost ~= prefill FLOPs / GPU FLOPs
```

### 5.6 Custom Triton kernels

| Kernel | Why |
|---|---|
| **Flash-style attention** | Never materializes the full `n x n` attention matrix. Memory goes from O(n^2) to O(n) |
| **Paged attention (decode)** | Reads K and V directly through the block table |
| **Fused RMSNorm** | One kernel instead of several, fewer memory round trips |
| **Fused rotary embeddings (RoPE)** | Same reason |
| **Int8 / int4 dequant + matmul** | Read small weights, expand them on chip |

The trick behind flash attention is the **online softmax**. Process scores in tiles. Keep a running max `m`, running sum `l`, and running output `o`. For each new tile of scores `x` with values `V_tile`:

```
m_new = max(m, max(x))
l_new = l * e^(m - m_new) + sum(e^(x - m_new))
o_new = o * e^(m - m_new) + sum(e^(x - m_new) * V_tile)
final output = o / l
```

Every kernel gets a correctness test against PyTorch and a speed benchmark.

### 5.7 Quantization

Symmetric quantization with group size `g`:

```
int8:  scale s = max|w| / 127      q = round(w / s)      w ~= q x s
int4:  scale s = max|w| / 7        (two values packed per byte)
```

Rounding error is roughly uniform on `[-s/2, s/2]`, so its variance is `s^2 / 12`. Smaller groups mean smaller `s` and less error, but more scales to store. We measure the tradeoff with a **perplexity vs. memory** chart. Halving the bytes roughly doubles the decode ceiling from 5.1.

### 5.8 Speculative decoding

A small **draft** model (for example Qwen 2.5 0.5B) guesses `gamma` tokens. The big **target** model (for example Qwen 2.5 7B) checks all of them in one forward pass.

If each guess is accepted with probability `alpha`:

```
expected tokens per target pass = (1 - alpha^(gamma + 1)) / (1 - alpha)
```

Acceptance rule (keeps the output distribution **exactly** the same as the target model alone):

```
accept draft token x with probability  min(1, p(x) / q(x))
on rejection, sample from  normalize(max(0, p - q))
```

where `p` is the target's distribution and `q` is the draft's. VRTX tunes `gamma` live from the observed acceptance rate.

### 5.9 Sampling

Implemented by hand (already in `engine/sampling.py`):

- **Temperature:** divide logits by `T`
- **Top-k:** keep the `k` most likely tokens
- **Top-p (nucleus):** keep the smallest set whose probabilities sum to at least `p`

---

## 6. VRTX Cloud in depth

### 6.1 Components

```
+------------------------- CONTROL PLANE NODE (x3) --------------------------+
|                                                                            |
|  FastAPI API  -->  propose change  -->  RAFT LOG  -->  commit on majority  |
|                                            |                               |
|                                            v                               |
|                               STATE MACHINE (SQLite)                       |
|                  deployments | replicas | nodes | api keys | builds        |
|                                            |                               |
|            +-------------------------------+---------------------+         |
|            v                               v                     v         |
|       SCHEDULER                       AUTOSCALER              BUILDER      |
|   (where replicas go)          (how many replicas)      (git -> image)     |
+----------------------------------------------------------------------------+
```

Only the **leader** runs the scheduler and autoscaler. If it dies, a new leader is elected and picks up from the replicated log.

### 6.2 Raft consensus

- **Leader election:** followers wait a randomized timeout (for example 150 to 300 ms). If no heartbeat arrives, they become candidates and request votes. Randomization prevents split votes.
- **Log replication:** the leader appends commands and sends them to followers.
- **Commit rule:** an entry is committed once a majority has it:

```
majority = floor(n / 2) + 1        3 nodes -> 2,  5 nodes -> 3
```

- **Safety:** a node only votes for a candidate whose log is at least as up to date as its own.

**Demo moment:** kill the leader in the middle of a deploy and watch the cluster elect a new one and finish the deploy.

### 6.3 Scheduling is bin packing

Each node has capacity in several dimensions: GPU memory, CPU, RAM. Each replica needs some of each. Fitting them is **multi-dimensional bin packing**, which is NP-hard. We implement and compare:

| Algorithm | Notes |
|---|---|
| First-Fit Decreasing | Sort by size, put each item in the first bin that fits. In 1D, uses at most `11/9 x OPT + 6/9` bins |
| Best-Fit | Put each item where it leaves the least leftover space |
| Scored heuristic | Weighted score of leftover GPU memory, fragmentation, and spreading replicas across nodes |
| ILP solver (small clusters only) | Finds the true optimum, used to measure how close the heuristics get |

### 6.4 Autoscaling with queueing theory

Model each deployment as an **M/M/c queue**:

- `lambda` = request arrival rate
- `mu` = requests per second one replica can serve
- `c` = number of replicas
- `a = lambda / mu` (offered load), utilization `rho = a / c` (must be < 1)

**Erlang C** gives the probability a request has to wait:

```
                 (a^c / c!) x (c / (c - a))
C(c, a) = -----------------------------------------------
           sum_{k=0}^{c-1} (a^k / k!)  +  (a^c / c!) x (c / (c - a))
```

Then:

```
expected wait      W_q = C(c, a) / (c x mu - lambda)
tail of the wait   P(wait > t) = C(c, a) x e^(-(c x mu - lambda) x t)
p99 wait           t_99 = ln(100 x C(c, a)) / (c x mu - lambda)     (when C > 0.01)
```

**The autoscaler picks the smallest `c` where predicted p99 latency is under the SLO.**

Extras:

- **Little's Law** `L = lambda x W` to sanity check predictions against live metrics.
- **Holt-Winters forecasting** of `lambda` so we scale up **before** a spike.
- **PID control with hysteresis** so replica counts don't flap:

```
u(t) = Kp x e(t) + Ki x integral(e) + Kd x de/dt
```

- **Scale to zero** when idle, plus cold-start tricks (keep weights in page cache, snapshot loaded models).

### 6.5 Routing

- **Consistent hashing** on the first N tokens of the prompt, with virtual nodes. Requests with the same system prompt land on the replica that already has it cached. Adding a replica only moves about `1/n` of keys.
- **Power of two choices** as a comparison: pick two random replicas, send to the less loaded one. Max load drops from about `log n / log log n` to about `log log n`.
- **Measure it:** prefix cache hit rate with consistent hashing vs. random routing.

### 6.6 Deploying apps

```
vrtx deploy  -->  upload code  -->  builder makes Docker image
             -->  scheduler picks node  -->  worker starts container
             -->  gateway gives it a URL  -->  health checks  -->  live
```

- Isolation with namespaces and cgroups (Docker), stretch goal Firecracker.
- Every app gets a URL like `myapp.vrtx.local`.
- Rolling deploys: start new replica, health check, shift traffic, stop old one.

---

## 7. VRTX Ops in depth

```
logs + metrics --> COLLECTOR --> ANOMALY DETECTION --> INCIDENT --> AI DEBUGGER --> dashboard
                                   z-scores                          (VRTX Engine)
                                   log clustering
```

- **Metric anomalies:** rolling z-score `z = (x - mean) / std` over a window. Alert when `|z| > 3`.
- **Log clustering:** embed error messages, group similar ones so 10,000 identical errors become one incident.
- **AI debugger:** builds a prompt from the stack trace, recent logs, recent deploys and metrics, sends it to a model served by VRTX Engine, and returns:
  - what broke
  - the likely cause
  - a suggested fix
- **First real feature:** a GPU spend guard that auto-stops idle replicas.

---

## 8. API design

### Inference API (OpenAI compatible)

Any app that works with the OpenAI SDK can point at VRTX by changing the base URL.

| Method | Path | What it does |
|---|---|---|
| POST | `/v1/chat/completions` | Chat. `stream: true` returns Server-Sent Events |
| POST | `/v1/completions` | Raw text completion |
| GET | `/v1/models` | List loaded models |
| GET | `/health` | Liveness |
| GET | `/metrics` | Prometheus metrics |

### Control plane API

| Method | Path | What it does |
|---|---|---|
| POST | `/api/deployments` | Create a model or app deployment |
| GET | `/api/deployments` | List deployments |
| PATCH | `/api/deployments/{id}` | Change replicas, SLO, model |
| DELETE | `/api/deployments/{id}` | Tear down |
| GET | `/api/nodes` | Nodes, capacity, health |
| GET | `/api/cluster` | Raft leader, term, commit index |
| POST | `/api/keys` | Create an API key |
| GET | `/api/incidents` | Incidents from VRTX Ops |
| WS | `/ws/logs/{deployment}` | Live log stream |
| WS | `/ws/metrics` | Live metrics stream |

### Example deployment spec

```yaml
name: qwen-chat
type: model
model: Qwen/Qwen2.5-7B-Instruct
quantization: int8
speculative:
  draft: Qwen/Qwen2.5-0.5B-Instruct
  gamma: auto
scaling:
  min_replicas: 0
  max_replicas: 4
  slo_p99_ms: 800
resources:
  gpu_memory_gb: 12
```

---

## 9. Dashboard wireframes

### 9.1 Overview

```
+------------------------------------------------------------------------------+
|  VRTX                       Overview  Deployments  Playground  Ops  Bench    |
+------------------------------------------------------------------------------+
|                                                                              |
|  +--------------+  +--------------+  +--------------+  +--------------+      |
|  | Tokens/sec   |  | Active users |  | p99 latency  |  | GPU memory   |      |
|  |   1,842      |  |     126      |  |   640 ms     |  |   71%        |      |
|  +--------------+  +--------------+  +--------------+  +--------------+      |
|                                                                              |
|  Throughput (last 15 min)                       Cluster                      |
|  +------------------------------------+   +------------------------------+  |
|  |          __/\__      /\            |   | Leader:  cp-2   term 14      |  |
|  |   __/\__/      \____/  \___        |   | cp-1  follower   healthy     |  |
|  |  /                          \__    |   | cp-2  LEADER     healthy     |  |
|  +------------------------------------+   | cp-3  follower   healthy     |  |
|                                           +------------------------------+  |
|  Nodes                                                                       |
|  +------------------------------------------------------------------------+ |
|  | node      GPU      GPU mem            replicas   status                 | |
|  | gpu-a     T4       [#######---] 70%   3          healthy                | |
|  | gpu-b     T4       [####------] 40%   1          healthy                | |
|  | cpu-1     none     n/a                2 apps     healthy                | |
|  +------------------------------------------------------------------------+ |
+------------------------------------------------------------------------------+
```

### 9.2 Deployments

```
+------------------------------------------------------------------------------+
|  Deployments                                        [ + New deployment ]     |
+------------------------------------------------------------------------------+
|  name          type    model / image          replicas   p99     status      |
|  qwen-chat     model   Qwen2.5-7B (int8)      3 / 4      612ms   running     |
|  draft-small   model   Qwen2.5-0.5B           1 / 2      90ms    running     |
|  todo-api      app     git: main@a1f3c        2 / 5      45ms    running     |
|  docs-site     app     git: main@77b0e        0 / 2      n/a     scaled to 0 |
+------------------------------------------------------------------------------+

  qwen-chat
  +----------------------------------+  +-----------------------------------+
  | Replicas vs traffic              |  | Autoscaler decision log           |
  |  lambda  ___/^^^\___             |  | 14:02  lambda=38/s  c: 2 -> 3     |
  |  c       _|‾‾‾‾|_                |  |        predicted p99 910ms > SLO  |
  |                                  |  | 14:09  forecast spike, c: 3 -> 4  |
  +----------------------------------+  +-----------------------------------+
  [ Scale ]  [ Logs ]  [ Edit spec ]  [ Delete ]
```

### 9.3 Playground

```
+------------------------------------------------------------------------------+
|  Playground          model: [ qwen-chat v ]   temp [0.7]  top-p [0.9]        |
+------------------------------------------------------------------------------+
|  System: You are a helpful assistant.                                        |
|  ------------------------------------------------------------------------    |
|  You:   explain paged attention like I'm 5                                   |
|  VRTX:  Imagine your toy box is split into little bins...|  (streaming)      |
|                                                                              |
|  +--------------------------------------------------------------+ [ Send ]   |
|  | type a message                                               |            |
|  +--------------------------------------------------------------+            |
|  TTFT 84 ms   |   61 tok/s   |   prefix cache hit: yes   |   replica gpu-a/2 |
+------------------------------------------------------------------------------+
```

### 9.4 Ops / incidents

```
+------------------------------------------------------------------------------+
|  Ops                                                       3 open incidents  |
+------------------------------------------------------------------------------+
|  ! todo-api crash loop             14:21   412 similar log lines             |
|  ! gpu-b memory spike  z = 4.2     14:18                                     |
|  ~ qwen-chat p99 above SLO         14:02   resolved by autoscaler            |
+------------------------------------------------------------------------------+
|  todo-api crash loop                                                         |
|  +------------------------------------------------------------------------+  |
|  | AI diagnosis (via qwen-chat on VRTX Engine)                            |  |
|  |                                                                        |  |
|  | What broke: the app exits on startup with KeyError: 'DATABASE_URL'.    |  |
|  | Likely cause: deploy a1f3c added a database but no env var was set.    |  |
|  | Fix: vrtx env set todo-api DATABASE_URL=...  then redeploy.            |  |
|  +------------------------------------------------------------------------+  |
|  [ View logs ]  [ Roll back to 77b0e ]  [ Mark resolved ]                    |
+------------------------------------------------------------------------------+
```

### 9.5 Benchmarks

```
+------------------------------------------------------------------------------+
|  Benchmarks        GPU: [ T4 v ]   model: [ Qwen2.5-0.5B v ]                 |
+------------------------------------------------------------------------------+
|  Throughput vs concurrent users                                              |
|  tok/s                                                                       |
|   |                                 ___________ vLLM                         |
|   |                        ________/____________ VRTX                        |
|   |               ________/                                                  |
|   |     _________/                                                           |
|   |____/____________________________________________ HF naive               |
|   +-------------------------------------------------------- users            |
|                                                                              |
|  Roofline                         | Spec decoding speedup vs gamma            |
|  (achieved vs ridge point)        | (bars for gamma = 1..8)                   |
+------------------------------------------------------------------------------+
```

---

## 10. CLI wireframe

```
$ vrtx login
$ vrtx models deploy Qwen/Qwen2.5-7B-Instruct --name qwen-chat --quant int8 --slo 800ms
  Scheduling on gpu-a ........ done
  Loading weights (7.6 GB) ... done
  Healthy at https://qwen-chat.vrtx.local/v1

$ vrtx deploy ./todo-api
  Building image ............. done (38s)
  Live at https://todo-api.vrtx.local

$ vrtx ps
  NAME        TYPE   REPLICAS  P99    STATUS
  qwen-chat   model  3/4       612ms  running
  todo-api    app    2/5       45ms   running

$ vrtx logs todo-api --follow
$ vrtx explain todo-api        # asks VRTX Ops why it is failing
$ vrtx cluster status          # Raft leader, term, node health
$ vrtx bench qwen-chat --users 1,8,32,128
```

---

## 11. Repo structure

```
VRTX/
├── README.md               <- this file
├── requirements.txt
├── .gitignore
├── engine/                 VRTX Engine
│   ├── sampling.py         temperature / top-k / top-p (done)
│   ├── naive_engine.py     Month 1 baseline engine (done)
│   ├── scheduler.py        continuous batching (Month 2)
│   ├── kv_cache.py         paged KV blocks + prefix cache (Month 2)
│   ├── model_runner.py     our own forward pass (Month 2-3)
│   ├── kernels/            Triton kernels (Month 3)
│   ├── quant/              int8 / int4 (Month 3)
│   └── speculative/        draft + verify (Month 4)
├── gateway/                FastAPI front door
│   └── server.py           OpenAI-compatible API (done, Month 1)
├── cloud/                  VRTX Cloud
│   ├── raft/               consensus (Month 6)
│   ├── api/                control plane REST API (Month 5)
│   ├── scheduler/          bin packing (Month 6)
│   ├── autoscaler/         Erlang C + forecasting (Month 7)
│   ├── router/             consistent hashing (Month 7)
│   ├── builder/            git -> image (Month 5)
│   └── agent/              worker node agent (Month 5)
├── ops/                    VRTX Ops (Month 8)
├── cli/                    `vrtx` command (Month 5+)
├── dashboard/              Next.js web UI (Month 8)
├── benchmarks/
│   └── bench_naive.py      Month 1 baseline numbers (done)
├── tests/
├── docs/                   design notes, math derivations, blog drafts
└── scripts/                setup helpers
```

---

## 12. Build plan, month by month

Each month ends with something that runs and a number you can measure.

### Phase 1: VRTX Engine (Months 1 to 4)

**Month 1: Baseline and API**
- [x] Project folder, README, starter code
- [x] Run Qwen 2.5 0.5B with our own sampling loop (`engine/naive_engine.py`)
- [x] OpenAI-compatible FastAPI server with streaming (`gateway/server.py`)
- [x] Benchmark harness: time to first token, tokens/sec, latency percentiles
- [x] Load test with 1, 8, 32 users and record how badly the naive engine does
- [x] Push to GitHub, set up GitHub Actions running pytest
- **Milestone number:** baseline tokens/sec on a T4: **35.3 tok/s** (Hugging Face `model.generate`, one request at a time, fp16)

**Month 2: Continuous batching and paged KV cache**
- [x] Write our own model forward pass (embedding, RMSNorm, RoPE, attention, MLP) using HF weights
- [x] Block allocator and block tables
- [x] Iteration-level scheduler with chunked prefill
- [x] Preemption: recompute (swap to CPU still to do)
- [x] T4 benchmark vs. batch 1 and naive Hugging Face
- **Milestone number:** throughput at 32 users vs. Month 1 (target 5x or more)

**Month 2 results (Kaggle Tesla T4, fp16, Qwen2.5-0.5B-Instruct, 64 requests x 128 output tokens, 256 MB KV cache)**

| Setup | Throughput | vs. batch 1 | vs. naive HF | Wall time | Engine steps |
|---|---|---|---|---|---|
| Hugging Face `model.generate`, one request at a time | 35.3 tok/s | | 1.00x | 232.1 s | |
| VRTX, max batch 1 | 40.8 tok/s | 1.00x | 1.16x | 200.8 s | 8192 |
| VRTX, max batch 8 | 105.1 tok/s | 2.58x | 2.98x | 77.9 s | 1024 |
| VRTX, max batch 32 | 127.0 tok/s | 3.11x | 3.60x | 64.5 s | 258 |
| VRTX, max batch 64 | **130.4 tok/s** | 3.20x | **3.69x** | 62.8 s | 133 |

Headline: **3.7x the throughput of naive Hugging Face** on a T4, with zero preemptions. Median time to first token at batch 64 is 0.73 s versus 99 s when requests wait in line at batch 1.

Milestone status: 3.6x at 32 users, short of the 5x target. The reason is visible in the step times. One step takes about 25 ms at batch 1, 76 ms at batch 8, 250 ms at batch 32 and 470 ms at batch 64, so step time grows almost linearly with batch size. On a GPU the weight reads should be shared and nearly free per extra sequence, so the cost is per sequence work: attention still runs as a Python loop, one gather and one attention call per sequence per layer (64 x 24 = 1536 small launches per step). That is exactly what the Month 3 paged attention kernel replaces, and it is why throughput flattens between batch 32 and 64.

Other runs: 4-core Linux CPU, 8 users: 3.1 to 18.2 tok/s (5.8x). Apple M4 GPU (mps, fp32), 16 users: 10.4 to 39.0 tok/s (3.7x). Raw JSON in `benchmarks/results/`.

**Month 3: Kernels and quantization**
- [ ] Triton: fused RMSNorm, RoPE, flash-style prefill attention, paged decode attention
- [ ] Int8 and int4 weight quantization with group scales
- [ ] Perplexity vs. memory chart, roofline chart
- **Milestone number:** kernel speedups vs. PyTorch, memory saved

**Month 4: Speculative decoding and prefix caching**
- [ ] Draft/target verification with the exact rejection rule
- [ ] Adaptive gamma
- [ ] Prefix hashing, reference counts, copy-on-write
- [ ] Full benchmark writeup vs. HF naive and vLLM
- **Milestone number:** headline tokens/sec and concurrent users

### Phase 2: VRTX Cloud and Ops (Months 5 to 8)

**Month 5: Workers and deploys**
- [ ] Worker agent (start/stop containers, report GPU memory)
- [ ] Control plane API (single node first)
- [ ] Builder: code folder to Docker image
- [ ] Gateway routes `*.vrtx.local` to apps
- [ ] `vrtx deploy`, `vrtx ps`, `vrtx logs`
- **Milestone:** deploy an app and a model with one command each

**Month 6: Raft and scheduling**
- [ ] Raft: elections, log replication, commit, persistence
- [ ] Move control plane state onto the Raft log
- [ ] Bin packing: FFD, Best-Fit, scored heuristic, ILP comparison
- **Milestone:** kill the leader mid-deploy, cluster recovers

**Month 7: Autoscaling and routing**
- [ ] Erlang C autoscaler with SLO target
- [ ] Holt-Winters forecast, PID with hysteresis, scale to zero
- [ ] Consistent hashing router, power-of-two-choices comparison
- **Milestone:** p99 stays under SLO during a 10x traffic spike

**Month 8: Ops and dashboard**
- [ ] Metrics and log collector
- [ ] Z-score anomalies, log clustering
- [ ] AI debugger on VRTX Engine
- [ ] Next.js dashboard (all screens in section 9)
- [ ] Chaos tests: random node kills, network delay
- **Milestone:** a crash gets detected and explained automatically

### Phase 3: Ship it (Month 9)

- [ ] Clean up code and docs
- [ ] Technical blog post with all the charts
- [ ] 2 to 3 minute demo video
- [ ] Resume, LinkedIn, and portfolio updates

---

## 13. Free compute plan

| Need | Free source | Notes |
|---|---|---|
| Laptop work (API, scheduler logic, Raft, dashboard) | Your Mac | Use Qwen 2.5 0.5B on CPU/MPS. Slow but fine for logic |
| Daily GPU dev | **Lightning AI** | ~80 free GPU hours/month, persistent VS Code-style workspace with SSH |
| Daily GPU dev | **Modal** Starter plan | $30/month free credits, real datacenter GPUs, per-second billing |
| Backup GPU | **Kaggle** notebooks | Weekly GPU quota (commonly ~30 hrs, varies). Pick the T4 |
| Backup GPU | **Google Colab** free | Throttles at peak times |
| Big final benchmarks | **AMD Developer Cloud** | Apply for complimentary MI300X hours (192 GB GPU). Credits expire fast, use at the end |
| Big final benchmarks | New-account credits | Google Cloud, Azure, RunPod, Lambda trial credits |
| Big swing | Startup credits via Bhat-Tec LLC | Modal, RunPod and others run startup credit programs |
| Multi-machine control plane demo | **Oracle Cloud Always Free** | Free ARM VMs, no GPU needed for Raft nodes |
| Code, CI | GitHub | Free |
| Domain | Subdomain of therawebdesign.com | e.g. `vrtx.therawebdesign.com` |

Check each provider's current terms before relying on them; free tiers change.

**Rules:**
1. Develop on the smallest model that shows the behavior.
2. Always set auto-shutdown on GPU sessions.
3. Save benchmark results and checkpoints to GitHub, since free sessions can get cut off.
4. Compare systems on the same hardware. "4x faster than HF on a T4" is just as strong as on an H100.

---

## 14. Benchmarking plan

**Metrics:**
- **TTFT** (time to first token)
- **TPOT** (time per output token)
- **Throughput** (output tokens/sec across all users)
- **Latency percentiles** (p50, p95, p99)
- **Max concurrent users** at a given p99 SLO
- **GPU memory** used and wasted
- **Prefix cache hit rate**
- **Spec decoding** acceptance rate and speedup

**Workloads:**
- Short chat (100 in / 200 out)
- Long context (4K in / 200 out)
- Shared system prompt (tests prefix caching)
- Bursty traffic (tests autoscaler)

**Compared systems:** HF `transformers` naive generate, VRTX at each milestone, vLLM.

**Charts for the writeup:** throughput vs. users, latency CDF, roofline, KV memory waste before/after paging, perplexity vs. quantization, spec decoding speedup vs. gamma, autoscaler replicas vs. traffic, Raft recovery timeline.

---

## 15. The math cheat sheet

| Concept | Formula | Where it is used |
|---|---|---|
| Decode ceiling | `tok/s ~= bandwidth / model bytes` | Engine design |
| Roofline ridge | `B > peak FLOPs / bandwidth` | Batch size target |
| KV per token | `2 x L x H_kv x d x bytes` | Memory planning |
| Online softmax | `l_new = l e^(m - m_new) + sum e^(x - m_new)` | Flash attention kernel |
| Quant scale | `s = max|w| / 127`, error var `s^2/12` | Int8 / int4 |
| Spec decoding | `E = (1 - a^(g+1)) / (1 - a)` | Draft length |
| Spec acceptance | `min(1, p/q)`, resample `norm(max(0, p - q))` | Exactness |
| Raft majority | `floor(n/2) + 1` | Commit rule |
| FFD bound | `<= 11/9 OPT + 6/9` | Scheduler |
| Erlang C wait | `W_q = C(c,a) / (c mu - lambda)` | Autoscaler |
| Wait tail | `P(W > t) = C e^(-(c mu - lambda) t)` | p99 target |
| Little's Law | `L = lambda W` | Sanity checks |
| Token bucket | `tokens = min(cap, tokens + r dt)` | Rate limiting |
| Z-score | `z = (x - mean) / std` | Anomaly detection |
| PID | `u = Kp e + Ki int(e) + Kd de/dt` | Smooth scaling |
| Two choices | max load ~ `log log n` | Load balancing |

---

## 16. Risks and how we handle them

| Risk | Plan |
|---|---|
| Can't beat vLLM | We aren't trying to beat it overall. Targets: 10x+ over naive HF, within a measured % of vLLM on one workload |
| GPU access runs out | Stack several free tiers; most of Phase 2 needs no GPU at all |
| Triton kernels are hard to debug | Always test against a PyTorch reference; start with the simplest kernel (RMSNorm) |
| Raft has subtle bugs | Write tests that randomly drop/delay messages and kill nodes |
| Scope creep | Each month has one milestone number. Stretch goals stay stretch |
| Mac has no CUDA | Logic on the Mac, GPU code on Lightning AI / Modal / Kaggle |

---

## 17. What "done" looks like

- Public GitHub repo with clean code, tests, and this README
- Benchmark charts with real numbers
- A demo video showing:
  1. Live token streaming in the playground
  2. Traffic spike and the autoscaler adding replicas
  3. Killing the Raft leader and the cluster recovering
  4. An app crashing and VRTX Ops explaining why
- A technical blog post
- The resume line from section 1 with real values for X and Y

---

## 18. Getting started today

On your Mac, from this folder:

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server (downloads Qwen 2.5 0.5B the first time, ~1 GB)
uvicorn gateway.server:app --port 8000

# 4. In another terminal, chat with it (streaming)
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain GPUs in one sentence"}],"stream":true}'

# 5. Get your first baseline number
python -m benchmarks.bench_naive
```

Then:

```bash
git init
git add .
git commit -m "VRTX: Month 1 starter"
# create a repo on GitHub named vrtx and push
```

---

## 19. Reading list

Read these in roughly this order as you reach each month:

1. *Attention Is All You Need* (Vaswani et al., 2017): the transformer
2. Andrej Karpathy, *nanoGPT* and *Let's build GPT* video: write a model from scratch
3. *FlashAttention* (Dao et al., 2022): tiling and online softmax
4. *Efficient Memory Management for LLM Serving with PagedAttention* (Kwon et al., 2023): vLLM's paper
5. *Orca: A Distributed Serving System for Transformer-Based Generative Models* (2022): continuous batching
6. *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., 2023)
7. Triton tutorials (official docs): vector add, softmax, matmul, fused attention
8. *In Search of an Understandable Consensus Algorithm* (Ongaro and Ousterhout, 2014): Raft
9. The Raft visualization at raft.github.io
10. Mor Harchol-Balter, *Performance Modeling and Design of Computer Systems*: queueing theory
11. *Large-scale cluster management at Google with Borg* (2015): scheduling at scale
12. Martin Kleppmann, *Designing Data-Intensive Applications*
