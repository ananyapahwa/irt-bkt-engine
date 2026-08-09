# BKT Engine

> **Bayesian Knowledge Tracing** — a production-ready, fully modular Hidden Markov Model engine for tracking student skill mastery in real-time, with optional IRT-driven dynamic parameter derivation.

---

## Table of Contents

1. [What is BKT?](#1-what-is-bkt)
2. [The Hidden Markov Model (HMM) Structure](#2-the-hidden-markov-model-hmm-structure)
3. [The Four Core Parameters](#3-the-four-core-parameters)
4. [The Two-Stage Update Algorithm](#4-the-two-stage-update-algorithm)
   - [Stage 1 — Conditional Update (Bayes' Theorem)](#stage-1--conditional-update-bayes-theorem)
   - [Stage 2 — Learning Transition](#stage-2--learning-transition)
5. [IRT → BKT Parameter Bridge](#5-irt--bkt-parameter-bridge)
   - [The 2PL Item Characteristic Curve](#the-2pl-item-characteristic-curve)
   - [Dynamic P(G) from IRT](#dynamic-pg-from-irt)
   - [Dynamic P(S) from IRT](#dynamic-ps-from-irt)
   - [Difficulty-Scaled P(T)](#difficulty-scaled-pt)
   - [IRT Cold-Start P(L₀)](#irt-cold-start-pl₀)
6. [Guess Detection](#6-guess-detection)
7. [Numerical Stability & Clamping](#7-numerical-stability--clamping)
8. [Mastery Thresholding](#8-mastery-thresholding)
9. [Architecture](#9-architecture)
10. [Quick Start](#10-quick-start)
11. [Running Tests](#11-running-tests)
12. [Parameter Derivation Reference](#12-parameter-derivation-reference)

---

## 1. What is BKT?

**Bayesian Knowledge Tracing** (Corbett & Anderson, 1994) models a student's knowledge of a skill as a **binary latent variable**:

```
L_t ∈ {0, 1}     where  1 = "Knows the skill"
                         0 = "Does not know the skill"
```

Because we can never directly observe whether a student "knows" something — we can only observe whether they answered correctly or not — BKT frames the problem as a **Hidden Markov Model (HMM)**:

- The **hidden state** is `L_t` (mastered or not)
- The **observable** is each answer (correct or incorrect)
- We track a **probability distribution** `P(L_t)` over the hidden state, updating it each time a new observation arrives

This gives every student a skill-level probability score **between 0 and 1** that responds dynamically to their performance.

---

## 2. The Hidden Markov Model (HMM) Structure

```
Time:          t-1              t              t+1
               
Hidden state:  L_{t-1}  ──T──▶  L_t   ──T──▶  L_{t+1}
                                 │
                         P(G), P(S)
                                 │
                                 ▼
Observable:             O_t ∈ {Correct, Incorrect}
```

**Two types of relationships define the HMM:**

| Relationship | Direction | Parameter |
|---|---|---|
| **Emission** | Hidden state → Observable | P(G), P(S) |
| **Transition** | State at time t → State at time t+1 | P(T) |

The key insight: the model can only become *more* knowledgeable over time (no forgetting in classic BKT). A student who knows a skill stays knowing it; a student who doesn't know has probability P(T) of learning it after each practice opportunity.

---

## 3. The Four Core Parameters

Every skill in the system is governed by exactly four parameters:

### P(L₀) — Initial Knowledge Prior

```
P(L₀) ∈ (0, 1)
```

The probability that a student **already knows** the skill before any instruction or observation. This is the HMM's initial state distribution.

- **Without IRT data:** Defaults to **0.20** (20% prior knowledge — conservative estimate for untaught secondary-school skills; Corbett & Anderson, 1994)
- **With IRT data:** Seeded from the IRT mastery initializer's Beta-Bernoulli shrinkage formula (see Section 5)

---

### P(T) — Transition / Learn Rate

```
P(T) ∈ (0, 1)
```

The probability that a student **transitions from Not Known → Known** after a single practice opportunity. This governs how fast the BKT curve climbs.

```
P(L_t | L_{t-1} = 0) = P(T)          [probability of learning in one step]
P(L_t | L_{t-1} = 1) = 1.0           [once known, stays known — no forgetting]
```

- **Without IRT data:** Defaults to **0.10** (mid-point of the 0.05–0.40 empirical range; Baker et al., 2008)
- **With IRT data:** Scaled by concept difficulty via sigmoid normalization (see Section 5)

---

### P(G) — Guess Rate

```
P(G) ∈ (0, 0.5)
```

The probability that a student who **does NOT know** the skill answers correctly anyway (lucky guess, distractor elimination, pattern recognition).

```
P(Correct | L = 0) = P(G)
```

- **Without IRT data:** Defaults to **1/N_options = 0.25** for a 4-option MCQ (theoretical uniform random-guess probability)
- **With IRT data:** Derived dynamically per item from the 2PL curve (see Section 5)
- **Degeneracy constraint:** P(G) must be strictly < 0.5, otherwise a non-knowing student is more likely to answer correctly than a knowing student with high slip — the model becomes uninformative

---

### P(S) — Slip Rate

```
P(S) ∈ (0, 0.5)
```

The probability that a student who **DOES know** the skill answers incorrectly (careless error, misread, time pressure, transcription mistake).

```
P(Incorrect | L = 1) = P(S)
P(Correct   | L = 1) = 1 - P(S)
```

- **Without IRT data:** Defaults to **0.10** (mid-point of the 0.05–0.15 empirical range; Baker et al., 2008)
- **With IRT data:** Derived dynamically per item from the 2PL curve (see Section 5)
- **Degeneracy constraint:** P(G) + P(S) < 1.0 must always hold

---

## 4. The Two-Stage Update Algorithm

Every time a student answers a question on skill `k`, the engine executes two sequential mathematical operations on the current state `P(L_{t-1})`:

---

### Stage 1 — Conditional Update (Bayes' Theorem)

This is a direct application of Bayes' theorem: given the observation (correct or incorrect), what is the posterior probability that the student knows the skill?

#### If the answer is **CORRECT**:

The joint probability of being in each state AND producing a correct answer:

```
P(L=1 AND Correct) = P(L_{t-1}) × P(Correct | L=1) = P(L_{t-1}) × (1 - P(S))
P(L=0 AND Correct) = (1 - P(L_{t-1})) × P(Correct | L=0) = (1 - P(L_{t-1})) × P(G)
```

Normalizing by the total probability of a correct answer (law of total probability):

```
                         P(L_{t-1}) × (1 - P(S))
P(L_t | Correct) = ─────────────────────────────────────────────────────
                   P(L_{t-1}) × (1 - P(S)) + (1 - P(L_{t-1})) × P(G)
```

**Intuition:** A correct answer increases our belief that the student knows the skill. The strength of this update depends on:
- How high P(L_{t-1}) already is (base rate)
- How low P(S) is (knowing students rarely make mistakes → correct = strong evidence)
- How low P(G) is (non-knowing students rarely guess correctly → correct = strong evidence)

---

#### If the answer is **INCORRECT**:

```
P(L=1 AND Incorrect) = P(L_{t-1}) × P(Incorrect | L=1) = P(L_{t-1}) × P(S)
P(L=0 AND Incorrect) = (1 - P(L_{t-1})) × P(Incorrect | L=0) = (1 - P(L_{t-1})) × (1 - P(G))
```

Normalizing:

```
                           P(L_{t-1}) × P(S)
P(L_t | Incorrect) = ─────────────────────────────────────────────────────────
                     P(L_{t-1}) × P(S) + (1 - P(L_{t-1})) × (1 - P(G))
```

**Intuition:** An incorrect answer decreases our belief, but the drop is cushioned by P(S). A student who was at 90% mastery and slips once stays well above 50% — the slip parameter prevents catastrophic drops from single errors.

---

### Stage 2 — Learning Transition

After incorporating the observation, we apply the **learning transition**: the probability that a student who still didn't know the skill at time t has now learned it:

```
P(L_t) = P(L_t | Obs) + (1 - P(L_t | Obs)) × P(T)
```

Breaking this down:

```
P(L_t) = [probability already known after obs] + [probability just learned]
       = P(L_t | Obs)                          + (1 - P(L_t | Obs)) × P(T)
                                                  └─────────────────────────┘
                                                  probability of still not knowing
                                                  × probability of learning in this step
```

**Note:** The transition only applies to the "not known" portion of the probability mass. If `P(L_t | Obs) = 1.0`, no transition is needed. This ensures that learning always pushes P(L) upward (or leaves it unchanged).

---

### Complete Update in One Place

```python
# Stage 1 — Conditional update
if is_correct:
    numerator   = P_L_prev × (1 - P_S)
    denominator = P_L_prev × (1 - P_S) + (1 - P_L_prev) × P_G
else:
    numerator   = P_L_prev × P_S
    denominator = P_L_prev × P_S + (1 - P_L_prev) × (1 - P_G)

P_conditional = numerator / denominator

# Stage 2 — Learning transition
P_L_new = P_conditional + (1 - P_conditional) × P_T

# Clamp to (0.001, 0.999)
P_L_new = clamp(P_L_new, 0.001, 0.999)
```

---

## 5. IRT → BKT Parameter Bridge

The classical BKT model uses static, skill-level constants for P(G) and P(S). This engine extends BKT with an **IRT hook**: when the orchestrator has student ability θ and item parameters (a, b) from the IRT engine, it derives **per-item dynamic parameters** instead of using static skill-level values.

---

### The 2PL Item Characteristic Curve

All IRT-based derivations depend on the **Two-Parameter Logistic (2PL) ICC**:

```
P(correct | θ, a, b) = 1 / (1 + exp(−a(θ − b)))   =   σ(a(θ − b))
```

Where:

| Symbol | Name | Meaning |
|---|---|---|
| `θ` | Student ability | Logit-scale latent ability; from IRT engine's `ThetaResult.theta` |
| `a` | Discrimination | How sharply the ICC separates knowers from non-knowers |
| `b` | Difficulty | Item difficulty on the same logit scale as θ |
| `σ` | Logistic sigmoid | `1 / (1 + exp(-x))` |

**Key properties:**
- `θ = b`: P(correct) = 0.5 exactly (student at chance level)
- `θ >> b`: P(correct) → 1 (easy item for strong student)
- `θ << b`: P(correct) → 0 (very hard item for weak student)
- Higher `a` → steeper curve → item better discriminates strong from weak students

---

### Dynamic P(G) from IRT

```
P(G)_eff = P_G_BASE × (1 − P_2PL(θ, a, b)) + P_G_FLOOR
```

where:
- `P_G_BASE = 1/N_options = 0.25` (4-option MCQ default)
- `P_G_FLOOR = P_G_BASE / N_options = 0.0625` (physical minimum)
- `P_2PL = σ(a(θ − b))` = IRT predicted probability of correct

**Why this formula works:**

| P_2PL value | Scenario | P(G)_eff |
|---|---|---|
| P_2PL → 1.0 | Easy item, strong student | → `P_G_FLOOR ≈ 0.063` (minimal) |
| P_2PL = 0.5 | Student at chance level | → `0.25 × 0.5 + 0.063 = 0.188` |
| P_2PL → 0.0 | Hard item, weak student | → `P_G_BASE = 0.25` (maximum) |

**Intuition:** When IRT already predicts the student is very likely to answer correctly, a correct answer carries almost zero "guess" component. When IRT predicts near-certain failure, any correct answer was probably a lucky guess.

Result clamped to `[0.0625, 0.45]`.

---

### Dynamic P(S) from IRT

```
P(S)_eff = P_S_BASE × (1 − P_2PL(θ, a, b)) + P_S_FLOOR
```

where:
- `P_S_BASE = 0.10` (default slip rate)
- `P_S_FLOOR = P_S_BASE / 2 = 0.05`

**Intuition:** A high-ability student answering an easy item rarely slips — their slip rate collapses toward the floor. A low-ability student on a hard item has elevated effective slip even if they nominally "know" the skill, because the difficulty creates additional error risk.

| P_2PL value | Scenario | P(S)_eff |
|---|---|---|
| P_2PL → 1.0 | Easy for strong student | → `P_S_FLOOR = 0.05` |
| P_2PL = 0.5 | At chance level | → `0.10 × 0.5 + 0.05 = 0.10` |
| P_2PL → 0.0 | Hard for weak student | → `P_S_BASE = 0.10` |

Result clamped to `[0.05, 0.35]`.

---

### Difficulty-Scaled P(T)

P(T) is scaled by the **mean IRT difficulty of all items in this concept**:

```
P(T)_scaled = DEFAULT_P_T × σ(−mean_b) / σ(0)
```

Since `σ(0) = 0.5`, this simplifies to:

```
P(T)_scaled = DEFAULT_P_T × 2 × σ(−mean_b)
```

Where `mean_b` = mean difficulty across all items tagged to this concept (from Bloom-derived IRT difficulties: Remember → −2.0, Understand → −1.0, Apply → 0.0, Analyze → 1.0, Evaluate → 2.0, Create → 2.5).

**Behaviour across Bloom levels:**

| Concept type | mean_b | σ(−mean_b) | P(T) scaling | Result |
|---|---|---|---|---|
| All-Remember | −2.0 | σ(2.0) ≈ 0.88 | ×1.76 | `0.10 × 1.76 = 0.176` ↑ faster |
| All-Apply | 0.0 | σ(0.0) = 0.50 | ×1.00 | `0.10 × 1.00 = 0.100` = default |
| All-Evaluate | 2.0 | σ(−2.0) ≈ 0.12 | ×0.24 | `0.10 × 0.24 = 0.024` ↓ slower |

**Intuition:** Harder concepts require more practice opportunities to master. The sigmoid ensures P(T) is always positive (no absorbing states) and smoothly transitions across the difficulty spectrum.

Result clamped to `[0.01, 0.50]`.

---

### IRT Cold-Start P(L₀)

When IRT diagnostic data is available, P(L₀) is initialized via **Beta-Bernoulli shrinkage** — the same formula used in the IRT engine's `mastery_initializer.py`:

```
theta_implied = mean_i [ P_2PL(θ, a_ref, b_i) ]          # mean over concept's items
observed      = n_correct / n_attempted                    # raw diagnostic accuracy

weight        = n_attempted / (n_attempted + K)            # shrinkage weight; K = 3.0
P(L₀)         = weight × observed + (1 − weight) × theta_implied
```

**The shrinkage mechanism:**

```
n_attempted = 1  →  weight = 1/4  →  90% prior, 10% observed  (don't trust 1 item)
n_attempted = 3  →  weight = 3/6  →  50% prior, 50% observed
n_attempted = 9  →  weight = 9/12 →  25% prior, 75% observed  (trust observed data)
n_attempted = ∞  →  weight → 1    →  100% observed data dominates
```

**What each term contributes:**

- `theta_implied`: what we'd expect from this student given their overall IRT ability and this concept's difficulty (monotone in θ, monotone decreasing in b)
- `observed`: what was actually seen in the diagnostic quiz for this concept specifically
- `K = 3.0`: effective sample size of the theta-implied prior — a concept probed by 1–2 questions is pulled heavily toward the theta-implied prior

Result clamped to `[0.05, 0.95]`.

---

## 6. Guess Detection

The engine includes an IRT-aware guess classifier that **detects suspiciously fast correct answers** and dampens the upward mastery update for those observations.

### Two-condition logic

A correct answer is flagged as a probable guess when **BOTH** conditions hold:

```
Condition (a): response_time_ms < RT_GUESS_THRESHOLD_MS  (= 1500ms)
Condition (b): P_2PL(θ, a, b) < IRT_SURPRISE_THRESHOLD   (= 0.30)
```

Condition (b) uses the IRT model to ask: *was this correct answer surprising?* If the model predicted only a 20% chance of success but the student answered in 0.5 seconds, it is very likely a lucky guess — not genuine mastery.

**When flagged**, the effective P(G) is inflated:

```
P(G)_flagged = DEFAULT_P_G × P_G_INFLATED_MULTIPLIER = 0.25 × 3.0 = 0.75
```

This larger P(G) makes the conditional update produce a **much smaller upward revision** — the correct answer still increases P(L), but only weakly, reflecting that we don't believe it was a genuine demonstration of knowledge.

**Fallback:** If no IRT parameters are available, condition (b) is skipped and only the response-time criterion is used.

---

## 7. Numerical Stability & Clamping

### Why P(L) can never be exactly 0 or 1

Both the Stage 1 denominator and Stage 2 formula become undefined at the boundaries:
- At `P(L) = 0`: the Stage 1 denominator for a correct answer becomes `0 + (1−0) × P(G) = P(G)`, which is fine — but the posterior `P(L|correct) = 0 / P(G) = 0`. This is an **absorbing state**: no number of correct answers can ever raise P(L) above 0.
- At `P(L) = 1`: for an incorrect answer, the denominator is `1 × P(S) + 0 = P(S)`, posterior = `P(S)/P(S) = 1`. Another absorbing state — no incorrect answer can ever lower mastery.

BKT's update equations **require that P(L) remain in the open interval (0, 1)**. We enforce this by clamping after every update:

```
P(L)_final = clamp(P(L)_new, P_L_CLAMP_MIN, P_L_CLAMP_MAX)
           = clamp(P(L)_new, 0.001, 0.999)
```

### Logistic exponent overflow prevention

The 2PL ICC computes `exp(−a(θ−b))`. For large `|a(θ−b)|`, this can overflow or underflow. We clamp the exponent before calling `exp()`:

```
z = −a(θ − b)
z_clamped = clamp(z, −35.0, +35.0)
```

`exp(35) ≈ 1.6 × 10¹⁵`, which is safely within float64 range and represents a probability effectively equal to 0 or 1 at any operationally relevant precision.

### Degeneracy guard

Before every update, the engine verifies:

```
P(G) + P(S) < 1.0
```

If this fails, a correct answer provides *no information* about mastery (because `P(correct | knows) = 1 − P(S)` ≤ `P(G) = P(correct | doesn't know)`). The update is rejected with `InvalidParameterError`.

---

## 8. Mastery Thresholding

A student is declared **mastered** in a skill when:

```
P(L_t) ≥ MASTERY_THRESHOLD   (default: 0.95)
```

The 0.95 threshold is the de-facto standard in BKT literature (Corbett & Anderson, 1994; van de Sande, 2013) and maps to:

> *"At most a 5% residual probability of not knowing the skill — acceptable for skill-advancement decisions in classroom ITS deployments."*

### Forward simulation: expected attempts to mastery

For reporting/dashboards, the engine can simulate how many all-correct answers a student would need to reach mastery from their current P(L):

```python
while P_L < threshold and step < max_steps:
    # Apply one correct update
    num   = P_L × (1 − P_S)
    denom = P_L × (1 − P_S) + (1 − P_L) × P_G
    P_cond = num / denom
    P_L    = P_cond + (1 − P_cond) × P_T
    step  += 1
```

This forward simulation uses the skill's **current** P(G), P(S), P(T) and stops as soon as `P(L) ≥ threshold`.

---

## 9. Architecture

```
bkt-engine/
├── bkt/
│   ├── config.py       ← All constants with derivation rationale. No magic numbers elsewhere.
│   ├── exceptions.py   ← BKTError hierarchy — every failure mode gets its own type.
│   ├── parameters.py   ← BKTSkillParameters dataclass + IRT→BKT bridge functions.
│   ├── state.py        ← BKTState: immutable frozen snapshot of one student × skill.
│   ├── engine.py       ← BKTEngine: pure stateless update math. Zero side effects.
│   ├── mastery.py      ← Mastery policy: threshold check, guess detection, forward sim.
│   └── __init__.py     ← Clean public API re-exports.
├── examples/
│   ├── irt_bridge.py         ← IRT→BKT formula reference + runnable demo.
│   └── mock_orchestrator.py  ← End-to-end orchestrator demo with session transcript.
└── tests/
    ├── test_engine.py        ← Core BKT math + IRT hooks + guess detection tests.
    ├── test_parameters.py    ← IRT→BKT bridge monotonicity + formula spot-checks.
    ├── test_mastery.py       ← Threshold + guess detection + forward simulation tests.
    └── test_orchestrator.py  ← End-to-end IRT→BKT integration tests.
```

### Design principles

| Principle | Implementation |
|---|---|
| **Pure math, no side effects** | `BKTEngine` is a stateless class; `update()` returns a new `BKTState` without mutating the old one |
| **No magic numbers** | Every constant in `config.py` includes its derivation formula and literature citation |
| **No silent failures** | Every error condition has its own exception type; callers can catch specifically or broadly |
| **Decoupled from IRT** | The engine has zero IRT imports; IRT values enter only through function arguments |
| **Fully auditable** | `BKTUpdateResult` records every intermediate value (`p_conditional`, `p_g_used`, `was_flagged_guess`, etc.) |

---

## 10. Quick Start

### 1. No IRT data — pure BKT with defaults

```python
from bkt import BKTEngine, defaults

params = defaults(skill_id="ohms_law")
state  = BKTEngine.initial_state("student_1", "ohms_law", params)

# Answer a question (correct)
result = BKTEngine.update(state, is_correct=True, params=params)
print(f"P(L): {result.previous_p_l:.4f} → {result.new_state.p_l:.4f}")
print(f"Mastered: {result.new_state.is_mastered}")
```

### 2. With IRT data — dynamic parameters per item

```python
from bkt import BKTEngine, from_irt, seed_p_l0_from_irt

# IRT engine output: student theta + item (a, b)
theta = 0.65    # from ThetaResult.theta
a     = 1.20    # from QuestionIRTParameters.discrimination
b     = -0.80   # from QuestionIRTParameters.difficulty

# P(L0) from mastery initializer output
p_l0 = seed_p_l0_from_irt(
    theta=theta,
    bloom_difficulties=[-1.5, -0.8, 0.0],   # concept's item b-values
    n_correct=3, n_attempted=4,
)

params = from_irt(theta=theta, b=b, a=a, p_l0=p_l0, skill_id="ohms_law")
state  = BKTEngine.initial_state("student_1", "ohms_law", params)

# Per-item IRT hooks: inject dynamic P(G), P(S) at update time
from bkt import derive_p_g_from_irt, derive_p_s_from_irt

result = BKTEngine.update(
    state, is_correct=True, params=params,
    override_p_g=derive_p_g_from_irt(theta, b, a),
    override_p_s=derive_p_s_from_irt(theta, b, a),
    response_time_ms=4500,
    theta=theta, difficulty=b, discrimination=a,
)
print(f"P(G) used: {result.p_g_used:.4f}  (skill default: {params.p_g:.4f})")
print(f"P(L): {result.previous_p_l:.4f} → {result.new_state.p_l:.4f}")
```

### 3. Batch replay of a session history

```python
from bkt import BKTEngine, defaults

params = defaults(skill_id="circuit_analysis")
state  = BKTEngine.initial_state("student_2", "circuit_analysis", params)

observations = [
    (True,  8000),   # (is_correct, response_time_ms)
    (False, 12000),
    (True,  7500),
    (True,  6200),
    (True,  5100),
]

final_state, results = BKTEngine.batch_update(state, observations, params)
print(f"Final P(L): {final_state.p_l:.4f} | Mastered: {final_state.is_mastered}")
for i, r in enumerate(results):
    print(f"  Step {i+1}: {'✓' if r.is_correct else '✗'}  P(L): {r.previous_p_l:.4f} → {r.new_state.p_l:.4f}")
```

### 4. Mastery utilities

```python
from bkt import check_mastery, mastery_gap, expected_attempts_to_mastery

p_l = 0.72
print(check_mastery(p_l))                    # False
print(f"Gap to mastery: {mastery_gap(p_l):.4f}")  # 0.2300

# How many more correct answers needed?
n = expected_attempts_to_mastery(p_l, p_t=0.10, p_g=0.25, p_s=0.10)
print(f"~{n} more correct answers to mastery")
```

### 5. Run the full orchestrator demo

```bash
cd bkt-engine
python examples/mock_orchestrator.py
```

```bash
# Run the IRT bridge formula reference
python examples/irt_bridge.py
```

---

## 11. Running Tests

```bash
cd /path/to/irt-engine
source irt-engine/venv/bin/activate
python -m pytest bkt-engine/tests/ -v
```

**Expected output:** `98 passed in 0.08s`

### Test coverage

| File | What it tests |
|---|---|
| `test_engine.py` | BKT math formulas (hand-computed spot-checks), IRT hooks, guess detection, batch threading, invalid input guards |
| `test_parameters.py` | ICC formula, P(G)/P(S)/P(T) monotonicity, cold-start seeding, `from_irt()` factory |
| `test_mastery.py` | Threshold, gap, guess detection (all condition combinations), forward simulation |
| `test_orchestrator.py` | IRT seeding ordering, per-item overrides, full session mastery, unknown concept graceful handling |

---

## 12. Parameter Derivation Reference

| Parameter | Default | Formula / Source |
|---|---|---|
| `P(L₀)` | `0.20` | Corbett & Anderson (1994) — conservative secondary-school prior |
| `P(T)` | `0.10` | Mid-point of 0.05–0.40 empirical range; Baker et al. (2008) |
| `P(G)` | `1/4 = 0.25` | `1/N_options` — uniform random-guess probability for 4-option MCQ |
| `P(S)` | `0.10` | Mid-point of 0.05–0.15 empirical range; Baker et al. (2008) |
| `MASTERY_THRESHOLD` | `0.95` | Corbett & Anderson (1994); van de Sande (2013) |
| `P(G)_floor` | `1/16 = 0.0625` | `P(G)_default / N_options` — physical minimum per-item guess floor |
| `P(S)_floor` | `0.05` | `P(S)_default / 2` — scales if default is recalibrated |
| `P(G)_inflated` | `0.75` | `P(G)_default × 3` — 3× multiplier for guess-flagged answers |
| `P(L) clamp min` | `0.001` | Prevents absorbing state at 0 |
| `P(L) clamp max` | `0.999` | Prevents absorbing state at 1 |
| `RT_GUESS_THRESHOLD_MS` | `1500` | Matches `knowledge.service.ts` in quiz-portal |
| `IRT_SURPRISE_THRESHOLD` | `0.30` | Matches `knowledge.service.ts` in quiz-portal |
| `MASTERY_PRIOR_STRENGTH` | `3.0` | Matches `config.MASTERY_PRIOR_STRENGTH` in IRT engine |
| `Logistic exponent clamp` | `35.0` | `exp(35) ≈ 1.6×10¹⁵` — safely inside float64, represents P≈0 or P≈1 |

---

## References

1. Corbett, A. T., & Anderson, J. R. (1994). *Knowledge tracing: Modeling the acquisition of procedural knowledge.* User Modeling and User-Adapted Interaction, 4(4), 253–278.
2. Baker, R. S. J. d., Corbett, A. T., & Aleven, V. (2008). *More accurate student modeling through contextual estimation of slip and guess probabilities in Bayesian Knowledge Tracing.* Proceedings of ITS 2008.
3. van de Sande, B. (2013). *Properties of the Bayesian Knowledge Tracing model.* Journal of Educational Data Mining, 5(2), 1–10.
4. Birnbaum, A. (1968). *Some latent trait models and their use in inferring an examinee's ability.* In F. M. Lord & M. R. Novick, Statistical theories of mental test scores (pp. 397–479).
5. Ebel, R. L., & Frisbie, D. A. (1991). *Essentials of Educational Measurement* (5th ed.). Prentice Hall.
