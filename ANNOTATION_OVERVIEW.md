# TMC Project: Longitudinal Radiology Evidence System

**Date:** 2026-06-07
**Status:** Annotation pilot ready. Seeking reviewer.

---

## 1. What We Currently Have

### Pipeline

A working pipeline that ingests radiology reports from MIMIC-IV and produces a structured fact graph:

```
Radiology report
  → Rule + LLM extraction (individual mode, per-report)
  → Entity grounding (RadLex ontology, 3-tier: exact → fuzzy → LLM)
  → Append-only FactStore (SQLite + JSON)
  → Entity normalizer (dedup)
  → Status derivation
```

**What works well:**
- Event-level extraction: 285 events from 37 reports for patient 10000935, with dates, report IDs, negation flags, evidence text, and character offsets into the source report
- Incremental architecture: new reports can be appended without re-running old ones
- Evidence provenance: 89.5% of events have valid character spans into the source text

**What needs human review before it can be trusted:**
- Entity status logic (historically used `all()` check, now fixed to latest-event but needs validation)
- Entity identity (cross-anatomy collapse — e.g., all "fracture" across body parts merged into one entity)
- RadLex grounding (~40% of labels are semantically imprecise)
- Negation handling (event-level is correct, entity-level needs review)

### Annotation UI

A review workbench at `http://localhost:5173/` that displays:

- **Source report text** — the full radiology report for context
- **Extracted evidence** — the specific sentence the AI used
- **AI judgment** — what the system claims (present/absent, entity name, site, RadLex label)
- **Annotation form** — fields for the reviewer to correct each AI judgment

The UI has 120 events prepared from patient 10000935, filtered to key reports (early baseline 1-6, oncologic transition 24-31, late follow-up 37).

---

## 2. What Annotation We Need

### The Annotation Task

A reviewer (initially not necessarily a radiologist — any medically literate person can do the first pass) examines each AI-extracted event and answers:

| Field | What the reviewer decides |
|--------|--------------------------|
| **Review decision** | Is the finding present, absent, uncertain, or not a real finding? |
| **Corrected entity name** | Is the AI's entity name correct? If not, what should it be? |
| **Anatomical site** | Is the site/location correct? |
| **Laterality** | Is the side (left/right/bilateral) correct? |
| **Same as prior entity** | Is this the same lesion/finding as a previously seen entity? |
| **Temporal change** | Has this finding changed since the last report? (new/worsened/improved/resolved) |
| **Error type** | What did the AI get wrong? (context bleed, wrong negation, wrong grounding, missing evidence, over-fragmentation) |
| **Confidence** | How confident is the reviewer in their judgment? |

### What We're Measuring

Not "is the AI correct?" but specific, actionable metrics:

| Metric | What it answers |
|--------|----------------|
| **Evidence validity** | Is the extracted event supported by the source sentence? |
| **Negation accuracy** | Did the AI preserve present/absent correctly? |
| **Site/laterality accuracy** | Is anatomy specific enough for clinical review? |
| **Entity persistence** | Did the AI link the same finding across time? |
| **Current-status accuracy** | Is the entity rollup clinically defensible? |
| **Grounding safety** | Is the RadLex label correct enough to display? |

### Pilot Scope

- **1 patient:** 10000935 (37 radiology reports over 5.5 years)
- **10-20 reports:** key transition points (early baseline, cancer diagnosis, late follow-up)
- **120 events:** prioritized by clinical importance (metastases, nodules, fractures, effusions, obstructions, important negatives)
- **Time:** 60-120 minutes for a first-pass reviewer

---

## 3. Why We Need This

### The Problem We're Solving

A cancer patient accumulates dozens of radiology reports over years. Each report contains 5-20 findings — nodules, effusions, fractures, metastases, negatives. Tracking what changed between reports requires reading all of them sequentially and manually cross-referencing.

**The goal:** When a new report arrives, the system should show a doctor:
- What's new since the last report
- What's changed (bigger/smaller/resolved)
- What was negated (ruled out)
- Links to the source evidence sentences

### Why We Need Human Annotation First

Three layers of the current pipeline are not trustworthy enough to generate this autonomously:

1. **Entity status (was broken, now fixed but unvalidated).** Prior logic kept entities "active" forever if they were ever positive. A bowel obstruction surgically treated 4 years ago appeared as "active" because the condition `all(e.is_negated)` required every historical event to be negated. Fixed to use latest-date logic, but needs human verification.

2. **Entity identity (cross-anatomy collapse).** "No acute fracture" from a knee x-ray, "acute fracture of the lateral malleolus" from an ankle, and "surgical fracture of the right sixth rib" from a chest x-ray all merged into one entity called `finding_acute_fracture`. The system can extract correctly at the event level but merges incorrectly at the entity level.

3. **RadLex grounding (fuzzy matching noise).** The ontology grounder maps clinical terms to RadLex IDs using TF-IDF at threshold 0.82 across 46,000 concepts. This produces errors like "fibular fracture" → "peroneal vein" or "disc space narrowing" → "widened disk space sign" (opposite concept).

### What We Cannot Fix Without Annotation

- **Grounding accuracy:** We can raise fuzzy thresholds, but we cannot validate whether labels are clinically correct without a human looking at them.
- **Entity identity:** We can add anatomy-aware keys, but we cannot verify whether "the same lesion" was correctly linked across time without a human tracing the timeline.
- **Status correctness:** We can fix the derivation logic, but we cannot verify whether the output is clinically defensible without a human reviewing the latest evidence.

---

## 4. Long-Term Goal

### Phase 1: Review Accelerator (now)

```
New report → AI extracts all findings → Doctor reviews and corrects → Summary updates
```

The system does the grunt work (find every finding, date, measurement, negation). The doctor spends 10-15 minutes reviewing and correcting. The output is a reviewed, trustworthy timeline.

### Phase 2: Learning from Annotations (requires 100+ annotated events)

Each annotation is training data:

- **Corrected entity names** → improve the extraction prompt with few-shot examples
- **Corrected RadLex labels** → patch the grounding layer with hard overrides
- **Error type patterns** → auto-flag high-risk findings before human review
- **Per-modality accuracy** → show the doctor a confidence estimate per finding

The more annotations accumulate, the less correction is needed.

### Phase 3: Autonomous Summary (requires 500+ annotated events, fine-tuned model)

```
New report → AI extracts with high confidence → auto-approved findings appear immediately
           → Low-confidence findings go to review queue
           → Doctor reviews only the borderline cases (~15% of events)
```

The extraction model is fine-tuned on annotated data. Routine findings (clear measurements, unambiguous negatives) are auto-approved. Edge cases (borderline masses, ambiguous language, cross-report lesion linking) still go to review.

---

## 5. What We Already Know (Pre-Pilot Findings)

### Paired Mode: Not the Answer

We tested a paired extraction mode (processing reports as consecutive pairs for temporal consistency). After fixing the evaluation metric (argument-order bug) and adding current-report evidence constraints:

- Paired mode produced 59% noise (context bleed + over-fragmentation)
- Post-constraint paired mode LOST 34% of genuine findings compared to individual mode
- **Decision: Individual extraction is the default.** Longitudinal linking will be done as a separate step, not inside the extraction prompt.

### Status Logic: Fixed, Needs Validation

The `all(e.is_negated)` bug is fixed. Status is now derived from the latest date's event group. Pre-fix: 6 entities stuck at "active" with fully-negated latest events. Post-fix: 0 contradictions. Needs human review to verify the new logic produces clinically defensible states.

### Evidence Text: Now Available

All 283/285 events in the individual fact graph now have stored evidence text. The annotation UI shows it alongside the source report.

---

## 6. How to Start

1. Go to **http://localhost:5173/**
2. Click **Review** in the nav
3. Select **10000935_pilot** (120 events) or **10000935_self_test** (5 events)
4. For each event:
   - Read the source report (expandable at top)
   - Check the evidence sentence shown by the AI
   - Fill in the annotation form (2-3 minutes per event for first few, ~30 seconds once you get the rhythm)
5. Click **Save annotation** — saved as JSON in `outputs/annotation_pilot/annotations/`

The metrics dashboard will auto-generate from saved annotations: `docs/DEEP_AUDIT/13_HUMAN_PILOT_RESULTS.md`

---

## Summary

We are not building an autonomous radiologist. We are building a system that:
1. Extracts every finding from every report (automated, works today)
2. Shows a reviewer exactly where each finding came from (works today)
3. Learns from reviewer corrections to get better over time (annotation → training data)
4. Eventually auto-approves routine findings and only escalates edge cases (long-term)

The annotation you do today is the training data that makes phase 4 possible tomorrow.
