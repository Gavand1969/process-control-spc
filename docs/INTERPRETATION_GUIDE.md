# SPC Interpretation Guide

This guide explains how to read the output of this toolkit the way a quality engineer would: what each number means, when to act on it, and what the textbook thresholds imply about a process.

Targeted at: hiring managers, process engineers, and analysts new to SPC who want to evaluate the output of this project without first reading Montgomery.

---

## 1. Capability indices — what the numbers mean

| Index | Formula | Tells you about |
|-------|---------|-----------------|
| **Cp**  | (USL − LSL) / (6 σ<sub>within</sub>)                                | Short-term *potential* — how tight the process *could* be if perfectly centered. Ignores mean. |
| **Cpk** | min((USL − μ)/(3 σ<sub>within</sub>), (μ − LSL)/(3 σ<sub>within</sub>)) | Short-term *actual* — penalizes off-centering. |
| **Pp**  | (USL − LSL) / (6 σ<sub>overall</sub>)                                | Long-term potential — uses pooled std. dev., includes between-subgroup drift. |
| **Ppk** | min((USL − μ)/(3 σ<sub>overall</sub>), (μ − LSL)/(3 σ<sub>overall</sub>)) | Long-term actual — what the customer experiences. |

σ<sub>within</sub> = R̄ / d₂ &nbsp;·&nbsp; σ<sub>overall</sub> = sample std. dev. (ddof=1)

### Industry thresholds

| Index value | Verdict                        | Typical context                          |
|-------------|--------------------------------|------------------------------------------|
| < 1.00      | Inadequate — process produces defects | Immediate corrective action required |
| 1.00 – 1.33 | Marginal — 100% inspection often needed | Use only when no alternative exists |
| 1.33 – 1.67 | Good — typical AIAG acceptance        | Standard for ongoing production     |
| 1.67 – 2.00 | Excellent — "Six Sigma" target        | World-class manufacturing            |
| ≥ 2.00      | Exceptional                           | Aerospace / medical-device tier      |

### Reading the gap

| Comparison         | Interpretation                                                       |
|--------------------|----------------------------------------------------------------------|
| **Cp ≈ Cpk**       | Process is centered.                                                 |
| **Cp ≫ Cpk**       | Process is off-center — adjust the mean before tightening variance.  |
| **Cp ≈ Pp**        | Process is stable — within- and between-subgroup variation match.    |
| **Cp ≫ Pp**        | Special-cause variation is inflating long-term σ — investigate drift, tool wear, shift effects. |
| **Cpk ≥ 1.33, Ppk < 1.33** | Process is *capable* short-term but *not performing* long-term. Stabilize first, then center. |

> **In this toolkit's output**: Line B shows Cp=1.40, Pp=1.21. That gap is the deliberate quality event — the days-18-22 mean shift inflates σ<sub>overall</sub> without affecting σ<sub>within</sub> proportionally.

---

## 2. X-bar & R control limits — what counts as "control"

**Control limits are not specification limits.** They describe the *voice of the process*; spec limits describe the *voice of the customer*.

For subgroup size n=5:

```
UCL_x̄ = X̄̄ + A₂ · R̄    with A₂ = 0.577
LCL_x̄ = X̄̄ − A₂ · R̄
UCL_R = D₄ · R̄         with D₄ = 2.114
LCL_R = D₃ · R̄         with D₃ = 0 (for n ≤ 6)
```

Source: Montgomery, *Introduction to Statistical Quality Control*, 8th ed., Table VI.

A point inside the control limits is "in statistical control." A point outside means a special cause is likely present — even if the point is still inside the spec limits.

---

## 3. Western Electric rules — when to investigate

Beyond a single point exceeding ±3σ, eight pattern-based rules flag *non-random* behavior. All eight live in `src/rules.py`.

| Rule | Pattern                                          | Real-world interpretation                   |
|------|--------------------------------------------------|---------------------------------------------|
| 1    | 1 point beyond ±3σ                               | Major upset, measurement error, gross shift |
| 2    | 2 of 3 consecutive beyond ±2σ, same side         | Sustained mean shift starting               |
| 3    | 4 of 5 consecutive beyond ±1σ, same side         | Gradual drift / trend taking hold           |
| 4    | 8 consecutive same side of centerline            | Persistent recentering — new normal mean    |
| 5    | 6 consecutive trending up or down                | Tool wear, gradual fixture loosening, temperature creep |
| 6    | 15 consecutive within ±1σ                        | Stratification — subgroups mixing different streams, or fake/over-smoothed data |
| 7    | 14 consecutive alternating up-down               | Over-control — operator adjusting at every reading |
| 8    | 8 consecutive beyond ±1σ, both sides             | Mixture of two distributions — two machines, two material lots |

Rules 1–5 are applied by default in the generated report; rules 6–8 are available via the `rules=` argument to `detect_violations`.

---

## 4. Reading the included charts

### X-bar chart (`xbar_r_line_b.png`)
- Red triangles mark WECO violations. Cluster of triangles near subgroups 97–188 = the days 18–22 drift on Line B.
- Centerline (CL) is the grand mean of subgroup means, **not** the engineering target.
- The dashed orange lines are spec limits (USL/LSL), shown for context — they live in a different mathematical space than UCL/LCL.

### R chart (lower panel)
- Watches *within-subgroup variation*. Should be in control independently of the X-bar chart.
- An R-chart out-of-control while X-bar is in-control means **piece-to-piece variability** changed — often a fixture issue or measurement system problem.

### Histogram with spec limits
- Visual gut-check for normality and centering.
- The shaded red bands beyond USL/LSL highlight the rejection zone. If the bell tail extends into shading, your process is leaking defects.

### Capability gauge
- Color-coded by AIAG bands (red < 1.00, yellow 1.00–1.33, teal 1.33–1.67, green ≥ 1.67).
- Compare Cp vs Cpk (centering gap) and Cp vs Pp (stability gap) at a glance.

### Pareto of defect causes
- Top bars + cumulative line crossing 80% identify the "vital few" causes responsible for the majority of defects.
- This drives the corrective-action prioritization in the auto-generated control plan.

---

## 5. Action workflow for a quality engineer

1. **Look at Ppk first.** This is what the customer experiences.
2. **If Ppk < 1.33**, compare to Cpk:
   - If `Cpk ≈ Ppk` → variability is the issue → reduce common-cause variation (process redesign, better tooling).
   - If `Cpk ≫ Ppk` → stability is the issue → eliminate special causes (find what's drifting).
3. **Then look at Cp vs Cpk:**
   - If `Cp ≈ Cpk` → centered, focus on variance reduction.
   - If `Cp ≫ Cpk` → off-center, adjust the mean (often cheaper than reducing variance).
4. **Check control charts for OOC signals.** WECO rules will point you to *when* the shift started — overlay with shift logs, material lots, operator changes.
5. **Run the Pareto** to prioritize root-cause investigations.

---

## References

- Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8th ed. Wiley. Tables VI, VII.
- AIAG (2005). *Statistical Process Control (SPC) Reference Manual*, 2nd ed.
- Western Electric Company (1956). *Statistical Quality Control Handbook*.
- Wheeler, D. (2010). *Understanding Statistical Process Control*, 3rd ed., SPC Press.
