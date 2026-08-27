# Ranking and Decision Stress specification

This file records the behavior of HugSelect `v1.0.1` in implementation terms.

## Feasibility

- **Must Have:** compiled to an Elasticsearch filter. A candidate with missing
  evidence or non-matching evidence is excluded before ranking.
- **Won't Have:** compiled to an Elasticsearch `must_not` clause. A matching
  candidate is excluded; missing evidence passes because absence is not treated
  as a forbidden match.
- **Should Have / Could Have:** soft requirements. Missing evidence lowers the
  normalized match but does not make the candidate ineligible.

## Feature-match score

For each requested value `v`, the best indexed match `i` contributes:

```text
feature_match = 100 × Σv max_i(base_weight_i × priority_i × accuracy_i)
                      / Σv max_i(base_weight_i × priority_i)
```

Accuracy is `1.0` for a canonical exact match, `0.90` for a normalized gram
match, and `0.80 × confidence` for an accepted synonym match. Missing soft
evidence contributes zero to the numerator but remains possible evidence in the
denominator. Must Have and Won't Have are feasibility rules and do not receive
a soft-score weight after filtering.

Automatic extraction uses multiplier `1.4` for a strong preference and `1.0`
for a preference. Explicit Should Have and Could Have rows use `1.0` and `0.5`
respectively.

Base weights:

| Group | Feature | Weight |
|---|---|---:|
| Essential | task | 11.5 |
| Essential | domain | 10.5 |
| Essential | author | 2.5 |
| Essential | objective | 10.0 |
| Essential | model name | 8.0 |
| Preference | license | 8.0 |
| Preference | library | 1.8 |
| Preference | base model | 10.8 |
| Preference | dataset | 1.8 |
| Preference | language | 9.5 |
| Preference | metric | 1.0 |
| Functional | functional phrase | 12.0 |

Configured quality weights are Functional Suitability `2.0`; Compatibility,
Performance Efficiency, Reliability, and Flexibility `1.2`; and Interaction
Capability, Security, and Maintainability `1.0`. The released web builder sets
`enable_quality_dimensions=False`, so those stored signals are not silently
expanded into query clauses or the displayed feature-match score.

## Elasticsearch rank contribution

Elasticsearch combines feature evidence with logarithmic popularity functions:
likes use weight `0.8`, and 30-day downloads use `0.75`. Their total rank-
function boost is capped at `70`. The displayed feature-match percentage does
not include these popularity functions. Elasticsearch supplies the initial
candidate order; comparison-level ties are resolved by 30-day downloads, then
likes, then stable model ID ordering.

## Decision Stress configurations

Multipliers are ordered `(essential, preference, functional, quality)`:

| Configuration | Multipliers | Rationale |
|---|---|---|
| Starting priorities | `(1, 1, 1, 1)` | Reproduces the stored comparison evidence. |
| Essential first | `(2, 0.75, 1, 1)` | Tests stronger emphasis on core fit. |
| Preferences first | `(1, 2, 0.75, 1)` | Tests metadata and deployment preferences. |
| Functional first | `(1, 0.75, 2, 1)` | Tests stronger emphasis on capability evidence. |
| Strict essentials | `(2, 0.75, 0.75, 1)` plus exclusion | Tests a hard essential boundary. |

These are sensitivity scenarios, not new searches or benchmark runs.
