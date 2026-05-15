# Part 3 — Conflict-Resolving RAG: Sample Answers

Three queries against the same index, phrased differently to probe how retrieval and the synthetic seed interact. The `syn` column flags chunks from the synthetic sister seed (`synthetic_seed`); rows without `Y` are natural turns from the multi-persona corpus.

## Query 1 — spec-literal phrasing (natural retrieval; surfaces real corpus contradictions)

_Synthetic chunks in top-5: **0**_

### Query: `Did I mention anything about my sister?`

**Entity:** your sister  
**Confidence:** 0.428  
**Contradictions flagged:** 3

**Answer:**

> You've mentioned your sister across 5 occasions over day-buckets 3–6. Earlier (day-bucket 4): "Yes, I have two sisters. We're all pretty close, and we're really helping each other through this.". Later (day-bucket 5): "I have one sister. We're not very close, but I still love her.". Note: these accounts appear inconsistent on factual details.

**Top source chunks (after composite re-rank):**

| # | id | day | sent | cos | cosN | recN | emo | score | syn | excerpt |
|--:|----|----:|-----:|----:|-----:|-----:|----:|------:|:---:|---------|
| 1 | `c7659_t6` | 4 | +0.866 | 0.520 | 1.000 | 0.667 | 0.866 | **0.873** |   | User 1: I'm sure she will. Do you have any siblings?  /  User 2: Yes, I have two sisters. We're all pretty close, and we're really helping each other through t… |
| 2 | `c8252_t8` | 5 | +0.953 | 0.487 | 0.533 | 0.833 | 0.953 | **0.707** |   | User 1: I have one sister. We're not very close, but I still love her.  /  User 2: I'm sorry to hear that. I'm glad you still love her, though. |
| 3 | `c8988_t6` | 5 | +0.402 | 0.502 | 0.751 | 0.833 | 0.402 | **0.706** |   | User 1: Do you have any siblings?  /  User 2: Yes, I have two older sisters. |
| 4 | `c4911_t16` | 3 | +0.886 | 0.494 | 0.629 | 0.500 | 0.886 | **0.642** |   | User 1: Yes, I have a younger sister. She's a lot of fun to travel with.  /  User 2: That's great! I'm an only child, so I've always wanted a sibling. |
| 5 | `c9987_t10` | 6 | +0.735 | 0.460 | 0.138 | 1.000 | 0.735 | **0.516** |   | User 1: That sounds like a lot of fun. I did not have any siblings when I was growing up.  /  User 2: I am sorry to hear that. Do you have any pets? |

**NLI-flagged contradictions:**

| a_id | b_id | a_day | b_day | a_sent | b_sent | contradiction_p |
|------|------|------:|------:|-------:|-------:|----------------:|
| `c7659_t6` | `c8252_t8` | 4 | 5 | +0.866 | +0.953 | **0.993** |
| `c4911_t16` | `c9987_t10` | 3 | 6 | +0.886 | +0.735 | **0.858** |
| `c8988_t6` | `c9987_t10` | 5 | 6 | +0.402 | +0.735 | **0.772** |


## Query 2 — recency-cued phrasing (may pull synthetic late-bucket seed forward)

_Synthetic chunks in top-5: **1**_

### Query: `What did I say about my sister recently?`

**Entity:** your sister  
**Confidence:** 0.431  
**Contradictions flagged:** 1

**Answer:**

> You've mentioned your sister across 5 occasions over day-buckets 1–5. Earlier (day-bucket 4): "Yes, I have two sisters. We're all pretty close, and we're really helping each other through this.". Later (day-bucket 5): "I have one sister. We're not very close, but I still love her.". Note: these accounts appear inconsistent on factual details.

**Top source chunks (after composite re-rank):**

| # | id | day | sent | cos | cosN | recN | emo | score | syn | excerpt |
|--:|----|----:|-----:|----:|-----:|-----:|----:|------:|:---:|---------|
| 1 | `c8252_t8` | 5 | +0.953 | 0.505 | 0.772 | 0.833 | 0.953 | **0.827** |   | User 1: I have one sister. We're not very close, but I still love her.  /  User 2: I'm sorry to hear that. I'm glad you still love her, though. |
| 2 | `syn_sister_0` | 1 | +0.950 | 0.526 | 1.000 | 0.167 | 0.950 | **0.740** | Y | User 1: my sister called today, we talked for almost an hour. she's honestly amazing — i feel so lucky to have her in my life.  /  User 2: that's beautiful — s… |
| 3 | `c8462_t12` | 5 | +0.924 | 0.490 | 0.604 | 0.833 | 0.924 | **0.737** |   | User 1: That sounds beautiful. I'm sure your sister loved it.  /  User 2: She did! She said it was the best dress she's ever worn. |
| 4 | `c8788_t6` | 5 | +0.948 | 0.478 | 0.468 | 0.833 | 0.948 | **0.674** |   | User 1: That's great news! I'm glad she's doing better.  /  User 2: Thanks! I'm also glad that I have my two sisters. They're always there for me. |
| 5 | `c7659_t6` | 4 | +0.866 | 0.482 | 0.512 | 0.667 | 0.866 | **0.629** |   | User 1: I'm sure she will. Do you have any siblings?  /  User 2: Yes, I have two sisters. We're all pretty close, and we're really helping each other through t… |

**NLI-flagged contradictions:**

| a_id | b_id | a_day | b_day | a_sent | b_sent | contradiction_p |
|------|------|------:|------:|-------:|-------:|----------------:|
| `c8252_t8` | `c7659_t6` | 5 | 4 | +0.953 | +0.866 | **0.983** |


## Query 3 — temporal/relationship phrasing (closest match to synthetic affective chunks; resolver demo under controlled contradiction)

_Synthetic chunks in top-5: **0**_

### Query: `Tell me about my relationship with my sister over time`

**Entity:** your sister  
**Confidence:** 0.435  
**Contradictions flagged:** 2

**Answer:**

> You've mentioned your sister across 5 occasions over day-buckets 4–6. Earlier (day-bucket 4): "Yes, I have two sisters. We're all pretty close, and we're really helping each other through this.". Later (day-bucket 5): "That's awesome! I have two sisters and we're not as close as we used to be.". Note: these accounts appear inconsistent on factual details.

**Top source chunks (after composite re-rank):**

| # | id | day | sent | cos | cosN | recN | emo | score | syn | excerpt |
|--:|----|----:|-----:|----:|-----:|-----:|----:|------:|:---:|---------|
| 1 | `c8172_t14` | 5 | +0.685 | 0.543 | 1.000 | 0.833 | 0.685 | **0.887** |   | User 1: That's awesome! I have two sisters and we're not as close as we used to be.  /  User 2: I'm sorry to hear that. But I'm sure you can still find ways to… |
| 2 | `c8252_t8` | 5 | +0.953 | 0.514 | 0.744 | 0.833 | 0.953 | **0.812** |   | User 1: I have one sister. We're not very close, but I still love her.  /  User 2: I'm sorry to hear that. I'm glad you still love her, though. |
| 3 | `c7659_t6` | 4 | +0.866 | 0.478 | 0.425 | 0.667 | 0.866 | **0.586** |   | User 1: I'm sure she will. Do you have any siblings?  /  User 2: Yes, I have two sisters. We're all pretty close, and we're really helping each other through t… |
| 4 | `c10117_t6` | 6 | +0.974 | 0.437 | 0.067 | 1.000 | 0.974 | **0.528** |   | User 1: That's so nice of you! I'm sure your siblings appreciate that. I'm glad you have a good relationship with your siblings.  /  User 2: Thank you! I am gl… |
| 5 | `c8988_t6` | 5 | +0.402 | 0.470 | 0.353 | 0.833 | 0.402 | **0.507** |   | User 1: Do you have any siblings?  /  User 2: Yes, I have two older sisters. |

**NLI-flagged contradictions:**

| a_id | b_id | a_day | b_day | a_sent | b_sent | contradiction_p |
|------|------|------:|------:|-------:|-------:|----------------:|
| `c8172_t14` | `c7659_t6` | 5 | 4 | +0.685 | +0.866 | **0.990** |
| `c8252_t8` | `c7659_t6` | 5 | 4 | +0.953 | +0.866 | **0.983** |

