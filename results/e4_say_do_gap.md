# E4 say-do gap: verbal corruption detection vs score consequence

Share of reviews whose text explicitly complains about corruption-class
problems (regex over summary/strengths/weaknesses/questions), per arm and
variant. `base` is the false-positive rate of the same regex.

| arm | base | corrupt_numbers | shuffle_paragraphs | typos | pad_filler | drop_appendix |
|---|---|---|---|---|---|---|
| claude-opus-5-5 | 13/75 (17%) | 53/75 (71%) | 25/75 (33%) | 20/75 (27%) | 18/75 (24%) | 12/69 (17%) |
| deepseek-chat | 5/73 (7%) | 33/70 (47%) | 15/70 (21%) | 18/69 (26%) | 13/70 (19%) | 4/64 (6%) |
| deepseek-reasoner | 8/75 (11%) | 27/75 (36%) | 14/75 (19%) | 15/75 (20%) | 15/75 (20%) | 9/69 (13%) |
| gemini-3.8-flash | 1/50 (2%) | 15/50 (30%) | 4/49 (8%) | 0/50 (0%) | 3/50 (6%) | 0/46 (0%) |
