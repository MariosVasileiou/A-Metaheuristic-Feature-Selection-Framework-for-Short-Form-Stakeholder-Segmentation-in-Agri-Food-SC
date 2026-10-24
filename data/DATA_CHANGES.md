# De-identification: what was changed

The source file of the survey (the questionnaire export, 42 rows x 46 columns)
is not included in this repository. Two de-identified CSV files are derived
from it. The values of the 40 item responses are unchanged, so every
analytical result of the article reproduces exactly from the released data.

## Fields removed or transformed

| Source column               | In release            | Action |
|-----------------------------|-----------------------|--------|
| Timestamp (Χρονική σήμανση) | dropped               | Submission time is identifying; removed entirely. |
| Actor in Pilot case         | `pilot` = P1..P7      | Recoded to the codes P1 to P7; the grouping is preserved, so the leave-one-pilot-case-out analyses reproduce. The article reports some results by pilot name (Figures 1 and 13, and Section 3.8), so the codes are not a protection in themselves; see Residual risk. |
| Country of Organization     | `country` (unchanged) | Kept at country level; the country counts are reported in the article. |
| Age (integer)               | `age_band`            | Binned to `<35`, `35-44`, `45-54`, `55+`. Exact ages could identify respondents at n = 42, particularly at the ends of the age range. |
| Gender                      | `gender` (unchanged)  | Kept (binary as collected). |
| Organisation type           | `org_type`            | The three categories with a single respondent each (IT Solutions, Advisor, Engineering) pooled into `Other`; the five common types kept. |
| 40 Likert items             | `Q1`..`Q40`           | Renamed from the long survey-question headers to `Q1..Q40` in the original order; values unchanged (raw -4..+4 scale). The wording of each item is given in Appendix D of the article. |
| (none)                      | `respondent_id`       | Sequential pseudonym `R01..R42`. The source contains no direct identifiers (no names, e-mail addresses or free text), so the respondents were already pseudonymous; the row order is preserved, so the seeded repeated subsampling reproduces exactly. |

## Files

- `alliance_items.csv`: respondent_id, pilot, Q1..Q40 (drives every analytical exhibit).
- `alliance_demographics_coarsened.csv`: respondent_id, pilot, country, age_band, gender, org_type (drives Figures 1 and 2 only).
- The pilot cases appear as codes (P1 to P7) in the data files. Figures 1 and 13 of the article show the pilot names; the label file used to draw those versions of the figures is not part of the release.

## Residual risk

At n = 42, the combination of quasi-identifiers (country, age band, gender,
organisation type, pilot case) can still single out some respondents.
Coarsening reduces but does not eliminate the risk of re-identification, and
the conditions of use in `data/LICENSE` therefore prohibit any attempt to
identify individual respondents or their organisations.
