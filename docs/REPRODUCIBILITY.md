# Methods and reproducibility

## Research question

Does a judgment about the reliability of an LLM's own confidence add useful information about whether its answer is correct, beyond ordinary answer reassessment?

The study tests the predictive utility of elicited judgments. It does not directly identify an internal higher-order mechanism, measure privileged access to hidden states, or demonstrate improved real-world agent decisions.

## Signals and comparisons

Each item has a frozen answer and original probability `p`. Three separate review calls receive the question, options, frozen answer and original confidence, but no answer key or other review output:

- `q`: revised probability that the frozen answer is correct.
- `q2`: a second call with the same reassessment prompt.
- `r`: HIGH (1) or LOW (0) reliability of the original probability judgment.

All are independent requests to the same model identifier. This does not imply independence of reasoning, especially at temperature zero. The reliability prompt says explicitly that a low probability can be reliable and a high probability unreliable. See [the prompt examples](PROMPTS.md).

Five ridge-logistic scoring models use original confidence and domain indicators, with added signals and interactions as specified in the frozen protocol. Scaling and coefficients are estimated using only the 480 calibration items. The 160 development items informed exploratory development. The final 1,600-item evaluation uses the unchanged fitted mappings.

| Role | Comparison | Interpretation of a negative Brier difference |
|---|---|---|
| Primary | `p_r − p_q` | Reliability review improves on ordinary reassessment at equal review-call count |
| Matched two-call check | `p_q_r − p_q_q2` | Mixed reviews improve on repeated reassessment |
| Secondary | `p_q_r − p_q` | Reliability adds information to reassessment, with an extra call |
| Secondary | `p_r − p` | Reliability adds information beyond original confidence |

Brier score is mean squared probability error; lower is better. These are comparisons between fitted prediction rules, not raw confidence statements. The main intervals use 5,000 paired evaluation-item bootstrap samples stratified by domain, conditional on frozen scoring models. The post-hoc sensitivity analysis separately resamples calibration and evaluation items within domain and refits the original models in 2,000 replicates. It addresses calibration-sample variability, not generation variability. Secondary findings do not rescue the unsuccessful primary comparison.

## Data dictionary

| File | Contents |
|---|---|
| `results/pilot_records.json` | 480 calibration and 160 development observations |
| `results/evaluation_records.json` | 1,600 evaluation observations |
| `results/frozen_models.json` | Five frozen scoring models, coefficients and scalers |
| `results/evaluation_results.json` | Original scores, intervals, predictions, missingness and checks |
| `results/refit_results.json` | Original post-hoc refit sensitivity results |
| `results/pilot_protocol.json` | Recorded item-level pilot protocol |
| `results/evaluation_protocol.json` | Recorded evaluation protocol and source hashes |

Record fields: `id` is the stable study item identifier; `domain` is the subject; `split` identifies calibration, development or evaluation; `p`, `q`, `q2` are probabilities in [0,1]; `r` is binary reliability; `y` is binary correctness. Evaluation records also retain the selected option letter as `answer`. An individual error is not itself evidence that a probability was miscalibrated.

The dataset is MMLU-Pro at revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`. The original preparation logic and option-order safeguards are in `auditstudy/data.py`. No question text or model reasoning transcript is required to reproduce the saved-signal analysis.

## Running the release

Use the root README commands. `reproduce_results.py` is a release wrapper around the original pure analysis functions. It verifies the packaged file hashes and protocol bindings, then checks reproduced Brier/log-loss scores and bootstrap intervals against the original outputs. `--refit` additionally checks the broader uncertainty intervals. `--groups` regenerates descriptive subgroup results. All generated files go to the ignored `reproduced/` directory.

The original source files remain unchanged so that recorded protocol hashes can be checked. Dependency versions in `requirements.txt` match the verified local numerical environment. Tiny cross-platform numerical differences may remain; the verification tolerances are explicit in the wrapper.

## Collection versus analysis reproduction

This release supports offline reproduction of the completed experiment's numerical analysis. It does not include the local task databases, raw API responses or provider credentials needed to resume the historical collection job. The original collection modules are included for methodological transparency, not as a fresh one-command collection workflow.

The paper's appendix describes the original workspace command `python -m diagnostics.item_confidence_evaluation --score-only`, which requires those original staged databases. In this trimmed repository, use `python reproduce_results.py` instead. `config/study.json` is the original shared pilot configuration; the final evaluation overrides and frozen settings are recorded in `results/evaluation_protocol.json`.

## Provenance and limits

`release_manifest.json` hashes each shipped file except itself. It is an integrity inventory, not an independent timestamp or preregistration service. Protocol files were recorded locally; they do not imply external preregistration.

The source protocols, records and result JSON files are copied unchanged from the study. The release README, documentation and offline wrapper were added afterward. No new model responses were generated for this release, and evaluation data were not used to tune the published scoring models.

The report states the central limits: one model alias, possible pretraining contamination, limited subjects and prompt variants, and an inability to distinguish confidence-reliability awareness from prompt-induced answer re-evaluation. The post-hoc analysis remains explicitly post hoc.
