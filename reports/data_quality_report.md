# Data Quality Report

Generated from the verified UCI workbook by `credit_risk.data`.

## Structural checks

- Parsed shape: 30,000 rows and 25 columns.
- Unique customer IDs: 30,000.
- Exact duplicate rows including ID: 0.
- Duplicate rows after dropping ID: 35. These are not automatically removed because distinct IDs can represent different clients with identical observed values.
- No missing cells were detected after parsing the second header row.

## Target

- Non-default (`0`): 23,364.
- Default next month (`1`): 6,636.
- Observed default rate: 22.12%.

The target is moderately imbalanced. Accuracy alone is not an adequate evaluation metric; later stages will include ROC-AUC, PR-AUC, KS, Brier score, and calibration.

## Undocumented source codes

- `education`: [0, 5, 6].
- `marital_status`: [0].
- Repayment status fields:
  - `repayment_status_sep`: [-2, 0]
  - `repayment_status_aug`: [-2, 0]
  - `repayment_status_jul`: [-2, 0]
  - `repayment_status_jun`: [-2, 0]
  - `repayment_status_may`: [-2, 0]
  - `repayment_status_apr`: [-2, 0]

These values pass the project schema because they occur in the official artifact, but the UCI description does not define them. They remain distinct until a documented and tested recoding decision is made.

## Range observations

- Credit limits and ages are positive in all records.
- Previous payment amounts are non-negative in all records.
- Bill statement amounts include negative values. They are retained because credits or accounting adjustments are plausible and the source does not establish that they are errors.

## Leakage and availability review

- `customer_id` is an identifier and must not be used as a predictor.
- The target is the following month's default event; it is not an input feature.
- Billing, payment, and repayment-status fields precede the target and are not direct target leakage for this account-level prediction task.
- Those behavioral fields are generally unavailable for new-to-bank applicants. Results must be described as existing-customer behavioral risk, not as a faithful reconstruction of a new application decision.

## Validation limitation

The dataset does not expose repeated target snapshots or a reliable development/validation time axis. A stratified random split may be used later, but it must not be labeled out-of-time validation.
