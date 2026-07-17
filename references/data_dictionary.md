# Data Dictionary

This dictionary separates source documentation from project interpretation. Definitions and documented codes come from the UCI dataset page and original paper. Observed ranges and undocumented codes come from the official workbook downloaded on 2026-07-17.

## Target and identifier

| Raw field | Project field | Role | Definition | Observed values | Modeling treatment |
|---|---|---|---|---|---|
| `ID` | `customer_id` | Identifier | Row/client identifier | 1 to 30,000; unique | Excluded from model features |
| `default payment next month` | `default_next_month` | Target | Default payment in the following month | 0: 23,364; 1: 6,636 | Binary target; 1 is the adverse event |

The observed event rate is 22.12%. The target month follows the latest September 2005 account history, but the dataset does not provide a reusable sequence of monthly target snapshots.

## Customer and account attributes

| Raw field | Project field | Source definition | Documented codes / unit | Observed values or range | Availability and caveats |
|---|---|---|---|---|---|
| `LIMIT_BAL` | `credit_limit` | Granted credit, including individual and supplementary family credit | New Taiwan dollar | 10,000 to 1,000,000 | Existing-account attribute; may reflect earlier underwriting decisions |
| `SEX` | `sex` | Gender | 1 = male; 2 = female | 1, 2 | Sensitive demographic field; report subgroup performance and avoid causal claims |
| `EDUCATION` | `education` | Education level | 1 = graduate school; 2 = university; 3 = high school; 4 = others | 0, 1, 2, 3, 4, 5, 6 | Codes 0, 5, 6 are undocumented; retain and flag pending an explicit recoding policy |
| `MARRIAGE` | `marital_status` | Marital status | 1 = married; 2 = single; 3 = others | 0, 1, 2, 3 | Code 0 is undocumented; retain and flag |
| `AGE` | `age` | Age | Years | 21 to 79 | Sensitive demographic field; application-time availability is plausible |

## Repayment status history

| Raw field | Project field | Month |
|---|---|---|
| `PAY_0` | `repayment_status_sep` | September 2005 |
| `PAY_2` | `repayment_status_aug` | August 2005 |
| `PAY_3` | `repayment_status_jul` | July 2005 |
| `PAY_4` | `repayment_status_jun` | June 2005 |
| `PAY_5` | `repayment_status_may` | May 2005 |
| `PAY_6` | `repayment_status_apr` | April 2005 |

UCI documents `-1` as paid duly and positive integers as months of delay. The workbook actually contains `-2`, `-1`, `0`, and positive values up to `8`; `PAY_5` and `PAY_6` do not contain `1`. The meanings of `-2` and `0` are not defined on the UCI page, so the project preserves them as distinct observed categories and flags the documentation gap.

These variables are historical account behavior. They are appropriate for predicting an existing client's next-month payment outcome, but generally unavailable for a new-to-bank applicant.

## Bill statement amounts

| Raw field | Project field | Month | Observed range (NT$) |
|---|---|---|---:|
| `BILL_AMT1` | `bill_amount_sep` | September 2005 | -165,580 to 964,511 |
| `BILL_AMT2` | `bill_amount_aug` | August 2005 | -69,777 to 983,931 |
| `BILL_AMT3` | `bill_amount_jul` | July 2005 | -157,264 to 1,664,089 |
| `BILL_AMT4` | `bill_amount_jun` | June 2005 | -170,000 to 891,586 |
| `BILL_AMT5` | `bill_amount_may` | May 2005 | -81,334 to 927,171 |
| `BILL_AMT6` | `bill_amount_apr` | April 2005 | -339,603 to 961,664 |

Negative balances are retained. They may reflect credits, overpayments, refunds, or accounting adjustments; the source does not provide enough information to label them data errors.

## Previous payment amounts

| Raw field | Project field | Month | Observed range (NT$) |
|---|---|---|---:|
| `PAY_AMT1` | `payment_amount_sep` | September 2005 | 0 to 873,552 |
| `PAY_AMT2` | `payment_amount_aug` | August 2005 | 0 to 1,684,259 |
| `PAY_AMT3` | `payment_amount_jul` | July 2005 | 0 to 896,040 |
| `PAY_AMT4` | `payment_amount_jun` | June 2005 | 0 to 621,000 |
| `PAY_AMT5` | `payment_amount_may` | May 2005 | 0 to 426,529 |
| `PAY_AMT6` | `payment_amount_apr` | April 2005 | 0 to 528,666 |

All observed payment amounts are non-negative. Like repayment status and bills, these fields describe existing account behavior rather than a pure application snapshot.

## Initial modeling inclusion policy

- Exclude `customer_id` from predictors.
- Retain all 23 explanatory variables during audit and baseline stages.
- Do not silently map undocumented category codes to missing values or documented groups.
- Treat repayment statuses and demographic codes as categorical unless an explicitly tested transformation justifies another representation.
- State whether each experiment represents existing-customer behavioral risk or a hypothetical approval decision; do not conflate the two.
