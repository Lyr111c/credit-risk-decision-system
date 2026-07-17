# Data

## Dataset

- Name: Default of Credit Card Clients
- Creator: I-Cheng Yeh
- Provider: UCI Machine Learning Repository
- Dataset page: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- Dataset DOI: <https://doi.org/10.24432/C55S3H>
- Research paper DOI: <https://doi.org/10.1016/j.eswa.2007.12.020>
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)
- Scope: 30,000 Taiwan credit-card client records, 23 explanatory variables, and one binary target
- Target: default payment in the following month (`1` = yes, `0` = no)

The source page and license were checked on 2026-07-17. The repository does not redistribute the raw dataset. Keeping the source file out of Git makes provenance explicit and prevents accidental data-history bloat.

## Download

After installing the project dependencies, run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m credit_risk.data download
```

The command downloads the official UCI ZIP, verifies its SHA-256 digest, and extracts the original Excel file into `data/raw/`. Both files are excluded by `.gitignore`.

Verified source artifacts:

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| `default_of_credit_card_clients.zip` | 5,539,494 | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` |
| `default of credit card clients.xls` | 5,539,328 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |

The hashes describe the files retrieved from the official UCI URL on 2026-07-17. If UCI intentionally replaces an artifact, do not bypass the checksum: inspect the new file and update the recorded provenance in a reviewed commit.

## Local layout

```text
data/
|-- raw/          # Unmodified downloaded artifacts
|-- interim/      # Optional reproducible intermediate outputs
`-- processed/    # Optional model-ready outputs
```

Only `data/README.md` is versioned. Raw, intermediate, and processed tabular files remain local unless a later review explicitly approves redistribution.

## Source-specific caveats

- The workbook has two header rows. The canonical column names are on the second row, so it must be read with `header=1`.
- UCI documents no missing values, and the downloaded file contains no null cells after correct parsing.
- The file contains undocumented category codes: `EDUCATION` includes `0`, `5`, and `6`; `MARRIAGE` includes `0`; repayment status includes `-2` and `0`. They are retained and flagged instead of being silently recoded.
- There are 35 duplicate rows after dropping `ID`, but all IDs are unique. These records are not automatically deleted because different clients can share the same observed values and target.
- Negative bill amounts occur and can represent credits or adjustments. They are not automatically treated as invalid.

## Business limitations

- The records describe existing Taiwan credit-card clients in 2005, not current Chinese banking or consumer-finance applicants.
- Historical billing and repayment variables may not be available in a new-customer application workflow. They are usable for this account-level prediction exercise but limit any claim about true application underwriting.
- The dataset does not provide a reliable multi-period development/validation design. A random split must not be called out-of-time validation.
- Reject inference, macroeconomic cycles, funding costs, credit limits after intervention, and regulatory decision rules cannot be recovered from these data.
- Any approval, revenue, loss, or threshold analysis in this project is a transparent teaching scenario, not a measured business result.

See [`references/data_dictionary.md`](../references/data_dictionary.md) for field definitions, observed ranges, and availability notes.
