# Public-source company screening pilot

This pool contains **130 real company, operating-business or product-brand
records**, **38 screening queries**, and citations to **130 official pages**
accessed on **2026-10-02**. It uses independently authored concise factual
summaries, not PitchBook descriptions. It is a purposive development pilot.
Company revenue, employee counts, enterprise value, ownership and middle-market
status were not verified; every record explicitly reports that missingness.
Large enterprises, subsidiaries and software brands are included. This is not a
representative middle-market sample, and some entities share corporate parents.

The curator browsed official primary websites and official-domain search results.
Each nonempty business field has a source reference. Sources retain the exact URL,
displayed page title, named publisher, access date and own fact summary; no direct
quotations, company images or logos are copied. Sector and keywords are analyst
normalizations of those facts. Empty fields mean unknown. The official claims
were not independently audited. The bibliography stores transient web-tool
evidence IDs for audit convenience; URLs are the durable citations.

The descriptions vary in factual detail: every third record contains core
business plus products, while other records also include known customers and
capabilities. This deliberate variation measures short, incomplete descriptions.
Structured fields preserve facts omitted from the sparse description, so compare
representations with awareness that their information budgets differ. Keywords
are separate and sourced; gold constraints and judgments never enter description
text or embeddings. No embedding model was used to assign relevance labels.

Queries cover wood distributors and customer channels, HVAC coils versus
electronic coils, refrigeration, air quality, ISO and materials testing,
pediatric dentistry, dental organizations versus dental software, purchasing
groups, industrial and facility services, manufacturing, cold logistics,
staffing, vertical software, financial crime, and print/mail production.
They are written from the same evidence used to curate the pool, so lexical and
selection bias are possible. Most pools contain several positives and close
contrasts. The judgments are manually authored by one curator, with no independent
double-labeling or inter-rater reliability estimate.

Relevance **2** means an explicitly supported direct match, **1** a useful partial
or potential candidate, and **0** an explicit incompatible documented primary
offering or an affirmative excluded customer channel. A zero based on primary
focus does not claim the organization has no adjacent products or subsidiaries.
Interpret each rationale within that documented-focus scope. Customer lists are
non-exclusive. No mention of contractors, adult care or merchant customers does
not establish that those customers are absent.

All unjudged pairs stay **unknown**. `unknown_judgements.jsonl` additionally
records selected exclusion-sensitive unknowns, rather than manufacturing
negative qrels. Three queries (`pub-q-0001`, `pub-q-0012`, `pub-q-0037`) lack any
fully verified exclusion-compliant direct positive. They are marked
`exploratory_exclusion_probe` and are ineligible for strict direct-match recall;
their rel1 candidates have explicitly unresolved exclusions. The other 35 queries
have at least one direct match. Report **known direct-match recall (rel >= 2)**
separately from **known broad-candidate recall (rel >= 1)**. Neither is complete
corpus recall. Precision and MAP require a separate complete judgment set and must
not be claimed from these pooled qrels. For exploratory probes, report documented
exclusion violations and unknown retrieval counts instead of strict success.

At a pool size of 130, Recall@100 examines most of the corpus and may be almost
trivial. Smaller cutoffs and annotated hard-negative violations are more useful
pilot diagnostics. Complete production-scale retrieval, inter-company ranking
and PitchBook transfer have not been validated by this dataset.

`rationales.jsonl` maps every qrel and hard negative back to fields and official
citations. `authored_facts.psv` and `bibliography.json` are the reviewable inputs;
`build_curated.py` deterministically writes the runtime dataset and manifest.

```powershell
python data/curated/build_curated.py
python data/curated/validate_curated.py
```

The original summaries, queries and annotations are offered under CC BY 4.0 with
attribution to "Company Embedding Benchmark public-source curation
(2026-10-02)". Retain this card and `sources.jsonl`. This license excludes source
website prose, trademarks, company images and other third-party material; no
general source website republication permission is assumed. The original source
URLs remain subject to their respective terms.

Before choosing a production model, add licensed PitchBook descriptions and an
independently adjudicated screening set at realistic corpus size, resolve strict
exclusion evidence, and measure comparable quality and performance in the VDI.
Personal-computer results from this pool are non-decisional.
