# Dataset Overview

| Dataset | Corpus / queries | Labels | Appropriate use |
|---|---|---|---|
| controlled-company-screening-v1 | 2,400 / 120 | Exhaustive synthetic constraints | Repeatable hard-negative and representation diagnostics |
| public-company-screening-v1 | 130 / 38 | 215 pooled judgments; 4.35% coverage | Public-source development pilot |
| beir-scifact | 5,183 / 300 | Official test qrels | Standard retrieval comparability |

The controlled generator uses seed 20261002. It creates 12 underlying screening profiles and multiple paraphrases; it is not a realistic population of distinct middle-market businesses.

The public pilot spans industrials/distribution, technology, print/media, business and healthcare services. It contains 44 sparse and 86 more detailed summaries (7–32 words). It includes companies, brands and parent-related businesses. Company size and middle-market status are unverified.

Source-backed direct-match recall uses 35 eligible queries and 109 relevance-2 judgments. Candidate recall includes 51 relevance-1 judgments. Fifty-five explicit negatives are annotated near misses. Three exclusion-sensitive queries remain exploratory because no strict positive has been verified. Unjudged pairs remain unknown.

Descriptions, keywords and enriched fields have different input information budgets. Use description-only and description + keywords as the sparse-description baselines. Field-based runs must be labelled as enriched.

The public summaries and annotations are offered under CC BY 4.0, excluding source website prose and third-party marks. Retain citations and the detailed [curated card](../data/curated/README.md). SciFact claims/qrels use CC BY 4.0; abstracts use ODC-By 1.0. The GitHub release includes the source license.

Actual PitchBook exports require the user's licensed access and permission. Keep them and their derived results private and evaluate transfer separately.

