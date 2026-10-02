"""Validate IDs, evidence references, pooled qrels and file hashes for the pilot."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(name): return [json.loads(x) for x in (BASE/name).read_text(encoding="utf-8").splitlines() if x]
def unique(rows, key):
    mapped={r[key]:r for r in rows}
    assert len(mapped)==len(rows), "duplicate "+key
    assert all(k.isascii() and k for k in mapped), "non-ASCII ID"
    return mapped
companies=unique(read("companies.jsonl"),"company_id")
queries=unique(read("queries.jsonl"),"query_id")
sources=unique(read("sources.jsonl"),"source_ref")
rationales=unique(read("rationales.jsonl"),"rationale_id")
negatives=read("hard_negatives.jsonl")
unknowns=read("unknown_judgements.jsonl")
manifest=json.loads((BASE/"dataset.json").read_text(encoding="utf-8"))
assert len(companies)>=120 and 25<=len(queries)<=40
for cid,co in companies.items():
    assert co["track"]=="source_backed"
    assert co["attributes"]["verified_middle_market"] is None
    assert co["attributes"]["size_verification_status"]=="not_verified_middle_market"
    assert all(s in sources for s in co["source_refs"])
    assert "generation" not in co
    assert not any(x in co["description"] for x in ("rationale_id", "known_positive_ids","relevance="))
    for field, value in co["fields"].items():
        if value:
            assert co["field_source_refs"][field]
            assert all(s in sources and value in sources[s]["fact_summary"] for s in co["field_source_refs"][field])
        else:
            assert value is None
for s in sources.values():
    assert s["url"].startswith("https://") and s["title"] and s["publisher"]
    assert s["title"]!="Internal Error ()"
    assert s["accessed"]=="2026-10-02"
    assert s["short_quotations"]==[]
with (BASE/"qrels.tsv").open(encoding="utf-8",newline="") as f:
    qrels=list(csv.DictReader(f,delimiter="\t"))
pairs={}
for row in qrels:
    pair=(row["query_id"],row["company_id"])
    assert pair not in pairs, "duplicate query-company judgment"
    assert pair[0] in queries and pair[1] in companies
    row["relevance"]=int(row["relevance"])
    assert row["relevance"] in (0,1,2)
    pairs[pair]=row
    r=rationales[row["rationale_id"]]
    assert (r["query_id"],r["company_id"],r["relevance"])==(pair[0],pair[1],row["relevance"])
    assert set(r["source_refs"]).issubset(companies[pair[1]]["source_refs"])
    assert r["decision"] and r["supporting_fact_summary"]
    assert all(x in sources for x in r["source_refs"])
    assert all(x in companies[pair[1]]["fields"] for x in r["asserted_fields"])
    if queries[pair[0]]["category"]=="customer_exclusion" and row["relevance"]==1:
        assert r["unsupported_exclusion_status"]=="unknown"
assert len(rationales)==len(qrels)
for qid,q in queries.items():
    assert q["judgement_scope"]=="pooled" and q["judgement_completeness"]=="incomplete"
    direct={cid for (qid2,cid),r in pairs.items() if qid2==qid and r["relevance"]==2}
    partial={cid for (qid2,cid),r in pairs.items() if qid2==qid and r["relevance"]==1}
    zeros={cid for (qid2,cid),r in pairs.items() if qid2==qid and r["relevance"]==0}
    assert set(q["known_direct_match_ids"])==direct
    assert set(q["known_partial_match_ids"])==partial
    assert set(q["known_positive_ids"])==direct|partial
    assert set(q["hard_negative_ids"])==zeros
    assert q["strict_direct_match_recall_eligible"]==bool(direct)
    assert q["evaluation_role"]==("strict_screening" if direct else "exploratory_exclusion_probe")
    assert direct|partial, "no useful pooled candidates"
    assert all(s in sources for s in q["source_refs"])
negative_pairs=set()
for n in negatives:
    pair=(n["query_id"],n["company_id"])
    assert pair not in negative_pairs
    negative_pairs.add(pair)
    assert pairs[pair]["relevance"]==0
    assert n["rationale_id"]==pairs[pair]["rationale_id"]
    assert n["shared_facets"] and n["violated_constraints"]
    assert all(s in sources for s in n["source_refs"])
assert negative_pairs=={p for p,r in pairs.items() if r["relevance"]==0}
for u in unknowns:
    pair=(u["query_id"],u["company_id"])
    assert pair not in pairs
    assert u["company_id"] in queries[u["query_id"]]["explicit_unknown_ids"]
    assert u["judgment"]=="unknown"
for name,digest in manifest["file_hashes"].items():
    assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest, "hash mismatch: "+name
assert manifest["company_count"]==len(companies)
assert manifest["query_count"]==len(queries)
assert manifest["qrel_count"]==len(qrels)
assert manifest["hard_negative_count"]==len(negatives)
assert manifest["source_count"]==len(sources)
assert not manifest["judgements_complete"] and not manifest["precision_map_valid"]
assert not manifest["pitchbook_data"] and not manifest["embeddings_used_for_labels"]
words=[len(c["description"].split()) for c in companies.values()]
print(json.dumps({"status":"PASS","company_count":len(companies),"query_count":len(queries),
    "strict_evaluable_queries":sum(q["strict_direct_match_recall_eligible"] for q in queries.values()),
    "qrel_count":len(qrels),"relevance_counts":dict(Counter(r["relevance"] for r in qrels)),
    "hard_negatives":len(negatives),"sources":len(sources),"explicit_unknown_notes":len(unknowns),
    "judgement_coverage":len(pairs)/(len(companies)*len(queries)),
    "description_words_min":min(words),"description_words_max":max(words),
    "description_words_mean":round(sum(words)/len(words),1),
    "field_missing_counts":{f:sum(c["fields"][f] is None for c in companies.values()) for f in next(iter(companies.values()))["fields"]}},sort_keys=True))

