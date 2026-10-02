"""Build the manually authored public company pool; no embedding model assigns labels."""
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter
BASE = Path(__file__).resolve().parent
DATE = "2026-10-02"
F = ("core_business", "products_services", "customers_end_markets", "capabilities")
def write_jsonl(name, rows):
    (BASE/name).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n" for r in rows), encoding="utf-8")
def ids(s):
    return s.split() if s else []
# Each row is an independently authored query/judgment pool. Direct/partial/negative
# sets were chosen from reviewed source facts, before any retrieval was run.
# Zeroes describe a documented incompatible primary offering or an explicit exclusion.
SPECS = [
("wood-manufacturers-exclusion","customer_exclusion","Find hardwood lumber suppliers serving manufacturers; exclude suppliers documented as serving builders, contractors or DIY buyers.","","baillie midwest-hardwood","us-lumber austin-hardwoods","hardwoods paxton","hardwood lumber supplied to manufacturers","documented builder, contractor or DIY customer channel"),
("decorative-wood","product_customer","Distributors or producers of decorative hardwood lumber, plywood or architectural millwork for fabricators and design professionals.","hardwoods paxton","midwest-hardwood austin-hardwoods","conklin","","decorative architectural wood products","primary documented distribution is sheet metal and duct supplies"),
("builder-materials","product_customer","Companies whose documented business supplies structural wood building materials or components to professional construction buyers.","us-lumber builders-firstsource 84-lumber bluelinx","weyerhaeuser-distribution","coilcraft","","wood building materials for construction","primary documented manufacture is electronic inductors"),
("duct-metal","product_capability","Find distributors of metal sheets or coils and supplies used in HVAC duct fabrication.","conklin","ferguson johnstone","coilcraft","","HVAC distribution and duct fabrication supplies","documented coils are electronic inductors and RFID components"),
("hvac-coils","product_disambiguation","Companies supplying heat-transfer coils, fan coils or replacement coil equipment for HVAC; distinguish these from electronic magnetic coils.","cooney nortek titus","","coilcraft","","HVAC heat-transfer or fan-coil products","electronic RF inductors and RFID coils"),
("refrigeration","product_end_market","Companies whose documented primary offering is equipment for commercial or industrial refrigeration and cooling.","heatcraft evapco","nortek","modine-hvac","","refrigeration or evaporative cooling equipment","primary documented offering is heating equipment"),
("building-air","product_end_market","Companies supplying air filtration or ventilation equipment for commercial, industrial or institutional buildings.","camfil greenheck nortek","titus hart-cooley","brintons","","building air filtration and ventilation equipment","primary documented interior product is carpet"),
("iso-certification","standards_capability","Third-party technical organizations offering ISO or management-system certification and audit services.","nqa tuv-rheinland lrqa bureau-veritas","bsi tuv-sud intertek sgs","protolabs","","management-system certification and auditing","primary documented business is contract production of parts"),
("materials-test","product_capability","Companies whose primary described business tests, inspects or certifies materials and manufactured products.","element intertek bureau-veritas sgs tuv-sud","kiwa smithers","protolabs fictiv","","materials or product testing and inspection","primary documented service is producing custom parts"),
("food-compliance","sector_capability","Testing and certification organizations with documented food safety work or a broader compliance testing service relevant to food producers.","kiwa","sgs bureau-veritas intertek","dining-alliance foodbuy","","food compliance testing or relevant broad testing services","primary documented offering is procurement and purchasing discounts"),
("pediatric-dentistry","sector_customer","Find clinical pediatric dentistry providers delivering preventive or restorative dental care to children.","dentistry-children pediatric-associates dental-associates-kids pediatric-manchester dental-associates","","dentrix planet-dds","","clinical pediatric dental care","primary documented offering is practice-management software"),
("pediatric-exclusion","customer_exclusion","Pediatric dental care providers; exclude organizations documented to offer general family dentistry or orthodontic treatment for adults.","","dentistry-children pediatric-associates dental-associates-kids pediatric-manchester","dental-associates smile-doctors","heartland 42north","pediatric care with general/adult care exclusion unresolved","explicit general family dentistry or adult orthodontic treatment"),
("dental-support","business_model","Organizations whose documented business supports dentists and multi-location dental practices with administrative and growth services.","heartland smile-brands dentalcorp 42north","","dentrix planet-dds","","dental support organization and practice partnership services","primary documented business is dental software"),
("dental-software","vertical_software","Software vendors serving dental practices with scheduling, billing, imaging or multi-location practice management.","dentrix planet-dds","","heartland 42north","","dental practice software","primary documented business is a dental support organization"),
("healthcare-buying","sector_customer","Purchasing organizations or procurement groups serving hospitals, health systems or other healthcare sites.","healthtrust foodbuy","vizient omnia","procurated","","healthcare purchasing or supply-chain performance services","primary documented business is vendor-feedback software"),
("restaurant-buying","sector_customer","Buying groups and procurement providers helping restaurants manage food purchasing, rebates or supplier economics.","dining-alliance foodbuy","avendra","toast restaurant365","","restaurant group purchasing or procurement services","primary documented offering is restaurant operating software"),
("public-buying","sector_customer","Cooperative purchasing and procurement organizations serving local government, schools or other public institutions.","sourcewell omnia","foodbuy","procurated","","public-sector cooperative purchasing","primary documented business is supplier performance software"),
("industrial-maintenance","capability_business_model","Service businesses that perform industrial maintenance, inspection or repair on customers' operating assets.","ats team mistras","emcor","servicemax","","physical industrial maintenance, inspection or asset repair services","primary documented offering is field-service management software"),
("facility-maintenance","capability_end_market","Companies delivering commercial building mechanical services, facilities operations or HVAC equipment maintenance.","tdindustries emcor","ats","greenheck hart-cooley","","physical mechanical and facilities services","primary documented offering is HVAC equipment"),
("custom-parts","manufacturing_capability","Custom production providers offering CNC machining or related precision part manufacturing across industrial end markets.","protolabs fictiv xometry ensinger pace","","paxton","","CNC or precision machining and custom parts","documented materials and manufacturing focus is wood millwork"),
("metal-casting","manufacturing_capability","Providers offering aluminum, zinc or magnesium die casting and related precision machining or finishing.","pace fictiv xometry","protolabs","paxton","","metal die casting or related custom metal production","primary documented manufacturing is wood millwork"),
("cold-storage","logistics_capability","Companies providing physical cold storage and refrigerated or frozen supply-chain logistics.","lineage americold uscold rls","chrobinson","descartes wisetech","","cold warehousing or refrigerated logistics","primary documented business is logistics software"),
("fulfillment","logistics_business_model","Physical logistics providers offering ecommerce fulfillment, parcel delivery or direct-to-consumer distribution.","rls gxo dhl","lineage americold","descartes","","physical fulfillment or parcel delivery","primary documented business is logistics software"),
("freight","logistics_capability","Companies providing freight brokerage, freight forwarding or transportation management services for shippers.","chrobinson rls lineage","dhl","wisetech","","freight brokerage, forwarding or managed transport services","primary documented offering is supply-chain software"),
("healthcare-staffing","sector_customer","Recruiting and staffing service companies supplying nurses, allied-health professionals or physicians to healthcare employers.","amn aya","jackson","bullhorn avionte","","healthcare staffing services","primary documented offering is recruitment software"),
("industrial-staffing","sector_customer","Service providers recruiting industrial workers or maintenance technicians for manufacturing, construction and logistics employers.","aerotek ats","","bullhorn avionte","","industrial recruiting and workforce support","primary documented offering is software for staffing agencies"),
("trades-software","vertical_software","Business software for residential or commercial trade contractors covering jobs, quotes, dispatch, invoicing or payments.","servicetitan housecall jobber","servicemax","tdindustries","","trade-contractor business management software","primary documented business performs mechanical construction and facility services"),
("construction-software","vertical_software","Software vendors supporting construction project teams and builders with scheduling, project workflows, financials or client communication.","procore buildertrend","servicetitan","builders-firstsource 84-lumber","","construction project and builder workflow software","primary documented business supplies physical construction materials"),
("property-software","vertical_software","Software providers for property managers covering leasing, rent, accounting, maintenance or resident operations.","rent-manager appfolio buildium entrata realpage","","buildertrend","","property-management software","primary documented software focus is residential construction project execution"),
("legal-software","vertical_software","Legal technology providers supporting law practices with matters, documents, billing, clients or legal research.","clio mycase filevine thomson-reuters","","dentrix","","law-practice and legal workflow technology","primary documented practice-software domain is dentistry"),
("staffing-software","vertical_software","Software platforms for staffing agencies with recruiting, applicant tracking, client relationships, payroll or billing.","bullhorn avionte","ukg paylocity","aerotek","","staffing agency software and relevant HR tools","primary documented business is providing staffing services"),
("restaurant-software","vertical_software","Software providers for restaurant operations with point of sale, ordering, inventory, accounting or workforce tools.","toast restaurant365 lightspeed","","dining-alliance","","restaurant operating software","primary documented business is a restaurant buying group"),
("hr-payroll","horizontal_software","Vendors providing payroll and HR software for employers, including employee data, scheduling, benefits or onboarding.","ukg paylocity bamboohr gusto","avionte","robert-half","","payroll and HR technology","primary documented business is recruitment and staffing services"),
("logistics-software","vertical_software","Technology vendors providing fleet visibility, shipping management, customs compliance or logistics workflow software.","samsara descartes wisetech","chrobinson","dhl","","fleet or logistics technology","primary documented business is physical shipping and logistics services"),
("bank-fraud","customer_capability","Fraud and financial-crime technology for banks or financial institutions, including transaction monitoring, AML or investigations.","featurespace unit21 sardine nice-actimize feedzai","quantexa socure","taylor","","financial-institution fraud or financial-crime platform","documented fraud-prevention capability is secure printed documents"),
("merchant-fraud","customer_capability","Technology protecting ecommerce merchants or digital commerce from fraudulent purchases, chargebacks, account abuse or returns abuse.","riskified forter signifyd sift sardine","feedzai","taylor deluxe","","merchant or ecommerce fraud technology","documented business centers on printing, secure documents or payment services rather than ecommerce order-risk software"),
("bank-exclusion","customer_exclusion","Find bank fraud platforms; exclude any company explicitly documented as also serving merchants, merchant acquirers or payment service providers.","","unit21 nice-actimize quantexa","featurespace sardine feedzai","socure neterium","bank or financial-crime technology with merchant exclusion unresolved","explicit merchant, merchant-acquirer or payment-provider customer channel"),
("print-mail","media_production","Companies whose documented business includes physical commercial printing, direct mail, signs or branded printed materials.","quad taylor alphagraphics vistaprint","rrd cenveo deluxe lsc","hibu simon","","physical print, signs or mailing products and services","primary documented offering is digital marketing technology and services"),
]
SHARED = {
    "wood-manufacturers-exclusion":"lumber supply",
    "decorative-wood":"materials distribution",
    "builder-materials":"manufactured components",
    "duct-metal":"coils",
    "hvac-coils":"coils",
    "refrigeration":"temperature-control equipment",
    "building-air":"commercial building products",
    "iso-certification":"manufacturing and materials workflows",
    "materials-test":"manufactured products and materials",
    "food-compliance":"food-industry services",
    "pediatric-dentistry":"dental practices",
    "pediatric-exclusion":"dental and orthodontic care",
    "dental-support":"dental practice operations",
    "dental-software":"dental practice operations",
    "healthcare-buying":"procurement workflows",
    "restaurant-buying":"restaurant operations",
    "public-buying":"public procurement",
    "industrial-maintenance":"industrial maintenance workflows",
    "facility-maintenance":"building HVAC systems",
    "custom-parts":"custom manufactured parts",
    "metal-casting":"manufacturing and machining",
    "cold-storage":"logistics and supply chains",
    "fulfillment":"shipping and delivery workflows",
    "freight":"logistics and global trade",
    "healthcare-staffing":"staffing and recruitment",
    "industrial-staffing":"staffing and recruitment",
    "trades-software":"mechanical and contractor workflows",
    "construction-software":"construction and builders",
    "property-software":"real estate and building operations",
    "legal-software":"professional practice management software",
    "staffing-software":"staffing and recruitment",
    "restaurant-software":"restaurant operations",
    "hr-payroll":"workforce and recruiting",
    "logistics-software":"shipping and logistics",
    "bank-fraud":"fraud prevention",
    "merchant-fraud":"fraud protection and payment workflows",
    "bank-exclusion":"financial fraud technology",
    "print-mail":"marketing and brand communications",
}
bib = {r["company_id"]: r for r in json.loads((BASE/"bibliography.json").read_text(encoding="utf-8"))}
companies, sources = [], []
for index, line in enumerate((BASE/"authored_facts.psv").read_text(encoding="utf-8").splitlines()):
    if not line or line.startswith("#"):
        continue
    parts = line.split("|")
    if len(parts) != 9:
        raise ValueError((index+1, len(parts)))
    slug, name, url, sector, core, products, customers, capabilities, keywords = parts
    cid, src = "pub-co-"+slug, "pub-src-"+slug
    fields = dict(zip(F, [core, products or None, customers or None, capabilities or None]))
    # Deliberately vary factual completeness in the description without adding labels.
    detail = "sparse" if len(companies)%3 == 0 else "detailed"
    rendered = [core, products] if detail=="sparse" else [core, products, customers, capabilities]
    desc = " ".join(x for x in rendered if x)
    companies.append({
        "company_id":cid, "track":"source_backed", "name":name,
        "description":desc, "description_detail":detail,
        "keywords":keywords.split(","), "fields":fields,
        "field_source_refs":{k:[src] for k,v in fields.items() if v},
        "keyword_source_refs":[src],
        "attributes":{"sector":sector, "subsector":None, "offering":None,
            "capability":None, "customer_type":None, "end_market":None,
            "business_model":None, "geography":None, "negative_attributes":None,
            "verified_middle_market":None, "size_verification_status":"not_verified_middle_market"},
        "source_refs":[src],
        "assertion_notes":{"sector":"Analyst normalization of the described business; not a company financial-sector code.",
            "entity_scope":"Company, operating business or product brand as named on its official site; entities can share a corporate parent.",
            "exclusivity":"Customer lists are non-exclusive; unsupported absence is unknown."}
    })
    sources.append({
        "source_ref":src, "url":url, "title":bib[slug]["title"],
        "publisher":name, "accessed":DATE,
        "licence_terms_note":"Company website: no general republication license assumed. Dataset includes independently authored fact summaries and links; source prose, images and logos are not redistributed.",
        "fact_summary":" ".join(v for v in fields.values() if v),
        "short_quotations":[],
        "asserted_field_source_refs":{k:[src] for k,v in fields.items() if v},
        "retrieval_evidence_ref":bib[slug]["web_evidence_ref"],
        "retrieval_method":"Official company page read or official-domain web-search result; bibliography preserves the displayed title.",
    })
by_id = {r["company_id"]:r for r in companies}
queries, rationales, negatives, qrels, unknowns = [], [], [], [], []
for index, spec in enumerate(SPECS, 1):
    slug, category, text, direct, partial, zero, unjudged, positive_facet, violated_facet = spec
    qid = f"pub-q-{index:04d}"
    sets = {2:ids(direct), 1:ids(partial), 0:ids(zero)}
    all_ids = sum(sets.values(), [])
    assert len(all_ids)==len(set(all_ids)), qid
    exclusion = category=="customer_exclusion"
    must_not = [{"field":"customers_end_markets", "op":"evidence_supports", "values":[violated_facet]}] if exclusion else []
    queries.append({
        "query_id":qid, "slug":slug, "text":text, "category":category,
        "constraints":{"must":[{"field":"documented_offering", "op":"evidence_supports", "values":[positive_facet]}],
            "must_not":must_not},
        "source_refs":sorted({s for x in all_ids for s in by_id["pub-co-"+x]["source_refs"]}),
        "judgement_scope":"pooled", "judgement_completeness":"incomplete",
        "evaluation_role":"strict_screening" if sets[2] else "exploratory_exclusion_probe",
        "strict_direct_match_recall_eligible":bool(sets[2]),
        "broad_candidate_recall_eligible":bool(sets[2] or sets[1]),
        "known_positive_ids":["pub-co-"+x for rel in (2,1) for x in sets[rel]],
        "known_direct_match_ids":["pub-co-"+x for x in sets[2]],
        "known_partial_match_ids":["pub-co-"+x for x in sets[1]],
        "hard_negative_ids":["pub-co-"+x for x in sets[0]],
        "explicit_unknown_ids":["pub-co-"+x for x in ids(unjudged)],
        "exclusion_evidence_policy":"Only affirmative evidence of a forbidden channel supports a zero; lack of mention stays unknown.",
        "label_note":"Relevance 1 is a useful partial candidate. Exclusion-sensitive partials do not establish absence of the excluded channel." if exclusion else
            "Zeroes compare explicit documented primary offerings, not an exhaustive assertion that the company lacks all adjacent products.",
    })
    for relevance, slugs in sets.items():
        for co_slug in slugs:
            cid = "pub-co-"+co_slug
            co = by_id[cid]
            rat_id = qid+"--"+cid
            supported = {k:v for k,v in co["fields"].items() if v}
            if relevance==2:
                decision = "Direct match: the cited offerings and/or customer applications support "+positive_facet+"."
            elif relevance==1:
                decision = ("Useful partial candidate: "+positive_facet+
                    "; some requested specificity or exclusion evidence is unresolved. Do not interpret this as a verified strict match.")
            else:
                decision = "Judged incompatible documented focus or explicit exclusion: "+violated_facet+"."
            rationales.append({"rationale_id":rat_id, "query_id":qid, "company_id":cid,
                "relevance":relevance, "decision":decision,
                "supporting_fact_summary":" ".join(supported.values()),
                "asserted_fields":list(supported),
                "source_refs":co["source_refs"], "field_source_refs":co["field_source_refs"],
                "judgement_method":"human-authored screening judgment from independently paraphrased official-source facts",
                "confidence_scope":"documented primary offering/customer facts; does not verify size or exhaustive capabilities",
                "unsupported_exclusion_status":"unknown" if exclusion and relevance==1 else None})
            qrels.append({"query_id":qid, "company_id":cid, "relevance":relevance,
                "judgement_source":"curated_primary_source_review", "rationale_id":rat_id})
            if relevance==0:
                negatives.append({"query_id":qid, "company_id":cid,
                    "shared_facets":[SHARED[slug]],
                    "violated_constraints":[violated_facet], "rationale_id":rat_id,
                    "source_refs":co["source_refs"],
                    "negative_basis":"explicit forbidden customer channel" if exclusion else "incompatible explicitly documented primary offering"})
    for co_slug in ids(unjudged):
        cid = "pub-co-"+co_slug
        unknowns.append({"query_id":qid, "company_id":cid, "judgment":"unknown",
            "reason":"Required customer or exclusion evidence is missing; not a positive or negative qrel.",
            "source_refs":by_id[cid]["source_refs"]})
for name, rows in [("companies.jsonl",companies),("sources.jsonl",sources),("queries.jsonl",queries),
    ("hard_negatives.jsonl",negatives),("rationales.jsonl",rationales),("unknown_judgements.jsonl",unknowns)]:
    write_jsonl(name,rows)
with (BASE/"qrels.tsv").open("w",encoding="utf-8",newline="") as fh:
    writer=csv.DictWriter(fh,fieldnames=["query_id","company_id","relevance","judgement_source","rationale_id"],delimiter="\t",lineterminator="\n")
    writer.writeheader()
    writer.writerows(qrels)
hashed = ["companies.jsonl","queries.jsonl","qrels.tsv","hard_negatives.jsonl","rationales.jsonl","sources.jsonl","unknown_judgements.jsonl","authored_facts.psv","bibliography.json","build_curated.py","validate_curated.py","README.md"]
counts=Counter(r["relevance"] for r in qrels)
metadata={
    "schema_version":1,"dataset_id":"public-company-screening-v1","version":"1.0.0",
    "track":"source_backed","accessed":DATE,"created":DATE,
    "license":{"spdx":"CC-BY-4.0","scope":"Original fact-summary text, authored screening queries, and annotations only; source website content and third-party marks excluded.",
        "attribution":"Company Embedding Benchmark public-source curation (2026-10-02); retain sources.jsonl and this dataset card."},
    "source_urls":[s["url"] for s in sources],"source_revisions":None,
    "creation_command":"python data/curated/build_curated.py","seed":None,
    "corpus_count":len(companies),"company_count":len(companies),"query_count":len(queries),
    "source_count":len(sources),"qrel_count":len(qrels),"hard_negative_count":len(negatives),
    "relevance_counts":{str(k):v for k,v in sorted(counts.items())},
    "judgement_scope":"pooled","judgement_completeness":"incomplete",
    "judgements_complete":False,"exhaustive":False,
    "judged_pairs":len(qrels),"possible_pairs":len(companies)*len(queries),
    "judgement_coverage":len(qrels)/(len(companies)*len(queries)),
    "primary_metric_label":"known direct-match recall (relevance >= 2)",
    "secondary_metric_label":"known broad-candidate recall (relevance >= 1)",
    "strict_evaluable_query_count":sum(bool(q["known_direct_match_ids"]) for q in queries),
    "exploratory_query_count":sum(not q["known_direct_match_ids"] for q in queries),
    "precision_map_valid":False,"unjudged_semantics":"unknown; never implicit relevance zero",
    "size_status":"not_verified_middle_market",
    "pitchbook_data":False,"embeddings_used_for_labels":False,
    "selection":"Purposive official-source pool across industrials, distribution, technology, media/print, business and healthcare services; includes large enterprises and operating brands.",
    "description_detail_counts":dict(Counter(c["description_detail"] for c in companies)),
    "field_missing_counts":{f:sum(c["fields"][f] is None for c in companies) for f in F},
    "limitations":["Not PitchBook data or a representative middle-market sample.",
        "Pooled labels are incomplete and authored by one curator; no independent double labeling or inter-rater estimate.",
        "Official company claims have not been independently verified.",
        "Negative focus classifications describe the cited primary business; they do not establish that an organization has no other business.",
        "Strict customer exclusions remain unknown unless affirmative evidence resolves them.",
        "Some records are product brands or operating businesses sharing a corporate parent.",
        "Only short variable descriptions are source-backed; extended controlled renderings must not be presented as additional public facts.",
        "At 130 companies, Recall@100 searches most of the pool and has weak discrimination; this is a development pilot, not production-scale evidence.",
        "Personal-PC results are non-decisional; validate on licensed PitchBook descriptions and VDI measurements before deployment."],
    "file_hashes":{name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in hashed if (BASE/name).exists()},
}
(BASE/"dataset.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"companies":len(companies),"queries":len(queries),"sources":len(sources),
    "qrels":len(qrels),"relevance_counts":dict(counts),"hard_negatives":len(negatives),"unknown_notes":len(unknowns)},sort_keys=True))

