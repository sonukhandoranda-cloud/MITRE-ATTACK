"""
description_mapper.py  (v2 — fully data-driven, no hardcoded technique mappings)

Maps a free-text incident description to MITRE ATT&CK technique IDs, using
ONLY official MITRE data:

  1. Technique corpus (name + description)        -> semantic similarity layer
  2. Software/malware "uses" technique relationships -> alias/keyword layer
     (e.g. "Mimikatz" -> whatever techniques MITRE's own STIX data says
     Mimikatz implements, including all of MITRE's own listed aliases for
     that tool — nothing hand-typed by a human)

Pipeline:
    user description
        --(software-name alias match, sourced from MITRE relationships)-->
        --(semantic similarity vs MITRE technique text)-->
    technique IDs
        --(existing, UNCHANGED predict_top3())-->
    top-3 APT groups

No model retraining required.

Install (one-time):
    pip install sentence-transformers requests --break-system-packages

Run:
    python description_mapper.py
"""

import json
import pathlib
import re
import sys

import numpy as np
import requests

ROOT = pathlib.Path(".").resolve()
ARTIFACTS = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS / "attack_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

MITRE_ENTERPRISE_ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

# Empirically-derived reliability floor from find_min_coverage.py — used
# only to WARN the user, never to invent techniques.
DEFAULT_MIN_RELIABLE_TECHNIQUES = 18


# ─────────────────────────────────────────────────────────────────────────
# 1. Load the raw MITRE STIX bundle (cached locally after first download)
# ─────────────────────────────────────────────────────────────────────────

def load_raw_bundle():
    cache_path = CACHE_DIR / "enterprise-attack.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print("Downloading MITRE enterprise-attack STIX bundle (one-time)...")
    resp = requests.get(MITRE_ENTERPRISE_ATTACK_URL, timeout=60)
    resp.raise_for_status()
    bundle = resp.json()
    cache_path.write_text(json.dumps(bundle), encoding="utf-8")
    return bundle


# ─────────────────────────────────────────────────────────────────────────
# 2. Technique corpus (name + description) — for the semantic layer
# ─────────────────────────────────────────────────────────────────────────

def build_technique_corpus(bundle):
    corpus = {}          # technique_id -> "name. description"
    stix_id_to_tid = {}  # STIX internal id (attack-pattern--xxxx) -> "T1059"

    for obj in bundle["objects"]:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue

        tid = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tid = ref.get("external_id")
                break
        if not tid:
            continue

        stix_id_to_tid[obj["id"]] = tid

        name = obj.get("name", "")
        desc = obj.get("description", "")
        desc = re.sub(r"\(Citation:.*?\)", "", desc)
        corpus[tid] = f"{name}. {desc[:600]}"

    return corpus, stix_id_to_tid


# ─────────────────────────────────────────────────────────────────────────
# 3. Software/tool alias table — sourced ENTIRELY from MITRE relationships
#    (software "uses" technique), not hand-typed
# ─────────────────────────────────────────────────────────────────────────

def build_software_alias_table(bundle, stix_id_to_tid):
    """
    Returns {lowercase_name_or_alias: {"techniques": [...], "group_usage_count": N}},
    built purely from MITRE's own STIX objects:
      - tool / malware objects supply the names + x_mitre_aliases
      - relationship objects (relationship_type == "uses", software->technique)
        supply which techniques each piece of software implements
      - relationship objects (relationship_type == "uses", group->software)
        supply group_usage_count: how many distinct APT groups are recorded
        as using that software. This is MITRE's own data, used purely to
        detect GENERIC/widely-shared tools (e.g. Cobalt Strike, Mimikatz)
        which are not discriminative for attribution on their own.
    """
    cache_path = CACHE_DIR / "software_alias_table.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # id -> {name, aliases: set()}
    software_by_id = {}
    for obj in bundle["objects"]:
        if obj.get("type") in ("tool", "malware"):
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue
            names = {obj.get("name", "").lower()}
            for alias in obj.get("x_mitre_aliases", []) or []:
                names.add(alias.lower())
            software_by_id[obj["id"]] = names

    # software_id -> set(technique_id)   [software "uses" technique]
    software_techniques = {}
    # software_id -> set(group_stix_id)  [group "uses" software]
    software_group_users = {}

    for obj in bundle["objects"]:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "uses":
            continue
        src = obj.get("source_ref", "")
        tgt = obj.get("target_ref", "")

        if (src.startswith("tool--") or src.startswith("malware--")) and tgt.startswith("attack-pattern--"):
            tid = stix_id_to_tid.get(tgt)
            if tid and src in software_by_id:
                software_techniques.setdefault(src, set()).add(tid)

        elif src.startswith("intrusion-set--") and (tgt.startswith("tool--") or tgt.startswith("malware--")):
            if tgt in software_by_id:
                software_group_users.setdefault(tgt, set()).add(src)

    # flatten to name/alias -> {techniques, group_usage_count}
    alias_table = {}
    for sw_id, techniques in software_techniques.items():
        group_count = len(software_group_users.get(sw_id, set()))
        for name in software_by_id.get(sw_id, []):
            if not name or len(name) < 4:  # skip empty/very-short noise (short names
                # like "arp", "at", "ps" risk false matches)
                continue
            entry = alias_table.get(
                name, {"techniques": [], "group_usage_count": 0})
            merged_tids = sorted(set(entry["techniques"]) | techniques)
            entry["techniques"] = merged_tids
            entry["group_usage_count"] = max(
                entry["group_usage_count"], group_count)
            alias_table[name] = entry

    cache_path.write_text(json.dumps(alias_table), encoding="utf-8")
    return alias_table


# ─────────────────────────────────────────────────────────────────────────
# 3b. Technique discriminativeness (IDF), computed from the ACTUAL trained
#     model's data (X.joblib/y.joblib) — not MITRE's full dataset. This
#     tells us which techniques are rare/specific among the groups the
#     model actually learned to distinguish, vs. generic techniques that
#     appear in most groups and therefore carry little discriminating
#     power. Same idea as the TF-IDF strategy already noted in the
#     evaluation notebook, applied here to re-rank mapped techniques.
# ─────────────────────────────────────────────────────────────────────────

def build_technique_idf(feature_vocab_path=None, x_path=None, y_path=None):
    cache_path = CACHE_DIR / "technique_idf.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    import joblib
    feature_vocab = joblib.load(feature_vocab_path or (
        ARTIFACTS / "feature_vocab.joblib"))
    X = joblib.load(x_path or (ARTIFACTS / "X.joblib"))
    y = np.array(joblib.load(y_path or (ARTIFACTS / "y.joblib")))
    all_tids = sorted(feature_vocab, key=feature_vocab.get)

    # reconstruct each group's fullest signature (same method used
    # throughout this project) to get one row per REAL group, not per
    # augmented/dropout sample — augmentation duplicates would otherwise
    # skew document frequency toward whichever groups got more copies.
    group_rows = []
    for label in sorted(set(y)):
        idx = np.where(y == label)[0]
        row_sums = X[idx].sum(axis=1)
        group_rows.append(X[idx[np.argmax(row_sums)]])
    group_matrix = np.vstack(group_rows)  # (n_groups, n_features)

    n_groups = group_matrix.shape[0]
    # how many groups use each technique
    doc_freq = (group_matrix > 0).sum(axis=0)

    idf_raw = np.log((n_groups + 1) / (doc_freq + 1)) + 1
    idf_norm = (idf_raw - idf_raw.min()) / \
        (idf_raw.max() - idf_raw.min() + 1e-9)

    idf_table = {all_tids[i]: float(idf_norm[i]) for i in range(len(all_tids))}
    cache_path.write_text(json.dumps(idf_table))
    return idf_table


# ─────────────────────────────────────────────────────────────────────────
# 4. Embeddings for the technique corpus (semantic layer)
# ─────────────────────────────────────────────────────────────────────────

def build_technique_embeddings(corpus, model):
    cache_emb = CACHE_DIR / "technique_embeddings.npy"
    cache_ids = CACHE_DIR / "technique_ids.json"

    if cache_emb.exists() and cache_ids.exists():
        embeddings = np.load(cache_emb)
        tids = json.loads(cache_ids.read_text())
        if len(tids) == len(corpus):
            return tids, embeddings

    tids = list(corpus.keys())
    texts = [corpus[t] for t in tids]
    print(f"Embedding {len(texts)} technique descriptions (one-time)...")
    embeddings = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True)

    np.save(cache_emb, embeddings)
    cache_ids.write_text(json.dumps(tids))
    return tids, embeddings


# ─────────────────────────────────────────────────────────────────────────
# 5. One-call pipeline loader
# ─────────────────────────────────────────────────────────────────────────

def load_pipeline(model):
    bundle = load_raw_bundle()
    corpus, stix_id_to_tid = build_technique_corpus(bundle)
    alias_table = build_software_alias_table(bundle, stix_id_to_tid)
    tids, technique_embeddings = build_technique_embeddings(corpus, model)
    idf_table = build_technique_idf()
    print(f"Loaded {len(corpus)} techniques and {len(alias_table)} software "
          f"names/aliases from MITRE's own data.")
    return corpus, alias_table, tids, technique_embeddings, idf_table


# ─────────────────────────────────────────────────────────────────────────
# 6. Map a free-text description to technique IDs
# ─────────────────────────────────────────────────────────────────────────

# Tools used by more than this many distinct APT groups (per MITRE's own
# group-uses-software data) are considered "generic" — common frameworks
# like Cobalt Strike or Mimikatz that dozens of unrelated groups use, and
# therefore not discriminative for attribution on their own.
GENERIC_TOOL_GROUP_THRESHOLD = 8

# Max techniques a single software match is allowed to contribute — prevents
# one popular tool (e.g. Cobalt Strike, with 60+ linked techniques) from
# flooding the result and crowding out semantic matches from the rest of
# the description.
MAX_TECHNIQUES_PER_SOFTWARE = 8

# Reserve at least this many of top_k slots for semantic matches, so a
# description that names one generic tool still gets credit for whatever
# specific behavior (RDP, DNS tunneling, scheduled tasks, etc.) was
# actually described.
MIN_SEMANTIC_SLOTS = 20


def map_description_to_techniques(
    description, corpus, alias_table, tids, technique_embeddings, model,
    idf_table=None, top_k=30, similarity_threshold=0.30,
):
    """
    Returns (technique_ids, match_details, coverage_warning_or_None).

    Two matching layers, both sourced from MITRE's own data:
      A) software-name alias match (e.g. "cobalt strike" mentioned in text ->
         techniques MITRE's relationship data says it uses) — capped per
         software and down-weighted if the tool is used by many unrelated
         groups (generic/non-discriminative), using MITRE's own group-usage
         counts, not a hand-typed list of "common tools."
      B) semantic similarity between description sentences and MITRE's
         own technique name+description text — guaranteed a minimum number
         of result slots so tool mentions can't crowd it out entirely.

    Both layers' scores are then re-weighted by idf_table (if provided) —
    a discriminativeness score computed from the ACTUAL trained model's
    groups (build_technique_idf()). Techniques that are rare/specific
    among trained groups get a modest boost; techniques nearly every
    group shares get a modest penalty. This nudges ranking toward
    techniques that actually help distinguish groups, without changing
    which techniques get matched in the first place.
    """
    idf_table = idf_table or {}

    def idf_multiplier(tid):
        # neutral (1.0) if unknown; scaled into a NARROWER [0.85, 1.15] band —
        # a wider band let weak-but-rare semantic matches (score ~0.4) get
        # boosted above stronger, more relevant ones, occasionally pulling
        # toward small/narrow groups that happen to share one rare
        # technique rather than the correct broad group. This is a
        # deliberately gentle nudge, not a re-ranking override.
        idf_val = idf_table.get(tid, 0.5)
        return 0.85 + 0.3 * idf_val

    alias_matches = {}  # tid -> (score, reason)
    desc_lower = description.lower()

    # ── A) software alias match — longest names first, WORD-BOUNDARY match
    #    only (plain substring search causes false positives, e.g. the tool
    #    name "arp" matching inside "spearphishing") ──
    for name in sorted(alias_table.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(name) + r"\b"
        if not re.search(pattern, desc_lower):
            continue
        entry = alias_table[name]
        sw_techniques = entry["techniques"]
        group_usage = entry["group_usage_count"]

        is_generic = group_usage > GENERIC_TOOL_GROUP_THRESHOLD
        base_score = 0.55 if is_generic else 0.9
        reason_tag = (
            f"software match: '{name}' — used by {group_usage} known groups, "
            f"treated as generic/non-discriminative"
            if is_generic else
            f"software match: '{name}' (used by {group_usage} known group(s))"
        )

        # cap contribution so one tool can't fill the whole result
        capped = sw_techniques[:MAX_TECHNIQUES_PER_SOFTWARE]
        for tid in capped:
            weighted_score = base_score * idf_multiplier(tid)
            prev = alias_matches.get(tid, (0, ""))
            if weighted_score > prev[0]:
                alias_matches[tid] = (weighted_score, reason_tag)

    # ── B) semantic match ──
    sentences = [s.strip() for s in re.split(
        r"[.\n;]", description) if s.strip()]
    if not sentences:
        sentences = [description]

    sent_embeddings = model.encode(sentences, normalize_embeddings=True)
    sim_matrix = sent_embeddings @ technique_embeddings.T

    semantic_matches = {}  # tid -> (score, reason)
    for row in sim_matrix:
        best_idx = np.argsort(row)[::-1][:top_k]
        for i in best_idx:
            score = float(row[i])
            if score < similarity_threshold:
                continue
            tid = tids[i]
            weighted_score = score * idf_multiplier(tid)
            prev = semantic_matches.get(tid, (0, ""))
            if weighted_score > prev[0]:
                semantic_matches[tid] = (
                    weighted_score, f"semantic match ({score:.2f} × idf)")

    # ── merge: guarantee semantic matches a minimum number of slots ──
    semantic_ranked = sorted(semantic_matches.items(),
                             key=lambda kv: kv[1][0], reverse=True)
    alias_ranked = sorted(alias_matches.items(),
                          key=lambda kv: kv[1][0], reverse=True)

    reserved_semantic = semantic_ranked[:MIN_SEMANTIC_SLOTS]
    remaining_slots = max(0, top_k - len(reserved_semantic))

    # combine everything else and take the best of what's left, avoiding
    # duplicates already reserved
    reserved_tids = {tid for tid, _ in reserved_semantic}
    remaining_pool = [
        (tid, val) for tid, val in (alias_ranked + semantic_ranked)
        if tid not in reserved_tids
    ]
    # dedupe remaining_pool keeping best score per tid
    best_remaining = {}
    for tid, val in remaining_pool:
        if tid not in best_remaining or val[0] > best_remaining[tid][0]:
            best_remaining[tid] = val
    remaining_ranked = sorted(best_remaining.items(
    ), key=lambda kv: kv[1][0], reverse=True)[:remaining_slots]

    final = reserved_semantic + remaining_ranked
    final_ranked = sorted(final, key=lambda kv: kv[1][0], reverse=True)[:top_k]

    technique_ids = [tid for tid, _ in final_ranked]
    match_details = [
        {"technique_id": tid, "name": corpus.get(tid, "").split(".")[0],
         "score": round(score, 3), "reason": reason}
        for tid, (score, reason) in final_ranked
    ]

    coverage_warning = None
    if len(technique_ids) < DEFAULT_MIN_RELIABLE_TECHNIQUES:
        coverage_warning = (
            f"Only {len(technique_ids)} techniques were confidently identified "
            f"from this description (empirical reliability floor is ~"
            f"{DEFAULT_MIN_RELIABLE_TECHNIQUES}). Predictions may be less "
            f"reliable — consider adding more technical detail (specific "
            f"tools, persistence methods, C2 behavior, discovery commands)."
        )

    return technique_ids, match_details, coverage_warning


# ─────────────────────────────────────────────────────────────────────────
# 7. End-to-end: description -> technique IDs -> predict_top3()
# ─────────────────────────────────────────────────────────────────────────

def attribute_from_description(description, corpus, alias_table, tids, technique_embeddings, model, idf_table=None):
    sys.path.insert(0, str(ROOT))
    from inference import predict_top3  # UNCHANGED — existing model

    technique_ids, match_details, coverage_warning = map_description_to_techniques(
        description, corpus, alias_table, tids, technique_embeddings, model, idf_table=idf_table,
    )
    if not technique_ids:
        raise ValueError(
            "Could not confidently map this description to any known ATT&CK "
            "techniques. Try including more specific technical detail "
            "(tools used, persistence methods, C2 behavior, etc.)."
        )

    group_predictions = predict_top3(technique_ids)
    return technique_ids, match_details, group_predictions, coverage_warning


# ─────────────────────────────────────────────────────────────────────────
# Demo / CLI
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print("Loading sentence-embedding model (pretrained, no training needed)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    corpus, alias_table, tids, technique_embeddings, idf_table = load_pipeline(
        model)

    example_description = """
    The attacker sent a spearphishing email with a malicious attachment to
    an employee. Once opened, it ran an obfuscated PowerShell script that
    used Mimikatz to dump credentials from memory. The attacker then moved
    laterally via RDP to a domain controller and established persistence
    through a scheduled task. Cobalt Strike was used for command and
    control, with traffic beaconing out over DNS tunneling.
    """

    print("\n" + "=" * 70)
    print("EXAMPLE: description -> technique IDs -> group prediction")
    print("=" * 70)
    print(f"\nDescription:\n{example_description.strip()}")

    technique_ids, match_details, group_predictions, coverage_warning = attribute_from_description(
        example_description, corpus, alias_table, tids, technique_embeddings, model, idf_table=idf_table,
    )

    print(f"\nMapped technique IDs ({len(technique_ids)}):")
    for m in match_details:
        print(
            f"  {m['technique_id']:<12} {m['name']:<45} score={m['score']}  ({m['reason']})")

    if coverage_warning:
        print(f"\n⚠  {coverage_warning}")

    print(f"\nTop-3 predicted groups:")
    for r in group_predictions:
        print(f"  #{r['rank']}  {r['group']:<30} {r['confidence']:.4f}")
