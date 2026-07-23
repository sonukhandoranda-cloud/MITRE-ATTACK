
import pathlib
import sys
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(".").resolve()
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

from description_mapper import load_pipeline, attribute_from_description
from inference import predict_top3  # noqa: F401 (used internally by attribute_from_description)

# ─────────────────────────────────────────────────────────────────────────
# Load everything ONCE, fresh, for this run
# ─────────────────────────────────────────────────────────────────────────

print("Loading model + MITRE data pipeline (fresh, no cache)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus, alias_table, tids, technique_embeddings, idf_table = load_pipeline(model)

X = joblib.load(ARTIFACTS / "X.joblib")
y = np.array(joblib.load(ARTIFACTS / "y.joblib"))
feature_vocab = joblib.load(ARTIFACTS / "feature_vocab.joblib")
all_tids = sorted(feature_vocab, key=feature_vocab.get)
all_labels = sorted(set(y))

print(f"Ready. {len(all_labels)} trained groups, {len(alias_table)} software aliases.\n")


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def find_real_label(keywords):
    for label in all_labels:
        if any(k.lower() in label.lower() for k in keywords):
            return label
    return None


def get_real_signature(label):
    idx = np.where(y == label)[0]
    if len(idx) == 0:
        return set()
    row_sums = X[idx].sum(axis=1)
    full_row = X[idx[np.argmax(row_sums)]]
    return set(all_tids[i] for i, v in enumerate(full_row) if v > 0)


def analyze(mapped_ids, real_label):
    real_sig = get_real_signature(real_label)
    mapped_roots = sorted(set(t.split(".")[0] if "." in t else t for t in mapped_ids))
    tp = [t for t in mapped_roots if t in real_sig]
    fp = [t for t in mapped_roots if t not in real_sig]
    return {
        "unique_mapped": len(mapped_roots),
        "true_positives": len(tp),
        "false_positives": len(fp),
        "real_sig_size": len(real_sig),
        "precision": len(tp) / len(mapped_roots) * 100 if mapped_roots else 0,
        "coverage": len(tp) / len(real_sig) * 100 if real_sig else 0,
    }


# ─────────────────────────────────────────────────────────────────────────
# TEST CASES — add/edit freely
# ─────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "SolarWinds (elaborate)",
        "expected_keywords": ["APT29", "Cozy Bear"],
        "description": """
The attackers began with a supply chain compromise, gaining access to a
software vendor's build environment and inserting a backdoor into the
compilation process itself, so that a legitimate, digitally signed
software update silently included malicious code. This trojanized
update was distributed through the vendor's normal, trusted update
channel to thousands of customer organizations. Once installed, the
backdoor remained dormant for approximately two weeks, performing
extensive checks for security tools, sandboxes, and analysis
environments before activating, in order to evade automated detection.

When active, the malware used domain generation techniques and DNS
requests disguised as legitimate telemetry to identify high-value
targets and establish command and control, communicating over
encrypted channels designed to blend in with normal update-checking
traffic. In select high-value networks, the attackers deployed
additional, more capable backdoors delivered directly through the
first-stage implant, then began manual, hands-on-keyboard operations.

The attackers used stolen or forged authentication credentials to
access on-premises Active Directory Federation Services infrastructure,
allowing them to forge SAML tokens and impersonate any user in the
organization's cloud identity provider without needing that user's
actual password. This gave them the ability to access Microsoft 365
mailboxes, SharePoint, and other cloud services while appearing as
legitimate, already-authenticated users, bypassing multi-factor
authentication entirely since the forged tokens were trusted by the
identity provider.

Once inside cloud environments, they enumerated existing service
principals and application registrations, then added new credentials
to legitimate existing applications with broad permissions, granting
themselves persistent, hard-to-detect access to read email, download
files, and query directory information across the organization,
without creating new, more easily-noticed user accounts. They also
manipulated mailbox forwarding rules and application consent grants to
maintain long-term visibility into specific individuals' communications.

Throughout the intrusion, the attackers used PowerShell extensively
for reconnaissance and Windows Management Instrumentation for remote
execution and persistence, avoided writing files to disk where
possible by operating primarily in memory, and used code-signing
certificates to make malicious binaries appear legitimate to endpoint
security tools. They carefully limited their footprint in each
compromised environment, using unique infrastructure per target and
avoiding reused indicators, and disabled or evaded logging where
feasible. Investigators later noted the intrusion had likely been
active for many months before discovery, exfiltrating internal
documents, source code repositories, and email communications through
encrypted channels designed to mimic normal outbound traffic patterns.
""",
    },
    {
        "name": "Bangladesh Bank (elaborate)",
        "expected_keywords": ["Lazarus"],
        "description": """
The attackers gained initial access to the bank's network months
before the theft attempt, believed to have originated from a
spearphishing email sent to bank employees containing a malicious
attachment disguised as a job application. Once an employee opened
the document, it exploited a vulnerability to execute code and install
a backdoor, giving the attackers a persistent foothold inside the
bank's internal network without immediately alerting security staff.

Over the following weeks, the attackers conducted extensive internal
reconnaissance, mapping the network topology, identifying which
workstations had access to the SWIFT international payment messaging
system, and studying the bank's internal procedures for authorizing
and processing large wire transfers. They used legitimate
administrative tools already present on the network to move laterally
between systems, avoiding the need to introduce additional malware
that might trigger antivirus alerts, and dumped credentials from
memory on multiple machines to obtain higher-privileged access.

Once they reached systems with SWIFT access, the attackers installed
custom-built malware specifically engineered to interact with the
bank's SWIFT Alliance software, allowing them to monitor and later
manipulate the confirmation messages and transaction logs the bank
used to reconcile its accounts. On the night of the theft, they issued
approximately three dozen fraudulent transfer requests, instructing
correspondent banks to move funds to accounts they controlled in
another country, timed deliberately around a weekend and a regional
public holiday to maximize the delay before bank staff would notice
irregularities.

The custom malware actively intercepted and altered the confirmation
printouts and database records the bank used to verify transactions
had gone through correctly, deleting or modifying specific transaction
entries so that automated reconciliation systems would not flag the
fraudulent transfers, effectively blinding the bank's own monitoring
systems to what had happened. Most of the fraudulent transfers were
ultimately blocked or reversed after an alert correspondent bank
employee noticed a spelling irregularity in one of the transfer
instructions, but a portion of the funds were successfully moved
through casinos in another country to launder the proceeds before
authorities could intervene. Investigators later found the same
custom malware families and infrastructure had been used in
unsuccessful attempts against several other banks' SWIFT systems in
the preceding year.
""",
    },
]


# ─────────────────────────────────────────────────────────────────────────
# Run all test cases
# ─────────────────────────────────────────────────────────────────────────

results_summary = []

for case in TEST_CASES:
    print("=" * 78)
    print(f"TEST: {case['name']}")
    print("=" * 78)

    try:
        technique_ids, match_details, group_predictions, coverage_warning = attribute_from_description(
            case["description"], corpus, alias_table, tids, technique_embeddings, model, idf_table=idf_table,
        )
    except ValueError as e:
        print(f"✗ ERROR: {e}\n")
        results_summary.append({"name": case["name"], "status": "ERROR"})
        continue

    print(f"Mapped {len(technique_ids)} techniques.")
    if coverage_warning:
        print(f"⚠ {coverage_warning}")

    print("\nTop-3 predicted groups:")
    top3_names = []
    for r in group_predictions:
        print(f"  #{r['rank']}  {r['group']:<30} {r['confidence']:.4f}")
        top3_names.append(r["group"])

    if case["expected_keywords"]:
        real_label = find_real_label(case["expected_keywords"])
        if not real_label:
            print(f"\n⚠ Expected group not found in trained labels for keywords {case['expected_keywords']}")
            results_summary.append({"name": case["name"], "status": "GROUP NOT TRAINED"})
        else:
            hit = any(real_label.lower() in name.lower() or name.lower() in real_label.lower()
                      for name in top3_names)
            analysis = analyze(technique_ids, real_label)

            print(f"\nExpected group: {real_label}")
            print(f"Result: {'✓ IN TOP-3' if hit else '✗ NOT IN TOP-3'}")
            print(f"True positives: {analysis['true_positives']} / {analysis['unique_mapped']} mapped "
                  f"(precision {analysis['precision']:.0f}%)")
            print(f"Coverage of real signature: {analysis['coverage']:.0f}% "
                  f"({analysis['true_positives']}/{analysis['real_sig_size']})")

            results_summary.append({
                "name": case["name"],
                "status": "PASS" if hit else "FAIL",
                "true_positives": analysis["true_positives"],
                "precision": analysis["precision"],
                "coverage": analysis["coverage"],
            })
    else:
        print("\n(No expected-group check for this case — informational only.)")
        results_summary.append({"name": case["name"], "status": "INFO"})

    print()

# ─────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────

print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"{'Test':<35} {'Status':<20} {'TP':>5} {'Precision':>10} {'Coverage':>10}")
for r in results_summary:
    tp = r.get("true_positives", "—")
    prec = f"{r['precision']:.0f}%" if "precision" in r else "—"
    cov = f"{r['coverage']:.0f}%" if "coverage" in r else "—"
    print(f"{r['name']:<35} {r['status']:<20} {tp!s:>5} {prec:>10} {cov:>10}")