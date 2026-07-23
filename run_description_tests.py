"""
run_description_tests.py

Standalone test harness for the description -> technique -> group pipeline.
Runs entirely fresh each execution (no Streamlit, no @st.cache_resource),
so there's zero risk of stale-cache results like we hit in the app.

For each test case, it:
  1. Maps the description to technique IDs (via description_mapper.py)
  2. Runs predict_top3() (via inference.py — the unchanged trained model)
  3. Checks whether the expected group appears in the top-3
  4. Computes true-positive/false-positive counts against the expected
     group's REAL signature (pulled from X.joblib/y.joblib), so you get
     a precision/coverage number alongside the pass/fail — not just a
     yes/no, so you know WHY something failed if it does.

Run directly in VS Code (or terminal):
    python run_description_tests.py
"""

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
    {
        "name": "DNC intrusion (elaborate)",
        "expected_keywords": ["APT28", "Fancy Bear"],
        "description": """
Investigators identified two separate intrusions into the
organization's network, believed to be conducted independently by
different operators without apparent coordination between them. The
first intrusion began nearly a year before public disclosure, when
attackers sent spearphishing emails to staff containing links to a
fake webmail login page that closely mimicked the organization's
actual authentication portal, harvesting usernames and passwords when
targets attempted to log in.

Using these stolen credentials, the attackers gained initial access
and began methodically escalating privileges, compromising
administrator accounts and using them to access sensitive systems
containing internal communications, opposition research files, and
donor information. They installed multiple custom backdoors on
different machines to ensure redundant access even if one was
discovered and removed, and used legitimate remote access and
administrative protocols to move between systems in a way designed to
resemble normal IT administrative activity.

The attackers exfiltrated large volumes of internal documents and
email communications over an extended period, compressing and
encrypting files before transferring them out through command and
control channels designed to blend in with regular web traffic, using
compromised third-party infrastructure as relay points to obscure the
true destination of the stolen data. They periodically updated their
backdoors and changed command and control domains to avoid detection
as security tools were updated to recognize their earlier
infrastructure.

A separate group of attackers compromised the network months later,
apparently unaware of the first group's presence, using different
malware and different techniques, including keylogging tools and
screen capture utilities to monitor specific staff members' activity
in near real time. Both sets of attackers were eventually identified
and removed following a multi-week incident response effort, during
which investigators found extensive evidence of long-term, persistent
access, systematic harvesting of credentials across the network, and
careful operational security practices intended to make attribution
and detection more difficult, including the deliberate use of
infrastructure and tools associated with cybercriminal groups to
create false leads for investigators.
""",
    },
    {
        "name": "NotPetya (named malware)",
        "expected_keywords": ["Sandworm", "Telebots", "Voodoo Bear"],
        "description": """
The attackers compromised the update mechanism of a widely-used
accounting and tax preparation software product popular with
businesses operating in the country, gaining access to the vendor's
software update servers and modifying them to distribute a malicious
update to the software's install base. Because the software was
required for filing local tax documents, it was installed on a large
number of corporate networks, giving the attackers an unusually broad
and trusted initial foothold across many unrelated organizations
simultaneously.

Once a compromised update was installed and executed on a network, the
malware immediately began harvesting credentials from memory on the
infected machine, extracting both plaintext passwords and password
hashes belonging to any user who had recently logged into that system,
including domain administrator credentials if any administrator had
used the machine. It then used these harvested credentials, combined
with legitimate Windows administrative and remote execution tools
already present on the network, to spread automatically and rapidly to
every other reachable machine on the same network, without requiring
any further action from the attackers or any user interaction.

The malware, later identified as NotPetya, exploited a known but
frequently unpatched vulnerability in file-sharing protocols as a
secondary spreading mechanism for machines it could not reach using
stolen credentials alone, allowing it to compromise systems even where
credential harvesting failed. On each newly infected machine, it
disguised itself as ransomware, displaying a ransom note demanding
payment in cryptocurrency for file recovery, but the encryption
routine was deliberately and irreversibly destructive: it overwrote
the master boot record and encrypted disk structures in a way that
made the affected systems permanently unrecoverable, with no
functioning decryption mechanism actually existing regardless of
whether a ransom was paid.

The attack caused widespread, unintentional collateral damage far
beyond its apparent original target, disrupting shipping and logistics
companies, pharmaceutical manufacturers, and other multinational
corporations that had subsidiaries or business units using the
compromised accounting software, resulting in billions of dollars in
total damages globally and, in several cases, weeks of complete
operational shutdown while organizations rebuilt their entire IT
infrastructure from backups. Investigators later assessed the primary
objective had likely never been financial extortion at all, given the
irreversible nature of the encryption and the destructive design of
the malware, but rather deliberate, large-scale disruption disguised
as a criminal ransomware campaign.
""",
    },
    {
        "name": "Cobalt Strike / Mimikatz (control case)",
        "expected_keywords": None,  # no strong expectation, just checking behavior
        "description": """
The attacker sent a spearphishing email with a malicious attachment.
Once opened, it ran an obfuscated PowerShell script that used
Mimikatz to dump credentials from memory. The attacker then moved
laterally via RDP to a domain controller and established persistence
through a scheduled task. Cobalt Strike was used for command and
control, with traffic beaconing out over DNS tunneling.
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