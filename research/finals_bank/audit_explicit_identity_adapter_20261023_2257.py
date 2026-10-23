"""Independent SQL-to-identity parity audit; no writes to the canonical database."""
import copy
import hashlib
import importlib.util
import json
import pathlib
import sqlite3
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recovery'))
from explicit_identity_adapter_2257 import IdentityAdapter, PINS

EXPECTED_DB = '04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
# Independent expected actual tuples, individually established in 2113.
EXACT = {
    '7420260531030107': ('2026-06-03', '74', 7),
    '8620230123030111': ('2023-01-26', '86', 11),
    '8620230123030112': ('2023-01-26', '86', 12),
    '5520230123030105': ('2023-01-26', '55', 5),
    '5520230123030111': ('2023-01-26', '55', 11),
}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def reject(fn):
    try:
        fn()
    except (AssertionError, KeyError, TypeError, ValueError):
        return
    raise AssertionError('unverified identity accepted')

def main():
    db = ROOT / 'data/finals.sqlite'
    assert db.stat().st_size == 281206784 and sha(db) == EXPECTED_DB
    assert not any(pathlib.Path(str(db) + s).exists() for s in ['-journal', '-wal', '-shm'])
    adapter = IdentityAdapter()
    spec = importlib.util.spec_from_file_location('old_bank_selector', ROOT / 'recovery/apply_historical_bank.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    catalog = json.loads((ROOT / 'data/historical_bank_catalog.json').read_text())['records']
    old.validate(catalog)
    identities = []; tuples = []; mapped = []; negatives = []; coverage = Counter(); excluded = []
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as c:
        assert c.execute('pragma integrity_check').fetchall() == [('ok',)]
        rows = c.execute('select race_id,day,url,status,payload from races order by race_id').fetchall()
        for rid, day, url, status, raw in rows:
            p = json.loads(raw)
            if status == 'excluded':
                assert rid == '1120240912030011' and p['scope_review']['status'] == 'excluded_women_source_verified'
                excluded.append(rid); continue
            expected = EXACT.get(rid, (day, rid[:2], int(p['race']['race_no'])))
            assert expected[0] == day and 1 <= expected[2] <= 12
            identity = adapter.legacy(rid, day, url, raw)
            assert (identity['day'], identity['venue_code'], identity['race_no']) == expected
            assert identity['canonical_key'] == rid and identity['confirmed_kdreams_race_id'] == rid
            assert type(identity['original_race_no']) is type(p['race']['race_no'])
            assert identity['original_race_no'] == p['race']['race_no']
            bank = adapter.select_bank(catalog, identity, p['race']['track'])
            assert bank == old.select(catalog, rid, day, p['race']['track'])
            coverage['verified' if bank else 'unresolved'] += 1
            tuples.append(expected); identities.append(identity)
            if rid in EXACT:
                mapped.append(dict(identity, original_payload_sha256=hashlib.sha256(raw.encode()).hexdigest()))
                # No metadata substitution, raw coercion, or generic number correction.
                for tag, args in [
                    ('day', (rid, '2023-01-27', url, raw)),
                    ('URL', (rid, day, url.replace(rid, '9999999999999999'), raw)),
                    ('payload_byte', (rid, day, url, raw + ' ')),
                ]:
                    reject(lambda args=args: adapter.legacy(*args)); negatives.append(rid + ':' + tag)
                q = copy.deepcopy(p); q['race']['race_no'] = expected[2]
                reject(lambda: adapter.legacy(rid, day, url, json.dumps(q, ensure_ascii=False)))
                negatives.append(rid + ':raw_number_rewrite')
        assert len(rows) == 5354 and len(identities) == 5353 and len(excluded) == 1
    assert len(mapped) == 5 and coverage == {'verified': 230, 'unresolved': 5123}
    assert len(set(tuples)) == len(tuples), 'actual tuple collision among existing male rows'
    local_events = json.loads((ROOT / 'data/scope_review_1746/source_key_mapping_review.json').read_text())['events']
    local = [adapter.local(e) for e in local_events]
    local_tuples = [(i['day'], i['venue_code'], i['race_no']) for i in local]
    assert len(local) == len(set(local_tuples)) == 14 and not (set(local_tuples) & set(tuples))
    assert len({i['canonical_key'] for i in identities + local}) == 5367
    assert all(i['confirmed_kdreams_race_id'] is None for i in local)
    for field, value in [('local_key_namespace', 'kdreams'), ('confirmed_kdreams_race_id', '4420230603040009')]:
        q = copy.deepcopy(local_events[0]); q[field] = value
        reject(lambda: adapter.local(q)); negatives.append('local:' + field)
    # Even a self-consistent tuple/key/provider-key edit must not create a verified event.
    q = copy.deepcopy(local_events[0]); q['tuple']['race_no'] = 10
    q['local_composite_key'] = 'keirin:2023-06-06:44:10'
    q['observed_source_keys']['netkeiba_race_id'] = '202306064410'
    q['observed_source_keys']['netkeiba_actual_result_links'] = ['https://keirin.netkeiba.com/race/result/?race_id=202306064410']
    q['observed_source_keys']['oddspark_explicit_hidden_tuple']['raceNo'] = 10
    reject(lambda: adapter.local(q)); negatives.append('local:coherent_unobserved_event')
    # An encoded-looking ID outside the five exact reviewed rows is not a mapping.
    q = copy.deepcopy(p); fake = '4420230603040109'; q['race_id'] = q['race']['race_id'] = fake
    q['race']['race_no'] = 109; q['race']['race_date'] = '2023-06-06'
    q['race']['source_url'] = 'https://keirin.kdreams.jp/ogaki/racedetail/' + fake + '/'
    reject(lambda: adapter.legacy(fake, '2023-06-06', q['race']['source_url'], json.dumps(q)))
    negatives.append('unlisted_109_no_subtraction')
    ito = next(i for i in local if i['day'] == '2023-12-05' and i['venue_code'] == '37')
    assert adapter.select_bank(catalog, ito, '伊東競輪')['bank_version_id'] == '37-ito20231202-reviewed'
    assert sha(db) == EXPECTED_DB
    report = {
        'status': 'explicit_identity_interface_all_existing_male_parity_passed',
        'scope': 'Identity and bank-selection interface only; collection and complete payload SQL insertion remain unintegrated.',
        'canonical_DB_changed': False, 'SQL_rows_inserted': 0, 'new_completed_finals': 0,
        'all_finals_complete': False, 'network_requests': 0,
        'legacy_rows_scanned': 5354, 'male_identities_accepted': 5353,
        'excluded_women_preserved': excluded, 'exact_source_mappings': mapped,
        'original_ID_and_race_no_preserved': True, 'general_100_subtraction_rule': False,
        'actual_tuple_collisions': 0, 'local_vs_existing_male_collisions': 0,
        'local_identities_checked': 14, 'confirmed_local_KDreams_IDs': 0,
        'all5367_identity_keys_unique': True, 'existing_bank_selection_exact_parity': dict(coverage),
        'candidate_Ito_bank_verified_but_not_added_to_SQL_count': True,
        'mutations_rejected': negatives, 'evidence_pins': PINS,
        'source_code_sha256': sha(ROOT / 'recovery/explicit_identity_adapter_2257.py'),
        'unchanged_production_code_sha256': {name: sha(ROOT / name) for name in ['pipeline.py', 'recovery/apply_historical_bank.py', 'recovery/audit_historical_bank.py']},
        'remaining': ['complete payload adapter and raw provenance', 'readiness gate and isolated SQL insertion audit', '13 candidate historical banks', '17 empty planned history dates and explicit IDs', 'broader final-equivalent scope', '5123 existing historical banks'],
    }
    out = ROOT / 'data/scope_review_2257'; out.mkdir(exist_ok=True)
    (out / 'explicit_identity_parity_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k not in ['exact_source_mappings', 'evidence_pins', 'unchanged_production_code_sha256', 'mutations_rejected']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
