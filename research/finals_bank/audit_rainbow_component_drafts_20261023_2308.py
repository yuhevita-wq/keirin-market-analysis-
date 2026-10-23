"""Independent draft audit and isolated candidate-table round trip, never races SQL."""
import copy
import hashlib
import io
import json
import pathlib
import re
import sqlite3
import sys
import tarfile
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recovery'))
from build_rainbow_component_drafts_2308 import complete_row

def sha(b): return hashlib.sha256(b).hexdigest()
def norm(s): return ''.join(str(s).split())
def parse(b): return json.loads(b, parse_float=Decimal)
def dump(d): return json.dumps(d, ensure_ascii=False, sort_keys=True, default=str)

def raw_ref(ref):
    if 'archive' in ref:
        arc = ref['archive']; b = (ROOT / arc['path']).read_bytes()
        assert len(b) == arc['bytes'] and sha(b) == arc['sha256']
        with tarfile.open(fileobj=io.BytesIO(b), mode='r:xz') as t:
            assert len(t.getnames()) == len(set(t.getnames())) == arc['members']
            b = t.extractfile(ref['member']).read()
    else:
        b = (ROOT / ref['path']).read_bytes()
    assert len(b) == ref['bytes'] and sha(b) == ref['sha256']
    return b

SCOPE = parse((ROOT / 'data/scope_review_0457/rainbow_scope_independent_review.json').read_bytes())['events']
KEYS = parse((ROOT / 'data/scope_review_1746/source_key_mapping_review.json').read_bytes())['events']
CARDS = parse((ROOT / 'data/scope_review_0831/independent_card_results_review.json').read_bytes())['events']
LINES = parse((ROOT / 'data/scope_review_1307/independent_lines_review.json').read_bytes())['events']

def check(d, index, raw=None):
    e = SCOPE[index]; k = KEYS[index]; ir = CARDS[index]; line = LINES[index]
    assert d['schema'] == 'rainbow_component_draft_v1' and d['status'] == 'candidate_components_only'
    assert d['canonical_DB_inserted'] is False and d['all_finals_complete'] is False
    assert d['identity']['canonical_key'] == k['local_composite_key'] and d['identity']['confirmed_kdreams_race_id'] is None
    assert {f: d['identity'][f] for f in ['day', 'venue_code', 'race_no']} == k['tuple']
    assert d['observed_provider_keys'] == k['observed_source_keys']
    assert d['unconfirmed_KDreams_candidate_retained'] == e['candidate_kdreams_race_id']
    assert d['category'] == e['category']
    if raw is None:
        raw = {name: raw_ref(ref) for name, ref in d['component_raw_provenance'].items()}
    card = parse(raw['card']); odds = parse(raw['odds']); nk = parse(raw['independent_odds'])
    assert d['meeting_grade_raw'] == card['box']['open_grade_code'] and d['program_raw'] == card['race_program']
    assert len(d['entries']) == 9 and len(d['results']) == 9
    for n, entry in enumerate(d['entries'], 1):
        original = card['race_athletes'][str(n)]
        expected = next(r for r in ir['historical_entries_verified'] if r['car_no'] == n)
        assert entry['car_no'] == n and entry['registration_number'] == str(original['un_id']).zfill(6) == expected['registration_number']
        assert entry['name_raw'] == original['name'] and norm(entry['name_raw']) == expected['name']
        assert entry['term'] == str(original['graduate_period']) == expected['term']
        assert entry['age_at_race'] == original['age'] == expected['age']
        assert entry['class_raw'] == original['class_name'] + original['group_name']
        assert entry['prefecture_raw'] == original['pref_name']
        assert Decimal(entry['score_decimal']) == Decimal(original['average_point']) / 1000 == Decimal(expected['score_exact'])
        assert entry['finish_counts'] == expected['finish_counts'] and entry['winning_move_counts'] == expected['winning_move_counts']
        assert entry['gear_raw'] == original['gear_ratio'] and entry['gear_before_raw'] == original['gear_ratio_before'] and entry['style_raw'] == original['leg_quality']
        assert len(entry['history_slots']) == 3
        for h, slot in zip(entry['history_slots'], ['history_data1', 'history_data2', 'history_data3']):
            assert h['slot'] == slot and h['original'] == original[slot]
            assert h['date_adapter_status'] == 'pending_preserve_original_do_not_correct_year'
    assert [r['chariloto_raw'] for r in d['results']] == e['results']
    assert [r['netkeiba_raw'] for r in d['results']] == ir['results_verified']
    assert sorted(r['car_no'] for r in d['results']) == list(range(1, 10))
    for r in d['results']:
        n = r['car_no']; assert r['finish_text'] == r['chariloto_raw']['finish_text'] == r['netkeiba_raw']['finish_text']
        assert r['registration_number'] == d['entries'][n - 1]['registration_number']
    nk_key = ir['race_id_netkeiba']; record = nk['data']['nkrace_odds::' + nk_key]
    assert record['official_dt'].startswith(e['day'])
    for bet, field, count in [('sanrentan', 'list_9', 504), ('sanrenfuku', 'list_8', 84)]:
        expected = {}
        for index2, r in enumerate(odds['odds']):
            if r['bet_type'] == bet:
                cars = tuple(map(int, re.findall(r'\d+', r['deme'])))
                assert cars not in expected
                expected[cars] = (Decimal(str(r['rate'])), f'/odds/{index2}')
        independently_observed = {tuple(int(r[0][i:i+2]) for i in (0, 2, 4)): Decimal(r[1]) for r in record[field]}
        actual = {tuple(r['cars']): (Decimal(r['odds_decimal']), r['source_pointer']) for r in d['markets'][bet]}
        assert len(actual) == len(d['markets'][bet]) == count and actual == expected
        assert {c: rate for c, (rate, _) in actual.items()} == independently_observed
        assert all(r['is_valid_raw'] is True and r['on_final_raw'] is True for r in d['markets'][bet])
    assert d['published_line_groups'] == line['groups_chariloto'] == line['groups_netkeiba']
    assert d['published_line_tokens_raw'] == line['netkeiba_raw_tokens']
    expected_bank = {'status': 'unresolved', 'record': None}
    if e['day'] == '2023-12-05':
        expected_bank = {'status': 'verified_exact_event_candidate_not_SQL_applied', 'record': parse((ROOT / 'data/scope_review_1613/ito_candidate_bank_review.json').read_bytes())['bank_record']}
    assert d['historical_bank'] == expected_bank
    return raw

def reject(fn):
    try: fn()
    except (AssertionError, ValueError, KeyError): return
    raise AssertionError('bad draft accepted')

def main():
    db = ROOT / 'data/finals.sqlite'; db_before = db.read_bytes()
    assert len(db_before) == 281206784 and sha(db_before) == '04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as c:
        assert c.execute('pragma integrity_check').fetchall() == [('ok',)]
        assert c.execute('select count(*) from races').fetchone()[0] == 5354
    out = ROOT / 'data/scope_review_2308'
    review = parse((out / 'component_draft_review.json').read_bytes()); arc = review['archive']
    b = (ROOT / arc['path']).read_bytes(); assert sha(b) == arc['sha256'] and len(b) == arc['bytes']
    with tarfile.open(fileobj=io.BytesIO(b), mode='r:xz') as t:
        assert t.getnames() == ['component_drafts.json']
        b = t.extractfile('component_drafts.json').read()
    assert len(b) == arc['member_bytes'] and sha(b) == arc['member_sha256']
    drafts = parse(b)['events']; assert len(drafts) == 14
    cached = [check(d, index) for index, d in enumerate(drafts)]
    # Isolated table deliberately named candidate_drafts, with blocked-only status.
    # There is no races table and no path to the canonical DB in this connection.
    with sqlite3.connect(':memory:') as c:
        c.execute("create table candidate_drafts(local_key text primary key, status text check(status='blocked'), payload text not null)")
        native_drafts = json.loads(b)['events']
        for index, d in enumerate(drafts):
            text = dump(native_drafts[index]); c.execute('insert into candidate_drafts values(?,?,?)', (d['identity']['canonical_key'], 'blocked', text))
            assert parse(c.execute('select payload from candidate_drafts where local_key=?', (d['identity']['canonical_key'],)).fetchone()[0]) == d
            reject(lambda: complete_row(d))
        assert c.execute('select count(*) from candidate_drafts').fetchone()[0] == 14
        assert c.execute('pragma integrity_check').fetchall() == [('ok',)]
    mutations = []
    changes = [
        ('registration_ID', lambda d: d['entries'][0].update(registration_number='999999')),
        ('historical_score', lambda d: d['entries'][0].update(score_decimal='1000')),
        ('raw_history_year', lambda d: d['entries'][0]['history_slots'][0]['original'].update(open_from='20230603')),
        ('false_history_promotion', lambda d: d['entries'][0]['history_slots'][0].update(date_adapter_status='independently_verified_actual_date_and_raw_provenance')),
        ('actual_R', lambda d: d['identity'].update(race_no=10)),
        ('unconfirmed_KD_promotion', lambda d: d['identity'].update(confirmed_kdreams_race_id=d['unconfirmed_KDreams_candidate_retained'])),
        ('missing_odds', lambda d: d['markets']['sanrentan'].pop()),
        ('odds_value', lambda d: d['markets']['sanrentan'][0].update(odds_decimal='0.01')),
        ('DQ_to_rank', lambda d: d['results'][-1].update(finish_text='9')),
        ('line_boundary', lambda d: d['published_line_groups'].pop()),
        ('unverified_bank', lambda d: d['historical_bank'].update(status='verified_exact_event_candidate_not_SQL_applied')),
        ('complete_flag', lambda d: d.update(all_finals_complete=True)),
    ]
    for name, mutate in changes:
        d = copy.deepcopy(drafts[0]); mutate(d)
        reject(lambda: check(d, 0, cached[0])); reject(lambda: complete_row(d)); mutations.append(name)
    assert db.read_bytes() == db_before
    assert not any(pathlib.Path(str(db) + s).exists() for s in ['-wal', '-shm', '-journal'])
    report = {'status': 'component_drafts_independently_audited_all_blocked', 'events': 14, 'entries': 126, 'results': 126, 'market_combinations_exact_against_both_saved_providers': 8232, 'history_original_objects_exact_preserved': 378, 'independent_component_raw_locators_verified': 98, 'DQ_and_tied_finish_tokens_retained': True, 'candidate_table_roundtrip_rows': 14, 'candidate_table_storage': 'isolated in-memory SQLite; not canonical races; no SQLite artifact saved', 'complete_row_exports_rejected': 14, 'mutations_rejected': mutations, 'canonical_DB_unchanged': True, 'canonical_SQL_rows_inserted': 0, 'new_completed_finals': 0, 'all_finals_complete': False, 'production_compatibility_and_complete_payload_insertion': 'not implemented or claimed', 'draft_member_sha256': arc['member_sha256'], 'builder_sha256': sha((ROOT / 'recovery/build_rainbow_component_drafts_2308.py').read_bytes())}
    (out / 'independent_component_draft_audit.json').write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__': main()
