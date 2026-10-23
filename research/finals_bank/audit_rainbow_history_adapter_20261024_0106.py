"""Independent overlay audit against stored review chains; canonical SQL stays read-only."""
import copy
import hashlib
import io
import json
import pathlib
import sqlite3
import sys
import tarfile
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recovery'))
from adapt_rainbow_history_0106 import attach_verified_overlay, PINS
from build_rainbow_component_drafts_2308 import complete_row

def sha(b): return hashlib.sha256(b).hexdigest()
def enc(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
def load(p): return json.loads(p.read_bytes())
def unpack(arc):
    b = (ROOT / arc['path']).read_bytes(); assert len(b) == arc['bytes'] and sha(b) == arc['sha256']
    with tarfile.open(fileobj=io.BytesIO(b)) as t:
        assert t.getnames() == [arc['member']]; b = t.extractfile(arc['member']).read()
    assert len(b) == arc['member_bytes'] and sha(b) == arc['member_sha256']; return b

def review_chain():
    # Independent reconstruction, no use of the adapter's delta/apply helpers.
    for p, h in PINS.items(): assert sha((ROOT / p).read_bytes()) == h
    d = load(ROOT / 'data/scope_review_0914/independent_history_review.json')
    delta1 = load(ROOT / 'data/scope_review_1307/history_postponement_delta_review.json')
    manifest = load(ROOT / 'data/scope_review_1407/evidence_bundle_manifest.json')
    b = (ROOT / 'data/scope_review_1407/evidence_bundle.tar.xz').read_bytes()
    assert sha(b) == manifest['bundle_sha256'] and len(b) == manifest['bundle_bytes']
    member = 'current_data/unpacked/finals_ai/data/scope_review_1407/history_abnormal_delta_review.json'
    with tarfile.open(fileobj=io.BytesIO(b)) as t: b = t.extractfile(member).read()
    assert sha(b) == manifest['members'][member]['sha256']
    delta2 = json.loads(b)
    for delta in [delta1, delta2]:
        for ch in delta['delta']:
            e = d['events'][ch['event_index']]; a = e['entries'][ch['entry_index']]
            assert (e['day'], e['venue_code'], a['car_no'], a['registration_number']) == (ch['day'], ch['venue_code'], ch['car_no'], ch['registration_number'])
            a[ch['field']][ch['slot_index']] = ch['after']
    return d

def validate(event, draft, expected, firstday, gap):
    assert event['local_key'] == draft['identity']['canonical_key']
    assert event['canonical_SQL_rows_inserted'] == 0 and event['all_finals_complete'] is False
    assert len(event['entries']) == 9
    e_entries = {a['car_no']: a for a in expected['entries']}
    first_entries = {a['car_no']: a for a in firstday['entries']}
    for entry, actual in zip(draft['entries'], event['entries']):
        assert actual['car_no'] == entry['car_no'] and actual['registration_number'] == entry['registration_number']
        proof = e_entries[entry['car_no']]; f = first_entries[entry['car_no']]
        assert proof['registration_number'] == f['registration_number'] == actual['registration_number']
        slots = actual['adapted_history_slots']; assert len(slots) == 3
        assert [s['slot'] for s in slots] == ['history_data1', 'history_data2', 'history_data3']
        assert [s['original'] for s in slots] == [s['original'] for s in entry['history_slots']]
        current = slots[0]
        assert current['kind'] == 'current_meeting_context_not_participation' and current['actual_start_date'] is None
        assert current['rider_firstday_participation_inferred'] is False
        assert current['status'] == f['status'] and current['meeting_firstday'] == f['meeting_firstday']
        assert current['proof'] == {'OP_firstday': firstday['OP_firstday'], 'NK_firstday': firstday['NK_firstday']}
        for result, prior in zip(slots[1:], proof['Chariloto_prior_slots']):
            assert result['slot'] == prior['field'] and result['original'] == prior['raw_history'] and result['status'] == prior['status']
            if prior['status'] == 'empty_planned_slot_date_unresolved':
                assert result['kind'] == 'planned_slot_date_unresolved'
                assert result['actual_start_date'] is None and result['independent_group'] is None
            else:
                assert result['kind'] == 'actual_prior_meeting' and result['independent_group'] == prior['independent_group']
                assert result['actual_start_date'] == prior['independent_group']['start_date']
        assert actual['original_OP_prior_slots'] == proof['prior_OP_slots']
        expected_details = [x['detailed_result_resolution'] for p in proof['prior_OP_slots'] for x in p.get('result_differences', []) if 'detailed_result_resolution' in x]
        assert [r['detail'] for r in actual['abnormal_detail_sources']] == expected_details
        if actual['registration_number'] == '015690' and event['local_key'] == 'keirin:2023-12-17:85:09':
            alt = actual['OP_empty_cell_alternative']; assert alt is not None
            assert alt['group'] == gap['NK_raw_group'] and alt['records'] == gap['records']
            assert alt['original_OP_status'] == 'source_cell_empty_retained_no_fill'
            assert actual['original_OP_prior_slots'][1] == {'status': 'unresolved_empty_OP_slot', 'raw': '\n\n'}
        else: assert actual['OP_empty_cell_alternative'] is None

def reject(fn):
    try: fn()
    except (AssertionError, ValueError, KeyError): return
    raise AssertionError('bad history accepted')

def main():
    db = ROOT / 'data/finals.sqlite'; before = db.read_bytes()
    assert len(before) == 281206784 and sha(before) == '04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as c:
        assert c.execute('pragma integrity_check').fetchall() == [('ok',)] and c.execute('select count(*) from races').fetchone()[0] == 5354
    out = ROOT / 'data/scope_review_0106'; review = load(out / 'history_adapter_review.json')
    overlay_b = unpack(review['archive']); overlay = json.loads(overlay_b)
    component_b = unpack(overlay['base_component_archive']); components = json.loads(component_b)['events']
    full = review_chain()
    first = load(ROOT / 'data/scope_review_1545/current_firstday_independent_review.json')
    gap = load(ROOT / 'data/scope_review_1417/alternate_history_gap_review.json')
    def tuple_key(e): return (e['day'], e['venue_code'], e['race_no'])
    hm = {tuple_key(e): e for e in full['events']}; fm = {tuple_key(e): e for e in first['events']}
    for event, component in zip(overlay['events'], components):
        k = tuple_key(component['identity']); validate(event, component, hm[k], fm[k], gap)
    attached = attach_verified_overlay(component_b, overlay_b)
    # Removing only the added extension must recover the complete original draft.
    stripped = copy.deepcopy(attached)
    for d in stripped:
        del d['history_adapter_status']
        for a in d['entries']: del a['history_adapter']
    assert stripped == components
    kinds = Counter(s['kind'] for e in overlay['events'] for a in e['entries'] for s in a['adapted_history_slots'])
    assert kinds == {'current_meeting_context_not_participation': 126, 'actual_prior_meeting': 235, 'planned_slot_date_unresolved': 17}
    for d in attached: reject(lambda d=d: complete_row(d))
    with sqlite3.connect(':memory:') as c:
        c.execute("create table candidate_drafts(local_key text primary key, status text check(status='blocked'), payload text)")
        for d in attached:
            c.execute('insert into candidate_drafts values(?,?,?)', (d['identity']['canonical_key'], 'blocked', enc(d).decode()))
            assert json.loads(c.execute('select payload from candidate_drafts where local_key=?', (d['identity']['canonical_key'],)).fetchone()[0]) == d
        assert c.execute('select count(*) from candidate_drafts').fetchone()[0] == 14
    negatives = []
    event = overlay['events'][0]; d = components[0]; k = tuple_key(d['identity'])
    trials = [
        ('wrong_registration', lambda e: e['entries'][0].update(registration_number='999999')),
        ('raw_year_rewrite', lambda e: e['entries'][0]['adapted_history_slots'][0]['original'].update(open_from='20230603')),
        ('firstday_participation', lambda e: e['entries'][0]['adapted_history_slots'][0].update(rider_firstday_participation_inferred=True)),
        ('meeting_as_prior', lambda e: e['entries'][0]['adapted_history_slots'][0].update(kind='actual_prior_meeting')),
        ('prior_start_date', lambda e: e['entries'][0]['adapted_history_slots'][1].update(actual_start_date='2022-04-19')),
        ('prior_result_ID', lambda e: e['entries'][0]['adapted_history_slots'][1]['independent_group']['results'][0].update(href='https://keirin.netkeiba.com/db/result/?race_id=202304191109&player_id=99999')),
    ]
    for name, change in trials:
        e = copy.deepcopy(event); change(e); reject(lambda: validate(e, d, hm[k], fm[k], gap)); negatives.append(name)
    # Select an actual unresolved planned slot; never assume a fixed car position.
    ei, ai, si = next((ei, ai, si) for ei, e in enumerate(overlay['events']) for ai, a in enumerate(e['entries']) for si, s in enumerate(a['adapted_history_slots']) if s['kind'] == 'planned_slot_date_unresolved')
    for name, values in [('planned_date_guess', {'actual_start_date': '2023-05-23'}), ('planned_promotion', {'kind': 'actual_prior_meeting', 'status': 'independently_verified_explicit_full_date'})]:
        e = copy.deepcopy(overlay['events'][ei]); e['entries'][ai]['adapted_history_slots'][si].update(values)
        k2 = tuple_key(components[ei]['identity']); reject(lambda: validate(e, components[ei], hm[k2], fm[k2], gap)); negatives.append(name)
    altered = copy.deepcopy(overlay); altered['events'][0]['entries'][0]['adapted_history_slots'][1]['actual_start_date'] = '2099-01-01'
    reject(lambda: attach_verified_overlay(component_b, enc(altered))); negatives.append('changed_overlay_bytes')
    reject(lambda: attach_verified_overlay(component_b + b' ', overlay_b)); negatives.append('changed_component_bytes')
    # Preserve the postponed blank row and exact abnormal finish details.
    ei, ai, si = next((ei, ai, si) for ei, e in enumerate(overlay['events']) for ai, a in enumerate(e['entries']) for si, s in enumerate(a['adapted_history_slots']) if (s.get('independent_group') or {}).get('excluded_non_result_rows'))
    e = copy.deepcopy(overlay['events'][ei]); e['entries'][ai]['adapted_history_slots'][si]['independent_group']['excluded_non_result_rows'] = []
    k2 = tuple_key(components[ei]['identity']); reject(lambda: validate(e, components[ei], hm[k2], fm[k2], gap)); negatives.append('postponement_blank_row_deleted')
    ei, ai = next((ei, ai) for ei, e in enumerate(overlay['events']) for ai, a in enumerate(e['entries']) if a['abnormal_detail_sources'])
    e = copy.deepcopy(overlay['events'][ei]); e['entries'][ai]['abnormal_detail_sources'][0]['detail']['explicit_result_detail_raw'] = '病気欠場'
    k2 = tuple_key(components[ei]['identity']); reject(lambda: validate(e, components[ei], hm[k2], fm[k2], gap)); negatives.append('abnormal_detail_replaced')
    assert db.read_bytes() == before and not any(pathlib.Path(str(db) + s).exists() for s in ['-wal', '-shm', '-journal'])
    report = {'status': 'history_overlay_independently_audited_candidate_only', 'events': 14, 'raw_slots_exact_preserved': 378, 'actual_prior_slots_connected': 235, 'meeting_firstday_context_only': 126, 'empty_planned_unresolved_retained': 17, 'OP_empty_raw_cell_preserved_and_alternative_attached': 1, 'original_component_recovered_exact_by_removing_extension': True, 'isolated_candidate_roundtrip_rows': 14, 'complete_row_exports_rejected': 14, 'mutations_rejected': negatives, 'canonical_DB_changed': False, 'canonical_SQL_rows_inserted': 0, 'new_completed_finals': 0, 'network_requests': 0, 'all_finals_complete': False, 'builder_sha256': sha((ROOT / 'recovery/adapt_rainbow_history_0106.py').read_bytes()), 'overlay_member_sha256': review['archive']['member_sha256'], 'limitation': 'Candidate-history attachment and its round trip only; not a complete-payload insertion audit or global completion.'}
    (out / 'independent_history_adapter_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)); print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__': main()
