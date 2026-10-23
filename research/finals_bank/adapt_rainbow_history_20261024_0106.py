"""Attach previously verified history to component drafts, preserving every raw slot.

Offline only. No canonical SQL, no guessed year, no promotion of planned slots.
The output is a lossless overlay; it is not a completed race payload.
"""
import copy
import datetime
import hashlib
import io
import json
import pathlib
import sys
import tarfile
from collections import Counter
from urllib.parse import urlparse, parse_qs

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recovery'))
from build_rainbow_component_drafts_2308 import Sources

PINS = {
 'data/scope_review_0914/independent_history_review.json': '29b103743774fe14811d9e385a1cd090dec834da946763094072c60f6ce76409',
 'data/scope_review_1307/history_postponement_delta_review.json': '1b39580e5d88f9822426d8f4eeac7fa5aedaa359aed02996db2a01909a58ddec',
 'data/scope_review_1407/evidence_bundle_manifest.json': '53db0a19a15a1a494847ef7868a1795b577612ec2d7b4a316741dc584deb4754',
 'data/scope_review_1545/current_firstday_independent_review.json': 'fccdce7146c78efbdd69f1c61fb4e8d1ab88f15caff6af2f3e74fdffa27bf005',
 'data/scope_review_1417/alternate_history_gap_review.json': '41bb2e0c02d2486be1e4cf97b8fea6196419735f75be60558ed21b321f47c759',
 'data/scope_review_0914/raw_manifest.json': 'f1c162c631e4dcbf593907bb4656c34a1cc282b700fc4bf144d1eae06cfb30af',
 'data/scope_review_1417/raw_manifest.json': '24ac92769eea7e0bb3a6606f3becb4d90ed0c6c5a0b14b50d15594cabb4b6cd6',
 'data/scope_review_2308/component_draft_review.json': 'a6a89a9b976804c491c18791e159d3033d63b795852b18cab41cfa437abd3e19',
}
NORMAL = 'independently_verified_explicit_full_date'
ABNORMAL = 'independently_verified_explicit_full_date_with_abnormal_detail'
EMPTY = 'empty_planned_slot_date_unresolved'
OVERLAY_MEMBER_SHA = '53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522'

def sha(b): return hashlib.sha256(b).hexdigest()
def enc(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
def load(p): return json.loads(p.read_bytes())

def apply_review_delta(full, delta):
    for change in delta['delta']:
        e = full['events'][change['event_index']]
        a = e['entries'][change['entry_index']]
        assert e['day'] == change['day'] and e['venue_code'] == change['venue_code']
        assert a['car_no'] == change['car_no'] and a['registration_number'] == change['registration_number']
        field = change['field']; assert field in ['prior_OP_slots', 'Chariloto_prior_slots']
        assert a[field][change['slot_index']]['status'] == change['before_status']
        a[field][change['slot_index']] = copy.deepcopy(change['after'])
    for name in ['independent_prior_slots_status', 'Chariloto_slot_date_status']:
        full[name] = copy.deepcopy(delta[name])

def inputs():
    for p, h in PINS.items(): assert sha((ROOT / p).read_bytes()) == h, p
    sources = Sources()
    full = load(ROOT / 'data/scope_review_0914/independent_history_review.json')
    original_groups = [copy.deepcopy(a['NK_all_raw_groups']) for e in full['events'] for a in e['entries']]
    d1307 = load(ROOT / 'data/scope_review_1307/history_postponement_delta_review.json')
    assert d1307['baseline_report_sha256'] == PINS[d1307['baseline_report']]
    apply_review_delta(full, d1307)
    bundle_path = 'data/scope_review_1407/evidence_bundle.tar.xz'
    manifest = load(ROOT / 'data/scope_review_1407/evidence_bundle_manifest.json')
    raw = (ROOT / bundle_path).read_bytes()
    assert len(raw) == manifest['bundle_bytes'] and sha(raw) == manifest['bundle_sha256']
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:xz') as t:
        assert len(t.getnames()) == len(set(t.getnames())) == len(manifest['members'])
        assert set(t.getnames()) == set(manifest['members'])
        bundle = {}
        for m in t.getmembers():
            b = t.extractfile(m).read(); ref = manifest['members'][m.name]
            assert len(b) == ref['bytes'] and sha(b) == ref['sha256']; bundle[m.name] = b
    sources.archives[bundle_path] = bundle
    sources.refs[bundle_path] = {'path': bundle_path, 'bytes': len(raw), 'sha256': sha(raw), 'members': len(bundle)}
    delta_member = 'current_data/unpacked/finals_ai/data/scope_review_1407/history_abnormal_delta_review.json'
    d1407 = json.loads(bundle[delta_member])
    assert d1407['baseline_report_sha256'] == PINS[d1407['baseline_report']]
    apply_review_delta(full, d1407)
    assert original_groups == [a['NK_all_raw_groups'] for e in full['events'] for a in e['entries']]
    counts = Counter(s['status'] for e in full['events'] for a in e['entries'] for s in a['Chariloto_prior_slots'])
    assert counts == {NORMAL: 230, ABNORMAL: 5, EMPTY: 17}
    for path, digest, mani in [
        ('data/scope_review_0914/independent_history_raw.tar.xz', '6e9c07f6b552b16ba501fd1520347e85ac0dbf6d16d7f3d3020b4da8404c4985', 'data/scope_review_0914/raw_manifest.json'),
        ('data/scope_review_1417/alternate_history_gap_raw.tar.xz', '6c5004feeecf715ac6b74ce7ed726d7a75ac0f9be8bde26a895c390d205632da', 'data/scope_review_1417/raw_manifest.json'),
    ]: sources.archive(path, digest, mani)
    index = {}
    for arc, files in sources.archives.items():
        for member, b in files.items(): index.setdefault(sha(b), sources.member_ref(arc, member))
    review = load(ROOT / 'data/scope_review_2308/component_draft_review.json')
    arc = review['archive']; raw = (ROOT / arc['path']).read_bytes()
    assert len(raw) == arc['bytes'] and sha(raw) == arc['sha256']
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        assert t.getnames() == [arc['member']]; b = t.extractfile(arc['member']).read()
    assert len(b) == arc['member_bytes'] and sha(b) == arc['member_sha256']
    drafts = json.loads(b)['events']
    return full, load(ROOT / 'data/scope_review_1545/current_firstday_independent_review.json'), load(ROOT / 'data/scope_review_1417/alternate_history_gap_review.json'), drafts, index, arc

def group_check(g, registration, target_day):
    start = datetime.date.fromisoformat(g['start_date']).isoformat()
    assert start < target_day and g['group_id'] == start.replace('-', '') + g['venue_code']
    for r in g['results'] + g.get('excluded_non_result_rows', []):
        day = datetime.date.fromisoformat(r['date']).isoformat(); assert start <= day < target_day
        assert type(r['race_no']) is int and 1 <= r['race_no'] <= 12
        assert r['race_id'] == day.replace('-', '') + g['venue_code'] + f"{r['race_no']:02d}"
        u = urlparse(r['href']); q = parse_qs(u.query)
        assert u.hostname == 'keirin.netkeiba.com' and u.path == '/db/result/'
        assert q['race_id'] == [r['race_id']] and q['player_id'] == [str(int(registration))]

def overlay_event(draft, history, firstday, gap, index):
    ident = draft['identity']; day = ident['day']
    assert all(history[f] == firstday[f] == ident[f] for f in ['day', 'venue_code', 'race_no'])
    by_car = {r['car_no']: r for r in history['entries']}
    first_car = {r['car_no']: r for r in firstday['entries']}
    assert firstday['OP_firstday']['date'] == firstday['NK_firstday']['date'] <= day
    source_ref = index[history['source_meta']['sha256']]
    assert source_ref['bytes'] == history['source_meta']['bytes']
    out = []
    for entry in draft['entries']:
        n = entry['car_no']; h = by_car[n]; f = first_car[n]; reg = entry['registration_number']
        assert reg == h['registration_number'] == f['registration_number']
        assert ''.join(entry['name_raw'].split()) == ''.join(h['name'].split()) == ''.join(f['name'].split())
        assert entry['term'] == f['term']
        originals = {s['slot']: s['original'] for s in entry['history_slots']}
        assert originals['history_data1'] == f['raw_Chariloto_history_data1']
        assert f['rider_firstday_participation_inferred'] is False and not originals['history_data1']['short_boxes']
        slots = [{'slot': 'history_data1', 'kind': 'current_meeting_context_not_participation', 'original': originals['history_data1'], 'meeting_firstday': f['meeting_firstday'], 'actual_start_date': None, 'rider_firstday_participation_inferred': False, 'status': f['status'], 'proof': {'OP_firstday': firstday['OP_firstday'], 'NK_firstday': firstday['NK_firstday']}}]
        for s in h['Chariloto_prior_slots']:
            field = s['field']; assert s['raw_history'] == originals[field]
            if s['status'] == EMPTY:
                assert not originals[field]['short_boxes']
                slots.append({'slot': field, 'kind': 'planned_slot_date_unresolved', 'original': originals[field], 'status': EMPTY, 'actual_start_date': None, 'independent_group': None})
            else:
                assert s['status'] in [NORMAL, ABNORMAL]
                g = s['independent_group']; group_check(g, reg, day)
                assert s['candidate_full_dates'] == [g['start_date']]
                assert len(originals[field]['short_boxes']) == len(g['results'])
                for raw_row, result in zip(originals[field]['short_boxes'], g['results']): assert raw_row['race_num'] == result['race_no']
                slots.append({'slot': field, 'kind': 'actual_prior_meeting', 'original': originals[field], 'status': s['status'], 'actual_start_date': g['start_date'], 'independent_group': copy.deepcopy(g)})
        assert [s['slot'] for s in slots] == ['history_data1', 'history_data2', 'history_data3']
        details = []
        for p in h['prior_OP_slots']:
            for difference in p.get('result_differences', []):
                if 'detailed_result_resolution' in difference:
                    detail = difference['detailed_result_resolution']; ref = index[detail['source_sha256']]
                    assert ref['bytes'] == detail['source_bytes'] and detail['registration_number'] == reg
                    details.append({'detail': detail, 'source': ref})
        alternative = None
        if (day, ident['venue_code'], n, reg) == ('2023-12-17', '85', 6, '015690'):
            assert gap['alternate_NK_group_independently_verified'] is True
            assert h['prior_OP_slots'][1]['status'] == 'unresolved_empty_OP_slot'
            assert gap['OP_prior2_slot_raw'] == h['prior_OP_slots'][1]['raw']
            group_check(gap['NK_raw_group'], reg, day)
            alternative = {'status': 'independent_alternative_verified_original_OP_cell_empty_retained', 'original_OP_status': gap['OP_prior2_status'], 'group': gap['NK_raw_group'], 'records': gap['records'], 'sources': [index[s['sha256']] for s in gap['sources']]}
        out.append({'car_no': n, 'registration_number': reg, 'adapted_history_slots': slots, 'original_OP_prior_slots': copy.deepcopy(h['prior_OP_slots']), 'abnormal_detail_sources': details, 'OP_empty_cell_alternative': alternative})
    return {'local_key': ident['canonical_key'], 'entries': out, 'history_source': source_ref, 'independent_card_source': draft['component_raw_provenance']['independent_card'], 'firstday_NK_result_source': draft['component_raw_provenance']['independent_result'], 'status': 'history_adapter_applied_to_candidate_only', 'canonical_SQL_rows_inserted': 0, 'all_finals_complete': False}

def attach_verified_overlay(component_bytes, overlay_bytes):
    """Return new candidate objects; verify exact bound inputs before attachment."""
    assert sha(overlay_bytes) == OVERLAY_MEMBER_SHA
    overlay = json.loads(overlay_bytes); base = overlay['base_component_archive']
    assert sha(component_bytes) == base['member_sha256'] and len(component_bytes) == base['member_bytes']
    assert overlay['review_pins'] == PINS and overlay['all_finals_complete'] is False
    drafts = json.loads(component_bytes)['events']; by_key = {e['local_key']: e for e in overlay['events']}
    assert len(drafts) == len(by_key) == 14
    for draft in drafts:
        event = by_key[draft['identity']['canonical_key']]
        entries = {a['car_no']: a for a in event['entries']}; assert len(entries) == 9
        for entry in draft['entries']:
            a = entries[entry['car_no']]; assert a['registration_number'] == entry['registration_number']
            assert [s['original'] for s in a['adapted_history_slots']] == [s['original'] for s in entry['history_slots']]
            entry['history_adapter'] = copy.deepcopy(a)
        draft['history_adapter_status'] = 'attached_candidate_only_unresolved_retained'
    return drafts

def main():
    history, firstdays, gap, drafts, index, base = inputs()
    # Join by explicit actual tuple; record order is not a matching rule.
    hs = {(e['day'], e['venue_code'], e['race_no']): e for e in history['events']}
    fs = {(e['day'], e['venue_code'], e['race_no']): e for e in firstdays['events']}
    events = []
    for d in drafts:
        k = tuple(d['identity'][f] for f in ['day', 'venue_code', 'race_no'])
        events.append(overlay_event(d, hs[k], fs[k], gap, index))
    counts = Counter(s['kind'] for e in events for a in e['entries'] for s in a['adapted_history_slots'])
    assert counts == {'current_meeting_context_not_participation': 126, 'actual_prior_meeting': 235, 'planned_slot_date_unresolved': 17}
    out = ROOT / 'data/scope_review_0106'; out.mkdir(exist_ok=True)
    payload = enc({'schema': 'rainbow_history_overlay_v1', 'base_component_archive': base, 'events': events, 'review_pins': PINS, 'all_finals_complete': False})
    arc = out / 'history_adapter_overlay.tar.xz'
    with tarfile.open(arc, 'w:xz', preset=9) as t:
        info = tarfile.TarInfo('history_adapter_overlay.json'); info.size = len(payload); info.mtime = 0; t.addfile(info, io.BytesIO(payload))
    with tarfile.open(arc) as t: assert t.getnames() == ['history_adapter_overlay.json'] and t.extractfile('history_adapter_overlay.json').read() == payload
    report = {'status': 'history_adapter_connected_existing_evidence_not_new_race_completion', 'events': 14, 'raw_history_slots_preserved': 378, 'meeting_context_dates_not_rider_participation': 126, 'actual_prior_meeting_slots': 235, 'planned_slots_unresolved': 17, 'OP_prior_slots_original_status': history['independent_prior_slots_status'], 'OP_empty_cell_alternatives_attached_without_fill': 1, 'source_pins': PINS, 'archive': {'path': str(arc.relative_to(ROOT)), 'bytes': arc.stat().st_size, 'sha256': sha(arc.read_bytes()), 'member': 'history_adapter_overlay.json', 'member_bytes': len(payload), 'member_sha256': sha(payload)}, 'canonical_DB_changed': False, 'canonical_SQL_rows_inserted': 0, 'new_completed_finals': 0, 'network_requests': 0, 'all_finals_complete': False, 'remaining': ['17 planned-slot dates/explicit IDs', '13 candidate historical banks', 'full registry/scope/history raw provenance integration', 'complete payload schema/readiness and isolated complete insertion audit', 'broader final-equivalent scope and existing 5123 bank gaps']}
    (out / 'history_adapter_review.json').write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)); print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__': main()
