"""Offline component normalization. Drafts are deliberately ineligible for races SQL.

No networking, canonical database writes, fabricated KD IDs, or historical
date correction. Source records stay in their original files and archives.
"""
import copy
import hashlib
import io
import itertools
import json
import pathlib
import re
import sys
import tarfile
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'recovery'))
from explicit_identity_adapter_2257 import IdentityAdapter

PINS = {
 'data/scope_review_0457/rainbow_scope_independent_review.json': '53552457be52018be9f3b6c8c9258e1acb9fa92d8bab5568431dc06605591bdb',
 'data/scope_review_0831/independent_card_results_review.json': '1e0774557f609a4461d4f670d8ccdb901d64f4d2e9b710c0cabf5dfbf4d60b40',
 'data/scope_review_0806/independent_odds_review.json': '0d9644ce9e2a98a8b50d5310d3eac4bf12ea252787705aec1117fe87121c05de',
 'data/scope_review_1307/independent_lines_review.json': '520903a2da88125583d498db5bdea1af5718515e9c174df7921c241800fab0ab',
 'data/scope_review_1746/source_key_mapping_review.json': '755a2c57cbfb5b4c4e66a326d089ba81005b7af780e36d457072d06e3865673a',
 'data/scope_review_0831/raw_manifest.json': '47661ef1ca2ff32aa6cacf647b7db9c526a5652bbddc46f7ce5a615214c3d18f',
 'data/scope_review_1307/lines_raw_manifest.json': '384965eb34d67167293f0307c650f3c0e86ed529765fe71072d467d585a761c1',
 'data/scope_review_1613/ito_candidate_bank_review.json': '0c44b0f1cf3f22e14c905c567b867938d11b33671ecfa1468cd28d9b1daf87ac',
}

def sha(b): return hashlib.sha256(b).hexdigest()
def norm(s): return ''.join(str(s).split())
def encoded(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str).encode()
def load(p): return json.loads(p.read_bytes(), parse_float=Decimal)
def key(e): return e['day'].replace('-', '') + e['venue_code'] + f"{e['race_no']:02d}"

class Sources:
    def __init__(self):
        self.refs = {}
        self.archives = {}

    def direct(self, name, expected=None):
        raw = (ROOT / name).read_bytes()
        h = sha(raw)
        if expected is not None:
            assert h == expected['sha256'] and len(raw) == expected['bytes'], name
        ref = {'path': name, 'sha256': h, 'bytes': len(raw)}
        self.refs[name] = ref
        return raw, ref

    def archive(self, name, expected_sha, manifest_path):
        raw = (ROOT / name).read_bytes(); assert sha(raw) == expected_sha
        manifest = load(ROOT / manifest_path)
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:xz') as t:
            members = t.getmembers(); assert len(members) == len(manifest)
            assert len({m.name for m in members}) == len(members)
            assert {m.name for m in members} == set(manifest)
            files = {}
            for m in members:
                assert m.isfile()
                b = t.extractfile(m).read(); expected = manifest[m.name]
                assert len(b) == expected['bytes'] and sha(b) == expected['sha256']
                files[m.name] = b
        self.archives[name] = files
        self.refs[name] = {'path': name, 'sha256': sha(raw), 'bytes': len(raw), 'members': len(files)}
        return files

    def member_ref(self, archive, member):
        b = self.archives[archive][member]
        return {'archive': self.refs[archive], 'member': member, 'sha256': sha(b), 'bytes': len(b)}

def assemble():
    sources = Sources()
    for p, h in PINS.items(): assert sha((ROOT / p).read_bytes()) == h, p
    scope = load(ROOT / 'data/scope_review_0457/rainbow_scope_independent_review.json')['events']
    reviews = {}
    for label, path in [('card', 'data/scope_review_0831/independent_card_results_review.json'), ('odds', 'data/scope_review_0806/independent_odds_review.json'), ('line', 'data/scope_review_1307/independent_lines_review.json')]:
        reviews[label] = {key(e): e for e in load(ROOT / path)['events']}
    identities = load(ROOT / 'data/scope_review_1746/source_key_mapping_review.json')['events']
    adapter = IdentityAdapter()
    cards_archive = 'data/scope_review_0831/independent_card_results_raw.tar.xz'
    lines_archive = 'data/scope_review_1307/independent_lines_raw.tar.xz'
    sources.archive(cards_archive, 'b067a5fcb5629030f5440bc0efc88e368b753c4236a5a50fee8af372c270bbe2', 'data/scope_review_0831/raw_manifest.json')
    sources.archive(lines_archive, '9411b28b394f64f2c7c93da0433048cef6c03ef4b93eeeba7e9e3acf341f121c', 'data/scope_review_1307/lines_raw_manifest.json')
    drafts = []
    for e, ident_event in zip(scope, identities):
        k = key(e); assert ident_event['tuple'] == {f: e[f] for f in ['day', 'venue_code', 'race_no']}
        identity = adapter.local(ident_event)
        independent = reviews['card'][k]; line = reviews['line'][k]; odds_review = reviews['odds'][k]
        direct = {}
        for ref in e['source_files']:
            path = 'data/scope_review_0457/' + ref['path']
            raw, locator = sources.direct(path, ref)
            kind = 'card' if 'card' in pathlib.Path(path).name else 'odds' if 'odds' in pathlib.Path(path).name else 'result'
            assert kind not in direct
            direct[kind] = (raw, locator)
        assert set(direct) == {'card', 'odds', 'result'}
        card = json.loads(direct['card'][0]); odds = json.loads(direct['odds'][0], parse_float=Decimal)
        for box in [card['box'], card['race_program'], card['race_status']]:
            assert box['open_day'] == e['day'] and box['vel_code'] == e['venue_code'] and box['race_num'] == e['race_no']
        assert card['box']['is_girls'] is False and card['box']['on_abort'] is False and card['box']['is_invalid'] is False
        assert card['race_program']['id'] == ident_event['observed_source_keys']['chariloto_program_status_id']
        checked = {r['car_no']: r for r in independent['historical_entries_verified']}
        official = {r['car_no']: r for r in e['entry_identities']}
        entries = []
        for num, r in sorted(card['race_athletes'].items(), key=lambda kv: int(kv[0])):
            n = int(num); ir = checked[n]; reg = str(r['un_id']).zfill(6)
            assert n == r['entry_num'] and reg == ir['registration_number'] == official[n]['registration_number']
            assert norm(r['name']) == ir['name'] == norm(official[n]['name'])
            assert str(r['graduate_period']) == ir['term'] == official[n]['term']
            assert r['age'] == ir['age']
            score = Decimal(r['average_point']) / 1000
            assert score == Decimal(ir['score_exact'])
            counts = [r[f] for f in ['first_count', 'second_count', 'third_count', 'out_rank_count']]
            moves = [r[f] for f in ['nige_count', 'makuri_count', 'sashi_count', 'mark_count']]
            assert counts == ir['finish_counts'] and moves == ir['winning_move_counts']
            histories = [{'slot': f'history_data{i}', 'original': copy.deepcopy(r[f'history_data{i}']), 'date_adapter_status': 'pending_preserve_original_do_not_correct_year'} for i in [1, 2, 3]]
            entries.append({'car_no': n, 'registration_number': reg, 'name_raw': r['name'], 'term': str(r['graduate_period']), 'age_at_race': r['age'], 'class_raw': r['class_name'] + r['group_name'], 'prefecture_raw': r['pref_name'], 'score_decimal': str(score), 'finish_counts': counts, 'winning_move_counts': moves, 'gear_raw': r['gear_ratio'], 'gear_before_raw': r['gear_ratio_before'], 'style_raw': r['leg_quality'], 'history_slots': histories, 'source_pointer': '/race_athletes/' + num})
        assert [r['car_no'] for r in entries] == list(range(1, 10))
        results = []
        nk = {r['car_no']: r for r in independent['results_verified']}
        regs = {r['car_no']: r['registration_number'] for r in entries}
        for r in e['results']:
            n = r['car_no']; assert r['finish_text'] == nk[n]['finish_text'] and regs[n] == nk[n]['registration_number'].zfill(6)
            results.append({'car_no': n, 'registration_number': regs[n], 'finish_text': r['finish_text'], 'chariloto_raw': r, 'netkeiba_raw': nk[n]})
        assert sorted(r['car_no'] for r in results) == list(range(1, 10))
        markets = {}
        for bet, ordered in [('sanrentan', True), ('sanrenfuku', False)]:
            rows = []
            for index, row in enumerate(odds['odds']):
                if row['bet_type'] != bet: continue
                cars = tuple(map(int, re.findall(r'\d+', row['deme'])))
                rate = Decimal(str(row['rate'])); assert rate == Decimal(str(row['normalize_rate'])) and rate.is_finite() and rate > 0
                assert row['is_valid'] is True and row['on_final'] is True
                rows.append({'cars': list(cars), 'odds_decimal': str(rate), 'is_valid_raw': row['is_valid'], 'on_final_raw': row['on_final'], 'source_pointer': f'/odds/{index}'})
            expected = set(itertools.permutations(range(1, 10), 3) if ordered else itertools.combinations(range(1, 10), 3))
            assert len(rows) == len(expected) and {tuple(r['cars']) for r in rows} == expected
            assert odds_review['markets'][bet]['count'] == len(rows) and odds_review['markets'][bet]['exact_decimal_odds_equal'] is True
            markets[bet] = rows
        assert line['groups_chariloto'] == line['groups_netkeiba'] == e['published_line_groups']
        assert sorted(n for group in line['groups_netkeiba'] for n in group) == list(range(1, 10))
        nk_odds_path = f'data/scope_review_0806/{k}_odds.json'
        _, nk_odds_ref = sources.direct(nk_odds_path, odds_review['source'])
        refs = {kind: ref for kind, (_, ref) in direct.items()}
        refs.update({'independent_card': sources.member_ref(cards_archive, k + '_card.html'), 'independent_result': sources.member_ref(cards_archive, k + '_result.html'), 'independent_line': sources.member_ref(lines_archive, k + '.html'), 'independent_odds': nk_odds_ref})
        bank = {'status': 'unresolved', 'record': None}
        if (e['day'], e['venue_code'], e['race_no']) == ('2023-12-05', '37', 9):
            bank = {'status': 'verified_exact_event_candidate_not_SQL_applied', 'record': json.loads((ROOT / 'data/scope_review_1613/ito_candidate_bank_review.json').read_bytes())['bank_record']}
        drafts.append({'schema': 'rainbow_component_draft_v1', 'identity': identity, 'observed_provider_keys': ident_event['observed_source_keys'], 'unconfirmed_KDreams_candidate_retained': e['candidate_kdreams_race_id'], 'category': e['category'], 'meeting_grade_raw': card['box']['open_grade_code'], 'program_raw': card['race_program'], 'entries': entries, 'results': results, 'markets': markets, 'published_line_groups': line['groups_netkeiba'], 'published_line_tokens_raw': line['netkeiba_raw_tokens'], 'historical_bank': bank, 'component_raw_provenance': refs, 'review_pins': PINS, 'status': 'candidate_components_only', 'canonical_DB_inserted': False, 'all_finals_complete': False})
    assert len(drafts) == 14
    return drafts, sources.refs

def blockers(draft):
    # Successful component conversion alone never grants complete-row eligibility.
    pending = [f"car{r['car_no']}:{h['slot']}" for r in draft['entries'] for h in r['history_slots'] if h['date_adapter_status'] != 'independently_verified_actual_date_and_raw_provenance']
    reasons = ['complete_payload_independent_insertion_audit_pending', 'history_raw_provenance_adapter_pending']
    if pending: reasons.append('history_slots_not_yet_adapted:' + str(len(pending)))
    if draft['historical_bank']['status'] != 'verified_exact_event_candidate_not_SQL_applied': reasons.append('historical_bank_unresolved')
    return reasons

def complete_row(draft):
    reasons = blockers(draft)
    if reasons: raise ValueError('ineligible complete race row: ' + ','.join(reasons))
    raise NotImplementedError('complete payload insertion is not implemented')

def main():
    drafts, refs = assemble()
    out = ROOT / 'data/scope_review_2308'; out.mkdir(exist_ok=True)
    payload = encoded({'schema': 'rainbow_component_drafts_v1', 'events': drafts, 'all_finals_complete': False})
    archive = out / 'component_drafts.tar.xz'
    with tarfile.open(archive, 'w:xz', preset=9) as t:
        info = tarfile.TarInfo('component_drafts.json'); info.size = len(payload); info.mtime = 0
        t.addfile(info, io.BytesIO(payload))
    with tarfile.open(archive, 'r:xz') as t:
        assert t.getnames() == ['component_drafts.json'] and t.extractfile('component_drafts.json').read() == payload
    report = {'status': 'fourteen_component_drafts_created_not_complete_payloads', 'canonical_DB_changed': False, 'canonical_SQL_rows_inserted': 0, 'new_completed_finals': 0, 'all_finals_complete': False, 'network_requests': 0, 'events': [{'local_key': d['identity']['canonical_key'], 'readiness': 'blocked', 'blockers': blockers(d)} for d in drafts], 'entries': 126, 'result_rows': 126, 'market_combinations': 8232, 'original_history_slots_retained': 378, 'all_14_complete_row_exports_rejected': True, 'source_refs': refs, 'review_pins': PINS, 'archive': {'path': str(archive.relative_to(ROOT)), 'bytes': archive.stat().st_size, 'sha256': sha(archive.read_bytes()), 'member': 'component_drafts.json', 'member_bytes': len(payload), 'member_sha256': sha(payload)}, 'limitations': ['Component drafts use a separate schema and are not production-compatible full payloads.', '235 reviewed prior slots and 126 meeting-firstday observations have not yet been applied by the history adapter; 17 planned slots remain unresolved.', '13 candidate banks, global scope and existing 5123 bank gaps remain unresolved.']}
    for d in drafts:
        try: complete_row(d)
        except ValueError: pass
        else: raise AssertionError('incomplete draft exported')
    (out / 'component_draft_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k not in ['source_refs', 'review_pins', 'events']}, ensure_ascii=False))

if __name__ == '__main__': main()
