"""Read-only identity interface. Never rewrites provider IDs or original payloads.

This interface is separate from collection and SQL insertion. A valid identity
does not imply a complete race, historical bank, or permission to insert it.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'recovery/review_key_adapter_1918.py': '8be612e5ca2c4611bb53c93a4ca4a585c93a5c927655709e7ef272c950f3bd24',
    'data/scope_review_1746/source_key_mapping_review.json': '755a2c57cbfb5b4c4e66a326d089ba81005b7af780e36d457072d06e3865673a',
    'data/scope_review_2113/kochi_actual_number_review.json': '26821a2c614487f14b6c26db807a73d27624e4ebc215fcc85af6ede2fa5f2cdc',
    'data/scope_review_2113/legacy4_actual_number_review.json': 'e2d0a02971c4803fd21b25597bc8301ab163be0d9e932db26a72acb2c4c37786',
    'data/scope_review_2113/raw.tar.xz': '2e5e2f0d1a45aa945ad4afaf753f75742a21caa0c07f4648a0fe3e4e04be834d',
    'data/scope_review_2113/legacy4_raw.tar.xz': '74c474a8e9666c149004cb8b124b39dba3e49d0be0e2eb0f1533276b71a7f5d4',
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

class IdentityAdapter:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        for name, expected in PINS.items():
            if sha((self.root / name).read_bytes()) != expected:
                raise ValueError('identity evidence mismatch: ' + name)
        spec = importlib.util.spec_from_file_location('identity_rules_1918', self.root / 'recovery/review_key_adapter_1918.py')
        self.rules = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.rules)
        k = json.loads((self.root / 'data/scope_review_2113/kochi_actual_number_review.json').read_text())
        four = json.loads((self.root / 'data/scope_review_2113/legacy4_actual_number_review.json').read_text())
        k = dict(k, legacy_race_id=k['legacy_id_retained'])
        self._mappings = {e['legacy_race_id']: e for e in [k] + four['events']}
        if len(self._mappings) != 5:
            raise ValueError('expected five individually verified mappings')
        events = json.loads((self.root / 'data/scope_review_1746/source_key_mapping_review.json').read_text())['events']
        self._local_events = {e['local_composite_key']: e for e in events}
        if len(self._local_events) != 14:
            raise ValueError('expected fourteen observed local identities')

    def legacy(self, race_id, day, url, raw_payload):
        """raw_payload is the exact stored SQL text, not reserialized JSON."""
        if type(raw_payload) is not str:
            raise TypeError('exact stored payload text required')
        p = json.loads(raw_payload)
        if race_id not in self._mappings:
            identity = self.rules.legacy_identity(race_id, day, url, p)
            return dict(identity, original_race_no=p['race']['race_no'], number_basis='observed_legacy_payload')
        m = self._mappings[race_id]
        if sha(raw_payload.encode('utf-8')) != m['legacy_payload_sha256']:
            raise ValueError('mapped original payload changed')
        r = p['race']
        a = self.rules.actual(**{'day': m['actual_tuple']['day'], 'venue': m['actual_tuple']['venue_code'], 'rno': m['actual_tuple']['race_no']})
        if not (day == a['day'] == r['race_date'] and race_id[:2] == a['venue_code']
                and race_id == p['race_id'] == r['race_id']
                and type(r['race_no']) is type(m['raw_race_no_retained'])
                and r['race_no'] == m['raw_race_no_retained']
                and '/racedetail/' + race_id + '/' in url
                and urlparse(url).path == urlparse(r['source_url']).path):
            raise ValueError('mapping must match exact original row and observed tuple')
        return dict(a, canonical_key=race_id, provider='kdreams_observed_legacy',
                    confirmed_kdreams_race_id=race_id, original_race_no=r['race_no'],
                    number_basis='independently_verified_exact_source_mapping_2113',
                    original_payload_sha256=m['legacy_payload_sha256'])

    def local(self, event):
        # Shape validation alone cannot establish an unobserved new event.
        key = event.get('local_composite_key')
        if key not in self._local_events or event != self._local_events[key]:
            raise ValueError('unverified or changed local source mapping')
        return self.rules.local_identity(copy.deepcopy(event))

    def select_bank(self, records, identity, track):
        return self.rules.select_explicit(records, identity, track)
