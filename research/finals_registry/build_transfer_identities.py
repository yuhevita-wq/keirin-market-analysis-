"""Join explicit JKA transfer IDs to already verified official term records."""
import pathlib,json,collections
def build(data):
    sources=json.loads((data/'registry_official_sources.json').read_text());moves=json.loads((data/'registry_official_transfers.json').read_text());groups=collections.defaultdict(list);out=[]
    for r in moves:groups[r['registration_number']].append(r)
    for number,events in groups.items():
        events.sort(key=lambda x:x['effective_date'])
        for i,r in enumerate(events):
            terms={s['term'] for s in sources if s['registration_number']==number and s['name']==r['name']}
            if len(terms)!=1:continue
            if i:assert events[i-1]['new_prefecture']==r['old_prefecture'],(number,'transfer chain')
            start=events[i-1]['effective_date'] if i else '2023-01-01'
            if start>=r['effective_date']:continue
            term=next(iter(terms));term_sources=[s for s in sources if s['registration_number']==number and s['name']==r['name'] and s['term']==term]
            out.append({'registration_number':number,'name':r['name'],'term':term,'prefecture':r['old_prefecture'],'source_url':r['source_url'],'source_line':r['source_line'],'source_kind':'official_transfer_and_registration_identity','valid_from':start,'valid_until_exclusive':r['effective_date'],'transfer':r,'term_evidence':[{'source_url':s['source_url'],'source_line':s['source_line'],'registration_number':number,'name':s['name'],'term':s['term']} for s in term_sources],'note':'Registration ID explicitly in dated JKA old/new prefecture notice; term independently verified using the same registration ID. Identity only.'})
    seen={(s['source_url'],s['source_line'],s['prefecture']) for s in sources}
    for r in out:
        if (r['source_url'],r['source_line'],r['prefecture']) not in seen:sources.append(r);seen.add((r['source_url'],r['source_line'],r['prefecture']))
    (data/'registry_official_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2));(data/'registry_transfer_identities.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('historical identity intervals',len(out))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);build(p.parse_args().data)
