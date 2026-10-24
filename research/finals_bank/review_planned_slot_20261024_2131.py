import hashlib,json,re,zipfile,tarfile,io
from pathlib import Path
from lxml import html
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'new_planned_2131'
PINS={'015869_OP_profile.html':'748c231d88bf649f678bde7f808d0efb519231bce6c55f39f39d2a263a4e5826','OP_015869_history_page1.html':'5499295d38792f1e5865c84957fa8027b1631ae23876a7185c2bad041dd4ecf5','WT_2026061646_cyclists.html':'a7dd1063e24bf58566536634fe5b7790734b95dc1d5bb47a9388debd3e05366a'}
def sources():
 out={}
 for n,h in PINS.items():
  b=(SRC/n).read_bytes();assert hashlib.sha256(b).hexdigest()==h
  mp=SRC/(n+'.meta.json') if n!='015869_OP_profile.html' else SRC/'015869_OP_profile.meta.json'
  m=json.loads(mp.read_text());assert m['status']==200 and m['sha256']==h and m['bytes']==len(b)
  out[n]=(b,m)
 return out
def build():
 ss=sources();profile=ss['015869_OP_profile.html'][0].decode()
 assert re.search(r'params.playerCd\s*=\s*"015869"',profile)
 assert 'url : "PlayerDetailHistory.do"' in profile and 'raceHistoryUpdateForm_Paging(1)' in profile
 hm=ss['OP_015869_history_page1.html'][1]
 assert hm['request_form']=={'playerCd':'015869','page':'1'}
 op=html.fromstring(ss['OP_015869_history_page1.html'][0].decode('utf-8'))
 rows=[r for r in op.xpath('//tr') if [x.text_content().strip() for x in r.xpath('./td')][:2]==['2026/6/16-6/18','富山']]
 assert len(rows)==1
 cells=[x.text_content().strip() for x in rows[0].xpath('./td')]
 assert cells==['2026/6/16-6/18','富山','ＣＴＣなら３分前まで買える杯','負傷欠場']
 wt=html.fromstring(ss['WT_2026061646_cyclists.html'][0].decode())
 aa=wt.xpath('//a[@href="/keirin/cyclist/015869"]');assert len(aa)==1
 at=''.join(aa[0].text_content().split());assert '杉浦颯太' in at and '127期' in at
 text=''.join(wt.text_content().split());assert '2026年6月16日' in text and '6月18日' in text and '富山' in text
 z=zipfile.ZipFile(ROOT/'restored_2131/finals_checkpoint_updated.zip')
 b=z.read('finals_ai/data/scope_review_0106/history_adapter_overlay.tar.xz')
 assert hashlib.sha256(b).hexdigest()=='c81b9260ef8e3fb8f5d2186fc6477e45fdd953cec5c1a4118d69513628290248'
 t=tarfile.open(fileobj=io.BytesIO(b));raw=t.extractfile('history_adapter_overlay.json').read()
 assert hashlib.sha256(raw).hexdigest()=='53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522'
 old=json.loads(raw)
 ev=next(x for x in old['events'] if x['local_key']=='keirin:2026-06-30:23:09')
 en=next(x for x in ev['entries'] if x['registration_number']=='015869' and x['car_no']==2)
 slot=next(x for x in en['adapted_history_slots'] if x['slot']=='history_data2')
 assert slot['kind']=='planned_slot_date_unresolved' and slot['original']=={'grade':'F2','open_from':'19880616','short_boxes':[],'vel_code':'46','vel_name':'富　山','vel_short_name':'富'}
 return {'schema':'planned_slot_dated_source_review_v1','local_key':ev['local_key'],'car_no':2,'registration_number':'015869','slot':'history_data2','original_slot':slot,'reviewed_scheduled_start_date':'2026-06-16','reviewed_scheduled_end_date':'2026-06-18','date_and_explicit_ID_providers':['oddspark_explicit_profile_request_history_row','winticket_explicit_event_entrant_profile_anchor'],'OP_raw_cells':cells,'WT_explicit_anchor':{'href':'/keirin/cyclist/015869','text':at},'withdrawal_evidence':{'raw':'負傷欠場','provider':'oddspark','independently_confirmed':False},'actual_start_date':None,'actual_participation_inferred':False,'historical_stats_imported':False,'original_slot_changed':False,'canonical_DB_changed':False,'canonical_saved':False,'all_finals_complete':False,'new_completed_finals':0,'canonical_planned_unresolved':17,'pending_date_proof_added':1,'source_manifest':{n:{'sha256':PINS[n],'bytes':len(b),'metadata':m} for n,(b,m) in ss.items()}}
if __name__=='__main__':
 r=build();(ROOT/'planned_slot_review_2131.json').write_text(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print('review passed; canonical changed=0; completed=0')
