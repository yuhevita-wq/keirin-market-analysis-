import json,hashlib,re,copy,tarfile,zipfile,io,html as H
from html.parser import HTMLParser
from pathlib import Path
P=Path(__file__).resolve().parent
class Parser(HTMLParser):
 def __init__(self):
  super().__init__();self.links=[];self.a=None;self.text=[];self.skip=0
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag in ('script','style'):self.skip+=1
  if tag=='a':self.a=[d.get('href'),[]]
 def handle_data(self,s):
  if not self.skip:self.text.append(s)
  if self.a:self.a[1].append(s)
 def handle_endtag(self,tag):
  if tag in ('script','style'):self.skip-=1
  if tag=='a' and self.a:self.links.append((self.a[0],''.join(self.a[1])));self.a=None
def expected():
 d=P/'new_planned_2131';pins={'015869_OP_profile.html':'748c231d88bf649f678bde7f808d0efb519231bce6c55f39f39d2a263a4e5826','OP_015869_history_page1.html':'5499295d38792f1e5865c84957fa8027b1631ae23876a7185c2bad041dd4ecf5','WT_2026061646_cyclists.html':'a7dd1063e24bf58566536634fe5b7790734b95dc1d5bb47a9388debd3e05366a'}
 bodies={n:(d/n).read_bytes() for n in pins}
 for n,b in bodies.items():assert hashlib.sha256(b).hexdigest()==pins[n]
 meta={}
 for n,b in bodies.items():
  m=json.loads((d/(n+'.meta.json') if n!='015869_OP_profile.html' else d/'015869_OP_profile.meta.json').read_text())
  assert m['status']==200 and m['sha256']==pins[n] and m['bytes']==len(b);meta[n]=m
 assert meta['OP_015869_history_page1.html']['request_form']=={'playerCd':'015869','page':'1'}
 s=bodies['015869_OP_profile.html'].decode()
 assert 'params.playerCd = "015869"' in s and 'url : "PlayerDetailHistory.do"' in s and 'raceHistoryUpdateForm_Paging(1)' in s
 rows=[]
 for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>',bodies['OP_015869_history_page1.html'].decode(),re.S):
  cells=[H.unescape(re.sub('<[^>]+>','',c)).strip() for c in re.findall(r'<td\b[^>]*>(.*?)</td>',tr,re.S)]
  if cells[:2]==['2026/6/16-6/18','富山']:rows.append(cells)
 assert len(rows)==1 and rows[0]==['2026/6/16-6/18','富山','ＣＴＣなら３分前まで買える杯','負傷欠場']
 w=Parser();w.feed(bodies['WT_2026061646_cyclists.html'].decode())
 links=[(u,''.join(t.split())) for u,t in w.links if u=='/keirin/cyclist/015869'];assert len(links)==1 and '杉浦颯太' in links[0][1] and '127期' in links[0][1]
 text=''.join(''.join(w.text).split());assert '2026年6月16日' in text and '6月18日' in text and '富山' in text
 z=zipfile.ZipFile(P/'restored_2131/finals_checkpoint_updated.zip');b=z.read('finals_ai/data/scope_review_0106/history_adapter_overlay.tar.xz')
 assert hashlib.sha256(b).hexdigest()=='c81b9260ef8e3fb8f5d2186fc6477e45fdd953cec5c1a4118d69513628290248'
 t=tarfile.open(fileobj=io.BytesIO(b));b=t.extractfile('history_adapter_overlay.json').read();assert hashlib.sha256(b).hexdigest()=='53a19c888f8063062d7541ee8879e95796d2999c6c57d5e597260a0c8d765522'
 j=json.loads(b);matches=[s for e in j['events'] if e['local_key']=='keirin:2026-06-30:23:09' for en in e['entries'] if en['car_no']==2 and en['registration_number']=='015869' for s in en['adapted_history_slots'] if s['slot']=='history_data2'];assert len(matches)==1
 return matches[0],rows[0],links[0],{n:{'sha256':pins[n],'bytes':len(b),'metadata':meta[n]} for n,b in bodies.items()}
def verify(r):
 old,cells,link,manifest=expected()
 assert r['local_key']=='keirin:2026-06-30:23:09' and r['registration_number']=='015869' and r['car_no']==2 and r['slot']=='history_data2'
 assert r['original_slot']==old and r['reviewed_scheduled_start_date']=='2026-06-16' and r['reviewed_scheduled_end_date']=='2026-06-18'
 assert r['OP_raw_cells']==cells and r['WT_explicit_anchor']=={'href':link[0],'text':link[1]} and r['source_manifest']==manifest
 assert r['date_and_explicit_ID_providers']==['oddspark_explicit_profile_request_history_row','winticket_explicit_event_entrant_profile_anchor']
 assert r['withdrawal_evidence']=={'raw':'負傷欠場','provider':'oddspark','independently_confirmed':False}
 assert r['actual_start_date'] is None and r['new_completed_finals']==0 and r['canonical_planned_unresolved']==17 and r['pending_date_proof_added']==1
 for k in ['actual_participation_inferred','historical_stats_imported','original_slot_changed','canonical_DB_changed','canonical_saved','all_finals_complete']:assert r[k] is False
 return True
if __name__=='__main__':
 r=json.loads((P/'planned_slot_review_2131.json').read_text());verify(r)
 changes={'wrong_ID':('registration_number','015870'),'wrong_date':('reviewed_scheduled_start_date','1988-06-16'),'actual_participation':('actual_participation_inferred',True),'canonical_save_claim':('canonical_saved',True),'complete_claim':('all_finals_complete',True),'current_stats_import':('historical_stats_imported',True),'canonical_count_decrement':('canonical_planned_unresolved',16),'invent_actual_date':('actual_start_date','2026-06-16')}
 rejected=[]
 for label,(k,v) in changes.items():
  m=copy.deepcopy(r);m[k]=v
  try:verify(m)
  except AssertionError:rejected.append(label)
  else:raise AssertionError(label)
 assert len(rejected)==8
 out={'schema':'independent_planned_slot_source_audit_v1','producer_imported':False,'raw_parser':'stdlib HTMLParser plus independent table-cell regex','source_date_ID_pair_verified':True,'single_provider_withdrawal_preserved':True,'mutations_rejected':rejected,'canonical_DB_changed':False,'canonical_saved':False,'all_finals_complete':False,'new_completed_finals':0}
 (P/'independent_planned_slot_audit_2131.json').write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print('independent audit passed; 8 mutations rejected')
