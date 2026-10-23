import pathlib,json,hashlib,copy,importlib.util,sqlite3
root=pathlib.Path('current_data/unpacked/finals_ai');d=root/'data/scope_review_1206';spec=importlib.util.spec_from_file_location('review1206',root/'recovery/review_rainbow_history_venue13_1206.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.validated_venue13_alias()
mut=[]
for field,value in [('vel_code','14'),('open_day','2023-11-25'),('race_num',8)]:
 p=pathlib.Path('raw_alias_1206/chariloto.html');orig=p.read_bytes();mp=d/'raw_manifest.json';mo=mp.read_bytes()
 try:
  c=json.loads(orig);c['box'][field]=value;b=json.dumps(c,ensure_ascii=False).encode();p.write_bytes(b);manifest=json.loads(mo);manifest[p.name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};mp.write_text(json.dumps(manifest))
  try:m.validated_venue13_alias()
  except AssertionError:mut.append({'mutation':field,'rejected':True})
  else:raise RuntimeError('mutation accepted')
 finally:p.write_bytes(orig);mp.write_bytes(mo)
slot={'venue_header':'平F2','results':[{'month':11,'day':24,'finish_text':'2'},{'month':11,'day':25,'finish_text':'1'},{'month':11,'day':26,'finish_text':'1'}]}
g={'venue_code':'14','venue_name':'いわき平','group_id':'2023112414','start_date':'2023-11-24','results':[{'date':'2023-11-'+str(day),'race_id':'202311'+str(day)+'1409','finish_text':rank} for day,rank in [(24,'2'),(25,'1'),(26,'1')]]}
assert m.match_slot(slot,[g])['status']=='unresolved_date_venue_sequence';mut.append({'mutation':'other_venue_code14','rejected':True})
g['venue_code']='13';g['venue_name']='other';assert m.match_slot(slot,[g])['status']=='unresolved_date_venue_sequence';mut.append({'mutation':'other_venue_name','rejected':True})
(d/'mutation_review.json').write_text(json.dumps(mut,indent=2))
bp=root/'data/scope_review_0914/independent_history_review.json';base=json.loads(bp.read_text());new=json.loads(pathlib.Path('history_review_1206.json').read_text());reconstructed=copy.deepcopy(base);delta=[]
for ei,(be,ne) in enumerate(zip(base['events'],new['events'])):
 for ai,(ba,na) in enumerate(zip(be['entries'],ne['entries'])):
  assert ba['registration_number']==na['registration_number']
  for field in ['prior_OP_slots','Chariloto_prior_slots']:
   for si,(bs,ns) in enumerate(zip(ba[field],na[field])):
    if bs!=ns:
     delta.append({'event_index':ei,'entry_index':ai,'slot_index':si,'day':ne['day'],'venue_code':ne['venue_code'],'race_no':ne['race_no'],'car_no':na['car_no'],'registration_number':na['registration_number'],'field':field,'before_status':bs['status'],'after':ns})
     reconstructed['events'][ei]['entries'][ai][field][si]=ns
for k in ['independent_prior_slots_status','Chariloto_slot_date_status']:reconstructed[k]=new[k]
assert reconstructed==new
report={k:v for k,v in new.items() if k not in ['events','scope_note']};report.update(baseline_report=str(bp.relative_to(root)),baseline_report_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),delta=delta,delta_reconstruction_equals_full_review=True,raw_archive='data/scope_review_1206/venue13_raw.tar.xz',raw_archive_sha256=hashlib.sha256((d/'venue13_raw.tar.xz').read_bytes()).hexdigest(),new_raw_members=4,source_note='Chariloto box13 2023-11-26 9R + explicit15049/currenthistory1平 11/24R5rank2/11/25R5rank1; OP same actualcode/date/R/ID and currentday history; saved NK explicit15049 group2023112413/racekeys match. Venue alias only. Empty raw venue codes and malformed raw years are retained unchanged. No rider-ID inference. No score or bank applicability extension.',failed_discovery_request={'url':'https://www.oddspark.com/keirin/RaceList.do?joCode=13&kaisaiBi=20231126&raceNo=12','result':'web inaccessible; actual saved NK final9R used; no retry12'},unresolved_preserved='8 OP落/NK棄 differences;1 OPempty;17 emptyplanned;6 nonempty mapping;currentfirstday126singleprovider;line/bank/key/scope pending')
(d/'history_alias_delta_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
p=root/'data/merged_integrity.json';merged=json.loads(p.read_text());merged['independent_rainbow_history_review_1206']={k:new[k] for k in ['independent_prior_slots_status','Chariloto_slot_date_status','canonical_db_changed','all_finals_complete']};merged['independent_rainbow_history_review_1206']['report']='data/scope_review_1206/history_alias_delta_review.json';p.write_text(json.dumps(merged,ensure_ascii=False,indent=2))
db=root/'data/finals.sqlite';assert db.stat().st_size==281206784 and hashlib.sha256(db.read_bytes()).hexdigest()=='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91'
assert not any(db.with_name(db.name+x).exists() for x in ['-wal','-shm','-journal'])
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);assert c.execute('pragma integrity_check').fetchone()==('ok',);assert c.execute('select count(*) from races').fetchone()==(5354,);c.close()
print('mutations',len(mut),'delta',len(delta),'DB healthy unchanged')
print(json.dumps({k:v for k,v in report.items() if k in ['independent_prior_slots_status','Chariloto_slot_date_status']},ensure_ascii=False))
