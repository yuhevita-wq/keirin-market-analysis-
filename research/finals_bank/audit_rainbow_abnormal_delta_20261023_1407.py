"""Audit explicit historical fall-retirement evidence and lossless delta to 1307.
Run from workspace cwd with saved 0831/0914/1206/1407 archives extracted.
"""
import pathlib,json,copy,hashlib,importlib.util,sqlite3,tarfile
ROOT=pathlib.Path('current_data/unpacked/finals_ai');OUT=ROOT/'data/scope_review_1407'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 spec=importlib.util.spec_from_file_location('r1407',ROOT/'recovery/review_rainbow_history_abnormal_1407.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.review();new=json.loads(pathlib.Path('history_review_1407.json').read_text())
 original=ROOT/'data/scope_review_0914/independent_history_review.json';base=json.loads(original.read_text());bp=ROOT/'data/scope_review_1307/history_postponement_delta_review.json';d=json.loads(bp.read_text());assert sha(original.read_bytes())==d['baseline_report_sha256']
 for c in d['delta']:base['events'][c['event_index']]['entries'][c['entry_index']][c['field']][c['slot_index']]=c['after']
 for k in ['independent_prior_slots_status','Chariloto_slot_date_status']:base[k]=d[k]
 changed=[];recon=copy.deepcopy(base);e=next(x for x in new['events'] if x['day']=='2023-12-05');a=e['entries'][4];s=next(s for s in a['prior_OP_slots'] if s['status'].endswith('explicit_fall_retirement_detail'));nk=s['NK_group']['results'][0];rid=int(a['registration_number']);args=('落',nk,rid,a['name'],115);assert m.explicit_abnormal_detail(*args)
 tests=[]
 for name,row,op in [('other_race',dict(nk,race_id='202310194711'),'落'),('other_abnormal_type',nk,'失')]:
  assert m.explicit_abnormal_detail(op,row,rid,a['name'],115) is None;tests.append({'mutation':name,'not_resolved_by_global_token_mapping':True})
 p=pathlib.Path('raw_abnormal_1407/202310194712.html');mp=p.with_suffix('.json');manifest=OUT/'raw_manifest.json';pb=p.read_bytes();mb=mp.read_bytes();manb=manifest.read_bytes()
 for label,old,value in [('explicit_detail','落車棄権','病気欠場'),('profile_ID','id=15366','id=15367'),('race_key','value="202310194712"','value="202310194711"')]:
  try:
   assert old.encode() in pb;b=pb.replace(old.encode(),value.encode());p.write_bytes(b);meta=json.loads(mb);meta.update(bytes=len(b),sha256=sha(b));mp.write_text(json.dumps(meta,ensure_ascii=False));mani=json.loads(manb);mani[p.name]={'bytes':len(b),'sha256':sha(b)};mani[mp.name]={'bytes':mp.stat().st_size,'sha256':sha(mp.read_bytes())};manifest.write_text(json.dumps(mani))
   try:m.explicit_abnormal_detail(*args)
   except AssertionError:tests.append({'mutation':label,'rejected_after_recomputed_meta_and_manifest_hash':True})
   else:raise AssertionError('accepted '+label)
  finally:p.write_bytes(pb);mp.write_bytes(mb);manifest.write_bytes(manb)
 for ei,(be,ne) in enumerate(zip(base['events'],new['events'])):
  for ai,(ba,na) in enumerate(zip(be['entries'],ne['entries'])):
   assert ba['registration_number']==na['registration_number'] and ba['NK_all_raw_groups']==na['NK_all_raw_groups']
   for field in ['prior_OP_slots','Chariloto_prior_slots']:
    for si,(bs,ns) in enumerate(zip(ba[field],na[field])):
     if bs!=ns:
      changed.append({'event_index':ei,'entry_index':ai,'slot_index':si,'day':ne['day'],'venue_code':ne['venue_code'],'car_no':na['car_no'],'registration_number':na['registration_number'],'field':field,'before_status':bs['status'],'after':ns});recon['events'][ei]['entries'][ai][field][si]=ns
 for k in ['independent_prior_slots_status','Chariloto_slot_date_status']:recon[k]=new[k]
 assert recon==new and len(changed)==13
 assert new['independent_prior_slots_status']=={'independently_verified_dates_and_results':243,'independently_verified_dates_and_results_with_explicit_fall_retirement_detail':8,'unresolved_empty_OP_slot':1}
 assert new['Chariloto_slot_date_status']=={'independently_verified_explicit_full_date':230,'empty_planned_slot_date_unresolved':17,'independently_verified_explicit_full_date_with_abnormal_detail':5}
 rawmanifest=json.loads(manb)
 with tarfile.open(OUT/'abnormal_results_raw.tar.xz') as t:
  assert len(t.getnames())==len(set(t.getnames()))==16 and set(t.getnames())==set(rawmanifest)
  for name,v in rawmanifest.items():
   b=t.extractfile(name).read();assert len(b)==v['bytes'] and sha(b)==v['sha256']
 report={k:v for k,v in new.items() if k not in ['events','scope_note']};report.update(baseline_report=str(bp.relative_to(ROOT)),baseline_report_sha256=sha(bp.read_bytes()),reconstruction='First apply baseline1307 delta to its declared0914 full report, then apply this delta; update top counters from each delta. Exact equality to regenerated full1407 verified.',delta=changed,delta_reconstruction_equals_full_review=True,all_original_NK_groups_preserved=True,mutations=tests,raw_archive='data/scope_review_1407/abnormal_results_raw.tar.xz',raw_archive_sha256=sha((OUT/'abnormal_results_raw.tar.xz').read_bytes()),raw_archive_bytes=(OUT/'abnormal_results_raw.tar.xz').stat().st_size,note='Eight exact racekey/explicit-ID pairs only: NK result finish棄 with explicit margin落車棄権 resolves OP history落. Both raw tokens and all original rows remain. No global落=棄 mapping; no numeric rank fill; no malformed raw date repair. Rider names never infer IDs.')
 if pathlib.Path('db_recovery_1407.json').exists():report['working_DB_corruption']=json.loads(pathlib.Path('db_recovery_1407.json').read_text())
 (OUT/'history_abnormal_delta_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 db=ROOT/'data/finals.sqlite';assert db.stat().st_size==281206784 and sha(db.read_bytes())=='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91';assert not any(db.with_name(db.name+s).exists() for s in ['-wal','-shm','-journal']);c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);assert c.execute('pragma integrity_check').fetchone()==('ok',) and c.execute('select count(*) from races').fetchone()==(5354,);c.close();print('13 field delta, 5 mutations, archive16members, DB unchanged healthy')
if __name__=='__main__':main()
