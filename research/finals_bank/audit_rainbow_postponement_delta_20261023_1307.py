"""Reproduce 1307 full history review, then validate narrow exclusion and save lossless delta.
Extract 0831/0914/1206 archives to their documented workspace folders before running.
"""
import pathlib,json,hashlib,copy,importlib.util,sqlite3
ROOT=pathlib.Path('current_data/unpacked/finals_ai');OUT=ROOT/'data/scope_review_1307'
def main():
 spec=importlib.util.spec_from_file_location('review1307',ROOT/'recovery/review_rainbow_history_postponement_1307.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.review()
 n=json.loads(pathlib.Path('history_review_1307.json').read_text());a=next(e for e in n['events'] if e['day']=='2026-06-30')['entries'][0];s=a['Chariloto_prior_slots'][1];assert s['status']=='independently_verified_explicit_full_date';g=next(g for g in a['NK_all_raw_groups'] if g['group_id']=='2026053174');before=copy.deepcopy(g);v=m.actual_result_comparison_group(g);assert g==before and len(g['results'])==4 and len(v['results'])==3 and len(v['excluded_non_result_rows'])==1
 assert v['results'][-1]['date']=='2026-06-03' and v['results'][-1]['finish_text']=='1'
 mut=[]
 for name,fun in [('other_group',lambda x:x.update(group_id='2026053074')),('blank_other_key',lambda x:x['results'][2].update(race_id='202606027408')),('nonblank_finish',lambda x:x['results'][2].update(finish_text='1'))]:
  x=copy.deepcopy(g);fun(x);z=m.actual_result_comparison_group(x);assert len(z['results'])==4 and 'excluded_non_result_rows' not in z;mut.append({'mutation':name,'unrelated_or_real_result_preserved':True})
 for name,fun in [('wrong_venue',lambda x:x.update(venue_code='75')),('wrong_row_date',lambda x:x['results'][2].update(date='2026-06-01')),('missing_rescheduled_result',lambda x:x['results'][-1].update(finish_text=''))]:
  x=copy.deepcopy(g);fun(x)
  try:m.actual_result_comparison_group(x)
  except AssertionError:mut.append({'mutation':name,'rejected':True})
  else:raise AssertionError(name)
 bp=ROOT/'data/scope_review_0914/independent_history_review.json';base=json.loads(bp.read_text());recon=copy.deepcopy(base);delta=[]
 for ei,(be,ne) in enumerate(zip(base['events'],n['events'])):
  for ai,(ba,na) in enumerate(zip(be['entries'],ne['entries'])):
   assert ba['registration_number']==na['registration_number'] and ba['NK_all_raw_groups']==na['NK_all_raw_groups']
   for field in ['prior_OP_slots','Chariloto_prior_slots']:
    for si,(bs,ns) in enumerate(zip(ba[field],na[field])):
     if bs!=ns:
      delta.append({'event_index':ei,'entry_index':ai,'slot_index':si,'day':ne['day'],'venue_code':ne['venue_code'],'car_no':na['car_no'],'registration_number':na['registration_number'],'field':field,'before_status':bs['status'],'after':ns});recon['events'][ei]['entries'][ai][field][si]=ns
 for k in ['independent_prior_slots_status','Chariloto_slot_date_status']:recon[k]=n[k]
 assert recon==n and len(delta)==23
 assert n['entry_identities_verified']==126 and n['Chariloto_slot_date_status']=={'independently_verified_explicit_full_date':230,'empty_planned_slot_date_unresolved':17,'unresolved_slot_mapping_or_abnormal_text':5}
 r={k:v for k,v in n.items() if k not in ['events','scope_note']};r.update(baseline_report=str(bp.relative_to(ROOT)),baseline_report_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),delta=delta,delta_reconstruction_equals_full_review=True,new_network_history_raw=0,note='Only 2026053174/202606027407 empty finish is removed from comparison under saved postponement notice; all raw groups/rows retained. No numeric finish fill or date offset. OP and Chariloto comparisons use same evidence-backed view. Raw anomalous dates unchanged.')
 (OUT/'history_postponement_delta_review.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));(OUT/'history_mutation_review.json').write_text(json.dumps(mut,indent=2))
 db=ROOT/'data/finals.sqlite';assert db.stat().st_size==281206784 and hashlib.sha256(db.read_bytes()).hexdigest()=='04c4de11b6937c3b3a499f042410786cfdde4c2892ca0af46999ca059c5e2d91';assert not any(db.with_name(db.name+x).exists() for x in ['-wal','-shm','-journal']);c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);assert c.execute('pragma integrity_check').fetchone()==('ok',) and c.execute('select count(*) from races').fetchone()==(5354,);c.close();print('23 field lossless delta, 6 mutations, raw groups unchanged, DB healthy unchanged')
if __name__=='__main__':main()
