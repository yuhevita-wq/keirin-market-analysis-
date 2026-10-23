"""Deterministic extension of SHA-pinned saved1307 reviewer; no copied baseline code."""
import pathlib
root=pathlib.Path('current_data/unpacked/finals_ai');s=(root/'recovery/review_rainbow_history_postponement_1307.py').read_text();assert __import__('hashlib').sha256(s.encode()).hexdigest()=='356256e111ab468463552a66c8ecd00b4500a35c192904c49420f18d9826a5cf'
insert='''def explicit_abnormal_detail(op_finish,nk_row,rid,name,term):
 if op_finish==nk_row['finish_text']:return {'raw_tokens_exact':True}
 if (op_finish,nk_row['finish_text'])!=('落','棄'):return None
 allowed={('202310194712',15366),('202308303705',13868),('202505094405',15323),('202511273605',14942),('202511048607',15922),('202605214404',14775),('202605026304',14775),('202604296209',15178)}
 k=nk_row['race_id']
 if (k,rid) not in allowed:return None
 q=urllib.parse.parse_qs(urllib.parse.urlparse(nk_row['href']).query);assert q['race_id']==[k] and q['player_id']==[str(rid)]
 manifest=json.loads((ROOT/'data/scope_review_1407/raw_manifest.json').read_text());p=pathlib.Path('raw_abnormal_1407')/(k+'.html');mp=p.with_suffix('.json')
 for f in [p,mp]:
  b=f.read_bytes();v=manifest[f.name];assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 meta=json.loads(mp.read_text());b=p.read_bytes();assert meta['status']==200 and len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256']
 u=urllib.parse.urlparse(meta['url']);assert u.scheme=='https' and u.netloc=='keirin.netkeiba.com' and u.path=='/race/result/' and urllib.parse.parse_qs(u.query)=={'race_id':[k]} and meta['final_url']==meta['url']
 doc=html.fromstring(b);assert doc.xpath('//input[@name="plp-raceid"]/@value')==[k]
 found=[]
 for row in doc.xpath('//table[@id="All_Result_Table"]//tr[td]'):
  cs=row.xpath('./td');assert len(cs)==8
  links=cs[3].xpath('.//a/@href');assert len(links)==1
  actual_id=urllib.parse.parse_qs(urllib.parse.urlparse(links[0]).query)['id'][0]
  if int(actual_id)==rid:found.append(cs)
 assert len(found)==1;cs=found[0];assert [norm(x) for x in cs[3].xpath('.//img/@alt')]==[norm(name)]
 terms=re.findall(r'(\\d+)期',norm(cs[3].text_content()));assert terms==[str(int(term))]
 assert norm(cs[0].text_content())=='棄' and '落車棄権' in norm(cs[4].text_content())
 assert norm(cs[4].text_content()) in ['-(落車棄権)','−(落車棄権)']
 return {'raw_tokens_exact':False,'OP_raw_finish':op_finish,'NK_history_raw_finish':nk_row['finish_text'],'NK_result_raw_finish':cs[0].text_content(),'explicit_result_detail_raw':cs[4].text_content(),'race_id':k,'registration_number':str(rid).zfill(6),'result_car_no':int(norm(cs[2].text_content())),'source_member':p.name,'source_sha256':meta['sha256'],'source_bytes':meta['bytes'],'resolution':'event_and_explicit_ID_specific_fall_retirement_detail; no global token equivalence or raw replacement'}

'''
s=s.replace('def match_slot(slot,groups):',insert+'def match_slot(slot,groups,rid=None,name=None,term=None):')
s=s.replace("g=matches[0];different=[]", "g=matches[0];different=[];unresolved=[]")
s=s.replace("if op['finish_text']!=nk['finish_text']:different.append({'date':nk['date'],'OP':op['finish_text'],'NK':nk['finish_text']})", "if op['finish_text']!=nk['finish_text']:\n   evidence=explicit_abnormal_detail(op['finish_text'],nk,rid,name,term);rec={'date':nk['date'],'OP':op['finish_text'],'NK':nk['finish_text'],'detailed_result_resolution':evidence};different.append(rec)\n   if evidence is None:unresolved.append(rec)")
s=s.replace("'independently_verified_dates_and_results' if not different else 'dates_matched_result_text_discrepancy_unresolved'", "('independently_verified_dates_and_results_with_explicit_fall_retirement_detail' if different else 'independently_verified_dates_and_results') if not unresolved else 'dates_matched_result_text_discrepancy_unresolved'")
s=s.replace("match_slot(op_slot(c),groups) for c", "match_slot(op_slot(c),groups,rid,a['name'],a['graduate_period']) for c")
s=s.replace("norm(x.get('note',''))==n['finish_text']", "explicit_abnormal_detail(norm(x.get('note','')),n,rid,a['name'],a['graduate_period']) is not None")
s=s.replace("s['status']=='independently_verified_dates_and_results'", "s['status'] in ['independently_verified_dates_and_results','independently_verified_dates_and_results_with_explicit_fall_retirement_detail']")
s=s.replace("'independently_verified_explicit_full_date' if len(verified)==1", "('independently_verified_explicit_full_date_with_abnormal_detail' if any(n['finish_text']=='棄' for n in verified[0]['results']) else 'independently_verified_explicit_full_date') if len(verified)==1")
s=s.replace('history_review_1307.json','history_review_1407.json')
exec(compile(s,__file__,'exec'),globals())
