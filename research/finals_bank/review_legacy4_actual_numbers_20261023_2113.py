"""Four exact legacy-number mappings via observed day-card links and independent results.
Retain source fields and provider IDs; no arithmetic correction and no SQL writes.
"""
import pathlib,json,re,hashlib,sqlite3,tarfile,io,copy,urllib.parse,importlib.util
ROOT=pathlib.Path('current_data/unpacked/finals_ai');OUT=ROOT/'data/scope_review_2113'
spec=importlib.util.spec_from_file_location('kochi',ROOT/'recovery/review_kochi_actual_number_2113.py');k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)
CONFIG=[('8620230123030111','86',11,'Ａ級決勝','766b5b68d31651fa90d0212cc7366b5013b81d43d2063709f2c1ce0d0242c261'),('8620230123030112','86',12,'Ｓ級決勝','5ba4ccdca156d14b9301f10154a9120692aec156811cc034f691ecf582996f4a'),('5520230123030105','55',5,'Ａ級チ決勝','a9133a267e89b6fb2b3eebe41cefb2dbbd58bca3b994c3baf2e307f80ac04cea'),('5520230123030111','55',11,'Ａ級決勝','70fb2c0945ae8c5f96132457b145e1f9c0335f16da5c47affd2c49db86f5853d')]
def verify(files,conf,p):
 rid,venue,rno,label,ph=conf;key='20230126'+venue+f'{rno:02d}';assert p['race_id']==p['race']['race_id']==rid and p['race']['race_date']=='2023-01-26'
 dayname='op_beppu_day' if venue=='86' else 'op_wakayama_day';day,dm=k.body(files,dayname);op,om=k.body(files,key+'_card');nk,nm=k.body(files,key+'_result')
 url='https://www.oddspark.com/keirin/RaceList.do?joCode='+venue+'&kaisaiBi=20230126&raceNo='+str(rno)
 assert dm['url']==dm['final_url']=='https://www.oddspark.com/keirin/AllRaceList.do?joCode='+venue+'&kaisaiBi=20230126'
 hit=[]
 for a in day.xpath('//a[contains(@href,"RaceList.do")]'):
  if urllib.parse.urljoin(dm['url'],a.get('href'))==url and k.norm(a.getparent().text_content()).startswith('第'):hit.append(k.norm(a.getparent().text_content()))
 assert hit==[k.norm('第'+str(rno)+'R'+label)]
 assert om['url']==om['final_url']==url and nm['url']==nm['final_url']=='https://keirin.netkeiba.com/race/result/?race_id='+key
 for field,val in [('joCode',venue),('kaisaiBi','20230126'),('raceNo',str(rno))]:assert op.xpath('//input[@id="'+field+'"]/@value')==[val]
 assert nk.xpath('//input[@name="plp-raceid"]/@value')==[key]
 ot=k.norm(op.findtext('.//title'));nt=k.norm(nk.findtext('.//title'));track='別府' if venue=='86' else '和歌山';nlabel='チャレンジ決勝' if rno==5 else '決勝'
 assert '2023年1月26日(木)'+track+'競輪'+str(rno)+'R' in ot and k.norm(label) in ot
 assert '2023年01月26日'+str(rno)+'R'+nlabel in nt and track+'競輪' in nt
 ath={int(a['car_no']):a for a in p['entries']};refs={int(a['car_no']):a for a in p['results']};assert set(ath)==set(refs)==set(range(1,8));opids={}
 for row in op.xpath('//table[@summary="出走表"]/tr[td]'):
  cs=row.xpath('./td');ns=[c for c in cs if re.fullmatch(r'no[1-9]',c.get('class',''))]
  if not ns:continue
  assert len(ns)==1;car=int(k.norm(ns[0].text_content()));assert car not in opids;a=ath[car];links=cs[-8].xpath('.//a');assert len(links)==1;u=urllib.parse.urlparse(links[0].get('href'));assert u.path.endswith('/PlayerDetail.do');number=urllib.parse.parse_qs(u.query)['playerCd'][0];reg=a['rider_registration_number'];assert re.fullmatch(r'[0-9]{6}',reg) and int(number)==int(reg)
  source=urllib.parse.urlparse(a['rider_source_id']);assert source.hostname in ['keirin.jp','www.keirin.jp'] and urllib.parse.parse_qs(source.query)['snum']==[reg]
  assert k.norm(links[0].text_content())==k.norm(a['player_name']);age=re.search(r'(\d+)歳/(\d+)期',k.norm(cs[-8].text_content()));assert age and age.groups()==(str(a['age']),str(a['term']));assert k.norm(cs[-6].text_content())==a['class'];opids[car]={'registration_number':reg,'name':k.norm(a['player_name']),'term':str(a['term']),'OP_profile_href':links[0].get('href')}
 assert set(opids)==set(ath);out=[]
 for row in nk.xpath('//table[@id="All_Result_Table"]//tr[td]'):
  cs=row.xpath('./td');assert len(cs)==8;car=int(k.norm(cs[2].text_content()));assert car not in [x['car_no'] for x in out];a=ath[car];links=cs[3].xpath('.//a/@href');assert len(links)==1;u=urllib.parse.urlparse(links[0]);assert u.hostname=='keirin.netkeiba.com' and u.path=='/db/profile/' and int(urllib.parse.parse_qs(u.query)['id'][0])==int(a['rider_registration_number'])
  assert [k.norm(s) for s in cs[3].xpath('.//img/@alt')]==[k.norm(a['player_name'])];info=k.norm(cs[3].text_content());age=re.search(r'(\d+)歳',info);cl=re.search(r'(\d+)期([ASL][123])',info);assert age and cl and age[1]==str(a['age']) and cl.groups()==(str(a['term']),a['class'])
  rank=re.sub(r'着$','',k.norm(cs[0].text_content()));assert rank==str(refs[car]['finish_text']);out.append(dict(opids[car],car_no=car,NK_profile_href=links[0],finish_text=rank))
 assert len(out)==7 and {x['car_no'] for x in out}==set(ath)
 return {'legacy_race_id':rid,'raw_race_no_retained':p['race']['race_no'],'actual_tuple':{'day':'2023-01-26','venue_code':venue,'race_no':rno},'actual_OP_card_href':url,'day_card_link_parent_label':hit[0],'observed_NK_race_id':key,'riders':sorted(out,key=lambda x:x['car_no']),'source_records':{'day':dm,'card':om,'result':nm},'legacy_payload_sha256':ph,'general_100_subtraction_rule':False,'KD_encoding_cause_claimed':False}
def main():
 db=ROOT/'data/finals.sqlite';assert db.stat().st_size==281206784 and k.sha(db.read_bytes())==k.DBSHA;assert not any(pathlib.Path(str(db)+s).exists() for s in ['-wal','-shm','-journal'])
 archive=OUT/'legacy4_raw.tar.xz'
 if archive.exists():
  with tarfile.open(archive) as t:
   assert len(t.getnames())==len(set(t.getnames()))==20;files={m.name:t.extractfile(m).read() for m in t.getmembers()}
 else:files={p.name:p.read_bytes() for root in ['raw_legacy4_2113','raw_discovery_2113'] for p in pathlib.Path(root).iterdir() if p.is_file()}
 assert len(files)==20
 reviews=[];muts=[]
 with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as c:
  assert c.execute('pragma integrity_check').fetchall()==[('ok',)] and c.execute('select count(*) from races').fetchone()[0]==5354
  for conf in CONFIG:
   raw=c.execute('select payload from races where race_id=?',(conf[0],)).fetchone()[0];assert k.sha(raw.encode())==conf[4];p=json.loads(raw);reviews.append(verify(files,conf,p));key='20230126'+conf[1]+f'{conf[2]:02d}'
   for kind in ['hidden_key','explicit_ID','rank']:
    doc,meta=k.body(files,key+'_result')
    if kind=='hidden_key':doc.xpath('//input[@name="plp-raceid"]')[0].set('value','202301269912')
    elif kind=='explicit_ID':doc.xpath('//table[@id="All_Result_Table"]//a')[0].set('href','https://keirin.netkeiba.com/db/profile/?id=99999')
    else:doc.xpath('//table[@id="All_Result_Table"]//tr[td]')[0].xpath('./td')[0].text='9着'
    b=k.html.tostring(doc,encoding='utf8');meta.update(sha256=k.sha(b),bytes=len(b));f=dict(files);f[key+'_result.html']=b;f[key+'_result.json']=json.dumps(meta).encode()
    try:verify(f,conf,p)
    except AssertionError:muts.append(conf[0]+':'+kind)
    else:raise AssertionError('wrong evidence accepted')
 assert len(reviews)==4 and len(muts)==12
 if not archive.exists():
  with tarfile.open(archive,'w:xz',preset=9) as t:
   for name,b in sorted(files.items()):
    m=tarfile.TarInfo(name);m.size=len(b);t.addfile(m,io.BytesIO(b))
 manifest={name:{'bytes':len(b),'sha256':k.sha(b)} for name,b in sorted(files.items())}
 with tarfile.open(archive) as t:assert {m.name:{'bytes':m.size,'sha256':k.sha(t.extractfile(m).read())} for m in t.getmembers()}==manifest
 assert k.sha(db.read_bytes())==k.DBSHA
 report={'status':'four_exact_legacy_actual_number_mappings_independently_verified','events':reviews,'raw_members':manifest,'archive':{'bytes':archive.stat().st_size,'sha256':k.sha(archive.read_bytes())},'mutations_rejected':muts,'verified_legacy_mappings_this_report':4,'combined_with_separate_Kochi_report':5,'remaining_of_original5':0,'canonical_DB_changed':False,'SQL_rows_inserted':0,'new_completed_finals':0,'all_finals_complete':False,'prior_1918_report_raw_values_unchanged':True,'note':'Exact source-specific mappings only. All 35 identities/results across five races need preservation of provider IDs/raw race_no. Full SQL payload adapter and bank/scope work remain incomplete.'}
 (OUT/'legacy4_actual_number_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True));print(json.dumps({a:b for a,b in report.items() if a not in ['events','raw_members']},ensure_ascii=False))
if __name__=='__main__':main()
