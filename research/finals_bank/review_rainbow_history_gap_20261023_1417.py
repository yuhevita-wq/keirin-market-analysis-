"""Verify an alternate independent source for one empty OP prior-history slot.
No blank slot is filled, and no canonical DB row is inserted.
"""
import pathlib,json,hashlib,urllib.parse,re,unicodedata,copy,tarfile,io,lzma,importlib.util
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai');RAW=pathlib.Path('raw_gap_1417');OUT=ROOT/'data/scope_review_1417'
def norm(x):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(x)))
def checked(p,meta):
 b=p.read_bytes();assert len(b)==meta['bytes'] and hashlib.sha256(b).hexdigest()==meta['sha256'];return b

def source(stem):
 p=RAW/(stem+'.html');m=json.loads(p.with_suffix('.json').read_text());b=checked(p,m);assert m['status']==200 and m['url']==m['final_url'];return b,m

def actual(day,card,result,nkfinish):
 assert (card['box']['open_day'],card['box']['vel_code'],card['box']['race_num'])==(day,'73',3)
 assert card['box']['is_girls'] is False and card['box']['on_abort'] is False and card['box']['is_invalid'] is False
 athletes=list(card['race_athletes'].values());a=[a for a in athletes if int(a['un_id'])==15690];assert len(a)==1;a=a[0];assert a['entry_num']==1 and norm(a['name'])=='斉藤樂' and a['graduate_period']=='123' and a['pref_name']=='宮城'
 doc=html.fromstring(result);sec=doc.xpath('//*[@id="race3"]');assert len(sec)==1;table=sec[0].xpath('.//table[.//th[normalize-space()="選手名"] and .//th[normalize-space()="着"]]');assert len(table)==1;rows=table[0].xpath('./tbody/tr');allcars=[];target=None
 for row in rows:
  cs=row.xpath('./td');assert len(cs)==12;car=int(norm(cs[1].text_content()));allcars.append(car)
  if car==1:target=cs
 assert len(allcars)==len(set(allcars)) and set(allcars)==set(map(int,card['race_athletes']))
 assert target is not None;cs=target
 assert norm(cs[2].text_content())==norm(a['name']) and int(norm(cs[3].text_content()))==a['age'] and norm(cs[4].text_content())==a['pref_name'] and norm(cs[5].text_content())==a['graduate_period'] and norm(cs[6].text_content())==norm(a['class_name']+a['group_name'])
 finish=norm(cs[0].text_content());assert (day,finish,nkfinish) in [('2023-10-26','1','1'),('2023-10-27','落','棄')]
 return {'date':day,'venue_code':'73','race_no':3,'car_no':1,'registration_number':'015690','name':a['name'],'term':a['graduate_period'],'Chariloto_finish_raw':cs[0].text_content(),'NK_history_finish_raw':nkfinish,'result_cells_raw':[c.text_content() for c in cs]}

def main():
 op=pathlib.Path('raw_independent_0831/202312178509_card.html');om=json.loads(op.with_suffix('.meta.json').read_text());o=html.fromstring(checked(op,om));assert o.xpath('//input[@id="joCode"]/@value')==['85'] and o.xpath('//input[@id="kaisaiBi"]/@value')==['20231217']
 row=o.xpath('//table[@summary="出走表"]/tr[td[@class="no6"]]');assert len(row)==1;cs=row[0].xpath('./td');href=cs[-8].xpath('.//a/@href');assert len(href)==1 and urllib.parse.parse_qs(urllib.parse.urlparse(href[0]).query)['playerCd']==['015690'];assert norm(cs[-1].text_content())==''
 np=pathlib.Path('raw_history_0914/202312178509.html');nm=json.loads(np.with_suffix('.meta.json').read_text());n=html.fromstring(checked(np,nm));nr=n.xpath('//tr[@id="player-6"]');assert len(nr)==1 and nr[0].xpath('.//a[@id="name_15690_202312178509"]')
 spec=importlib.util.spec_from_file_location('history1307',ROOT/'recovery/review_rainbow_history_postponement_1307.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);groups=h.group_rows(nr[0],15690,'2023-12-17');g=groups[1];assert g['group_id']=='2023102673' and g['venue_code']=='73' and g['venue_name']=='小松島';assert [(r['race_id'],r['finish_text']) for r in g['results']]==[('202310267303','1'),('202310277303','棄')]
 records=[];sources=[];inputs=[]
 for day,finish in [('2023-10-26','1'),('2023-10-27','棄')]:
  k=day.replace('-','');b,cm=source(k+'_card');result,rm=source(k+'_result');u=urllib.parse.urlparse(cm['url']);assert u.netloc=='www.chariloto.com' and u.path=='/api/keirin/race_info' and urllib.parse.parse_qs(u.query)=={'open_day':[day],'vel_code':['73'],'race_num':['3']};assert rm['url']=='https://www.chariloto.com/keirin/results/73/'+day+'/'
  c=json.loads(b);records.append(actual(day,c,result,finish));sources.extend([cm,rm]);inputs.append((day,c,result,finish))
 b,dm=source('202310277303_detail');d=html.fromstring(b);assert dm['url']=='https://keirin.netkeiba.com/race/result/?race_id=202310277303' and d.xpath('//input[@name="plp-raceid"]/@value')==['202310277303'];found=[]
 for rr in d.xpath('//table[@id="All_Result_Table"]//tr[td]'):
  cc=rr.xpath('./td');links=cc[3].xpath('.//a/@href')
  if links and urllib.parse.parse_qs(urllib.parse.urlparse(links[0]).query).get('id')==['15690']:found.append(cc)
 assert len(found)==1;cc=found[0];assert norm(cc[0].text_content())=='棄' and int(norm(cc[2].text_content()))==1 and [norm(x) for x in cc[3].xpath('.//img/@alt')]==['斉藤樂'] and re.findall(r'(\d+)期',norm(cc[3].text_content()))==['123'] and norm(cc[4].text_content()) in ['-(落車棄権)','−(落車棄権)'];sources.append(dm)
 tests=[];day,c,b,finish=inputs[0]
 for label,mut in [('wrong_actual_day',lambda x:x['box'].update(open_day='2023-10-25')),('wrong_explicit_ID',lambda x:x['race_athletes']['1'].update(un_id=15691))]:
  x=copy.deepcopy(c);mut(x)
  try:actual(day,x,b,finish)
  except AssertionError:tests.append({'mutation':label,'rejected':True})
  else:raise AssertionError(label)
 doc=html.fromstring(b);rr=doc.xpath('//*[@id="race3"]//table[.//th[normalize-space()="選手名"] and .//th[normalize-space()="着"]]//tbody/tr')[0];rr.xpath('./td')[0].text='2'
 try:actual(day,c,html.tostring(doc),finish)
 except AssertionError:tests.append({'mutation':'changed_actual_finish','rejected':True})
 else:raise AssertionError('rank mutation accepted')
 report={'all_finals_complete':False,'canonical_db_changed':False,'target_final':'202312178509','target_car_no':6,'registration_number':'015690','OP_prior2_slot_raw':cs[-1].text_content(),'OP_prior2_status':'source_cell_empty_retained_no_fill','alternate_NK_group_independently_verified':True,'group_id':'2023102673','full_start_date':'2023-10-26','records':records,'NK_raw_group':g,'NK_actual_finish_raw':cc[0].text_content(),'NK_actual_detail_raw':cc[4].text_content(),'sources':sources,'original_history_source':nm,'original_OP_source':om,'mutations':tests,'note':'Alternate actual-race evidence verifies the specific missing prior2 group across Chariloto and netkeiba. OP empty cell and all planned slots remain unchanged; this is not a claim that OP supplied the missing history or a proof of global history completeness. IDs are explicit, never inferred from names.'}
 (OUT/'alternate_history_gap_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in RAW.iterdir()};(OUT/'raw_manifest.json').write_text(json.dumps(manifest,indent=2));buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w',format=tarfile.USTAR_FORMAT) as t:
  for p in sorted(RAW.iterdir()):
   x=tarfile.TarInfo(p.name);x.size=p.stat().st_size;t.addfile(x,io.BytesIO(p.read_bytes()))
 (OUT/'alternate_history_gap_raw.tar.xz').write_bytes(lzma.compress(buf.getvalue(),preset=9|lzma.PRESET_EXTREME))
 with tarfile.open(OUT/'alternate_history_gap_raw.tar.xz') as t:
  assert len(t.getnames())==len(set(t.getnames()))==10 and set(t.getnames())==set(manifest)
  for name,v in manifest.items():
   b=t.extractfile(name).read();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 print('specific alternate prior2 group verified, original blank retained, mutations3 rejected, raw10members lossless')
if __name__=='__main__':main()
