"""Independent source binding/date audit; geometry is locked manual raster review.

This does not claim two geometry providers or automatic raster/OCR verification.
"""
import copy,datetime,gzip,hashlib,io,json,sqlite3,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup
import fitz
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/bank_sources_1227'
PINS={'data/scope_review_0552/integrated_candidates.tar.xz': 'c1a030bc66b78fec4bda307046402f9db15fa67dd386845b8d19862cd3f29e3a', 'data/scope_review_1613/ito_candidate_bank_review.json': '0c44b0f1cf3f22e14c905c567b867938d11b33671ecfa1468cd28d9b1daf87ac', 'data/bank_research_0242/ito_news.xml.gz': '43f61e9057a0ef1c87d6e9fa7e053397dbbda8913c523f8185e22af027f41687', 'data/bank_research_0242/ito_news_xml_meta.json': '99e8598b0378e328401267bbe2c6b9621a08f4bb5cb0da5a4aa80864ffd120e0', 'data/bank_sources_1227/ito20251211.pdf': '181254589a44b6b9a7dfd35d78c4eb4aa6de9158fc4d0b217eef7a874f56b573', 'data/bank_sources_1227/ito20251211.meta.json': 'c242b12358fffea45dd46fe9f61fe197bd84c9fbab150db468bd9c947f8a99a4'}
EXPECTED_GEOMETRY={'circumference_m':333,'minashi_straight_m':46.6,'home_width_m':11.0,'back_width_m':9.3,'center_width_m':7.8,'home_cant_dms':[3,26,1],'center_cant_dms':[34,41,9]}
URL='https://www.itokeirin.com/race/pdf_20251211.pdf?2025121001_01'
def h(b):return hashlib.sha256(b).hexdigest()
def js(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def arc(path):
 b=(ROOT/path).read_bytes();assert h(b)==PINS[path]
 with tarfile.open(fileobj=io.BytesIO(b)) as t:assert len(t.getnames())==len(set(t.getnames()));return {m.name:t.extractfile(m).read() for m in t.getmembers()}
def raw(ref):
 if 'archive' in ref:
  ar=ref['archive'];original=(ROOT/ar['path']).read_bytes();assert len(original)==ar['bytes'] and h(original)==ar['sha256']
  with tarfile.open(fileobj=io.BytesIO(original)) as t:assert len(t.getnames())==len(set(t.getnames()));b=t.extractfile(ref['member']).read()
 else:b=(ROOT/ref['path']).read_bytes()
 assert len(b)==ref['bytes'] and h(b)==ref['sha256'];return b
def validate(doc,source,old):
 assert doc['schema']=='rainbow_candidate_dated_bank_delta_v1' and doc['status']=='candidate_exact_event_primary_source_bank_reviewed_not_SQL_inserted' and doc['input_pins']==PINS
 ident={'canonical_key':'keirin:2025-12-14:37:09','confirmed_kdreams_race_id':None,'day':'2025-12-14','provider':'local_composite','race_no':9,'venue_code':'37'};assert doc['identity']==ident
 e=next(e for e in source['events'] if e['identity']==ident);refs=e['component_raw_provenance'];assert doc['actual_event_sources']==refs;card=json.loads(raw(refs['card']));assert doc['actual_card_box']==card['box']
 box=card['box'];assert (box['open_day'],box['vel_code'],box['race_num'],box['vel_name'])==('2025-12-14','37',9,'伊東') and '75' in box['sub_name'];assert not box['is_girls'] and not box['on_abort'] and not box['is_invalid']
 op=BeautifulSoup(raw(refs['independent_card']).decode(),'html.parser');hidden={x['name']:x.get('value') for x in op.find_all('input',attrs={'name':True})};assert (hidden['joCode'],hidden['kaisaiBi'],hidden['raceNo'])==('37','20251214','9')
 for a in e['entries']:
  car=a['car_no'];reg=a['registration_number'];c=card['race_athletes'][str(car)];assert c['un_id']==int(reg) and c['entry_num']==car;assert len(op.find_all('a',href='./PlayerDetail.do?playerCd='+reg))==1
 meta=json.loads((OUT/'ito20251211.meta.json').read_bytes());assert doc['raw_retrieval_meta']==meta and meta['url']==meta['final_url']==URL and meta['status']==200 and meta['bytes']==8404111 and meta['sha256']=='181254589a44b6b9a7dfd35d78c4eb4aa6de9158fc4d0b217eef7a874f56b573'
 pub=doc['official_news_proof'];assert pub['stored_path']=='data/bank_research_0242/ito_news.xml.gz' and pub['stored_sha256']==PINS[pub['stored_path']] and pub['stored_bytes']==(ROOT/pub['stored_path']).stat().st_size
 news=gzip.decompress((ROOT/pub['stored_path']).read_bytes());assert len(news)==pub['raw_bytes']==310189 and h(news)==pub['raw_sha256']=='67c8a24bb6f210a6a8c2dbd95875dd400bfa82ebef7df598be538b829a8d4c96'
 parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True));tree=ET.fromstring(news,parser);nodes=[x for x in tree.findall('news') if any(c.tag is ET.Comment and '/race/pdf_20251211.pdf?2025121001_01' in (c.text or '') for c in x)];assert len(nodes)==1
 n=nodes[0];assert n.findtext('date')=='2025/12/10' and n.findtext('contents')=='12月　レース展望[PDF]を掲載しました。';given=ET.fromstring(pub['dated_news_node'],ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)));expected_node=copy.deepcopy(n);expected_node.tail=None;given.tail=None;assert ET.tostring(given)==ET.tostring(expected_node)
 assert pub['publication_date']=='2025-12-10' and pub['link_kind']=='commented_alink_in_saved_official_XML_not_current_active_link'
 record=doc['bank_record'];assert record['bank_version_id']=='37-ito20251211-reviewed' and record['venue_code']=='37' and record['track']=='伊東競輪' and record['valid_from']=='2025-12-11' and record['valid_until_exclusive']=='2025-12-15' and record['geometry']==EXPECTED_GEOMETRY and record['review_status']=='verified_within_source_event'
 assert record['validity_basis']=='Only explicitly reviewed Dec11–14 2025 source event; no interval extrapolation'
 assert record['sources']==[{'source_url':URL,'raw_file':'bank_sources_1227/ito20251211.pdf','bytes':8404111,'sha256':'181254589a44b6b9a7dfd35d78c4eb4aa6de9158fc4d0b217eef7a874f56b573','location':'One page, lower-left blue bank diagram, seven explicitly labelled values; event panel states Dec11Thu–14Sun,75th,and Dec14Sun9R Rainbow challenge final.'}]
 manual=doc['manual_visual_review'];assert manual['geometry']==EXPECTED_GEOMETRY and manual['PDF_page']==1 and manual['printed_circumference_preserved_without_333_33_correction'] is True and manual['raster_only_no_extractable_text'] is True and manual['independent_geometry_provider_count']==1
 assert manual['source_event_text']=='伊東温泉競輪 開設75周年記念 椿賞争奪戦 GIII / 12月11日(木)〜14日(日)' and manual['candidate_label_text']=='12/14(日) 第9R レインボーカップ チャレンジファイナル'
 assert manual['source_year_basis']=='2025/12/10 dated original official news XML commented alink plus 75th event and all four printed weekday/date pairs; no year inferred from file name alone'
 assert record['visual_review']==manual['method']=='Manual visual transcription from original raster-only official PDF; independently rendered with PyMuPDF and Poppler, not OCR or automated geometry extraction'
 assert manual['PDF_page_points']==[1264.25,915.5900268554688] and manual['geometry_clip_points']==[0,613.4453179931641,379.275,915.5900268554688]
 days=[datetime.date(2025,12,11)+datetime.timedelta(days=i) for i in range(4)];assert [d.weekday() for d in days]==[3,4,5,6] and record['valid_from']<=ident['day']<record['valid_until_exclusive']
 remaining=[v for v in old['other_13_candidate_banks_unresolved'] if not (v['day']=='2025-12-14' and v['venue_code']=='37')];assert len(remaining)==12 and doc['other_candidate_banks_unresolved']==remaining
 assert doc['candidate_bank_verified_count']==2 and doc['candidate_bank_unresolved_count']==12 and doc['existing_SQL_bank_verified_count']==230 and doc['existing_SQL_bank_unresolved_count']==5123 and doc['eligible_existing_SQL_race_not_applied']=='3720251211040012'
 assert doc['catalog_changed'] is doc['canonical_DB_changed'] is doc['all_finals_complete'] is doc['full_payload_SQL_adapter_verified'] is doc['complete_payload_insertion_audit'] is False
 assert doc['canonical_SQL_inserts']==doc['new_completed_races']==doc['successful_raw_refetches']==0 and doc['new_raw_requests']==1
def main():
 for p,v in PINS.items():assert h((ROOT/p).read_bytes())==v
 source=json.loads(arc('data/scope_review_0552/integrated_candidates.tar.xz')['integrated_candidates.json']);old=json.loads((ROOT/'data/scope_review_1613/ito_candidate_bank_review.json').read_bytes());doc=json.loads((OUT/'ito2025_candidate_bank_delta.json').read_bytes());validate(doc,source,old)
 pdf=(OUT/'ito20251211.pdf').read_bytes();d=fitz.open(stream=pdf,filetype='pdf');assert len(d)==1 and d[0].get_text()=='' and abs(d[0].rect.width-1264.25)<.01;d.close()
 # Geometry constants above are the separately visually re-read Poppler diagram,
 # compared with the pinned primary source. This is not automatic OCR.
 c=sqlite3.connect((ROOT/'data/finals.sqlite').resolve().as_uri()+'?mode=ro',uri=True);assert c.execute('pragma integrity_check').fetchall()==[('ok',)] and c.execute('select count(*) from races').fetchone()[0]==5354
 eligible=[]
 for rid,day,rawtext in c.execute("select race_id,day,payload from races where status!='excluded' and day>='2025-12-11' and day<'2025-12-15' and race_id like '37%'"):
  p=json.loads(rawtext);assert p['race']['track']=='伊東競輪' and p.get('bank') is None;eligible.append(rid)
 c.close();assert eligible==['3720251211040012'];rejected=[]
 cases=[('wrong_day',lambda x:x['identity'].__setitem__('day','2025-12-15')),('wrong_venue',lambda x:x['identity'].__setitem__('venue_code','38')),('wrong_R',lambda x:x['identity'].__setitem__('race_no',12)),('missing_dimension',lambda x:x['bank_record']['geometry'].pop('home_width_m')),('333_33_correction',lambda x:x['bank_record']['geometry'].__setitem__('circumference_m',333.33)),('source_year_change',lambda x:x['official_news_proof'].__setitem__('publication_date','2024-12-10')),('period_extension',lambda x:x['bank_record'].__setitem__('valid_until_exclusive','2026-01-01')),('false_two_geometry_providers',lambda x:x['manual_visual_review'].__setitem__('independent_geometry_provider_count',2)),('candidate_unresolved_false_promotion',lambda x:x.__setitem__('candidate_bank_unresolved_count',0)),('false_SQL_application',lambda x:x.__setitem__('canonical_DB_changed',True)),('complete_flag',lambda x:x.__setitem__('all_finals_complete',True))]
 for name,fn in cases:
  bad=copy.deepcopy(doc);fn(bad)
  try:validate(bad,source,old)
  except (AssertionError,KeyError,ValueError):rejected.append(name)
  else:raise AssertionError(name)
 badpdf=pdf[:-1]+bytes([pdf[-1]^1]);assert h(badpdf)!=PINS['data/bank_sources_1227/ito20251211.pdf'];rejected.append('original_PDF_byte_change')
 report={'status':'independent_candidate_bank_source_binding_audit_passed','candidate_key':doc['identity']['canonical_key'],'candidate_bank_verified_count':2,'candidate_bank_unresolved_count':12,'eligible_existing_SQL_races_not_applied':eligible,'geometry_source_provider_count':1,'geometry_review_method':'manual visual review of original raster; no automatic geometry extraction or independent second provider claim','official_XML_commented_link_not_active':True,'mutations_rejected':rejected,'catalog_changed':False,'canonical_DB_changed':False,'canonical_SQL_inserts':0,'new_completed_races':0,'all_finals_complete':False,'full_payload_SQL_adapter_verified':False,'complete_payload_insertion_audit':False,'independent_implementation':'ElementTree/BeautifulSoup and separately locked visual geometry; producer lxml not imported','second_SQLite_artifact_saved':False}
 (OUT/'independent_ito2025_bank_audit.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
