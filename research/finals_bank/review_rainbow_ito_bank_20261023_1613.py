"""Verify one pending Rainbow final's bank against its own dated official source.
No SQL insertion, catalog mutation, period extension, or inferred provider ID.
Geometry remains a SHA-locked manual visual transcription, never claimed OCR.
"""
import pathlib,json,hashlib,copy,datetime,importlib.util,re
from lxml import html
ROOT=pathlib.Path('current_data/unpacked/finals_ai');DATA=ROOT/'data';OUT=DATA/'scope_review_1613'
FINGERPRINT='53d50965942388957fe630ccd91748767458ebe312ed4be51f12b956bb81f626'
def sha(b):return hashlib.sha256(b).hexdigest()
def compact(t):return re.sub(r'\s+','',t)
def module():
 s=importlib.util.spec_from_file_location('cards0831',ROOT/'recovery/review_rainbow_independent_card_results_0831.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verify(event,rec,review,pdf,card,op,nk,official):
 assert review['key']=='ito20231202' and review['year']==2023
 assert (event['day'],event['venue_code'],event['race_no'],event['category'])==('2023-12-05','37',9,'A12')
 assert rec['bank_version_id']=='37-ito20231202-reviewed' and rec['review_status']=='verified_within_source_event'
 assert rec['venue_code']==review['venue_code']==event['venue_code'] and rec['track']==review['track']=='伊東競輪'
 assert rec['valid_from']==review['valid_from']=='2023-12-02' and rec['valid_until_exclusive']==review['valid_until_exclusive']=='2023-12-06'
 assert rec['valid_from']<=event['day']<rec['valid_until_exclusive']
 fields={'circumference_m','minashi_straight_m','home_width_m','back_width_m','center_width_m','home_cant_dms','center_cant_dms'}
 assert set(rec['geometry'])==set(review['geometry'])==fields and rec['geometry']==review['geometry']
 assert review['review_method'].startswith('Manual visual transcription')
 assert len(pdf)==review['bytes'] and sha(pdf)==review['sha256'];source=rec['sources'][0]
 for k in ['source_url','raw_file','sha256','bytes']:assert source[k]==review[k]
 start=datetime.date.fromisoformat(rec['valid_from']);end=datetime.date.fromisoformat(rec['valid_until_exclusive']);days=[start+datetime.timedelta(days=i) for i in range((end-start).days)]
 assert ['月火水木金土日'[d.weekday()] for d in days]==review['weekdays']
 b=card['box'];assert (b['open_day'],b['vel_code'],b['vel_name'],b['race_num'])==(event['day'],'37','伊東',9)
 assert b['sub_name']=='第73周年　椿賞争奪戦' and b['is_end_day'] and not b['is_girls'] and not b['on_abort'] and not b['is_invalid']
 for f,v in [('joCode','37'),('kaisaiBi','20231205'),('raceNo','9')]:assert op.xpath('//input[@id="'+f+'"]/@value')==[v]
 assert nk.xpath('//input[@name="plp-raceid"]/@value')==['202312053709']
 text=compact(html.fromstring(official).text_content())
 assert '椿賞争奪戦' in text and '12月5日' in text and '第9レースで実施されたレインボーカップA級ファイナル' in text
 return {'status':'candidate_historical_bank_verified_for_exact_source_event','target':{'day':event['day'],'venue_code':'37','race_no':9,'category':'A12'},'bank_record':rec,'manual_source_review':{'method':review['review_method'],'reviewed_records_sha256':FINGERPRINT,'source_pdf_sha256':sha(pdf),'source_pdf_bytes':len(pdf),'source_page':2,'geometry_manual_values_rechecked_from_original_PDF_media_box':True,'event_year_dates_manual_reading':'令和5年12月2日〜5日','geometry_values_not_automatic_OCR':True},'source_scope':'Only this actual event/day/venue and actual 9R; no wider period or other venue','canonical_DB_changed':False,'existing_verified_bank_count_increment':0,'candidate_is_complete_SQL_row':False,'all_finals_complete':False}
def main():
 OUT.mkdir(exist_ok=True);m=module();raw=(DATA/'bank_sources_0313/reviewed_records.json').read_bytes();assert sha(raw)==FINGERPRINT;review=next(r for r in json.loads(raw) if r['key']=='ito20231202');pdf=(DATA/review['raw_file']).read_bytes();catalog=json.loads((DATA/'historical_bank_catalog.json').read_text())['records'];rec=next(r for r in catalog if r['bank_version_id']=='37-ito20231202-reviewed')
 scope=json.loads((DATA/'scope_review_0457/rainbow_scope_independent_review.json').read_text());e=next(e for e in scope['events'] if e['day']=='2023-12-05')
 for v in e['source_files']:
  b=(DATA/'scope_review_0457'/v['path']).read_bytes();assert len(b)==v['bytes'] and sha(b)==v['sha256']
 manifest=json.loads((DATA/'scope_review_0831/raw_manifest.json').read_text())
 for part in ['card','result']:
  for suffix in ['.html','.meta.json']:
   n='202312053709_'+part+suffix;v=manifest[n];b=(pathlib.Path('raw_independent_0831')/n).read_bytes();assert len(b)==v['bytes'] and sha(b)==v['sha256']
 card=json.loads(m.cardfile(e).read_text());op,om=m.doc('202312053709','card');nk,nm=m.doc('202312053709','result');official=(DATA/'scope_review_0313/official_day4_report.html').read_bytes();meta=json.loads((DATA/'scope_review_0313/official_day4_report.json').read_text());assert len(official)==meta['bytes'] and sha(official)==meta['sha256']=='f2d06c167189d64fb064eb6f9aa6bbf03d427e979dead65b0e7321df287dbd6d'
 report=verify(e,rec,review,pdf,card,op,nk,official);tests=[]
 for mode in ['day_outside','wrong_venue','wrong_R','missing_geometry','changed_geometry','PDF_byte_change','wrong_source_year']:
  ev=copy.deepcopy(e);rc=copy.deepcopy(rec);rv=copy.deepcopy(review);pr=pdf
  if mode=='day_outside':ev['day']='2023-12-06'
  elif mode=='wrong_venue':ev['venue_code']='38'
  elif mode=='wrong_R':ev['race_no']=12
  elif mode=='missing_geometry':rc['geometry'].pop('home_width_m')
  elif mode=='changed_geometry':rc['geometry']['circumference_m']=333.33
  elif mode=='PDF_byte_change':pr=pdf[:-1]+bytes([pdf[-1]^1])
  else:rv['year']=2024
  try:verify(ev,rc,rv,pr,card,op,nk,official)
  except (AssertionError,KeyError,ValueError):tests.append({'mutation':mode,'rejected':True})
  else:raise AssertionError('accepted '+mode)
 report['sources']={'OP_actual_target':om,'NK_actual_target':nm,'official_held_report':meta,'Chariloto_card':e['source_files'][0]};report['mutations']=tests;report['new_raw_fetches']=0
 report['other_13_candidate_banks_unresolved']=[{'day':x['day'],'venue_code':x['venue_code'],'race_no':x['race_no'],'status':'unresolved_no_matching_dated_catalog_evidence'} for x in scope['events'] if x['day']!='2023-12-05']
 assert len(report['other_13_candidate_banks_unresolved'])==13
 (OUT/'ito_candidate_bank_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('exact-event pending candidate bank verified1; existing SQL bank count unchanged230; other13 unresolved; mutations7 rejected')
if __name__=='__main__':main()
