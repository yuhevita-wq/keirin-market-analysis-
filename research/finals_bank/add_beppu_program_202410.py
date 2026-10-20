"""Add only the bank period explicitly covered by the recovered official event program."""
import argparse, pathlib, json, hashlib, subprocess, re

def main(data):
    evidence=data/'bank_sources_0848'
    page=evidence/'beppu_program_page2.pdf'
    text=subprocess.check_output(['pdftotext','-layout',str(page),'-']).decode()
    compact=re.sub(r'\s+','',text)
    for token in ['59.96','33度41分24秒','2度51分45秒','10.0','9.0','8.0','10/11','10/14']:
        assert token in compact, ('source value missing',token)
    manifest=json.loads((evidence/'manifest.json').read_text())
    for item in manifest['saved_files']:
        p=evidence/item['file']; assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    source={'source_url':'https://beppu-keirin.net/raceinfo/pdf/20240913_86.pdf?ver=1',
        'raw_file':'bank_sources_0848/beppu_program_page2.pdf','sha256':hashlib.sha256(page.read_bytes()).hexdigest(),
        'bytes':page.stat().st_size,'original_file_sha256':manifest['beppu_original_sha256'],
        'original_file_bytes':6100416,'original_pdf_pages_1based':[2],
        'archive_url':'https://web.archive.org/web/20250804123524id_/https://beppu-keirin.net/raceinfo/pdf/20240913_86.pdf',
        'archive_memento_datetime':'Mon, 04 Aug 2025 12:35:24 GMT',
        'location':'Page2: bank table lower left and event header. Year2024 explicitly appears on page1; saved full-page rendering in same evidence directory.',
        'evidence_format':'Unmodified source page extracted; page1 rendering preserves explicit event year. Original full PDF SHA retained.'}
    record={'bank_version_id':'86-program-20241011','track':'別府競輪','venue_code':'86',
        'valid_from':'2024-10-11','valid_until_exclusive':'2024-10-15',
        'validity_basis':'Official event program explicitly dated2024-10-11 through14. Apply to that event only; do not extend the bank table to the separate statistical period2023-11 through2024-07.',
        'geometry':{'circumference_m':400,'minashi_straight_m':59.96,'home_width_m':10.0,'back_width_m':9.0,'center_width_m':8.0,'home_cant_dms':[2,51,45],'center_cant_dms':[33,41,24]},
        'sources':[source],'review_status':'verified_within_source_event',
        'visual_review':'Full page2 rendered and bank label/value correspondence checked; page1 rendering confirms2024year and event dates.'}
    p=data/'historical_bank_catalog.json';catalog=json.loads(p.read_text())
    existing=[r for r in catalog['records'] if r['bank_version_id']==record['bank_version_id']]
    if existing: assert existing==[record]
    else: catalog['records'].append(record)
    from apply_historical_bank import validate,select
    validate(catalog['records'])
    assert select(catalog['records'],'8620241011040012','2024-10-14','別府競輪')==record
    for day,track,rid in [('2024-10-15','別府競輪','8620241011040012'),('2024-10-10','別府競輪','8620241011040012'),('2024-10-14','玉野競輪','8620241011040012'),('2024-10-14','別府競輪','6120241011040012')]:
        assert select(catalog['records'],rid,day,track) is None
    p.write_text(json.dumps(catalog,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=pathlib.Path);main(p.parse_args().data)
