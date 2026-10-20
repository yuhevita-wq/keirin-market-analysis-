import pathlib,json,sqlite3,datetime
def age(birth,day):
    b=datetime.date.fromisoformat(birth);d=datetime.date.fromisoformat(day);return d.year-b.year-((d.month,d.day)<(b.month,b.day))
def apply(data):
    records=json.loads((data/'registry_short_term_official.json').read_text());byname={x['short_name']:x for x in records};assert len(byname)==len(records)
    c=sqlite3.connect(data/'finals.sqlite');resolved=[];rejected=[]
    for rid,day,raw in c.execute("select race_id,day,payload from races where status!='excluded'").fetchall():
        d=json.loads(raw);changed=False
        for e in d.get('entries',[]):
            name=''.join(e.get('player_name','').split());r=byname.get(name)
            if not r or e.get('rider_source_id'):continue
            if not (str(e['term'])=='999' and e['prefecture']=='外国' and str(e['age'])==str(age(r['birth_date'],day)) and day>=r['registered_on'] and e['class']=='S2' and r['full_name'].endswith(name)):
                rejected.append({'race_id':rid,'name':name,'reason':'foreign scope, age, class or registration effective date mismatch'});continue
            e.update(rider_source_id=r['profile_url'],rider_registration_number=r['registration_number'],rider_id_status='verified_official_short_term_name_age_scope',rider_identity_evidence={'matching_fields':['published_surname','age_at_race_from_official_birth_date','foreign_short_term_scope','class','registration_effective_date'],'official_sources':[r],'local_sentinels':{'term':'999','prefecture':'外国'},'use':'registration identity only; historical stats unchanged'});changed=True;resolved.append({'race_id':rid,'car_no':e['car_no'],'registration_number':r['registration_number'],'age_at_race':age(r['birth_date'],day)})
        if changed:c.execute('update races set payload=? where race_id=?',(json.dumps(d,ensure_ascii=False),rid))
    c.commit();assert c.execute('pragma integrity_check').fetchall()==[('ok',)];c.close();report={'resolved':resolved,'rejected':rejected,'count':len(resolved),'all_finals_complete':False};(data/'registry_short_term_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);apply(p.parse_args().data)
