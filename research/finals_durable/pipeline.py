"""Finals-only collection. Raw pages, explicit gaps, resumable SQLite state."""
from __future__ import annotations
import argparse, concurrent.futures as cf, datetime as dt, gzip, hashlib, itertools, json, math, re, sqlite3, sys, threading, time, unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from bs4 import BeautifulSoup
from keirin_market_analysis import s_yosen_2025 as base
from keirin_market_analysis.line_formation_2025 import parse_line_formation_html
from keirin_market_analysis.results_payouts_2025 import parse_result_page, result_url
from keirin_market_analysis.historical_final_odds import parse_market_header, parse_trifecta, parse_trio, odds_url
from keirin_market_analysis.g3_history import nearest_published_grades, parse_title_meta
ROOT=Path(__file__).resolve().parent
LOCAL=threading.local()
def norm(s):return unicodedata.normalize('NFKC',str(s)).strip()
def dump(x):return json.dumps(x,ensure_ascii=False)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def final_label(label):
 t=norm(label)
 return ('決勝' in t and not re.search(r'準.*決勝',t) and not any(k in t for k in ['ガールズ','女子','L級'])) or ('グランプリ' in t and not any(k in t for k in ['ガールズ','女子']))
def category(label):
 t=norm(label)
 if 'チャレンジ' in t:return 'A3'
 if 'A級' in t:return 'A12'
 if 'S級' in t or 'グランプリ' in t:return 'S'
 return 'unresolved'
def db_open(data):
 data.mkdir(parents=True,exist_ok=True)
 c=sqlite3.connect(data/'finals.sqlite',timeout=60);c.execute('PRAGMA journal_mode=DELETE')
 c.executescript('''CREATE TABLE IF NOT EXISTS days(day TEXT PRIMARY KEY,status TEXT,payload TEXT,error TEXT,updated TEXT);
 CREATE TABLE IF NOT EXISTS races(race_id TEXT PRIMARY KEY,day TEXT,url TEXT,label TEXT,grade TEXT,category TEXT,status TEXT,payload TEXT,error TEXT,updated TEXT);
 CREATE TABLE IF NOT EXISTS banks(venue TEXT PRIMARY KEY,status TEXT,payload TEXT,updated TEXT);
 CREATE TABLE IF NOT EXISTS cancellations(day TEXT,event_card_url TEXT,race_no INTEGER,label TEXT,evidence TEXT,PRIMARY KEY(day,event_card_url,race_no));''');return c
def get(url,data):
 p=data/'raw'/(hashlib.sha256(url.encode()).hexdigest()+'.html.gz');p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  with gzip.open(p,'rt',encoding='utf-8') as f:return f.read()
 if not hasattr(LOCAL,'session'):
  LOCAL.session=base.make_session();LOCAL.last=0
 time.sleep(max(0,0.4-(time.monotonic()-LOCAL.last)))
 r=LOCAL.session.get(url,timeout=35);LOCAL.last=time.monotonic();r.raise_for_status();r.encoding='utf-8';s=r.text
 tmp=p.with_name(p.name+f'.{threading.get_ident()}.tmp')
 with gzip.open(tmp,'wt',encoding='utf-8') as f:f.write(s)
 tmp.replace(p)
 return s
def discover_html(html,day,url):
 s=BeautifulSoup(html,'lxml');out={};unresolved=[];cancelled=[]
 if not s.select('td.name') and not any(x in s.get_text() for x in ['開催はありません','開催がありません']):
  raise ValueError('no program labels: discovery unverified')
 for table in s.find_all('table'):
  container=table.find_parent('div',class_='kaisai-program_table')
  if container and 'none' in container.get('class',[]):continue
  rows=table.find_all('tr')
  for i,row in enumerate(rows):
   cells=base.expand_row_cells(row)
   for col,cell in enumerate(cells):
    if 'name' not in cell.get('class',[]):continue
    label=base.normalize_text(cell.get_text(' ',strip=True))
    if not final_label(label):continue
    anchor=None
    for later in rows[i+1:]:
     cs=base.expand_row_cells(later)
     if col<len(cs):anchor=cs[col].find('a',href=re.compile('/racedetail/'))
     if anchor:break
    if anchor is None:
     block=cell.find_parent('div',class_='kaisai-list');event_link=block.select_one('li.active a[href*="/racecard/"]') if block else None
     later_cells=base.expand_row_cells(rows[i+1]) if i+1<len(rows) else []
     cancelled_cell=later_cells[col] if col<len(later_cells) else None
     if cancelled_cell and '中止' in cancelled_cell.get_text() and event_link:
      cancelled.append({'label':label,'race_no':col+1,'event_card_url':event_link['href'],'source_url':url,'evidence':base.normalize_text(cancelled_cell.get_text())})
     else:unresolved.append({'label':label,'column':col+1,'reason':'final_label_without_link','event_card_url':event_link['href'] if event_link else None})
     continue
    u=base.canonical_race_url(url,anchor['href']);m=re.search(r'/racedetail/(\d{16})/',u)
    if not m:unresolved.append({'label':label,'reason':'invalid_race_id'});continue
    grades=nearest_published_grades(anchor)
    block=anchor.find_parent('div',class_='kaisai-list')
    if block and block.get('id'):
     nav=s.find('a',href='#'+block['id'])
     if nav:grades=set(re.findall(r'(GP|G[123]|F[12])',norm(nav.get_text()).upper()))
    out[m[1]]={'race_id':m[1],'day':day,'url':u,'label':label,'grade':next(iter(grades)) if len(grades)==1 else 'unknown','category':category(label)}
 return {'finals':list(out.values()),'cancelled_finals':cancelled,'unresolved':unresolved,'source_url':url,'captured':now(),'program_label_count':len(s.select('td.name'))}
def discover_one(day,data):
 u='https://keirin.kdreams.jp/kaisai/'+day.replace('-','/')+'/'
 try:return day,discover_html(get(u,data),day,u),None
 except Exception as e:return day,None,repr(e)
def discover(c,data,start,end,workers):
 days=[];d=dt.date.fromisoformat(start);e=dt.date.fromisoformat(end)
 while d<=e:
  day=d.isoformat();r=c.execute('SELECT status FROM days WHERE day=?',(day,)).fetchone()
  if not r or r[0]!='verified':days.append(day)
  d+=dt.timedelta(days=1)
 with cf.ThreadPoolExecutor(max_workers=workers) as ex:
  for day,p,err in ex.map(lambda d:discover_one(d,data),days):
   status='error' if err else ('unresolved' if p['unresolved'] else 'verified')
   c.execute('INSERT OR REPLACE INTO days VALUES(?,?,?,?,?)',(day,status,dump(p),err,now()))
   if p:
    for r in p.get('cancelled_finals',[]):
     c.execute('INSERT OR REPLACE INTO cancellations VALUES(?,?,?,?,?)',(day,r['event_card_url'],r['race_no'],r['label'],dump(r)))
    for r in p['finals']:
     c.execute('INSERT OR IGNORE INTO races VALUES(?,?,?,?,?,?,?,NULL,NULL,?)',(r['race_id'],day,r['url'],r['label'],r['grade'],r['category'],'pending',now()))
     if r['grade']!='unknown':c.execute('UPDATE races SET grade=? WHERE race_id=?',(r['grade'],r['race_id']))
   c.commit();print('discover',day,status,len(p['finals']) if p else 0,flush=True)
def page_identity(soup):
 title=base.normalize_text(soup.title.get_text(' ',strip=True) if soup.title else '')
 _,event,label=parse_title_meta(soup)
 if not label:
  m=re.search(r'\|\s*(.*?)\s*(\d+)R\s+([^|]+)\|',title)
  if m:event,label=m[1].strip(),m[3].strip()
 if not label:raise ValueError('race title label unparsed: '+title)
 if not final_label(label):raise ValueError('not a final: '+label)
 return event,label
def collect_one(row,data):
 rid,day,url,label,grade,cat=row[:6];previous=json.loads(row[6]) if len(row)>6 and row[6] else None;errors={};p={'race_id':rid,'discovered_day':day,'expected_label':label,'grade':grade,'category':cat,'captured':now()}
 if previous:p=previous.copy()
 try:
  if previous and previous.get('entries') and previous.get('race') and previous.get('line'):
   p=previous;p['grade']=grade;p['category']=cat;entries=p['entries']
  else:
   h=get(url,data);s=BeautifulSoup(h,'lxml');event,actual_label=page_identity(s)
   ref=base.RaceRef(day,int(rid[-4:]),url)
   meta,entries=base.extract_entries(h,ref,require_target_label=False)
   meta.update(race_type=actual_label,event_name=event,meeting_grade=grade)
   if meta['race_date']!=day:raise ValueError('discovery/date mismatch')
   p['race']=meta;p['category']=category(actual_label);p['entries']=entries
   # Profile links are immutable source IDs where published. No guessed identity merge.
   _,rows=base.pick_base_racecard_table(s)
   links={}
   for tr in rows:
    nc=tr.select_one('td.num');rc=tr.select_one('td.rider')
    if not nc or not rc:continue
    cm=re.search('[1-9]',nc.get_text());a=rc.find('a',href=True)
    if cm and a:links[int(cm[0])]=urljoin(url,a['href'])
   for ent in entries:
    ent['race_type']=actual_label;ent['rider_source_id']=links.get(ent['car_no']);ent['rider_id_status']='published_profile_link' if ent['rider_source_id'] else 'unresolved'
   try:
    line=parse_line_formation_html(h)
    if line['status']=='published' and set(line['riders'])!={e['car_no'] for e in entries}:raise ValueError('line cars != entry cars')
    p['line']=line
   except Exception as e:errors['line']=repr(e)
   # Keep detailed race-specific histories as raw tables; never use retrospective current profile stats.
   p['racecard_tables']=[{'class':t.get('class',[]),'rows':[[base.normalize_text(c.get_text(' ',strip=True)) for c in tr.find_all(['th','td'],recursive=False)] for tr in t.find_all('tr')]} for t in s.select('table.racecard_table')]
 except Exception as e:
  return rid,'error',p,{'entries':repr(e)}
 try:
  if not p.get('results'):
   results,payouts=parse_result_page(get(result_url(url),data));p['results']=results;p['payouts']=payouts
 except Exception as e:errors['results']=repr(e)
 try:
  os=BeautifulSoup(get(odds_url(url),data),'lxml');p['odds_header']=parse_market_header(os)
  cars=[e['car_no'] for e in p['entries']]
  p['trifecta']=[{'cars':list(k),'odds':v[0],'status':v[1]} for k,v in parse_trifecta(os,cars).items()]
  p['trio']=[{'cars':list(k),'odds':v[0],'status':v[1]} for k,v in parse_trio(os,cars).items()]
  # The old parser pads absent trio combinations as unavailable: never count those as observed odds.
  missing=[v['cars'] for v in p['trio'] if v['status']!='available']
  if missing:errors['trio_coverage']=f'{len(missing)} combinations unobserved/unavailable; not inferred'
  if any(v['status']!='available' for v in p['trifecta']):errors['trifecta_coverage']='unavailable selections'
 except Exception as e:errors['odds']=repr(e)
 if not p.get('line') or p['line']['status']!='published':errors.setdefault('line','not_published')
 if p['category']=='unresolved':
  classes={norm(e.get('class','')) for e in p['entries']}
  if classes and all(x.startswith('S') for x in classes):p['category']='S'
  elif classes and all(x=='A3' for x in classes):p['category']='A3'
  elif classes and all(x in {'A1','A2'} for x in classes):p['category']='A12'
  elif classes and all(x.startswith('L') for x in classes):return rid,'excluded',p,{'scope':'women_only'}
 if any(not e.get('rider_source_id') for e in p['entries']):errors['rider_identity']='profile links missing; manual registration-ID mapping required'
 return rid,('partial' if errors else 'collected'),p,errors
def collect(c,data,workers,limit=0):
 rows=c.execute("SELECT race_id,day,url,label,grade,category,payload FROM races WHERE status!='excluded' AND (payload IS NULL OR COALESCE(json_array_length(json_extract(payload,'$.trifecta')),0)=0 OR COALESCE(json_array_length(json_extract(payload,'$.trio')),0)=0 OR COALESCE(json_array_length(json_extract(payload,'$.results')),0)=0) ORDER BY COALESCE(updated,''),day,race_id").fetchall()
 if limit:rows=rows[:limit]
 with cf.ThreadPoolExecutor(max_workers=workers) as ex:
  futures={ex.submit(collect_one,r,data):r[0] for r in rows}
  for future in cf.as_completed(futures):
   try:rid,status,p,err=future.result()
   except Exception as exc:
    print('collect_worker_error',futures[future],repr(exc),flush=True);continue
   c.execute('UPDATE races SET status=?,payload=?,error=?,updated=?,category=? WHERE race_id=?',(status,dump(p),dump(err),now(),p.get('category','unresolved'),rid));c.commit()
   print('collect',rid,status,err,flush=True)
def audit(c,data,start,end):
 expected=(dt.date.fromisoformat(end)-dt.date.fromisoformat(start)).days+1
 days=dict(c.execute('SELECT status,count(*) FROM days WHERE day BETWEEN ? AND ? GROUP BY status',(start,end)))
 races=dict(c.execute('SELECT status,count(*) FROM races WHERE day BETWEEN ? AND ? GROUP BY status',(start,end)))
 groups=[dict(zip(['year','category','grade','count'],r)) for r in c.execute('SELECT substr(day,1,4),category,grade,count(*) FROM races GROUP BY 1,2,3')]
 field_counts={}
 for key in ['entries','results','trifecta','trio']:
  field_counts[key]=c.execute("SELECT count(*) FROM races WHERE day BETWEEN ? AND ? AND json_array_length(json_extract(payload,?))>0",(start,end,'$.'+key)).fetchone()[0]
 field_counts['published_line']=c.execute("SELECT count(*) FROM races WHERE day BETWEEN ? AND ? AND json_extract(payload,'$.line.status')='published'",(start,end)).fetchone()[0]
 report={'as_of':now(),'field_presence_races':field_counts,'start':start,'end':end,'expected_calendar_days':expected,'days':days,'races':races,'by_year_class_grade':groups,'all_finals_complete':False,'bank_rows':c.execute('SELECT count(*) FROM banks').fetchone()[0], 'cancelled_finals':c.execute('SELECT count(*) FROM cancellations').fetchone()[0], 'completeness_note':'Calendar discovery needs independent official-calendar reconciliation. Collected != complete; bank properties and rider registry mapping are separate gates.'}
 (data/'coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(dump(report),flush=True);return report
def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=['discover','collect','run','audit']);a.add_argument('--start',default='2023-01-01');a.add_argument('--end',default='2026-10-19');a.add_argument('--workers',type=int,default=4);a.add_argument('--limit',type=int,default=0);a.add_argument('--data',type=Path,default=ROOT/'data');x=a.parse_args()
 if x.start<'2023-01-01' or x.end>=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date().isoformat():raise ValueError('only completed historical dates from 2023-01-01')
 c=db_open(x.data)
 if x.command in ['discover','run']:discover(c,x.data,x.start,x.end,x.workers)
 if x.command in ['collect','run']:collect(c,x.data,x.workers,x.limit)
 audit(c,x.data,x.start,x.end)
if __name__=='__main__':main()
