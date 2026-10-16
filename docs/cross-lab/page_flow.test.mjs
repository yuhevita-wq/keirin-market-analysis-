// Execute the actual page module with a small DOM adapter. This verifies its
// model/selection/render/save wiring; it is not a browser layout test.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

test('page flow keeps ranking, review, core, auxiliary and saved decision separate',async()=>{
  const html=fs.readFileSync(new URL('./index.html',import.meta.url),'utf8');
  const source=html.match(/<script type="module">([\s\S]*?)<\/script>/)[1]
    .replace(/from "(\.\/[^\"]+)"/g,(_,p)=>'from '+JSON.stringify(new URL(p,import.meta.url).href));
  const races=JSON.parse(fs.readFileSync(new URL('./fixtures/nara_20261015_withdrawn.json',import.meta.url))).races;
  const nodes=new Map();let current;
  const make=()=>({value:'',checked:false,innerHTML:'',textContent:'',className:'',ownerDocument:null,
    addEventListener(){},append(){},replaceChildren(){}});
  const document={getElementById(id){if(!nodes.has(id))nodes.set(id,make());const n=nodes.get(id);n.ownerDocument=document;return n;},
    createElement(){return make();},querySelectorAll(){return [];},querySelector(selector){
      const no=Number(selector.match(/data-car="(\d+)"/)[1]);
      const field=selector.match(/data-stat="([^\"]+)"/)?.[1]||'style';
      return {value:String(current.entries.find(e=>e.no===no)[field])};
    }};
  const previousDocument=globalThis.document,previousFetch=globalThis.fetch;
  globalThis.document=document;
  globalThis.fetch=async path=>({ok:true,json:async()=>String(path).includes('live-race-data')?{races}:JSON.parse(fs.readFileSync(new URL(path,import.meta.url),'utf8'))});
  try{
    const module=await import('data:text/javascript;base64,'+Buffer.from(source+'\nexport {load,analyze};export const record=()=>lastRecord;').toString('base64'));
    for(const r of races){
      current=r;document.getElementById('raceUrl').value='https://keirin.kdreams.jp/nara/racedetail/'+r.race_id+'/';await module.load();
      document.getElementById('useMarketModel').checked=true;
      document.getElementById('autoModel').value='joint';
      document.getElementById('comparisonObjective').value='HIT_PROBABILITY';
      await module.analyze();
      const rendered=document.getElementById('analysis').innerHTML,record=module.record();
      assert.ok(record,document.getElementById('analysis').textContent);
      assert.match(rendered,/主採用（モデル暫定1位）/);
      assert.match(rendered,/全3組を同じ展開/);
      assert.equal((rendered.match(/data-line-comparison=/g)||[]).length,3);
      assert.match(rendered,/比較数値の1位（暫定主採用候補）/);
      assert.match(rendered,/2着の競合/);
      assert.equal((rendered.match(/<article class="candidate">/g)||[]).length,11);
      assert.equal(record.roleConditions.length,3);
      assert.match(rendered,/出走前の役割条件/);
      assert.ok(record.roleConditions.every(x=>x.trainThrough==='2024-12-31'));
      assert.ok(record.selection.signature);
      assert.equal(record.comparison.selectedSignature,null);assert.equal(record.selection.basis.status,'MODEL_PROVISIONAL');
      assert.ok(record.comparison.rankedSignature);
      assert.doesNotThrow(()=>JSON.stringify(record));
    }
    const d=document.getElementById;
    d('primaryCross').value='45-45-12';
    for(const k of ['interaction','alternativeCross','sameLine','outside'])d(k).value='comparison';
    await module.analyze();
    assert.match(d('analysis').textContent,/表裏/);
    for(const k of ['forwardCoexistence','reverseCoexistence','forwardCompetition','reverseCompetition'])d(k).value='direction-specific comparison';
    await module.analyze();
    const record=module.record();
    assert.ok(record.selection.signature);
    assert.equal(record.selection.record.directions.length,2);
    assert.equal(record.selection.basis.status,'ANALYST_JUDGMENT');
    assert.match(d('analysis').innerHTML,/確認した事実/);
    assert.match(d('analysis').innerHTML,/（）付き3着補助/);
  }finally{globalThis.document=previousDocument;globalThis.fetch=previousFetch;}
});

