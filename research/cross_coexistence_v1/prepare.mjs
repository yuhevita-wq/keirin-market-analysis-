import fs from 'node:fs';
import {predictJoint} from '../../docs/cross-lab/joint_model.mjs';
import {analyzeCross,normalizeRace} from '../../docs/cross-lab/cross_engine.mjs';
import {coexistenceFeatures,FEATURE_NAMES} from '../../docs/cross-lab/coexistence.mjs';
const [input,prefix]=process.argv.slice(2),cards=JSON.parse(fs.readFileSync(input));
const bundle=JSON.parse(fs.readFileSync(new URL('../../docs/cross-lab/joint_model_frozen.json',import.meta.url)));
const fd=fs.openSync(prefix+'.f32','w'),meta=[];
for(const [idx,row] of cards.entries()){
  const f=predictJoint(row.input,bundle),pack=coexistenceFeatures(row.input,f),r=normalizeRace(row.input);
  const actualKey=row.target.slice(0,2).sort((a,b)=>a-b).join('-');
  const candidates=analyzeCross(row.input,{limit:1000}).candidates.map(c=>{
    const key=[...c.firstPair].sort((a,b)=>a-b).join('-');
    return {key:c.signature,pair:pack.records.findIndex(z=>z.key===key),p:c.tickets.reduce((s,t)=>s+f.probabilities[t.join('-')],0),
      hit:c.tickets.some(t=>t.join('-')===row.target.join('-'))};
  });
  fs.writeSync(fd,Buffer.from(new Float32Array(pack.records.flatMap(z=>z.features)).buffer));
  const third=r.lines.map((_,i)=>i).sort((a,b)=>pack.lineWin[b]-pack.lineWin[a])[2];
  meta.push({id:row.input.race_id,date:row.date,payout:row.payout,target:pack.records.findIndex(z=>z.key===actualKey),
    base:pack.records.map(z=>z.probability),pairs:pack.records.map(z=>z.pair),linePairs:pack.records.map(z=>z.linePair),
    lineRanks:pack.lineRank,lineWin:pack.lineWin,win:Object.fromEntries(pack.win),
    lines:r.lines,third,candidates});
  if(idx%2000===0)console.log(idx+'/'+cards.length);
}
fs.closeSync(fd);fs.writeFileSync(prefix+'.json',JSON.stringify({featureNames:FEATURE_NAMES,meta}));
