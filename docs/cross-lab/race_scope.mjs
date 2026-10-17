/** Resolve omitted race labels from every entrant's published class. */
export function resolveRaceClass(race){
 const classes=(race.entries||[]).map(e=>String(e.class||'').normalize('NFKC').toUpperCase());
 if(classes.length&&classes.every(c=>['SS','S1','S2'].includes(c)))return 'S';
 if(classes.length&&classes.every(c=>['A1','A2'].includes(c)))return 'A';
 if(classes.length&&classes.every(c=>c==='A3'))return 'A3';
 const type=String(race.race_type||'').normalize('NFKC');
 if(/S級/i.test(type))return 'S';
 if(/A級.*(?:チャレンジ|チ予|チ般|チ選|チ準|チ決)/i.test(type))return 'A3';
 if(/A級/i.test(type))return 'A';
 return 'unknown';
}
export function modelRaceType(type,cls){
 if(cls==='unknown')return type||'';
 return (cls==='A3'?'A級チャレンジ':cls+'級')+' '+String(type||'');
}
