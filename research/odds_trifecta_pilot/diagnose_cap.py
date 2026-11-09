#!/usr/bin/env python3
"""Safe diagnostic: does the public page parameter alter 3連単 historic coverage?

Three read-only requests for a single fixed race. No result inference.
"""
import hashlib
import json
from urllib.parse import urlencode
from research.odds_path_archive.collect import Source, matched

DAY="2026-11-08"
RACE=1

def market_info(response, track):
    r=matched(response,DAY,track,RACE)
    odds=r.get("odds_info") or {}
    pops=odds.get("time_pops") or {}
    times=odds.get("time_odds_times") or {}
    key=next((k for k in pops if str(k).lower().strip() in ("3rentan","trifecta","sanrentan","santan","3連単","tierce")),None)
    frames=pops.get(key,[]) if key else []
    stamps=times.get(key,[]) if key else []
    seen=set()
    frame_lens=[]
    hashes=[]
    for frame in frames:
        if not isinstance(frame,dict):
            raise ValueError("Unexpected frame")
        seen.update(frame.keys())
        frame_lens.append(len(frame))
        hashes.append(hashlib.sha256(json.dumps(frame,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:12])
    return {"market_key":key,"odds_info_keys":sorted(odds.keys()),"pool_keys":sorted(pops.keys()),
            "frame_count":len(frames),"frame_counts":frame_lens,"time_labels":stamps,
            "unique_tickets_across_frames":len(seen),"first_frame_digest":hashes[0] if hashes else None,
            "first_frame_tickets":set(frames[0]) if frames else set()}

def main():
    source=Source(pause=2.5)
    probe=[]
    catalog,_=source.api(DAY)
    tracks=catalog.get("kr_info",{}).get("track_info") or []
    if not tracks: raise ValueError("No tracks in actual API catalog")
    track=tracks[0]
    for page in (1,2,3):
        url="https://keirinodds.com/odds?"+urlencode(
            {"race_kind":"kr","race_date":DAY,"display_times":24,"page":page,
             "race_track":track,"race_no":RACE})
        raw,received=source.get(url)
        response=json.loads(raw)
        try:
            info=market_info(response,track)
        except ValueError as exc:
            returned=response.get("kr_info",{}).get("race_info") or []
            sample=[{"track":item.get("race_info",{}).get("track"),
                     "race":item.get("race_info",{}).get("race")} for item in returned[:2]]
            probe.append({"page":page,"error":str(exc),"returned_first_two":sample})
            continue
        ids=info.pop("first_frame_tickets")
        probe.append({"page":page,"source_sha256":hashlib.sha256(raw).hexdigest()[:16],
                      "first_frame_count":len(ids),"first_frame_tickets":ids,**info})
    success=[p for p in probe if "first_frame_tickets" in p]
    if not success:
        raise ValueError("No page returned expected race")
    union=set().union(*(p["first_frame_tickets"] for p in success))
    identical=all(p["first_frame_tickets"]==success[0]["first_frame_tickets"] for p in success[1:])
    for p in success: p.pop("first_frame_tickets")
    print(json.dumps({"race":f"{DAY} track={track} {RACE}R",
        "pages_with_correct_race":len(success),
        "pages_same_first_frame":identical,
        "first_frame_union_across_pages":len(union),
        "pages":probe,
        "conclusion":("NO_EXTRA_3RENTAN_FROM_PAGINATION" if len(success)==3 and identical else "PAGINATION_CANNOT_YET_EXTEND_HISTORY")},
        ensure_ascii=False))

if __name__=="__main__":
    main()
