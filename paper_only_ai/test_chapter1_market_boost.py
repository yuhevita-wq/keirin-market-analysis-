"""Boosted residual model's time-safe probability and payout invariants."""
import sys,types,importlib.util
from pathlib import Path
import numpy as np
mod=types.ModuleType("chapter1_market_error")
sys.modules["chapter1_market_error"]=mod
spec=importlib.util.spec_from_file_location("market_boost",
    Path(__file__).with_name("chapter1_market_boost.py"))
s=importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
base=np.array([0.4,0.35,0.25])
scores=np.array([0.7,0.2,0.1])
row=("2023-10-05","r1",np.zeros((3,17),dtype=np.float32),
     np.log(base),np.array([2.5,4.,5.]),0,np.log(scores))
assert np.allclose(s.blend(row,0),base)
assert np.allclose(s.blend(row,1),scores)
assert np.isclose(s.blend(row,0.5).sum(),1)
assert s.blend(row,1)[0]>s.blend(row,0)[0]
data={"3連単":[row],"3連複":[row]}
results=s.book_return(data,{"3連単":1.0,"3連複":1.0},0)
assert results["purchased_tickets"]>=1
assert results["winning_tickets"]==2
assert results["payout_yen"]==500.0
X,y=s.training_arrays([row[:-1]])
assert len(X)==3 and int(y.sum())==1 and X.shape[1]==19
print("BOOST_MARKET_RESIDUAL_SAFETY_TEST_PASSED")
