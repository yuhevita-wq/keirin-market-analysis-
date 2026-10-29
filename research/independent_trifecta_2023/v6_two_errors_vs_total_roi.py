#!/usr/bin/env python3
"""Postprocess a complete, frozen V6 3rentan census into TWO simultaneous errors.

Input is EXACT all-ticket frozen V6 census artifact from completed GitHub
Actions run 40942071797. All cells use the SAME cohort and a 100-JPY fixed
stake per all 3ren-tan order, not an arbitrarily selected 10%-band strategy.

Error 1: realized ROI in each p band MINUS the *total all-ticket* realized ROI
for the SAME cohort. This is descriptive market-price recovery concentration,
not in itself proof that V6 p is inaccurate.
Error 2: actual winner count MINUS sum(V6 p) for every ticket in band,
also relative count error and percentage-point frequency error.
Both signed and absolute differences shown, and no target-75% comparison.
Training 2023 Jan-Oct kept separate from already-inspected retrospective years.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

GROUPS=("2023_train","2023_dev","2024","2025","2026","2024_to_2026","TOTAL")
BANDS=("per_probability_band","fine_probability_band")

def process(original):
    assert original["state"]=="V6_ALL_TRIFECTA_ROI_75PCT_VS_ACTUAL_COMPLETE"
    assert original["all_tickets_no_probability_selection"] is True
    assert original["no_odds_file_read"] is True
    assert original["model_unchanged"] is True
    result={
        "state":"V6_FULL_TICKET_DUAL_ERROR_VS_ACTUAL_TOTAL_ROI_COMPLETE",
        "source_workflow_run":"https://github.com/yuhevita-wq/keirin-market-analysis-/actions/runs/40942071797",
        "frozen_model":original["model"],
        "benchmark_definition":"all 3ren-tan combinations at 100 JPY each for SAME period; actual settled winning payouts / total stake; NOT 75%",
        "two_concurrent_metrics":{
            "market_return_deviation":"band_actual_roi_pct - same_cohort_all_ticket_actual_roi_pct",
            "probability_calibration_deviation":"band_actual_winners - sum_unmodified_V6_probabilities",
        },
        "years_already_inspected_not_genuinely_unseen":True,
        "training_period_separate":True,
        "cohorts":{},
    }
    for group in GROUPS:
        total=original["full_population"][group]
        if total is None:continue
        roi_base=100*total["actual_settled_payout_yen"]/total["hypothetical_stake_yen"]
        n_total=total["all_tickets_in_probability_group"]
        detail={
            "races":total["races_with_probability_mass"],
            "tickets":n_total,
            "total_stake_yen":total["hypothetical_stake_yen"],
            "total_payout_yen":total["actual_settled_payout_yen"],
            "cohort_total_roi_pct":round(roi_base,7),
            "full_ticket_observed_wins":total["actual_winning_tickets"],
            "full_ticket_expected_wins":total["expected_winning_tickets"],
            "coarse_probability_bands":[],
            "fine_probability_bands":[],
        }
        for input_name,output_name in (("per_probability_band","coarse_probability_bands"),
                                       ("fine_probability_band","fine_probability_bands")):
            entries=original[input_name][group]
            n=0;hits=0;expected=0.;stake=0.;paid=0.
            for label,cell in entries.items():
                if cell is None:continue
                nt=cell["all_tickets_in_probability_group"]
                pr=float(cell["expected_winning_tickets"])
                actual=int(cell["actual_winning_tickets"])
                invested=cell["hypothetical_stake_yen"]
                returned=cell["actual_settled_payout_yen"]
                if invested!=100*nt:raise AssertionError("Non-uniform wager")
                roi=100*returned/invested
                diff=actual-pr
                row={
                    "probability_band":label,
                    "tickets":nt,
                    "predicted_hit_pct":round(100*pr/nt,8),
                    "realized_hit_pct":round(100*actual/nt,8),
                    "v6_expected_wins":round(pr,5),
                    "actual_wins":actual,
                    "actual_minus_expected_wins":round(diff,5),
                    "absolute_wins_error":round(abs(diff),5),
                    "hit_frequency_error_percentage_points":round(100*diff/nt,8),
                    "actual_over_expected_win_ratio":round(actual/pr,7) if pr else None,
                    "paid_yen":returned,
                    "staked_yen":invested,
                    "actual_roi_pct":round(roi,7),
                    "same_period_total_roi_pct":round(roi_base,7),
                    "actual_roi_minus_total_roi_percentage_points":round(roi-roi_base,7),
                    "absolute_roi_deviation_from_total_points":round(abs(roi-roi_base),7),
                }
                detail[output_name].append(row)
                n+=nt;hits+=actual;expected+=pr;stake+=invested;paid+=returned
            assert n==n_total,(group,input_name,"ticket count mismatch",n,n_total)
            assert hits==total["actual_winning_tickets"]
            assert abs(expected-total["expected_winning_tickets"])<.1
            assert stake==total["hypothetical_stake_yen"]
            assert abs(paid-total["actual_settled_payout_yen"])<.1
        result["cohorts"][group]=detail

    a=result["cohorts"]["2024_to_2026"]["coarse_probability_bands"]
    neg=[x for x in a if x["actual_minus_expected_wins"]<0]
    pos=[x for x in a if x["actual_minus_expected_wins"]>0]
    result["descriptive_observations"]={
        "2024_2026_roi_base_pct":result["cohorts"]["2024_to_2026"]["cohort_total_roi_pct"],
        "lowest_roi_band":min(a,key=lambda x:x["actual_roi_pct"])["probability_band"],
        "highest_roi_band":max(a,key=lambda x:x["actual_roi_pct"])["probability_band"],
        "underpredicted_win_count_bands":[x["probability_band"] for x in pos],
        "overpredicted_win_count_bands":[x["probability_band"] for x in neg],
    }
    result["interpretation_limits"]=[
       "The 10% illustration is NOT a selection rule; ALL tickets are evaluated.",
       "Each probability band uses the SAME total all-ticket ROI within the specified year/cohort as baseline, not 75%.",
       "All-ticket ROI is inherently independent of how the frozen V6 ranks tickets.",
       "Disagreement between ROI and overall ROI is not necessarily miscalibrated probabilities: market odds vary by band.",
       "Compare observed wins against summed frozen V6 probabilities to assess calibration; the ROI comparison diagnoses price structure.",
       "A complete probability distribution sums to one per race, so full-ticket predicted winners equal actual winners by construction and cannot alone validate calibration; use bands.",
       "Full fine bands are exploratory; low-count cells need uncertainty intervals and genuine future holdout to confirm.",
       "Archived settled payout is real after-the-fact return, not a verified pre-purchase quote.",
    ]
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    audit=json.loads(args.input.read_text(encoding="utf-8"))
    report=process(audit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for year in ("2023_train","2023_dev","2024","2025","2026","2024_to_2026"):
        r=report["cohorts"][year]
        print("V6_TWO_ERRORS_COHORT",json.dumps({
            "period":year,
            "races":r["races"],
            "tickets":r["tickets"],
            "baseline_total_roi":r["cohort_total_roi_pct"],
            "bands":r["coarse_probability_bands"],
        },ensure_ascii=False),flush=True)
    print("V6_FULL_ALL_TICKET_TWO_ERRORS_REPORT_VERIFIED",flush=True)

if __name__=="__main__":main()
