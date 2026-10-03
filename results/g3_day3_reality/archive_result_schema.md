# G3 archive CSV schema

## payouts.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'race_no', 'race_type', 'ticket_type', 'bet_code', 'combination', 'payout_yen', 'popularity', 'status', 'result_source_url', 'result_source', 'captured_at_utc', 'raw_text']
- sample: ['2820220104010001', '2022-01-04', '立川競輪', '1', 'Ｓ級一予選', '2車単', 'exacta', '3-2', '2820', '12', 'paid', 'https://keirin.kdreams.jp/tachikawa/racedetail/2820220104010001/?pageType=KS_RACE_CARD_PAGE_TYPE_SHOW_RESULT', '楽天Kドリームス 結果・払戻金', '2026-09-02T14:03:15.447063+00:00', '3-2 2,820円 (12)']

## result_failures.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'race_no', 'race_type', 'result_source_url', 'result_source', 'captured_at_utc', 'error']
- sample: []

## line_failures.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'race_no', 'source_url', 'error']
- sample: []

## odds_failures.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'race_no', 'race_type', 'odds_source_url', 'error']
- sample: ['2820220104030006', '2022-01-06', '立川競輪', '6', 'Ｓ級選抜', 'https://keirin.kdreams.jp/tachikawa/racedetail/2820220104030006/?pageType=odds', 'OddsParseError: trifecta fixed-first tables missing for cars [2]; found=[1, 3, 4, 5, 6, 7, 8, 9]']

## races.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'meeting_grade', 'race_no', 'race_type', 'start_time', 'deadline', 'entry_count', 'source_url', 'captured_at_utc', 'predicted_line_formation', 'line_status', 'line_provider', 'line_source']
- sample: ['2820220104010001', '2022-01-04', '立川競輪', 'G3', '1', 'Ｓ級一予選', '10:55', '10:50', '9', 'https://keirin.kdreams.jp/tachikawa/racedetail/2820220104010001/?pageType=result', '2026-09-02T13:23:40.735837+00:00', '1-9/2-7/5-3-6/8-4', 'published', 'アオケイ', '楽天Kドリームス 並び予想']

## entries.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'meeting_grade', 'race_no', 'race_type', 'start_time', 'deadline', 'source_url', 'captured_at_utc', 'car_no', 'player_name', 'player_profile', 'prefecture', 'age', 'term', 'class', 'style', 'gear', 'score', 's_count', 'b_count', 'nige_count', 'makuri_count', 'sashi_count', 'mark_count', 'first_count', 'second_count', 'third_count', 'outside_count', 'win_rate', 'top2_rate', 'top3_rate', 'prediction_mark', 'evaluation', 'raw_row_json', 'predicted_line_formation', 'line_status', 'line_provider', 'line_source', 'line_id', 'line_position', 'line_size', 'line_role']
- sample: ['2820220104010001', '2022-01-04', '立川競輪', 'G3', '1', 'Ｓ級一予選', '10:55', '10:50', 'https://keirin.kdreams.jp/tachikawa/racedetail/2820220104010001/?pageType=result', '2026-09-02T13:23:40.735837+00:00', '1', '阿部 拓真', '宮 城/31/107', '宮城', '31', '107', 'S2', '両', '3.93', '106.69', '3', '1', '0', '4', '3', '0', '6', '1', '6', '10', '26.0', '30.4', '56.5', '◎', '3', '{"cells": ["◎", "", "3", "1", "1", "阿部 拓真 宮 城/31/107", "S2", "両", "3.93", "106.69", "3", "1", "0", "4", "3", "0", "6", "1", "6", "10", "26.0", "30.4", "56.5"], "row_class": ["n1"]}', '1-9/2-7/5-3-6/8-4', 'published', 'アオケイ', '楽天Kドリームス 並び予想', '1', '1', '2', '自在']

## trio_final_odds.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'race_no', 'race_type', 'ticket_type', 'combination', 'odds', 'market_rank', 'odds_status', 'total_votes', 'odds_as_of', 'odds_phase', 'odds_source', 'odds_source_url', 'captured_at_utc']
- sample: ['2820220104010001', '2022-01-04', '立川競輪', '1', 'Ｓ級一予選', '3連複', '1=2=3', '17.2', '6', 'available', '82967', '2022/01/04 10:57', 'final', '楽天Kドリームス 確定オッズ', 'https://keirin.kdreams.jp/tachikawa/racedetail/2820220104010001/?pageType=odds', '2026-09-02T14:20:21.392054+00:00']

## trifecta_final_odds.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'race_no', 'race_type', 'ticket_type', 'combination', 'odds', 'market_rank', 'odds_status', 'total_votes', 'odds_as_of', 'odds_phase', 'odds_source', 'odds_source_url', 'captured_at_utc']
- sample: ['2820220104010001', '2022-01-04', '立川競輪', '1', 'Ｓ級一予選', '3連単', '1-2-3', '115.8', '38', 'available', '643347', '2022/01/04 10:57', 'final', '楽天Kドリームス 確定オッズ', 'https://keirin.kdreams.jp/tachikawa/racedetail/2820220104010001/?pageType=odds', '2026-09-02T14:20:21.392054+00:00']

## results.csv (18 zip)
- schema x18: ['race_id', 'race_date', 'track', 'race_no', 'race_type', 'car_no', 'player_name', 'finish_position', 'finish_text', 'margin', 'last200', 'winning_method', 'sb', 'result_comment', 'result_source_url', 'result_source', 'captured_at_utc', 'raw_row_json']
- sample: ['2820220104010001', '2022-01-04', '立川競輪', '1', 'Ｓ級一予選', '1', '阿部 拓真', '5', '5', '３/４車輪', '12.0', '', '', 'バック捲上', 'https://keirin.kdreams.jp/tachikawa/racedetail/2820220104010001/?pageType=KS_RACE_CARD_PAGE_TYPE_SHOW_RESULT', '楽天Kドリームス 結果・払戻金', '2026-09-02T14:03:15.447063+00:00', '{"cells": ["◎", "5", "1", "阿部 拓真", "３/４車輪", "12.0", "", "", "バック捲上"], "row_class": []}']

## failures.csv (18 zip)
- schema x18: ['stage', 'discovered_on', 'url', 'error']
- sample: ['race_parse', '2022-03-01', 'https://keirin.kdreams.jp/kochi/racedetail/7420220226040012/?pageType=result', 'CollectorError: page is not marked G3']

## excluded_girls.csv (10 zip)
- schema x10: ['race_date', 'track', 'race_no', 'race_type', 'url']
- sample: []

## cancelled_races.csv (1 zip)
- schema x1: ['race_id', 'race_date', 'track', 'race_no', 'race_type', 'status', 'source_url', 'captured_at_utc', 'note']
- sample: ['8620240725030008', '2024-07-27', '別府競輪', '8', 'Ｓ級特選', 'cancelled', 'https://keirin.kdreams.jp/beppu/racedetail/8620240725030008/?pageType=KS_RACE_CARD_PAGE_TYPE_SHOW_RESULT', '2026-09-01T10:14:58.647393+00:00', '2024-07-27 別府G3 3日目8Rは悪天候で中止。KDreams公式インフォメーション等で確認済み。車券は全返還。結果・払戻行は生成しない。']

