# 育成AI：GitHub保存済みデータへの正規接続

旧育成AIの train_week.py は2023年の一部アーカイブと固定の年別オッズ・結果ファイルしか読み取らなかった。その他の年や保管庫には接続していなかった。

## 元データの棚卸し

- data/audits/odds_universe_inventory.json：元出走表ソースに95,165の異なるレースID。出走表ソースが存在するだけでは完全レースを意味しない。
- data/audits/result_universe_v2.json：92,938の有効な着順候補（元の出走表IDと集合が同一とは限らない）。
- data/audits/odds_only_export.json：39,934レース分の完全な確定オッズ（3連単・3連複のみ）。2023年だけでも9,987レース。
- 旧GEN-007の2023年4月5日〜6月11日の61学習日の出走表3,869R中、結合完了は1,475R、完全オッズ未結合2,377R。これは同一日時・レースIDによる結合の問題であり、違う日時の価格を借りてはいけない。

## 接続口

研究ディレクトリ research/ai_nursery/source_bridge.py に共通APIを追加。

- joined_races(year, first=None, last=None) ：該当年のA級・チャレンジ・S級・Gレースのレースカード、3連単・3連複の全確定オッズと1〜3着実結果を16桁レースIDで厳密結合し、既存 feature_matrix が扱う辞書を返す。
- joined_races_for_training(start,end) ：将来の世代が2021〜2026上期を年をまたいで学習するための正規入口。2023年6月1〜7日と7月1〜7日の既存ホールドアウトへの学習を拒否する。
- 入力は data/2023, data/2024, data/2025, data/2026_h1, data/a_class_12, data/a3_challenge, data/grade_races にある出走表とアーカイブ。男子のみ。2026年下期は拒否。
- 3連単・3連複の価格は data/odds_only/odds_YYYY.jsonl.gz。data/audits/odds_only_export.json と照合して件数・SHA256を検査。未保存の券種や買い目は作らない。
- 着順は data/result_universe_v2/combined/results_YYYY.jsonl.gz。着順や払い戻しなどの結果列を出走表特徴から除く。レースIDの開催開始日を実施日と取り違えない。

## 監査と保存

source_connection_audit.py をGitHub Actionsで実行すると research/ai_nursery/source_connection/coverage.json と docs/ai-lab/nursery-source-connection.json に年別の元出走表・欠損結果・欠損完全オッズ・実際に結合したレース数、ソース選択とチェックサムを保存する。test_source_bridge.py は不正レース日、結果リーク、完全市場、監査の一致、ホールドアウトへの学習禁止を検証する。

**この段階は接続と監査のみ。AIの重みを更新しない。** 歴代GEN-001〜007のチェックポイント、学習時の結果と用紙は不変。

過去の確定オッズを締切前の価格として扱わない。2021〜22の出走表は主としてグレード戦ZIPで、全国全レースとは限らない。2023年の専用A3の3〜4月アーカイブは存在しない。暗号化済み7券種の秘密鍵を求めたり、非公開鍵をコミットしたりしない。

## 追加接続：回収済みレースと2層教師データ

以前の37,441Rの接続に加え、`data/unarchived_races_v1/batch_*` の gzip 圧縮出走表を統合する。新しい `supervised_races_year` は (A)着順＋完備した3連単・3連複オッズ (B)着順は確認できるが全オッズが無い出走表 を重複なしに返す。Aは既存の利益学習に利用できる。Bはオッズを前提としない将来の着順確率モデル用の入力であり、既存の `feature_matrix` に投入しないこと。6年分の年別結合・欠損・監査結果は `research/ai_nursery/source_connection/coverage-expanded.json` および `docs/ai-lab/nursery-source-connection-expanded.json`。全国カレンダーは開催レースの索引にとどまり、選手の出走表ではない。暗号化された券種データは復号手段が確認できないため価格として扱わない。既存AIの重みは不変。
