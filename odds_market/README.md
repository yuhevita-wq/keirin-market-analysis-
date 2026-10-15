# ODDS ONLY — 全国・全レース・全車券種・全組番

このサブシステムに保存するデータは **オッズのみ** です。出走表、選手、級班、ライン、予想、結果、払戻などの項目を付加しません。レース特定に必要なキーだけをオッズに添えます。

## 収集対象

発売のある全車券種・全組番を漏れなく扱います。

| 券種 | 通常の5車 | 通常の6車 | 通常の7車 | 通常の9車 |
| --- | ---: | ---: | ---: | ---: |
| 3連単 | 60 | 120 | 210 | 504 |
| 3連複 | 10 | 20 | 35 | 84 |
| 2車単 | 20 | 30 | 42 | 72 |
| 2車複 | 10 | 15 | 21 | 36 |
| ワイド | 10 | 15 | 21 | 36 |
| 2枠単 | 発売なし | 発売なし | 発売なし | 33 |
| 2枠複 | 発売なし | 発売なし | 発売なし | 18 |
| **合計** | **110** | **200** | **329** | **783** |

全国の全開催・全級班・全レースを対象にし、3-2-2などの型で選別しません。人気上位の数十点だけでは「全オッズ」と認定しません。実際に発売されていない券種は数値0で代用しません。途中欠場等で買える組番が変わる特殊レース、8車立て、4車以下はまだ網羅できません。これらを全件取得済みと偽らず取得エラーとして標準エラー出力します。

## オッズファイル以外は保存しない

`data/odds_only/YYYY/YYYY-MM-DD.csv.gz` に含まれる列:

```csv
race_id,race_type,bet_type,selection,odds_min,odds_max,odds_time_jst
```

**race_id** は結合用識別子、**race_type** はレース固有の見出しから照合した種別（例: A級一般）。**selection** は買い目（例: `1-2-3`）。オッズは倍数。ワイドのレンジだけ `odds_min` と `odds_max` が異なる。確定オッズ以外は取り込まず、人気・投票票数・払戻・選手情報・取得元文章は保存しない。

収集できない組番を推測・平均・補間で埋めず、完全収集できたレースのみ保存します。レース固有の見出しからレース種別を確認できない場合、ガールズを確実に除外できないため、そのレースは保存しません。同日ファイルに検証済みの完全な7券種オッズがある場合は、再開時にそのレースを再取得しません。

## 実行方法

公開元の自動収集・保存許諾を得た場合だけ実行してください。現在の実装は **実サイトに対する全賭式の動作確認が未完了** です。テストが通っても実サイトで全件取得できることを保証しません。

```bash
python -m pip install requests beautifulsoup4
python -m odds_market.test_collect_all_odds
python odds_market/collect_all_odds.py --start 2024-07-04 --end 2024-07-04 --out data/odds_only --pause 2 --accept-terms
```

GitHub Actions:
- [単体7券種取得・暗号化実証](https://github.com/yuhevita-wq/keirin-market-analysis-/actions/workflows/odds-seven-bet-authorized-pilot.yml)
- [認可済み全期間の暗号化・再開可能な定期収集](https://github.com/yuhevita-wq/keirin-market-analysis-/actions/workflows/odds-seven-bet-encrypted-backfill.yml)
- [7券種オフライン検証と従来の手動試験](https://github.com/yuhevita-wq/keirin-market-analysis-/actions/workflows/odds-market.yml)

**2026-10-15の実サイト検証:** 2024-07-04 小松島1R（9車）で7券種の全783組番を取得し、AES-256-GCMで暗号化保存。取得1件の実証にすぎず、全国網羅は別途監査する。定期収集は2021-01-01〜2026-06-30（ガールズ除外）を1回2日ずつ進め、実績が0件の失敗が3回続くと停止し、欠損日を記録して後から再試行する。

公開GitHubに生のオッズや暗号化秘密鍵は置かない。取得した完全オッズは `data/odds_ciphertext/YYYY/YYYY-MM-DD/RUN_ID.p7m` に**暗号文だけ**を保存する。進捗や失敗日のメタデータは `data/audits/odds_seven_bet_cursor.json`。レースID以外の選手名・結果・払戻・得点・ライン情報は分析ファイルに保存しない。

保管形式の復号（個人で秘密鍵を安全に管理すること）:

```bash
openssl cms -decrypt -binary -inform DER \
  -in data/odds_ciphertext/2024/2024-07-04/38713793808.p7m \
  -recip odds_market/vault_public_cert.pem \
  -inkey keirin_odds_private.pem \
  -out decoded_odds.csv.gz
```

`keirin_odds_private.pem` はチャット経由で別渡しする復号用の秘密鍵。**GitHubにアップロードしない。紛失すると暗号化済みオッズを復元できない。**

## 15分前と確定オッズ

過去に遡って確認できるのは原則として確定オッズです。15分前は今後のリアルタイム観測が必要であり、履歴からでっち上げたり確定オッズを15分前として扱ったりしません。

出典利用条件: [Kドリームスのサイトポリシー](https://keirin.kdreams.jp/info/sitepolicy/)。
