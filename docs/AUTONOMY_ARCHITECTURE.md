# 自律アーキテクチャ

このプロジェクトの自律アーキテクチャは、モデルを自動で公開するための仕組みではありません。目的は、学習 run、評価、公開判定、証跡、non-release 管理を一貫した contract に従って扱い、危険な公開を防ぐことです。

## 基本原則

- LLM 本体と自律運用を分ける
- 自律レイヤーはモデルの出力品質を「雰囲気」で判断しない
- release 判断は checkpoint-bound evidence に基づける
- raw corpus、clean corpus、private path、secret、optimizer state を公開物に混ぜない
- export / package / upload は明示承認なしに行わない

## パッケージ分離

```text
src/kobun_llm/       モデル本体の責務
src/kobun_autonomy/  監査、分類、公開判定、型付き contract の責務
```

`kobun_llm` は training loop、tokenizer、model、checkpoint I/O などを持ちます。

`kobun_autonomy` は release policy、non-release registry、augmentation audit、evaluation board 型、release evidence 型などを持ちます。

互換性のために `kobun_llm` 側に薄い shim が残る場合がありますが、新しい policy の権威は `kobun_autonomy` 側に置きます。

## 自律レイヤーが行うこと

自律レイヤーは次のような作業を担当します。

- run の状態を読む
- active / completed / failed / non-release を分類する
- evaluation board を構築する
- gate が通っているか確認する
- 次の安全な action を選ぶ
- non-release run を記録する
- review packet を sanitize して生成する
- release package に含めてはいけないものを検査する

## 自律レイヤーが行わないこと

自律レイヤーは次を行いません。

- LLM 生成文を学習コーパスとして作る
- reviewer text を test loss や leakage check の代わりにする
- release blocker を要約で隠す
- 古い checkpoint metadata を手で書き換えて release candidate にする
- 明示承認なしに Hugging Face へ upload する

## 型付き contract

`src/kobun_autonomy/types.py` は、自律運用で共有するデータ構造を型として定義します。

主な型の対象:

- backend
- run status
- release status
- run classification
- evaluation board
- release evidence
- autonomous action
- health status

これにより、script ごとにバラバラの dict を作るのではなく、同じ意味を同じ形で扱えるようにします。

## Gate の考え方

release candidate training や release package は、複数の gate を通る必要があります。

- preflight gate
- independent review gate
- checkpoint-bound quality gate
- source provenance gate
- tokenizer scope gate
- leakage / overlap gate
- package sanitization gate

これらは、モデルを強く見せるためではなく、公開してはいけない artifact を止めるための仕組みです。

## Non-release 管理

過去の失敗 run、古い run、証跡不足の run、途中停止した run は non-release として扱います。non-release record は durable な policy であり、後から都合よく release candidate に昇格させません。

release candidate が必要な場合は、古い checkpoint を修正するのではなく、新しい supervised run を開始します。

## Hugging Face との関係

Hugging Face への upload は、自律ループが勝手に行うものではありません。

自律レイヤーは、将来の upload に必要な evidence を整え、package が安全かを検査できます。しかし、最終的な export / upload は、人間が明示的に対象 checkpoint と公開先を承認した後にだけ行います。
