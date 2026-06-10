# 公開監査メモ

このメモは、Kobun Tiny LLM を世界に公開できる source repository として整えるための監査観点をまとめたものです。

## 公開形態

現在の公開形態は source-only です。

含めるもの:

- source code
- public metadata
- source manifest
- rule table
- compact eval fixture
- autonomy / release gate contract
- documentation

含めないもの:

- model weights
- checkpoints
- optimizer state
- raw corpus
- clean corpus
- train / validation / test corpus
- run logs
- release package
- private notes

## 監査した観点

公開前に確認する観点:

- 個人名や local machine path が入っていないか
- raw / clean corpus が追跡されていないか
- checkpoint や optimizer state が追跡されていないか
- generated logs や release package が追跡されていないか
- internal-only review text が公開 docs に混ざっていないか
- README と docs が source-only release として誤解を招かないか
- Hugging Face upload が自動実行されるように見えないか

## LLM と自律レイヤーの分離

公開時点での重要な設計判断は、LLM 本体と自律レイヤーを分けることです。

- `src/kobun_llm/` はモデル本体の領域
- `src/kobun_autonomy/` は公開判定、監査、non-release 管理、gate の領域

これにより、モデルの学習・推論コードと、公開安全性を判断する governance code を混同しないようにしています。

## 型付き自律 contract

自律レイヤーは、評価 board、run 分類、release evidence、autonomous action を型付きの構造として扱います。

型付き contract によって、次の点を監査しやすくします。

- どの run が release candidate か
- どの run が non-release として固定されているか
- どの gate が通っていて、どの gate が blocker か
- upload-ready と実際の upload が混同されていないか
- reviewer text が checkpoint-bound metric の代わりになっていないか

## Hugging Face 公開前の追加監査

モデル完成後に Hugging Face へ公開する場合は、GitHub 公開とは別に次を監査します。

- exact best checkpoint が正しいか
- model config と tokenizer が checkpoint と一致しているか
- model card が十分か
- eval results が checkpoint-bound か
- release package に禁止ファイルがないか
- safetensors など安全な形式を使っているか
- private path、secret、raw text、clean text が入っていないか
- upload が人間の明示承認に基づいているか

## 現在の結論

このリポジトリは、source-only public repository として公開する前提で整理されています。

モデル release はまだ行っていません。Hugging Face 公開は、完成した fresh checkpoint と release evidence がそろった後の別工程です。
