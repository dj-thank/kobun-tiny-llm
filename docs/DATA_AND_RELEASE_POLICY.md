# データと公開方針

このリポジトリは、source code と公開可能な metadata を安全に公開するためのものです。本文コーパスや学習済み checkpoint を配布する場所ではありません。

## 公開するもの

このリポジトリで追跡するもの:

- source metadata
- provenance manifest
- hash と split 情報
- 小さな evaluation fixture
- プロジェクト作成の文法・和歌ルール表
- tokenizer public vocabulary metadata
- rebuild / check / evaluate 用コード
- 自律運用と release gate の contract

これらは pipeline を検査し、再現性を高めるための public source asset です。

## 公開しないもの

このリポジトリで追跡しないもの:

- raw source download
- clean 済み source text
- train / validation / test corpus
- generated training snapshot
- run log
- model checkpoint
- optimizer state
- release bundle
- private note
- local absolute path
- secret / credential

## Source manifest の意味

source manifest は、どの資料を参照したか、どのような hash や provenance を持つかを記録するためのものです。

manifest はライセンス許諾ではありません。コーパスを再構築する人は、各上流ソースの利用条件を確認し、必要な範囲で自分の環境に本文を取得してください。

## 学習データの方針

このプロジェクトは、古典日本語モデルをゼロから学習することを前提にしています。

学習コーパスとして使わないもの:

- 外部 LLM が生成した本文
- hosted model output
- 評価 fixture の答え
- private note
- license や provenance が不明な本文

評価 fixture や rule table は、モデルの品質を確認するためのものです。学習コーパスの代替ではありません。

## Release evidence

公開候補には checkpoint-bound evidence が必要です。

必要な証跡:

- exact best checkpoint の identity
- checkpoint hash
- independent test language-model loss
- grammar / morphology / waka checks
- source と split の hash
- tokenizer scope check
- overlap / leakage check
- package sanitization check
- no-export default が守られていること

validation loss だけでは公開判断に使いません。

## Hugging Face 公開方針

モデルが完成したら、Hugging Face に上げることを目標にできます。

ただし、GitHub の source release と Hugging Face の model release は別物です。Hugging Face 公開は、次の条件を満たしてから行います。

- fresh supervised run が完了している
- exact best checkpoint が固定されている
- release gate と quality gate が通っている
- package に raw/clean corpus、logs、optimizer state、private path、secret が含まれていない
- model card にデータ方針、評価結果、制限、非保証範囲を書く
- 人間が対象 checkpoint と upload を明示的に承認する

「upload-ready」は、後で人間が公開判断できるだけの evidence が整っているという意味です。自動で export / upload したという意味ではありません。
