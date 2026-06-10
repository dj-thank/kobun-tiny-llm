# Kobun Tiny LLM

Kobun Tiny LLM は、古典日本語、特に中古和文・平安中期ごろの文体を対象に、小さな GPT 形式の言語モデルをゼロから学習するための研究用コードベースです。

このリポジトリの目的は、モデルそのものを大きく見せることではありません。学習パイプライン、トークナイザ方針、コーパス来歴、評価、リーク検査、公開判定を、外部から読める形で整理することを重視しています。

現時点で公開しているのは、ソースコード、メタデータ、ルール表、小さな評価 fixture、自律運用のための型とゲートです。モデル重み、チェックポイント、学習ログ、生コーパス、clean 済みコーパス、派生学習コーパス、optimizer state、release package は含めていません。

## 現在の状態

このリポジトリには、まだ公開用のモデルチェックポイントはありません。

ローカルで行った過去の試行は、公開リリース候補ではなく、内部的な実験証跡として扱います。公開モデルを作るには、次の条件を満たす新しい supervised run が必要です。

- ゼロから学習した fresh run であること
- exact best checkpoint に紐づいた評価があること
- test loss、文法、形態、和歌、リーク検査、トークナイザ範囲検査が通っていること
- 生テキストや派生コーパスを公開物に混ぜていないこと
- release gate が明示的に通っていること
- 人間が最終的に export / upload を承認していること

## リポジトリ構成

```text
src/kobun_llm/       LLM 本体、tokenizer、生成、文法制約、学習コード
src/kobun_autonomy/  自律運用、公開判定、non-release 管理、型付き gate
scripts/            データ準備、評価、監査、supervisor、release gate
data/               公開可能なメタデータ、ルール表、source manifest、評価 fixture
docs/               アーキテクチャ、データ方針、公開監査メモ
```

`kobun_llm` と `kobun_autonomy` は意図的に分けています。

`kobun_llm` は、モデル、tokenizer、学習、推論、checkpoint I/O など、LLM 本体に関係する領域です。

`kobun_autonomy` は、run の分類、non-release 記録、公開可否、監査、release gate、次アクション選択など、自律運用と公開判断に関係する領域です。

モデル本体は、自律運用レイヤーに依存しすぎないようにします。逆に、自律運用レイヤーはモデルや run を監視・分類できますが、モデルの学習データを作ったり、評価指標の代わりになったり、勝手に export / upload したりしてはいけません。

詳しくは [Autonomy Architecture](docs/AUTONOMY_ARCHITECTURE.md) を参照してください。

## インストール

Python 3.10 以上を想定しています。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

CPU だけで構文確認をする場合:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\kobun_llm\model.py src\kobun_llm\tokenizer.py src\kobun_llm\train.py
```

## 学習データの境界

この公開リポジトリには、学習用の本文データを含めていません。

source manifest は、どの公開・再取得可能な資料を参照したかを記録するためのものです。manifest はライセンス許諾そのものではありません。コーパスを再構築する場合は、各上流ソースの利用条件を確認してください。

このプロジェクトは、モデルをゼロから学習することを前提にしています。学習コーパスとして、外部 LLM の生成文や hosted model output を使う設計にはしていません。

## 評価の境界

このリポジトリには、公開しても安全な小さな評価 fixture とルール表を含めています。

公開リリースに使える評価証跡は、必ず checkpoint に紐づいている必要があります。validation loss だけでは公開判断に使いません。

公開候補の評価には、少なくとも次を含めます。

- independent test language-model loss
- grammar / morphology / waka の smoke・regression check
- train / eval overlap check
- split leakage check
- source provenance hash
- tokenizer scope check
- exact best checkpoint の identity

## 自律アーキテクチャ

自律レイヤーは、モデルではなく supervisor と evidence system です。

できること:

- evaluation board を作る
- run の状態を分類する
- 次に取るべき安全な action を選ぶ
- no-export default を守る
- review packet を sanitize して作る
- non-release run を durable record として管理する

してはいけないこと:

- LLM 生成文を学習コーパスとして使う
- checkpoint-bound metric の代わりに reviewer text を使う
- release blocker を要約で隠す
- 明示的な承認なしに model export / package / upload を行う

## Hugging Face への公開方針

モデルが完成したら、Hugging Face への公開を目標にできます。

ただし、このリポジトリの現在の状態は source-only 公開です。Hugging Face に上げるのは、次の条件がそろってからです。

- fresh supervised run が完了している
- exact best checkpoint が特定されている
- release gate と品質 gate が通っている
- 公開 package に生コーパス、clean コーパス、学習ログ、optimizer state、private path、secret が含まれていない
- model card に学習データの方針、評価結果、制限、非保証範囲を書く
- 最終的に人間が「この checkpoint を Hugging Face に上げる」と明示的に承認する

この方針により、開発中の run や不完全な checkpoint が誤って公開されることを防ぎます。

将来の公開先は、モデル完成後に別途決めます。公開する場合も、GitHub の source release と Hugging Face の model release は別物として扱います。

## 公開データ方針

このリポジトリで追跡するもの:

- source metadata と provenance manifest
- 小さな評価 fixture
- このプロジェクトで作成した文法・和歌ルール表
- tokenizer public vocabulary metadata
- rebuild / check / evaluate 用のコード
- 自律運用と release gate の型・契約

このリポジトリで追跡しないもの:

- raw source download
- clean 済み source text
- train / validation / test corpus
- 学習 snapshot
- run log
- checkpoint
- optimizer state
- release bundle
- private note

詳しくは [Data and Release Policy](docs/DATA_AND_RELEASE_POLICY.md) を参照してください。

## セキュリティ

credential、local machine path、private note、raw log、model checkpoint、optimizer state、generated release package、personal browser state を commit しないでください。

詳しくは [SECURITY.md](SECURITY.md) を参照してください。

## ライセンス

コードは [LICENSE](LICENSE) に従います。

manifest が参照する古典本文や外部資料には、それぞれ別の利用条件がある場合があります。本文データの再配布や再利用を行う場合は、必ず元資料の条件を確認してください。
