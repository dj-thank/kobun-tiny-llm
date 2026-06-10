# アーキテクチャ

Kobun Tiny LLM は、古典日本語向けの小さな GPT 形式モデルをゼロから学習するための研究用コードベースです。設計上の中心は、モデル本体と自律運用レイヤーを分けることです。

この分離により、モデルの学習・推論コードを読みやすく保ちつつ、公開判断、監査、run 管理、Hugging Face への将来公開準備を別の責務として扱えます。

## 全体像

```text
src/kobun_llm/       LLM 本体、tokenizer、生成、文法制約、学習、checkpoint I/O
src/kobun_autonomy/  自律運用、公開判定、non-release 記録、型付き gate
scripts/            データ準備、評価、監査、supervisor、release tooling
data/               公開可能な metadata、rule table、source manifest、eval fixture
docs/               設計、データ方針、公開監査メモ
```

## `kobun_llm`

`kobun_llm` はモデル本体のためのパッケージです。

主な責務:

- GPT 形式モデルの定義
- tokenizer と vocabulary の扱い
- 生成処理
- 文法・和歌制約の補助処理
- checkpoint の読み書き
- training loop
- device 選択

`kobun_llm` は、公開可否の判断や upload 判断を直接持たないようにします。モデル本体は、できるだけ「入力を受けて学習・推論する」純粋な領域に近づけます。

## `kobun_autonomy`

`kobun_autonomy` は、自律運用と release governance のためのパッケージです。

主な責務:

- run の分類
- non-release run の durable record
- release candidate の条件
- augmentation manifest の監査
- evaluation board の型
- release evidence の型
- supervisor / gate が使う共通 contract

自律レイヤーは、モデルを勝手に公開する仕組みではありません。むしろ逆で、未完成の run、古い checkpoint、証跡不足の artifact、private path や raw corpus を含む package が公開されないようにするための安全装置です。

## `scripts/`

`scripts/` は実行用の entrypoint 群です。データ準備、評価、監査、training supervisor、release gate、将来の export preparation などを担当します。

長期的には、`scripts/` は薄い CLI wrapper に寄せ、重要な policy や contract は `kobun_autonomy` に移す方針です。

## データと評価の境界

公開リポジトリには、本文コーパスや学習 snapshot を含めません。含めるのは、source manifest、hash、rule table、小さな evaluation fixture、再構築・監査用コードです。

公開用の評価は、必ず exact best checkpoint に紐づけます。validation loss のみ、または reviewer text のみでは release evidence になりません。

## Hugging Face 公開との関係

GitHub のこのリポジトリは source-only release です。Hugging Face へ上げる model release は別の段階です。

モデル完成後に Hugging Face 公開を行う場合は、次の流れを想定します。

1. fresh supervised run を完了する
2. exact best checkpoint を固定する
3. checkpoint-bound evaluation を通す
4. package に raw/clean corpus、logs、optimizer state、private path、secret が入っていないことを確認する
5. model card に学習方針、評価、制限、非保証範囲を書く
6. 人間が明示的に export / upload を承認する

この設計により、GitHub 上の source release と Hugging Face 上の model release を混同しないようにしています。
