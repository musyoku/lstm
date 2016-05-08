# Chainer implementation of LSTM

[この記事](http://musyoku.github.io/2016/04/10/Chainer%E3%81%A7LSTM%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%A8%E3%83%9F%E3%83%8B%E3%83%90%E3%83%83%E3%83%81%E5%AD%A6%E7%BF%92%E3%81%AE%E5%AE%9F%E8%A3%85/)で実装したコードです。

ChainerでLSTMを学習させるシンプルなコードになっています。

## 動作環境

- Chainer 1.8+

## 使い方

train_textフォルダ内の`train.py`を学習に用い、`validate.py`で文章生成を行えます。

train_text内に`text`フォルダを作成し、その中に文章データの`.txt`ファイルを入れてください。

テキストファイルは何個入れてもすべて自動的に読み込まれます。ファイル名も任意で構いません。

## オプション

- lstm_apply_batchnorm
	- [Recurrent Batch Normalization](http://arxiv.org/abs/1603.09025)を有効にします
	- 通常のLSTMに比べて学習速度は半分ほどに落ちますが、時間あたりの収束の速さは通常版を上回ります
	- デフォルトで有効です
- fc_output_type
	- LSTM出力を単語へ変換する全結合層の種類を指定します
	- 1か2のどちらかを指定します
	- 1(デフォルト)はソフトマックス層を用いて単語IDの分布を出力します
	- 2は直接単語埋め込みベクトルを出力し、二乗誤差による学習を行います
	- 2を使って学習させた場合、埋め込みベクトルの単語IDへの変換には`EmbedID.reverse()`を使います

## その他

以下のリポジトリの内容を含んでいます。

- [EmbedID拡張](https://github.com/musyoku/embed-id-extended)
- [Recurrent Batch Normalization](https://github.com/musyoku/recurrent-batch-normalization)

すでに入っているのでコピーする必要はありません。
