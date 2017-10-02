## Chainer implementation of LSTM

### 動作環境

- Chainer 2+
- Python 2 or 3

## 実行

### Penn Treebank

```
cd run/ptb
python3 train.py
```

### 任意のテキストデータ

#### 学習

```
cd run/text
python3 train.py -train PATH_TO_TRAINING_TEXTFILE -dev PATH_TO_DEVELOPMENT_TEXTFILE -test PATH_TO_TEST_TEXTFILE
```

単語がスペースで区切られている必要があります。

`-dev`と`-test`は指定しなくてもOKです。

#### 文章生成

```
cd run/text
python3 generate.py
```
