# SAM Audio Server

更新: 2026-09-22  
バージョン: 0.1.0（`VERSION`）

## 概要

混ざった音声から、テキストで指定した音だけを抜き出す HTTP API サーバーです。抜き出した音（target）と残り（residual）を返します。ブラウザのテストページ（`/test/`）からも使えます。

対象は英語の短い説明で指定します（例: `man speaking`）。時間区間の指定もできます。フルモデルでは映像とマスクによる視覚指定も使えます。Lite は音声とテキストのみです。

NVIDIA（CUDA）と AMD（ROCm、Radeon 780M を含む）で動かせます。使うモデルは起動時に ID で選びます。

## 配置

- Python モジュール: `modules/`
- モデルカタログ: `models/models.json`
- ダウンロード先: `models/`
- ブラウザテスト用: `test/`（`http://127.0.0.1:8765/test/`）

## 仮想環境

GPU 系統ごとに **別の venv フォルダ** を使います。この PC のように RTX 2060 SUPER と Radeon 780M が同居している場合、ROCm 用 `venv` のまま `run_cuda.bat` を実行しても NVIDIA GPU は見えません。

| 環境 | インストール | 起動 | venv |
|------|--------------|------|------|
| CPU | `install_cpu.bat` | `run_cpu.bat` | `venv-cpu` |
| RTX 2060 SUPER（CUDA） | `install_cuda.bat` | `run_cuda.bat` | `venv-cuda` |
| RX 7900 XTX（ROCm gfx1100） | `install_amd.bat` | `run_amd.bat` | `venv-amd` |
| Radeon 780M（ROCm gfx1103） | `install_780m.bat` | `run_780m.bat` | `venv` |

起動引数の例（`run_*.bat` の末尾に追加可能）:

```text
--host 0.0.0.0 --port 9000
--model large-lite
```

## モデル

カタログは `models/models.json` です。`--model` を省略すると JSON の `default`（`base-lite`）を使います。

| ID | 重み | Lite | 目安 |
| --- | --- | --- | --- |
| `base-lite` | base-tv BF16（約 3.6 GiB） | あり | 8GB GPU（既定） |
| `base` | 同上 | なし | フルモデル。8GB では溢れる |
| `large-lite` | large-tv BF16（約 6.9 GiB） | あり | 視覚なしの large |
| `large` | 同上 | なし | さらに VRAM が必要 |

Lite は視覚エンコーダと Ranker なし、BF16 です。`base` / `large` は同じ重みでも PE と Ranker を載せるため、8GB では溢れます。

```text
run_cuda.bat --model base-lite
run_cuda.bat --model large
run_780m.bat --model base-lite
```

未知の ID を渡すと、利用可能な ID 一覧を出して終了します。

AMD 向けの補足は `docs-rocm-windows.txt` を参照してください。DirectML は使用しません。

## API

- `GET /health` — 死活確認（`model_loaded` でモデル読み込み済みか確認）
- `GET /v1/info` — ロード済みモデル ID、device
- `POST /v1/separate` — `multipart/form-data`

| フィールド | 必須 | 内容 |
| --- | --- | --- |
| `audio` | 音声か映像のどちらか | 入力ミックス |
| `video` | 任意 | visual prompting 用映像 |
| `mask` | 任意 | 対象物体のマスク映像 |
| `description` | 推奨 | 抜き出したい音の説明（例: `man speaking`） |
| `anchors` | 任意 | JSON。例: `[["+", 6.3, 7.0]]` |
| `predict_spans` | 任意 | `true` でテキストから時間区間を自動予測 |
| `reranking_candidates` | 任意 | 候補数 |
| `output` | 任意 | `zip`（デフォルト）/ `target` / `residual` / `json` |

## ブラウザからテストする

1. 該当する `run_*.bat` でサーバーを起動する。
2. ブラウザで `http://127.0.0.1:8765/test/` を開く。
3. 「接続確認」を押し、`model_loaded` が `true` になるのを待つ。
4. 混ざった音声ファイルを選び、抜き出したい音を英語の短い語句で書く（例: `man speaking`）。
5. 「分離する」を押す。終わると **target**（抜き出した音）と **residual**（残り）が再生できる。

wav / flac / ogg はそのまま読めます。mp3 や映像から音を取る場合は、FFmpeg の実行ファイル（`ffmpeg.exe`）が PATH にあれば使います。torchcodec 用の FFmpeg DLL は不要です。

## サーバー起動（直接）

```text
venv\Scripts\python.exe -m modules.server --host 127.0.0.1 --port 8765 --device cuda --model base-lite
```

CUDA（RTX）のときは `venv-cuda\Scripts\python.exe` を使います。ROCm 環境でも GPU 利用の引数は `--device cuda` です。

## Credits

* [facebook/sam-audio](https://github.com/facebookresearch/sam-audio) - Segment Anything Model for Audio
* [aoiandroid/sam-audio-models](https://huggingface.co/aoiandroid/sam-audio-models) - BF16 safetensors 配布
