# EC LLMO Generator

商品ページ改善の9工程を自動生成するStreamlitアプリです。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windowsの場合:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## secrets設定

`.streamlit/secrets.toml.example` をコピーして `.streamlit/secrets.toml` を作成します。

```toml
app_password = "任意のパスワード"
openai_api_key = "OpenAI APIキー"
openai_model = "gpt-5.5"
```

`.streamlit/secrets.toml` はGitHubにコミットしないでください。

## 起動

```bash
streamlit run app.py
```

## できること

- 既存商品ページURLからタイトル、本文、価格らしき情報、画像URLらしき情報を取得
- 取得できない場合も手入力やCSVアップロードで続行
- 新規商品情報から9工程の生成
- 各工程ごとのタブ表示
- 選択工程だけ生成、または全工程まとめて生成
- 生成結果をMarkdownでダウンロード

## GitHub登録の流れ

```bash
git init
git add app.py requirements.txt README.md .gitignore .streamlit/secrets.toml.example
git commit -m "Initial EC LLMO generator"
git branch -M main
git remote add origin <GitHubのリポジトリURL>
git push -u origin main
```
