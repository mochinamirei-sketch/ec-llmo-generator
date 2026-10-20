import io
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
import anthropic


APP_TITLE = "EC LLMO Generator"

TASKS = {
    "01_gap_check": "1. ページ内の情報の齟齬を修正",
    "02_llmo_page": "2. LLMO向けにページ内容・構成・FAQなどを整備",
    "03_json_ld": "3. 構造化データを設定",
    "04_rich_results": "4. Googleリッチリザルトテスト用チェック項目",
    "05_search_console": "5. Google Search Console URL検査・登録チェック項目",
    "06_rakuten_pc": "6. 楽天PC用商品説明文HTML",
    "07_rakuten_sp": "7. 楽天スマホ用商品説明文HTML",
    "08_image_prompt": "8. 正方形JPEG商品説明画像プロンプト",
    "09_newsletter": "9. 本店メルマガ文章",
}

FORM_FIELDS = [
    ("product_name", "商品名"),
    ("sku", "SKU"),
    ("product_url", "商品URL"),
    ("image_urls", "商品画像URL"),
    ("price", "販売価格"),
    ("price_variants", "価格バリエーション"),
    ("sale_start", "販売開始日"),
    ("sale_end", "販売終了日"),
    ("availability", "在庫状態"),
    ("origin", "産地"),
    ("producer", "生産者"),
    ("brand", "ブランド"),
    ("seller", "販売者"),
    ("volume", "内容量"),
    ("variety", "品種・種類"),
    ("delivery_method", "配送方法"),
    ("shipping_terms", "送料条件"),
    ("bundle", "同梱可否"),
    ("date_request", "日付指定可否"),
    ("gift", "ギフト対応"),
    ("imperfect_reason", "訳あり理由"),
    ("storage", "保存方法"),
    ("return_deadline", "返品・破損対応期限"),
    ("strengths", "商品の強み"),
    ("cautions", "注意事項"),
    ("target", "ターゲット"),
    ("newsletter_terms", "本店メルマガ向け条件"),
]


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read Streamlit secrets first, then environment variables for local dev."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key.upper(), default)


def check_password() -> bool:
    app_password = get_secret("app_password")

    if not app_password:
        st.warning("st.secrets['app_password'] が未設定です。ローカル確認用として認証なしで続行します。")
        return True

    if st.session_state.get("authenticated"):
        return True

    st.title(APP_TITLE)
    st.caption("社内利用向けの簡易パスワード認証です。")
    password = st.text_input("パスワード", type="password")

    if st.button("ログイン", type="primary"):
        if password == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("パスワードが違います。")

    return False


def get_anthropic_client() -> Optional[anthropic.Anthropic]:
    # mailmagwrite と同じ大文字キー(ANTHROPIC_API_KEY)でも読めるようにしている
    api_key = get_secret("anthropic_api_key") or get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def normalize_text(text: str, limit: int = 12000) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)[:limit]


def extract_price_like_text(text: str, max_items: int = 20) -> List[str]:
    patterns = [
        r"(?:税込|税抜|送料込|送料無料|価格|販売価格|特別価格)?\s*[0-9０-９,，]+円(?:\s*\(税込\))?",
        r"[¥￥]\s*[0-9０-９,，]+",
        r"[0-9０-９,，]+\s*yen",
    ]
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    cleaned = [re.sub(r"\s+", " ", item).strip() for item in matches]
    return list(dict.fromkeys(item for item in cleaned if item))[:max_items]


def fetch_page(url: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "url": url,
        "title": "",
        "description": "",
        "text": "",
        "price_like": [],
        "image_urls": [],
        "og_image": "",
        "error": "",
    }

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; EC-LLMO-Generator/1.0)"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding

        soup = BeautifulSoup(response.text, "html.parser")
        result["title"] = soup.title.get_text(" ", strip=True) if soup.title else ""

        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description and meta_description.get("content"):
            result["description"] = meta_description["content"].strip()

        og_image = soup.find("meta", attrs={"property": "og:image"})
        if og_image and og_image.get("content"):
            result["og_image"] = urljoin(url, og_image["content"].strip())

        image_urls: List[str] = []
        for image in soup.find_all("img"):
            src = image.get("src") or image.get("data-src") or image.get("data-original")
            if src:
                image_urls.append(urljoin(url, src))

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        page_text = normalize_text(soup.get_text("\n", strip=True))

        result["text"] = page_text
        result["price_like"] = extract_price_like_text(page_text)
        result["image_urls"] = list(dict.fromkeys(image_urls))[:30]
    except Exception as exc:
        result["error"] = str(exc)

    return result


def read_uploaded_files(files: Iterable[Any]) -> str:
    chunks: List[str] = []
    for file in files or []:
        try:
            raw = file.getvalue()
            suffix = file.name.lower().rsplit(".", 1)[-1]

            if suffix == "csv":
                df = None
                for encoding in ("utf-8-sig", "cp932", "shift_jis"):
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                if df is None:
                    df = pd.read_csv(io.BytesIO(raw))
                chunks.append(f"--- uploaded csv: {file.name} ---\n{df.to_csv(index=False)}")
            else:
                text = raw.decode("utf-8-sig", errors="ignore")
                chunks.append(f"--- uploaded file: {file.name} ---\n{normalize_text(text, 16000)}")
        except Exception as exc:
            chunks.append(f"--- uploaded file: {file.name} ---\n読み取り失敗: {exc}")

    return "\n\n".join(chunks)[:24000]


def system_prompt() -> str:
    return """
あなたは日本向けEC商品ページ改善の専門家です。
ハチカッテの地方創生EC、食品EC、楽天、ShopServe、本店メルマガ、LLMO、構造化データに強い編集者として振る舞ってください。

文体と安全ルール:
- 日本語で書く
- 誇張しすぎないが、買いたくなる文章にする
- 女性30〜60代、富裕層、リピーター、良い商品なら価格を気にしない層を意識する
- 食品表示、効能表現、薬機法、景表法のリスクは安全側にする
- 根拠がない「無添加」「最高級」「日本一」「糖度保証」は断定しない
- 不明点は「要確認」と明記し、事実と推測を混ぜない
- 価格、在庫、販売期間、送料、返品条件は商品情報と矛盾させない
- HTMLはコードブロックで出す
- 構造化データは script type="application/ld+json" 形式で出す
- 楽天用HTMLはCSSやJavaScriptを使わず、center、table、font、br、hrなど基本タグ中心にする
- 本店HTMLはstyle属性を使って見栄えよくする
- 商品画像は画像生成そのものではなく、画像生成AIに渡せる完成プロンプトとして出す
""".strip()


def task_instruction(task_key: str) -> str:
    instructions = {
        "01_gap_check": """
ページ内の情報の齟齬を修正するための確認表を作ってください。
出力:
- 現状整理
- 齟齬・不足・要確認
- 表現リスク
- 購入前に迷いやすい点
- 修正後にページへ反映する文言案
""",
        "02_llmo_page": """
本店商品ページに使うLLMO向けのページ内容・構成・FAQを整備してください。
出力:
- 推奨ページ構成
- 本店用HTML(style属性あり)
- FAQ 5〜8件
- LLMが理解しやすい商品要約
- 内部リンク・アンカーテキスト案
""",
        "03_json_ld": """
Product構造化データを作成してください。
script type="application/ld+json" 付きで、コピペ可能なコードブロックにしてください。
価格バリエーションがある場合は offers 配列または AggregateOffer を適切に使い、要確認項目は無理に断定しないでください。
image は Product の必須項目です。必ず含めてください。
- 商品情報の「商品画像URL」(manual_product_info)に入力があれば、その中から商品が分かる画像を先頭からそのまま image に配列で入れてください。
- 入力がなければ scraped_page_info の og_image、image_urls から商品写真らしきものを選んでください。ロゴ・アイコン・バナーは使わないでください。
- 画像URLがどこにも見つからない場合も image を省略せず、"image": ["https://(要確認:商品画像URLを入れてください)"] のようなプレースホルダーを入れ、コードブロックの前後で「画像URLが未入力のため要確認」と明記してください。
- 画像URLは商品情報にあるものだけを使い、推測で作らないでください。
""",
        "04_rich_results": """
Googleリッチリザルトテストで検査するためのチェック項目を作成してください。
Product、Merchant listings、価格、在庫、送料、返品、画像、警告、エラーの観点を含めてください。
「image がありません」の警告・エラーが出た場合の直し方(画像URLの入れ方、複数画像の配列、絶対URLにすること)も含めてください。
""",
        "05_search_console": """
Google Search ConsoleでURL検査・インデックス登録をリクエストするためのチェック項目を作成してください。
公開URLテスト、クロール可能性、構造化データ、canonical、スマホ表示、インデックス登録リクエスト後の確認を含めてください。
""",
        "06_rakuten_pc": """
楽天PC用商品説明文を楽天で使えるHTMLで作成してください。
横幅は原則595px。CSSとJavaScriptは使わず、center、table、font、br、hrなど基本タグ中心にしてください。
写真を差し込む想定のコメントや見出しも入れてください。
""",
        "07_rakuten_sp": """
楽天スマホ用商品説明文を楽天で使えるHTMLで作成してください。
CSSとJavaScriptは使わず、スマホで縦読みしやすい短めの構成にしてください。
""",
        "08_image_prompt": """
文章を読まない購入者向けに、商品の魅力・特徴・内容量が一目で分かる約900×900pxの正方形JPEG商品説明画像プロンプトを作成してください。
画像生成AIにそのまま渡せる完成プロンプトにし、構図、文字量、色、写真表現、入れる文言、避ける表現も指定してください。
""",
        "09_newsletter": """
本店メルマガ文章を生成してください。
出力:
- 件名案 5本(15〜20字程度)
- プリヘッダー
- 本文(500字程度)
- おすすめ3ポイント
- CTA文
- 注意書き
""",
    }
    return instructions[task_key].strip()


def build_product_context(mode: str, form_data: Dict[str, Any], page_data: Optional[Dict[str, Any]], uploaded_text: str) -> str:
    context = {
        "mode": mode,
        "manual_product_info": form_data,
        "scraped_page_info": page_data or {},
        "uploaded_files": uploaded_text,
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def build_user_prompt(task_key: str, product_context: str) -> str:
    return f"""
以下の商品情報をもとに、指定工程の成果物を作成してください。

# 指定工程
{TASKS[task_key]}

# 工程別指示
{task_instruction(task_key)}

# 商品情報
{product_context}
""".strip()


def call_ai(task_key: str, product_context: str) -> str:
    model = get_secret("anthropic_model", "claude-sonnet-5-5")
    user_prompt = build_user_prompt(task_key, product_context)
    client = get_anthropic_client()

    if not client:
        return f"""Anthropic APIキーが未設定です。

`.streamlit/secrets.toml`(Streamlit CloudならSecrets)に `anthropic_api_key` を設定すると、この画面から直接生成できます。
当面は以下のプロンプトをClaude等に貼り付けて使えます。

## SYSTEM
{system_prompt()}

## USER
{user_prompt}
"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=system_prompt(),
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            types_seen = [getattr(b, "type", "?") for b in response.content]
            raise RuntimeError(
                "Claudeの応答にテキストが含まれていませんでした。"
                f" (model={model}, stop_reason={getattr(response, 'stop_reason', None)}, content={types_seen})"
            )
        return text
    except anthropic.AuthenticationError:
        return _error_text("Anthropic APIキーが正しくありません。Secretsの anthropic_api_key を確認してください。", user_prompt)
    except anthropic.RateLimitError as exc:
        return _error_text(f"リクエストが多すぎます。少し待ってからもう一度押してください。({exc})", user_prompt)
    except anthropic.BadRequestError as exc:
        if "credit balance" in str(exc).lower():
            return _error_text("Anthropic APIの残高が不足しています。console.anthropic.com の Billing でクレジットを追加してください。", user_prompt)
        return _error_text(str(exc), user_prompt)
    except Exception as exc:
        return _error_text(str(exc), user_prompt)


def _error_text(message: str, user_prompt: str) -> str:
    return f"""生成に失敗しました。

エラー:
`{message}`

## 再実行用プロンプト
{user_prompt}
"""


def text_input_with_default(label: str, key: str, default: str = "") -> str:
    return st.text_input(label, value=default, key=f"field_{key}")


def text_area_with_default(label: str, key: str, default: str = "", height: int = 90) -> str:
    return st.text_area(label, value=default, height=height, key=f"field_{key}")


def default_image_urls(page_data: Optional[Dict[str, Any]]) -> str:
    """og:image を先頭に、アイコンやロゴらしきものを除いた画像URLを最大5件返す。"""
    if not page_data:
        return ""
    skip_words = ("icon", "logo", "spacer", "blank", "banner", "btn", "button", "sprite", "arrow")
    urls: List[str] = []
    if page_data.get("og_image"):
        urls.append(page_data["og_image"])
    for image_url in page_data.get("image_urls", []):
        if any(word in image_url.lower() for word in skip_words):
            continue
        urls.append(image_url)
    return "\n".join(list(dict.fromkeys(urls))[:5])


def build_product_form(mode: str, page_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    st.subheader("商品情報")
    if mode == "既存商品ページから作る":
        st.caption("URL取得で足りない情報は、ここで手入力してください。CSVの内容も生成時に参照されます。")

    title_default = (page_data or {}).get("title", "")
    url_default = (page_data or {}).get("url", "")
    price_default = " / ".join((page_data or {}).get("price_like", [])[:5])
    image_default = default_image_urls(page_data)

    col1, col2 = st.columns(2)
    with col1:
        product_name = text_input_with_default("商品名", "product_name", title_default)
        sku = text_input_with_default("SKU", "sku")
        product_url = text_input_with_default("商品URL", "product_url", url_default)
        image_urls_text = text_area_with_default(
            "商品画像URL(1行に1件。メイン画像を先頭に)", "image_urls", image_default, height=110
        )
        price = text_input_with_default("販売価格", "price", price_default)
        price_variants = text_area_with_default("価格バリエーション", "price_variants")
        sale_start = text_input_with_default("販売開始日", "sale_start")
        sale_end = text_input_with_default("販売終了日", "sale_end")
        availability = st.selectbox(
            "在庫状態",
            ["InStock", "PreOrder", "OutOfStock", "SoldOut", "LimitedAvailability", "要確認"],
            key="field_availability",
        )
        origin = text_input_with_default("産地", "origin")
        producer = text_input_with_default("生産者", "producer")
        brand = text_input_with_default("ブランド", "brand")
        seller = text_input_with_default("販売者", "seller", "ハチカッテ")

    with col2:
        volume = text_input_with_default("内容量", "volume")
        variety = text_input_with_default("品種・種類", "variety")
        delivery_method = text_input_with_default("配送方法", "delivery_method")
        shipping_terms = text_area_with_default("送料条件", "shipping_terms")
        bundle = text_input_with_default("同梱可否", "bundle")
        date_request = text_input_with_default("日付指定可否", "date_request")
        gift = text_input_with_default("ギフト対応", "gift")
        imperfect_reason = text_area_with_default("訳あり理由", "imperfect_reason")
        storage = text_input_with_default("保存方法", "storage")
        return_deadline = text_input_with_default("返品・破損対応期限", "return_deadline", "到着後は早めに状態確認。破損時は到着後の期限内に連絡。")

    strengths = text_area_with_default("商品の強み", "strengths", height=130)
    cautions = text_area_with_default("注意事項", "cautions", height=110)
    target = text_area_with_default(
        "ターゲット",
        "target",
        "女性30〜60代。富裕層、リピーター、良い商品なら価格を気にしない層。大切な人への贈り物や季節のご褒美需要。",
        height=100,
    )
    newsletter_terms = text_area_with_default("本店メルマガ向け条件", "newsletter_terms", height=100)

    return {
        "商品名": product_name,
        "SKU": sku,
        "商品URL": product_url,
        "商品画像URL": image_urls_text,
        "販売価格": price,
        "価格バリエーション": price_variants,
        "販売開始日": sale_start,
        "販売終了日": sale_end,
        "在庫状態": availability,
        "産地": origin,
        "生産者": producer,
        "ブランド": brand,
        "販売者": seller,
        "内容量": volume,
        "品種・種類": variety,
        "配送方法": delivery_method,
        "送料条件": shipping_terms,
        "同梱可否": bundle,
        "日付指定可否": date_request,
        "ギフト対応": gift,
        "訳あり理由": imperfect_reason,
        "保存方法": storage,
        "返品・破損対応期限": return_deadline,
        "商品の強み": strengths,
        "注意事項": cautions,
        "ターゲット": target,
        "本店メルマガ向け条件": newsletter_terms,
    }


def render_page_fetcher() -> Optional[Dict[str, Any]]:
    st.subheader("既存商品ページ")
    url = st.text_input("商品URLを入力", key="page_url")

    if st.button("商品ページを取得", type="primary") and url:
        with st.spinner("ページを取得しています..."):
            st.session_state["page_data"] = fetch_page(url)

    page_data = st.session_state.get("page_data")
    if not page_data:
        return None

    if page_data.get("error"):
        st.error(f"取得できませんでした: {page_data['error']}")
        st.info("下の商品情報フォームに手入力するか、CSVをアップロードして続けられます。")
        return page_data

    st.success("ページ情報を取得しました。")
    col1, col2 = st.columns(2)
    with col1:
        st.write("ページタイトル")
        st.code(page_data.get("title") or "取得なし")
        st.write("価格らしき情報")
        st.write(page_data.get("price_like") or "取得なし")
    with col2:
        st.write("画像URLらしき情報")
        st.write(page_data.get("image_urls", [])[:8] or "取得なし")

    with st.expander("取得した本文テキスト"):
        st.text(page_data.get("text", "")[:8000])

    return page_data


def render_outputs() -> None:
    outputs: Dict[str, str] = st.session_state.get("outputs", {})
    st.subheader("生成結果")
    tabs = st.tabs(list(TASKS.values()))

    for tab, (task_key, task_label) in zip(tabs, TASKS.items()):
        with tab:
            content = outputs.get(task_key)
            if not content:
                st.info("まだ生成されていません。")
                continue

            st.markdown(content)
            st.download_button(
                "この工程をMarkdownでダウンロード",
                data=f"# {task_label}\n\n{content}",
                file_name=f"{task_key}.md",
                mime="text/markdown",
                key=f"download_{task_key}",
            )

    if outputs:
        all_markdown = "\n\n---\n\n".join(f"# {TASKS[key]}\n\n{value}" for key, value in outputs.items())
        st.download_button(
            "全工程をまとめてMarkdownでダウンロード",
            data=all_markdown,
            file_name="ec_llmo_generator_outputs.md",
            mime="text/markdown",
            key="download_all",
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not check_password():
        return

    st.title(APP_TITLE)
    st.caption("ハチカッテの商品ページ改善、LLMO、構造化データ、楽天HTML、商品画像プロンプト、メルマガを同じ流れで生成します。")

    mode = st.radio(
        "最初に選べるモード",
        ["既存商品ページから作る", "新規商品情報から作る"],
        horizontal=True,
    )

    page_data = render_page_fetcher() if mode == "既存商品ページから作る" else None

    uploaded_files = st.file_uploader(
        "CSVや商品情報ファイルをアップロード",
        type=["csv", "txt", "html"],
        accept_multiple_files=True,
        help="CSVはpandasで表として読み込みます。URL取得で不足した情報の補助に使えます。",
    )
    uploaded_text = read_uploaded_files(uploaded_files)
    if uploaded_text:
        with st.expander("アップロード内容の読み取り結果"):
            st.text(uploaded_text[:10000])

    form_data = build_product_form(mode, page_data)
    product_context = build_product_context(mode, form_data, page_data, uploaded_text)

    st.divider()
    st.subheader("生成")

    task_options = list(TASKS.keys())
    selected_tasks = st.multiselect(
        "生成する工程",
        options=task_options,
        default=[task_options[0]],
        format_func=lambda key: TASKS[key],
    )

    col1, col2 = st.columns(2)
    generate_selected = col1.button("選択した工程だけ生成", type="primary", disabled=not selected_tasks)
    generate_all = col2.button("全部まとめて生成")

    if "outputs" not in st.session_state:
        st.session_state["outputs"] = {}

    if generate_selected:
        for task_key in selected_tasks:
            with st.spinner(f"{TASKS[task_key]} を生成中..."):
                st.session_state["outputs"][task_key] = call_ai(task_key, product_context)

    if generate_all:
        for task_key in TASKS:
            with st.spinner(f"{TASKS[task_key]} を生成中..."):
                st.session_state["outputs"][task_key] = call_ai(task_key, product_context)

    st.divider()
    render_outputs()


if __name__ == "__main__":
    main()
