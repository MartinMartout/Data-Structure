import os
from datetime import datetime
import requests
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
from google import genai
import re

# 載入環境變數並設定 API 金鑰
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def get_chinese_font_file() -> str:
    """
    只檢查 Windows 系統字型資料夾中是否存在候選中文字型（TTF 格式）。
    若找到則回傳完整路徑；否則回傳 None。
    """
    fonts_path = r"C:\Windows\Fonts"
    candidates = ["kaiu.ttf"]  # 這裡以楷體為例，可依需要修改
    for font in candidates:
        font_path = os.path.join(fonts_path, font)
        if os.path.exists(font_path):
            print("找到系統中文字型：", font_path)
            return os.path.abspath(font_path)
    print("未在系統中找到候選中文字型檔案。")
    return None

# HW4


def create_table(pdf: FPDF, df: pd.DataFrame):
    col_width = (pdf.w - 2 * pdf.l_margin) / len(df.columns)
    row_height = 10
    font_size = 10

    pdf.set_font("ChineseFont", "B", font_size)
    pdf.set_fill_color(200, 200, 200)

    # 表頭
    for col in df.columns:
        pdf.cell(col_width, row_height, col, border=1, align='C', fill=True)
    pdf.ln(row_height)

    pdf.set_font("ChineseFont", "", font_size)

    # 內容
    for _, row in df.iterrows():
        for item in row:
            text = str(item)
            # 裁切長文字加省略號
            while pdf.get_string_width(text) > col_width - 2 and len(text) > 0:
                text = text[:-1]
            if pdf.get_string_width(text + "...") > col_width:
                text = text[:-3] + "..."
            pdf.cell(col_width, row_height, text, border=1, align='L')
        pdf.ln(row_height)


def parse_markdown_table(markdown_text: str) -> pd.DataFrame:
    """
    從 Markdown 格式的表格文字提取資料，返回一個 pandas DataFrame。
    例如，輸入：
      | start | end | text | 分類 |
      |-------|-----|------|------|
      | 00:00 | 00:01 | 開始拍攝喔 | 備註 |
    會返回包含該資料的 DataFrame。
    """
    lines = markdown_text.strip().splitlines()
    # 過濾掉空行
    lines = [line.strip() for line in lines if line.strip()]
    # 找到包含 '|' 的行，假設這就是表格
    table_lines = [line for line in lines if line.startswith("|")]
    if not table_lines:
        return None
    # 忽略第二行（分隔線）
    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    data = []
    for line in table_lines[2:]:
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if len(row) == len(headers):
            data.append(row)
    df = pd.DataFrame(data, columns=headers)
    return df
# hw4


def render_line_with_bold(pdf: FPDF, line: str):
    """處理內含 **粗體** 的句子轉換"""
    parts = re.split(r'(\*\*.*?\*\*)', line)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            text = part[2:-2]
            pdf.set_font("ChineseFont", "B", 12)
            pdf.write(8, text)
            pdf.set_font("ChineseFont", "", 12)
        else:
            pdf.write(8, part)
    pdf.ln(8)


def generate_pdf(text: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    font_path = get_chinese_font_file()
    pdf.add_font("ChineseFont", "", font_path, uni=True)
    pdf.add_font("ChineseFont", "B", font_path, uni=True)
    pdf.set_font("ChineseFont", "", 12)

    # 標題
    pdf.set_font("ChineseFont", "B", 16)
    pdf.cell(0, 12, "寶可夢對戰分析報告", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("ChineseFont", "", 12)

    lines = text.strip().splitlines()
    buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("**") and line.endswith("**"):
            if buffer:
                for buf in buffer:
                    render_line_with_bold(pdf, buf)
                pdf.ln(2)
                buffer = []
            title = line.strip("*").strip()
            pdf.set_font("ChineseFont", "B", 14)
            pdf.multi_cell(0, 10, title, align="L")
            pdf.set_font("ChineseFont", "", 12)
            pdf.ln(2)

        elif line.startswith("|") and "|" in line:
            buffer.append(line)
        else:
            if buffer and buffer[0].startswith("|"):
                df = parse_markdown_table("\n".join(buffer))
                if df is not None:
                    create_table(pdf, df)
                    pdf.ln(4)
                buffer = []
            buffer.append(line)

    # 最後緩衝段落處理
    if buffer:
        if buffer[0].startswith("|"):
            df = parse_markdown_table("\n".join(buffer))
            if df is not None:
                create_table(pdf, df)
        else:
            for buf in buffer:
                render_line_with_bold(pdf, buf)

    pdf_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_file)
    return pdf_file

# HW4


def gradio_handler(csv_file, user_prompt):
    print("進入 gradio_handler")
    if csv_file is not None:
        print("讀取 CSV 檔案")
        df = pd.read_csv(csv_file.name)
        total_rows = df.shape[0]
        block_size = 30
        block_responses = []

        # 分段送進 LLM 分析（避免 token 過長）
        for i in range(0, total_rows, block_size):
            block = df.iloc[i:i+block_size]
            block_csv = block.to_csv(index=False)
            prompt = (
                f"以下是CSV資料第 {i+1} 到 {min(i+block_size, total_rows)} 筆：\n"
                f"{block_csv}\n\n請根據以下規則進行分析並產出報表：\n{user_prompt}"
            )
            print("送出 prompt：")
            print(prompt)

            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[prompt]
            )
            block_response = response.text.strip()
            block_responses.append(block_response)

        # 合併所有分析結果為一份文字報告
        cumulative_response = "\n\n".join(block_responses)

        # 直接根據 AI 分析結果產出 PDF
        pdf_path = generate_pdf(text=cumulative_response)

        return cumulative_response, pdf_path


# HW4
default_prompt = """以下是寶可夢對戰的戰鬥過程紀錄，請針對每一列戰鬥紀錄進行以下分析：

1. 根據 `text` 欄位的內容，判斷該句話屬於以下哪一種類型（擇一）：
   - 寶可夢名稱
   - 寶可夢招式
   - 招式效果說明
   - 寶可夢狀態變更
   - 更換寶可夢
   - 其他

2. 根據 `回答完整性` 和 `回答內容評分` 這兩個欄位，計算整體平均分數，並依照分數區間（如 1–2、3、4–5）進行簡單分布統計。


最後，請以表格與條列方式整理報告重點，並加入簡要結論。
"""

with gr.Blocks() as demo:
    gr.Markdown("# CSV 報表生成器")
    with gr.Row():
        csv_input = gr.File(label="上傳 CSV 檔案")
        user_input = gr.Textbox(
            label="請輸入分析指令", lines=10, value=default_prompt)
    output_text = gr.Textbox(label="回應內容", interactive=False)
    output_pdf = gr.File(label="下載 PDF 報表")
    submit_button = gr.Button("生成報表")
    submit_button.click(fn=gradio_handler, inputs=[csv_input, user_input],
                        outputs=[output_text, output_pdf])

demo.launch()
