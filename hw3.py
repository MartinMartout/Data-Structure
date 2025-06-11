from playwright.sync_api import sync_playwright
import os
from dotenv import load_dotenv

# 讀取 .env 檔案
load_dotenv()
FB_EMAIL = os.getenv("FACEBOOK_EMAIL")
FB_PASSWORD = os.getenv("FACEBOOK_PASSWORD")

# 讀取 .env 檔案
load_dotenv()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # 顯示瀏覽器
    page = browser.new_page()

    print("啟動瀏覽器，進入 ChatGPT...")

    # 進入 ChatGPT
    page.goto("https://chatgpt.com/?model=gpt-4o")
    page.wait_for_timeout(5000)

    # 確認輸入框是否存在且可見
    try:
        # 使用更具體的 XPath 定位輸入框
        input_box = page.locator("//div[contains(@class, 'ProseMirror')]")
        input_box.wait_for(state="attached", timeout=10000)
        print("輸入框已載入")
    except Exception as e:
        print("無法找到輸入框或輸入框不可見：", e)
        browser.close()
        exit(1)

    # 模擬輸入問題
    input_box.fill("下一周天氣如何？")
    print("已輸入問題：下一周天氣如何？")

    # 正確觸發輸入事件
    page.evaluate("document.querySelector('div.ProseMirror').dispatchEvent(new Event('input', { bubbles: true }));")

    # 模擬按下 Enter 鍵
    page.keyboard.press("Enter")
    print("已提交查詢")

    # 等待回應
    page.wait_for_timeout(10000)
    print("已獲取回應")

    # 擷取回應內容
    try:
        response = page.locator("div[role='dialog']").text_content()
        print("ChatGPT 回應：", response)
    except Exception as e:
        print("無法取得回應：", e)

    # 保持瀏覽器開啟，方便查看
    input("瀏覽器保持開啟，按 Enter 關閉...")

    # 關閉瀏覽器
    browser.close()
    print("瀏覽器已關閉")