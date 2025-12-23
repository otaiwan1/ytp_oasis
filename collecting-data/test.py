import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service # 記得匯入 Service

def login(driver, username, password):
    # Note: If the 'state' token in this URL expires, switch to the main login URL.
    login_url = "https://www.myitero.com/"
    
    print(f"Navigating to: {login_url}")
    driver.get(login_url)

    wait = WebDriverWait(driver, 10) # 建議延長一點等待時間到 10 秒

    try:
        # 1. Wait for Username field using XPath
        print("Waiting for username field...")
        
        # [修正建議]: 原本的 @name='sabrinachou520...' 應該是誤植。
        # 通常登入框的 name 屬性會是 'username', 'email', 'user' 等。
        # 這裡嘗試使用通用的 'input' 標籤且 type='text' 或 'email'，或直接找 name='username'
        # 你可以根據實際網頁按 F12 檢查正確的 name
        user_field = wait.until(EC.presence_of_element_located(
            (By.XPATH, "/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/fieldset/input")
        ))
        
        user_field.clear()
        user_field.send_keys(username)
        print(f"Username entered: {username}")

        conti = wait.until(EC.presence_of_element_located(
            (By.XPATH, "/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/button")
        ))
        conti.click()

        time.sleep(3)
        # 2. Find Password field using XPath
        pass_field = driver.find_element(By.XPATH, "/html/body/div/div[2]/main/section/div/div/div/form/div[2]/div/div[2]/div[1]/input")
        
        pass_field.clear()
        pass_field.send_keys(password)
        print("Password entered.")

        # 3. Find Login Button using XPath
        # 使用 WebDriverWait 確保按鈕可點擊 (element_to_be_clickable) 會比單純 find_element 更穩定
        login_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "/html/body/div/div[2]/main/section/div/div/div/form/div[4]/button")
        ))
        
        login_btn.click()
        print("Login button clicked.")
        
        # 暫停一下確認是否登入成功 (實際專案建議改用 Wait 判斷登入後的元素)
        time.sleep(5)
        
    except Exception as e:
        print(f"Login failed: {e}")

def get_credentials(filepath="secret.txt"):
    """
    從檔案讀取帳號密碼
    格式假設：
    第一行：帳號
    第二行：密碼
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            if len(lines) >= 2:
                username = lines[0].strip()
                username = username[username.find('=') + 1:]
                password = lines[1].strip()
                password = password[password.find('=') + 1:]
                return username, password
            else:
                print("錯誤: credentials.txt 內容不足兩行 (帳號與密碼)")
                return None, None
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {filepath}，請確認檔案是否存在。")
        return None, None

if __name__ == "__main__":
    # 1. 讀取帳號密碼
    user, pwd = get_credentials()

    if user and pwd:
        # 2. 設定 Edge Driver
        # Selenium 4 以上版本通常不需要手動指定路徑，只要電腦有安裝 Edge 即可
        options = webdriver.EdgeOptions()
        service = Service(executable_path="msedgedriver.exe")
        # options.add_argument("--headless") # 如果不想看到視窗彈出，可取消此註解
        
        driver = webdriver.Edge(options=options, service=service)

        try:
            # 3. 呼叫登入函式
            login(driver, user, pwd)
        finally:
            # 結束後關閉視窗 (如果想要視窗留著，可以把下面這行註解掉)
            # driver.quit()
            pass
            
            # 若要讓視窗停留，改用 input 等待
            input("按 Enter 鍵結束程式並關閉瀏覽器...")
            driver.quit()