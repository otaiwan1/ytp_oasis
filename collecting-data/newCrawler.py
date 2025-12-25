import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service 

# Init driver (Global)
# 1. 使用 FirefoxOptions
options = webdriver.FirefoxOptions()

# 2. 設定 geckodriver 路徑 (假設檔名為 geckodriver.exe)
service = Service(executable_path="geckodriver.exe")

# 3. 初始化 Firefox Driver
driver = webdriver.Firefox(options=options, service=service)
print("Firefox Driver initialized successfully.")

def getCredentials(filePath="secret.txt"):
    try:
        with open(filePath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            if len(lines) >= 2:
                username = lines[0].strip()
                if '=' in username:
                    username = username[username.find('=') + 1:]
                
                password = lines[1].strip()
                if '=' in password:
                    password = password[password.find('=') + 1:]
                    
                return username, password
            else:
                print("Error: secret.txt content is insufficient.")
                return None, None
    except FileNotFoundError:
        print(f"Error: File {filePath} not found.")
        return None, None

def getElement(xpath):
    wait = WebDriverWait(driver, 10)
    return wait.until(EC.presence_of_element_located((By.XPATH, xpath)))

def getElementsLen(xpath):
    wait = WebDriverWait(driver, 10)
    return len(wait.until(EC.presence_of_all_elements_located((By.XPATH, xpath))))

def login(username, password):
    loginUrl = "https://www.myitero.com/"
    print(f"Navigating to: {loginUrl}")
    driver.get(loginUrl)

    try:
        print("Waiting for username field...")
        
        # 1. 輸入帳號
        userField = getElement("/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/fieldset/input")
        userField.clear()
        userField.send_keys(username)
        print(f"Username entered: {username}")

        # 2. 點擊繼續
        continueBtn = getElement("/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/button")
        continueBtn.click()

        # 等待頁面切換動畫
        time.sleep(3) 
        
        # 3. 輸入密碼
        passField = getElement("/html/body/div/div[2]/main/section/div/div/div/form/div[2]/div/div[2]/div[1]/input")
        passField.clear()
        passField.send_keys(password)
        print("Password entered.")

        # 4. 點擊登入
        loginBtn = getElement("/html/body/div/div[2]/main/section/div/div/div/form/div[4]/button")
        loginBtn.click()
        print("Login button clicked.")
        
    except Exception as e:
        print(f"Login failed: {e}")

    

import json

def getPatientIds():
    """
    透過攔截 Network Request (Fetch/XHR) 來取得資料
    """
    
    # 1. Inject Interceptor Script
    # 這段 JS 會攔截所有的 fetch 和 XHR 請求，並將回應存入 window.capturedData
    interceptor_script = """
    window.capturedData = [];
    
    // Intercept Fetch
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
        const response = await originalFetch(...args);
        const clone = response.clone();
        try {
            const data = await clone.json();
            window.capturedData.push({url: args[0], data: data});
        } catch(e) {}
        return response;
    };

    // Intercept XHR
    const originalXhrOpen = XMLHttpRequest.prototype.open;
    XMLHttpRequest.prototype.open = function(method, url) {
        this.addEventListener('load', function() {
            try {
                const data = JSON.parse(this.responseText);
                window.capturedData.push({url: url, data: data});
            } catch(e) {}
        });
        originalXhrOpen.apply(this, arguments);
    };
    """
    driver.execute_script(interceptor_script)
    print("Network interceptor injected.")

    # 2. 導航到 Patients 頁面 (觸發 API 請求)
    patients = getElement("/html/body/main/eup-home/div/div/div[2]/a[2]")
    patients.click()

    print("Starting data extraction with auto-scroll...")
    results = []
    
    try:
        # 定義容器 XPath
        containerXpath = "//eup-tbl/div" 
        tbodyXpath = f"{containerXpath}/table/tbody"
        
        # 等待表格出現
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.XPATH, f"{tbodyXpath}/tr")))

        # 取得捲動容器元素
        scrollableDiv = driver.find_element(By.XPATH, containerXpath)
        
        lastRowCount = 0
        retryCount = 0
        maxRetries = 3 

        while True:
            # A. 抓取目前所有的 tr (用來判斷是否到底)
            currentRowCount = getElementsLen(f"{tbodyXpath}/tr")
            
            print(f"Current rows loaded: {currentRowCount}")

            # B. 判斷是否已經到底
            if currentRowCount == lastRowCount:
                retryCount += 1
                print(f"Data count didn't increase. Retry {retryCount}/{maxRetries}...")
                
                if retryCount >= maxRetries:
                    print("Reached end of list (or data stopped loading).")
                    break
            else:
                retryCount = 0
                lastRowCount = currentRowCount
            
            # C. 執行 JavaScript 捲動
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollableDiv)
            
            # D. 等待新資料載入
            time.sleep(2.5)

        # --- 迴圈結束後，檢查攔截到的資料 ---
        print("Analyzing captured network data...")
        
        # 從瀏覽器取回 capturedData
        captured_data = driver.execute_script("return window.capturedData;")
        
        # 尋找包含 ID 的資料
        # 我們不知道 API 的確切格式，所以搜尋所有回應中的 UUID
        uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)
        
        found_ids = set()
        
        for item in captured_data:
            # 將資料轉為字串以便搜尋
            json_str = json.dumps(item['data'])
            
            # 搜尋 UUID
            matches = uuid_pattern.findall(json_str)
            for match in matches:
                # 過濾掉一些非 Patient ID 的 UUID (如果有已知的不相關 UUID 可以加在這裡)
                found_ids.add(match)
                
        results = list(found_ids)
        print(f"Total unique IDs extracted from network logs: {len(results)}")
            
        return results

    except Exception as e:
        print(f"Error getting data: {e}")
        return results

def downloadAllPatients(Ids):
    # Ids = [Ids[0], Ids[1], Ids[2]]
    ans = 0
    totalIds = len(Ids)
    for idcnt, id in enumerate(Ids, 1):
        try:
            driver.get(f"https://bff.cloud.myitero.com/doctors/patients/{id}/?isEvxEnabled=false")

            # wait until the first element appears
            getElement("/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[1]")
            rows = getElementsLen("/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr")
            print(rows)
            cnt = 0
            for i in range(1, rows + 1):
                txt = getElement(f"/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[{i}]/td[3]/span").text
                if txt == "Completed": cnt += 1
                print(f"no {idcnt}/{totalIds}: row {i}, {txt}")
            if cnt >= 2: ans += 1
        except Exception:
            print(Exception)
    
    print("Good", ans)
    

"""
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[1]
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr/td[3]/span
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[1]/td[3]/span
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[2]/td[3]/span
"""
if __name__ == "__main__":
    username, password = getCredentials()
    login(username, password)
    
    patientsIds = getPatientIds()
    downloadAllPatients(patientsIds)
    
    input("Press Enter to close browser...")
            
    driver.quit()