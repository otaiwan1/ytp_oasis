import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service 
from shutil import move
from os import makedirs

# Init driver (Global)
options = webdriver.FirefoxOptions()

service = Service(executable_path="geckodriver.exe")
downloadDir = r"C:\Users\11311\OneDrive\Desktop\downloaded"
options.set_preference("browser.download.folderList", 2) # 0:桌面, 1:預設, 2:自定義
options.set_preference("browser.download.dir", downloadDir)
options.set_preference("browser.download.useDownloadDir", True)
options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/zip")

driver = webdriver.Firefox(options=options, service=service)
print("Firefox Driver initialized successfully.")

def getCredentials(filePath="secret.txt"):
    with open(filePath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        username = lines[0].strip()
        if '=' in username:
            username = username[username.find('=') + 1:]
        
        password = lines[1].strip()
        if '=' in password:
            password = password[password.find('=') + 1:]
            
    return username, password

def getElement(xpath):
    wait = WebDriverWait(driver, 10)
    return wait.until(EC.presence_of_element_located((By.XPATH, xpath)))

def getElementsLen(xpath):
    wait = WebDriverWait(driver, 10)
    return len(wait.until(EC.presence_of_all_elements_located((By.XPATH, xpath))))

def tryClick(button, attempts = 5):
    for i in range(attempts):
        try:
            button.click()
            return
        except: time.sleep(3)
    assert 0


def login():
    username, password = getCredentials()
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
        time.sleep(2)
        
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

            # DEBUG
            # if currentRowCount >= 50: break

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

def ERR(idcnt, id, e):
    print(f"{e}\nERROR")
    with open("err.txt", "a", encoding = "utf-8") as file:
        file.write(f"{idcnt}:{id}\n{e}\n")
    driver.delete_all_cookies()
    login()
    time.sleep(3)

def downloadAllPatients(Ids):
    # Ids = [Ids[0], Ids[1], Ids[2]]
    totalIds = len(Ids)
    for idcnt, id in enumerate(Ids, 1):
        try:
            driver.get(f"https://bff.cloud.myitero.com/doctors/patients/{id}/?isEvxEnabled=false")

            # wait until the first element appears
            getElement("/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[1]")
            rows = getElementsLen("/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr")
            print(rows)
            
            try:
                nores = getElement(f"/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr/th").text
                if nores == "No matches were found": continue
            except: pass
            
            goodRows = []
            for i in range(1, rows + 1):
                txt = getElement(f"/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[{i}]/td[3]/span").text
                if txt == "Completed": 
                    goodRows.append(i)
                print(f"no {idcnt}/{totalIds}: row {i}, {txt}")

            def downloadRow(curRow):
                driver.get(f"https://bff.cloud.myitero.com/doctors/patients/{id}/?isEvxEnabled=false")
                row = getElement(f"/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[{curRow}]/th")
                expId = row.text

                tryClick(row)
                print(f"no {idcnt}/{totalIds}: trying to download row {curRow}")
                time.sleep(0.5)

                expButton = getElement(f"/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[{curRow + 2}]/th/button[3]")
                if expButton.text == "Viewer": expButton = getElement(f"/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[{curRow + 2}]/th/button[4]")
                if expButton.text != "Export": return
                tryClick(expButton)
                # print("pressed exp")
                time.sleep(0.5)

                dropButton = getElement(f"/html/body/main/eup-patientsorders/div/eup-orthocadexport/div/div/div/div/form/div[1]/div/div/a")
                tryClick(dropButton)
                # print("pressed drop")
                time.sleep(0.5)

                openShell = getElement(f"/html/body/main/eup-patientsorders/div/eup-orthocadexport/div/div/div/div/form/div[1]/div/div/ul/li[1]")
                openShell.click()
                # print("pressed openshell")
                time.sleep(0.5)

                showName = getElement(f"/html/body/main/eup-patientsorders/div/eup-orthocadexport/div/div/div/div/form/div[4]/tlk-checkbox/label/div[2]/div[1]")
                showName.click()
                # print("pressed show name")
                time.sleep(0.5)
            
                checkExpButton = getElement(f"/html/body/main/eup-patientsorders/div/eup-orthocadexport/div/div/div/div/form/div[5]/button[2]")
                checkExpButton.click()  
                # print("exp!!!!")          

                time.sleep(2)

                for _ in range(10):
                    try:
                        print(f"no {idcnt}/{totalIds}: Retrying")
                        preDown = getElement(f"/html/body/main/eup-patientsorders/eup-sticky-header/div/header/div[2]/div/div[3]/eup-download-notification/div/div")
                        tryClick(preDown, 3)
                        time.sleep(0.5)
                        cancelDown = getElement(f"/html/body/main/eup-patientsorders/eup-sticky-header/div/header/div[2]/div/div[3]/eup-download-notification/div/eup-export-downloads-progress-list/div/div/div/div[2]/div[3]")
                        tryClick(cancelDown, 3)
                        # print("Cancelled")
                        time.sleep(0.5)

                        driver.get(f"https://bff.cloud.myitero.com/doctors/patients/{id}/?selectedrow={curRow - 1}")
                        
                        time.sleep(3)
                        expButton = getElement(f"/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[{curRow + 2}]/th/button[3]")
                        if expButton.text == "Viewer": expButton = getElement(f"/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[{curRow + 2}]/th/button[4]")
                        tryClick(expButton)
                        # print("re-exp")
                        time.sleep(0.5)

                        dropButton = getElement(f"/html/body/main/eup-patientsorders/div/eup-orthocadexport/div/div/div/div/form/div[1]/div/div/a")
                        tryClick(dropButton)
                        # print("re-drop")
                        time.sleep(0.5)

                        openShell = getElement(f"/html/body/main/eup-patientsorders/div/eup-orthocadexport/div/div/div/div/form/div[1]/div/div/ul/li[1]")
                        openShell.click()
                        # print("re-openshell")
                        time.sleep(0.5)

                        showName = getElement(f"/html/body/main/eup-patientsorders/div/eup-orthocadexport/div/div/div/div/form/div[4]/tlk-checkbox/label/div[2]/div[1]")
                        showName.click()
                        # print("re-show name")
                        time.sleep(0.5)

                        checkExpButton = getElement(f"/html/body/main/eup-patientsorders/div/eup-orthocadexport/div/div/div/div/form/div[5]/button[2]")
                        checkExpButton.click()  
                        # print("re exppp")
                        time.sleep(5)
                    except Exception as e: 
                        # print(f"err {e}")
                        # time.sleep(3)
                        break


                try:
                    makedirs(f"{downloadDir}\\{id}", exist_ok = True)
                    move(f"{downloadDir}\\OrthoCAD_Export_{expId}.zip", f"{downloadDir}\\{id}")
                    print(f"no {idcnt}/{totalIds}: Download Completed at {downloadDir}\\{id}\\OrthoCAD_Export_{expId}.zip")
                except Exception as e:
                    ERR(idcnt, id, e)
            
            for goodRow in goodRows:
                downloadRow(goodRow)
            

        except Exception as e:
            ERR(idcnt, id, e)
    
    

"""
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[7]/th/button[3]
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[5]
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[7]/th/button[3]
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[4]
/html/body/main/eup-patientsorders/div/div/main/div/eup-tbl/div/table/tbody/tr[6]/th/button[3]
/html/body/main/eup-patientsorders/eup-sticky-header/div/header/div[2]/div/div[3]
/html/body/main/eup-patientsorders/eup-sticky-header/div/header/div[2]/div/div[3]/eup-download-notification
/html/body/main/eup-patientsorders/eup-sticky-header/div/header/div[2]/div/div[3]/eup-download-notification/div/div/div[2]
"""
if __name__ == "__main__":
    login()
    
    patientsIds = getPatientIds()
    downloadAllPatients(patientsIds)
    
    input("Press Enter to close browser...")
            
    driver.quit()