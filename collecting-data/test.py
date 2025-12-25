import time
import re
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service 

# Init driver (Global)
options = webdriver.FirefoxOptions()
service = Service(executable_path="geckodriver.exe")
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

def login(username, password):
    loginUrl = "https://www.myitero.com/"
    print(f"Navigating to: {loginUrl}")
    driver.get(loginUrl)

    try:
        print("Waiting for username field...")
        
        userField = getElement("/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/fieldset/input")
        userField.clear()
        userField.send_keys(username)
        print(f"Username entered: {username}")

        continueBtn = getElement("/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/button")
        continueBtn.click()

        time.sleep(3) 
        
        passField = getElement("/html/body/div/div[2]/main/section/div/div/div/form/div[2]/div/div[2]/div[1]/input")
        passField.clear()
        passField.send_keys(password)
        print("Password entered.")

        loginBtn = getElement("/html/body/div/div[2]/main/section/div/div/div/form/div[4]/button")
        loginBtn.click()
        print("Login button clicked.")
        
    except Exception as e:
        print(f"Login failed: {e}")

def debugPatientIds():
    """
    Debug version of getPatientIds
    """
    
    # 1. Inject Interceptor Script
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

    # 2. Navigate to Patients page
    patients = getElement("/html/body/main/eup-home/div/div/div[2]/a[2]")
    patients.click()

    print("Starting data extraction with auto-scroll...")
    results = []
    
    try:
        containerXpath = "//eup-tbl/div" 
        tbodyXpath = f"{containerXpath}/table/tbody"
        
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.XPATH, f"{tbodyXpath}/tr")))

        scrollableDiv = driver.find_element(By.XPATH, containerXpath)
        
        lastRowCount = 0
        retryCount = 0
        maxRetries = 3 

        while True:
            rows = driver.find_elements(By.XPATH, f"{tbodyXpath}/tr")
            currentRowCount = len(rows)
            
            print(f"Current rows loaded: {currentRowCount}")

            if currentRowCount == lastRowCount:
                retryCount += 1
                print(f"Data count didn't increase. Retry {retryCount}/{maxRetries}...")
                
                if retryCount >= maxRetries:
                    print("Reached end of list (or data stopped loading).")
                    break
            else:
                retryCount = 0
                lastRowCount = currentRowCount
            
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollableDiv)
            time.sleep(2.5)

        # --- Analysis ---
        print("\n--- DEBUG ANALYSIS ---")
        
        captured_data = driver.execute_script("return window.capturedData;")
        print(f"Total intercepted requests: {len(captured_data)}")
        
        uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)
        
        all_found_ids = []
        unique_ids = set()
        
        for i, item in enumerate(captured_data):
            url = item.get('url', 'unknown')
            json_str = json.dumps(item['data'])
            matches = uuid_pattern.findall(json_str)
            
            if matches:
                print(f"Request {i+1} ({url}): Found {len(matches)} UUIDs")
                all_found_ids.extend(matches)
                for m in matches:
                    unique_ids.add(m)
            else:
                print(f"Request {i+1} ({url}): No UUIDs found")

        print(f"\nTotal UUIDs found (including duplicates): {len(all_found_ids)}")
        print(f"Total Unique UUIDs: {len(unique_ids)}")
        print(f"Duplicates count: {len(all_found_ids) - len(unique_ids)}")
        
        return list(unique_ids)

    except Exception as e:
        print(f"Error getting data: {e}")
        return []

if __name__ == "__main__":
    username, password = getCredentials()
    login(username, password)
    
    debugPatientIds()
    
    input("Press Enter to close browser...")
            
    driver.quit()
