import time
import re
import json
import requests
from urllib.parse import urlparse, parse_qs
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

def get_ids_via_api():
    """
    Hybrid approach:
    1. Use Selenium to login and capture the API parameters AND HEADERS.
    2. Use Python requests with captured headers and cookies to fetch all data.
    """
    
    # 1. Inject Interceptor to catch the search URL and Headers
    interceptor_script = """
    window.capturedRequest = null;
    
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
        const url = args[0];
        const options = args[1] || {};
        
        if (url && url.includes("patients/search")) {
            // Capture URL and Headers
            window.capturedRequest = {
                url: url,
                headers: options.headers || {}
            };
            
            // If headers is a Headers object, convert to plain object
            if (window.capturedRequest.headers instanceof Headers) {
                const headerObj = {};
                window.capturedRequest.headers.forEach((value, key) => {
                    headerObj[key] = value;
                });
                window.capturedRequest.headers = headerObj;
            }
        }
        return originalFetch(...args);
    };
    
    // Note: XHR interception for headers is harder because headers are set via setRequestHeader
    // and not easily accessible on the XHR object itself later. 
    // Since modern Angular/React apps mostly use fetch, we rely on fetch interception.
    """
    driver.execute_script(interceptor_script)
    print("API Interceptor (URL + Headers) injected.")

    # 2. Navigate to Patients page to trigger the request
    try:
        patients = getElement("/html/body/main/eup-home/div/div/div[2]/a[2]")
        patients.click()
    except:
        print("Could not click Patients button, maybe already on page or different layout.")

    print("Waiting for API request to be captured...")
    
    captured_request = None
    for _ in range(20): # Wait up to 10 seconds
        captured_request = driver.execute_script("return window.capturedRequest;")
        if captured_request:
            break
        time.sleep(0.5)
    
    if not captured_request:
        print("Failed to capture API request. Aborting fast crawl.")
        return []

    target_url = captured_request['url']
    captured_headers = captured_request['headers']
    
    print(f"Captured API URL: {target_url}")
    print(f"Captured Headers keys: {list(captured_headers.keys())}")

    # 3. Extract Parameters
    parsed_url = urlparse(target_url)
    qs = parse_qs(parsed_url.query)
    
    company_id = qs.get('CompanyId', [None])[0]
    doctor_id = qs.get('DoctorId', [None])[0]
    
    if not company_id or not doctor_id:
        print("Could not extract CompanyId or DoctorId from URL.")
        return []
        
    print(f"Extracted - CompanyId: {company_id}, DoctorId: {doctor_id}")

    # 4. Prepare Session with Cookies AND Headers
    session = requests.Session()
    
    # A. Cookies
    selenium_cookies = driver.get_cookies()
    for cookie in selenium_cookies:
        session.cookies.set(cookie['name'], cookie['value'])
    
    # B. Headers
    # Add User-Agent
    session.headers.update({
        "User-Agent": driver.execute_script("return navigator.userAgent;")
    })
    
    # Add captured headers (Authorization, etc.)
    # Filter out some headers that requests/browser handles automatically to avoid conflicts
    # e.g., Content-Length, Host
    for key, value in captured_headers.items():
        if key.lower() not in ['content-length', 'host', 'connection', 'accept-encoding']:
            session.headers.update({key: value})

    # 5. Make the Direct API Call
    base_api_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    
    print("Attempting to fetch ALL data with PageSize=2000...")
    
    params = {
        "PageNumber": 1,
        "PageSize": 2000, 
        "SortField": "name",
        "CompanyId": company_id,
        "DoctorId": doctor_id,
        "IsShowAll": "false"
    }
    
    try:
        response = session.get(base_api_url, params=params)
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Regex extraction on the full response text
            text_data = json.dumps(data)
            uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)
            
            found_ids = set(uuid_pattern.findall(text_data))
            print(f"Total Unique IDs extracted via Fast API: {len(found_ids)}")
            
            return list(found_ids)
            
        else:
            print(f"API request failed: {response.text[:200]}")
            return []

    except Exception as e:
        print(f"Error during fast crawl: {e}")
        return []

if __name__ == "__main__":
    username, password = getCredentials()
    login(username, password)
    
    ids = get_ids_via_api()
    
    print(f"Final Result: {len(ids)} IDs collected.")
    
    # Optional: Save to file
    if ids:
        with open("patient_ids.txt", "w") as f:
            for pid in ids:
                f.write(f"{pid}\n")
        print("Saved to patient_ids.txt")

    input("Press Enter to close browser...")
    driver.quit()
