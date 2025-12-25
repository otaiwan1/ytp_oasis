import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service

# init driver
options = webdriver.EdgeOptions()
service = Service(executable_path="msedgedriver.exe")
driver = webdriver.Edge(options=options, service=service)
print("Driver initialized successfully.")

def get_credentials(filepath="secret.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            if len(lines) >= 2:
                username = lines[0].strip()
                # 保留你原本的 parsing 邏輯
                if '=' in username:
                    username = username[username.find('=') + 1:]
                
                password = lines[1].strip()
                if '=' in password:
                    password = password[password.find('=') + 1:]
                    
                return username, password
            else:
                print("錯誤: secret.txt 內容不足兩行")
                return None, None
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {filepath}")
        return None, None

def get_element(xpath: str):
    wait = WebDriverWait(driver, 10)
    return wait.until(EC.presence_of_element_located(
            (By.XPATH, "/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/fieldset/input")
        ))

def login(username, password):
    # 這裡直接使用全域變數 driver，不用再透過參數傳入
    
    login_url = "https://www.myitero.com/"
    print(f"Navigating to: {login_url}")
    driver.get(login_url)

    try:
        print("Waiting for username field...")
        
        # 建議：絕對路徑 XPath (/html/body...) 很脆弱，網頁改版容易壞
        # 如果未來失敗，請改用相對路徑 (例如 //input[@type='email'])
        user_field = get_element("/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/fieldset/input")
        
        user_field.clear()
        user_field.send_keys(username)
        print(f"Username entered: {username}")

        conti = get_element("/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/button")
        conti.click()

        # 等待密碼欄位出現
        time.sleep(3) 
        pass_field = get_element("/html/body/div/div[2]/main/section/div/div/div/form/div[2]/div/div[2]/div[1]/input")
        
        pass_field.clear()
        pass_field.send_keys(password)
        print("Password entered.")

        login_btn = get_element("/html/body/div/div[2]/main/section/div/div/div/form/div[4]/button")
        login_btn.click()
        print("Login button clicked.")
        
    except Exception as e:
        print(f"Login failed: {e}")

def get_data():
    pass

if __name__ == "__main__":
    user, pwd = get_credentials()
    login(user, pwd)
            
    input("按 Enter 鍵結束程式並關閉瀏覽器...")
            
    driver.quit()