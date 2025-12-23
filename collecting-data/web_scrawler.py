from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def login(driver, username, password):
    # Note: If the 'state' token in this URL expires, switch to the main login URL.
    login_url = "https://pre-login-app-prod-ap-northeast-1.iterocloud.com/pre-login"
    
    print(f"Navigating to: {login_url}")
    driver.get(login_url)

    wait = WebDriverWait(driver, 15)

    try:
        # 1. Wait for Username field using XPath
        # Strategy: Look for an input tag where the 'name' attribute is 'username' or similar
        print("Waiting for username field...")
        # UPDATE THIS STRING: Check if the site uses @name='userName', @name='email', or something else.
        user_field = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@name='userName']"))) 
        
        user_field.clear()
        user_field.send_keys(username)
        print("Username entered.")

        # 2. Find Password field using XPath
        # Strategy: Look for an input tag where the type is specifically 'password'
        # This is usually very reliable as there is often only one password field.
        pass_field = driver.find_element(By.XPATH, "//input[@type='password']")
        
        pass_field.clear()
        pass_field.send_keys(password)
        print("Password entered.")

        # 3. Find Login Button using XPath
        # Strategy: Look for a button (or div/span) that *contains* specific text
        # This is great because you don't need to know the button's cryptic ID.
        login_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Log In')]")
        
        login_btn.click()
        print("Login button clicked.")
        
    except Exception as e:
        print(f"Login failed: {e}")