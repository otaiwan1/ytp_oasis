from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import time
import os

load_dotenv()

def login(driver, username, password):
    # Note: If the 'state' token in this URL expires, switch to the main login URL.
    login_url = "https://www.myitero.com/"
    
    print(f"[Navigating to: {login_url}]")
    driver.get(login_url)

    wait = WebDriverWait(driver, 15)

    try:
        # 1. Wait for Username field using XPath
        # Strategy: Look for an input tag where the 'name' attribute is 'username' or similar
        print("[Waiting for username field...]")
        # UPDATE THIS STRING: Check if the site uses @name='userName', @name='email', or something else.
        user_xpath = "/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/fieldset/input"
        user_field = wait.until(EC.presence_of_element_located((By.XPATH, user_xpath))) 
        
        user_field.clear()
        user_field.send_keys(username)
        print("[Username entered.]")

        next_btn_xpath = "/html/body/app-root/div/div[2]/div[1]/div/app-pre-login-form/div/form/button"
        next_btn = driver.find_element(By.XPATH, next_btn_xpath)

        next_btn.click()
        print("[Next button clicked.]")

        time.sleep(5)

        # 2. Find Password field using XPath
        # Strategy: Look for an input tag where the type is specifically 'password'
        # This is usually very reliable as there is often only one password field.
        pass_xpath = "/html/body/div[1]/div[2]/main/section/div/div/div/form/div[2]/div/div[2]/div[1]/input"
        pass_field = driver.find_element(By.XPATH, pass_xpath)
        
        pass_field.clear()
        pass_field.send_keys(password)
        print("[Password entered.]")

        # 3. Find Login Button using XPath
        # Strategy: Look for a button (or div/span) that *contains* specific text
        # This is great because you don't need to know the button's cryptic ID.
        login_btn_xpath = "/html/body/div/div[2]/main/section/div/div/div/form/div[4]/button"
        login_btn = driver.find_element(By.XPATH, login_btn_xpath)
        
        login_btn.click()
        print("[Login button clicked.]")

        time.sleep(10)
        
    except Exception as e:
        print(f"[Login failed: {e}]")

# Configure Chrome options (essential for automated downloads)
options = webdriver.ChromeOptions()

# Set download directory and disable "Save As" prompt
prefs = {
    "download.default_directory": "/collecting-data/patients-images/",
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
}
options.add_experimental_option("prefs", prefs)

# Initialize the driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

login(driver, os.getenv("iTero_email_address"), os.getenv("iTero_password"))