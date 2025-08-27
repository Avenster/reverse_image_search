import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import os

def init_driver():
    options = uc.ChromeOptions()
    # Use only one profile folder
    user_data_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", f"chrome_data_{os.getpid()}")
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--window-size=1920,1080")
    # Minimal flags for testing
    # options.add_argument("--headless=new")  # Optional
    driver = uc.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

driver = init_driver()
url = "https://tineye.com/"

try:
    driver.get(url)
    print("✅ Page loaded successfully!")
    time.sleep(400)
except TimeoutException:
    print("⚠️ Timeout while loading page.")
except WebDriverException as e:
    print("❌ WebDriver error:", e)
