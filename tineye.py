import time
import csv
import cv2
import numpy as np
import requests
import os
import re
from pathlib import Path
import glob
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from collections import defaultdict
import argparse
import logging

# Logging Setup
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

DEFAULT_MAX_WORKERS = 8
DEFAULT_WAIT_TIMEOUT = 60
DEFAULT_ORB_FEATURES = 2000
DEFAULT_ORB_MIN_MATCHES = 10
DEFAULT_ORB_SIMILARITY_THRESHOLD = 0.01
DEFAULT_MAX_IMAGES = 50
DEFAULT_BATCH_DELAY = 5

def init_driver():
    options = uc.ChromeOptions()
    user_data_dir = os.path.join(os.path.expanduser("~"), ".tineye_uc_profile")
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.set_page_load_timeout(30)
    return driver

def get_image_files(folder_path):
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    folder_path = Path(folder_path)
    if not folder_path.exists():
        logging.error(f"Folder '{folder_path}' does not exist!")
        return []
    for ext in supported_extensions:
        image_files.extend(glob.glob(str(folder_path / ext)))
        image_files.extend(glob.glob(str(folder_path / ext.upper())))
    return sorted(set(image_files))

def upload_to_tineye(driver, image_path):
    try:
        driver.get("https://tineye.com/")
        logging.info("Please solve CAPTCHA manually if shown and press Enter in terminal...")
        input("Press Enter after solving CAPTCHA to continue...")
        file_input = WebDriverWait(driver, DEFAULT_WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        file_input.send_keys(os.path.abspath(image_path))
        logging.info("Waiting for TinEye to process image...")
        time.sleep(8)
        return True, get_match_count(driver)
    except Exception as e:
        logging.error(f"Error during upload: {str(e)}")
        return False, 0

def get_match_count(driver):
    try:
        time.sleep(2)
        page_text = driver.page_source.lower()
        match_pattern = r'(\d+)\s+match(?:es)?'
        matches = re.findall(match_pattern, page_text)
        if matches:
            count = int(matches[0])
            logging.info(f"TinEye found {count} matches")
            return count
        return 0
    except Exception as e:
        logging.error(f"Error getting match count: {str(e)}")
        return 0

def get_tineye_image_urls(driver, max_images=50):
    urls = []
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        similar_matches = driver.find_elements(By.CSS_SELECTOR, ".similar-match img")
        for img in similar_matches[:max_images]:
            src = img.get_attribute("src")
            if src and src.startswith("http"):
                urls.append(src)
        urls = list(dict.fromkeys(urls))
        logging.info(f"Extracted {len(urls)} unique image URLs")
        return urls[:max_images]
    except Exception as e:
        logging.error(f"Error extracting URLs: {str(e)}")
        return urls

def calculate_orb_similarity(img1_path, img2_url, orb_features=DEFAULT_ORB_FEATURES, min_matches=DEFAULT_ORB_MIN_MATCHES):
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        resp = requests.get(img2_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        img2 = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0.0, 0
        orb = cv2.ORB_create(nfeatures=orb_features)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None: return 0.0, 0
        FLANN_INDEX_LSH = 6
        flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m,n in matches if m.distance < 0.7*n.distance]
        num_good_matches = len(good_matches)
        similarity_score = num_good_matches / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        return similarity_score, num_good_matches
    except Exception:
        return 0.0, 0

def is_similar_image(img1_path, img2_url, orb_threshold=DEFAULT_ORB_SIMILARITY_THRESHOLD, min_matches=DEFAULT_ORB_MIN_MATCHES):
    orb_score, num_matches = calculate_orb_similarity(img1_path, img2_url, min_matches=min_matches)
    return orb_score >= orb_threshold and num_matches >= min_matches, orb_score, num_matches

def check_similarity_parallel(image_path, urls, orb_threshold, min_matches, max_workers):
    matched_urls = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(is_similar_image, image_path, url, orb_threshold, min_matches): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                is_match, orb_score, num_matches = future.result(timeout=20)
                if is_match:
                    matched_urls.append({'source_image': os.path.basename(image_path), 'matched_url': url, 'orb_score': orb_score, 'num_matches': num_matches})
            except Exception:
                continue
    return matched_urls

def process_image(driver, image_path, orb_threshold, min_matches, max_images, max_workers):
    success, match_count = upload_to_tineye(driver, image_path)
    if not success or match_count == 0:
        return []
    urls = get_tineye_image_urls(driver, max_images=min(match_count, max_images))
    if not urls: return []
    return check_similarity_parallel(image_path, urls, orb_threshold, min_matches, max_workers)

def save_results(all_matches, filename="tineye_matched_images.csv"):
    if not all_matches: return
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Image', 'Matched URL', 'ORB Score', 'Feature Matches'])
        for match in all_matches:
            writer.writerow([match['source_image'], match['matched_url'], f"{match['orb_score']:.4f}", match['num_matches']])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str, default='images')
    parser.add_argument('--output_file', type=str, default='tineye_matched_images.csv')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument('--max_images', type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument('--orb_min_matches', type=int, default=DEFAULT_ORB_MIN_MATCHES)
    parser.add_argument('--orb_similarity_threshold', type=float, default=DEFAULT_ORB_SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    image_files = get_image_files(args.images_folder)
    if not image_files:
        logging.error(f"No images found in '{args.images_folder}'!")
        return

    driver = init_driver()
    all_matches = []
    try:
        for idx, image_path in enumerate(image_files, 1):
            logging.info(f"[{idx}/{len(image_files)}] Processing {os.path.basename(image_path)}")
            matches = process_image(driver, image_path, args.orb_similarity_threshold, args.orb_min_matches, args.max_images, args.max_workers)
            all_matches.extend(matches)
            time.sleep(1)
        save_results(all_matches, args.output_file)
        logging.info(f"Processing complete! Total matches: {len(all_matches)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
