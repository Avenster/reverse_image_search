import time
import csv
import cv2
import numpy as np
import requests
import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import glob
from urllib.parse import unquote
import argparse
import logging

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless=new")  # Production: run headless
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

# -----------------------
# Get Images from Folder
# -----------------------
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
    return sorted(image_files)

# -----------------------
# Upload to Bing Visual Search
# -----------------------
def upload_to_bing(driver, image_path, timeout=20):
    try:
        driver.get("https://www.bing.com/images")
        WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#sbi_b, .sbi_b_prtl, [aria-label*='Visual Search'], [aria-label*='visual search']"))
        ).click()
        file_input = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        abs_path = os.path.abspath(image_path)
        file_input.send_keys(abs_path)
        WebDriverWait(driver, timeout).until(
            lambda d: "search?" in d.current_url or "detailV2" in d.current_url
        )
        return True
    except Exception as e:
        logging.warning(f"Upload to Bing failed for {image_path}: {e}")
        return False

# -----------------------
# Extract Image URLs from Bing Results
# -----------------------
def get_bing_image_urls(driver, max_images=40):
    urls = []
    try:
        # Scroll to load more images
        for _ in range(2):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
        # Try richImgLnk method
        rich_links = driver.find_elements(By.CSS_SELECTOR, "a.richImgLnk")
        for link in rich_links[:max_images]:
            m_data = link.get_attribute('data-m')
            if m_data:
                data = json.loads(m_data)
                if 'murl' in data:
                    urls.append(data['murl'])
                elif 'purl' in data:
                    urls.append(data['purl'])
        # Fallback: parse page source if few found
        if len(urls) < 10:
            page_source = driver.page_source
            murl_pattern = r'"murl":"(https?://[^"]+)"'
            purl_pattern = r'"purl":"(https?://[^"]+)"'
            urls.extend(re.findall(murl_pattern, page_source))
            urls.extend(re.findall(purl_pattern, page_source))
        # Clean URLs
        cleaned = []
        for url in urls:
            url = unquote(url)
            if url.startswith('http') and not ('bing.com/th' in url or 'mm.bing.net/th' in url):
                cleaned.append(url)
        return cleaned[:max_images]
    except Exception as e:
        logging.warning(f"Error extracting Bing URLs: {e}")
        return []

# -----------------------
# ORB Feature Matching
# -----------------------
def calculate_orb_similarity(img1_path, img2_url, min_matches=8, timeout=15):
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None: return 0.0, 0
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.bing.com/'
        }
        resp = requests.get(img2_url, timeout=timeout, headers=headers)
        if resp.status_code != 200: return 0.0, 0
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img2 is None: return 0.0, 0
        # Resize images for speed
        max_dim = 1024
        for img, shape in [(img1, img1.shape), (img2, img2.shape)]:
            h, w = shape
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return 0.0, 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        num_good_matches = len(good_matches)
        if num_good_matches >= min_matches:
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = mask.ravel().tolist()
                    num_inliers = sum(inliers)
                    if num_inliers >= min_matches:
                        similarity_score = num_inliers / min(len(kp1), len(kp2))
                        return min(similarity_score, 1.0), num_inliers
            except Exception:
                pass
        if num_good_matches > 0:
            similarity_score = num_good_matches / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
            return min(similarity_score, 1.0), num_good_matches
        return 0.0, 0
    except Exception:
        return 0.0, 0

# -----------------------
# Check Similarity (ORB only)
# -----------------------
def is_similar_image(img1_path, img2_url):
    orb_score, num_matches = calculate_orb_similarity(img1_path, img2_url, min_matches=8)
    is_match = (orb_score >= 0.008 and num_matches >= 8)
    return is_match, orb_score, num_matches

# -----------------------
# Process Single Image (Parallelized URL checks)
# -----------------------
def process_image(driver, image_path, max_images=40, max_workers=8):
    image_name = os.path.basename(image_path)
    logging.info(f"Processing: {image_name}")
    if not upload_to_bing(driver, image_path):
        logging.warning(f"Bing Visual Search upload failed for {image_name}")
        return []
    urls = get_bing_image_urls(driver, max_images=max_images)
    if not urls:
        logging.warning(f"No URLs found for {image_name}")
        return []
    matched_urls = []
    def check_url(url):
        is_match, orb_score, num_matches = is_similar_image(image_path, url)
        if is_match:
            return {
                'source_image': image_name,
                'matched_url': url,
                'orb_score': orb_score,
                'num_matches': num_matches
            }
        return None
    # Parallelize URL checks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(check_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            result = future.result()
            if result: matched_urls.append(result)
    logging.info(f"Found {len(matched_urls)} matches for {image_name}")
    return matched_urls

# -----------------------
# Save Results
# -----------------------
def save_results(all_matches, filename="bing_matched_images.csv"):
    if not all_matches:
        logging.info("No matches to save")
        return
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Image', 'Matched URL', 'ORB Score', 'Feature Matches'])
        for match in all_matches:
            writer.writerow([
                match['source_image'],
                match['matched_url'],
                f"{match['orb_score']:.4f}",
                match['num_matches']
            ])
    logging.info(f"Results saved to {filename}")

# -----------------------
# Main Function (Production CLI)
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Bing Visual Search ORB Matcher")
    parser.add_argument('--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('--output_file', type=str, default='bing_matched_images.csv', help='CSV file for output')
    parser.add_argument('--max_images', type=int, default=40, help='Max Bing result images to check')
    parser.add_argument('--max_workers', type=int, default=8, help='Max concurrent workers for matching')
    parser.add_argument('--batch_delay', type=float, default=2.0, help='Delay between images (seconds)')
    args = parser.parse_args()

    logging.info("Starting Bing Visual Search ORB Matcher (Production)")
    image_files = get_image_files(args.images_folder)
    if not image_files:
        logging.error(f"No images found in '{args.images_folder}'!")
        return
    driver = init_driver()
    all_matches = []
    try:
        for idx, image_path in enumerate(image_files, 1):
            matches = process_image(
                driver, image_path,
                max_images=args.max_images,
                max_workers=args.max_workers
            )
            all_matches.extend(matches)
            if idx < len(image_files):
                time.sleep(args.batch_delay)
        save_results(all_matches, args.output_file)
        logging.info(f"Processed {len(image_files)} images, total matches: {len(all_matches)}")
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()