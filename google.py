import time
import csv
import cv2
import numpy as np
import requests
import os
import logging
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

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
    # options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    return driver

# -----------------------
# Upload and Extract URLs
# -----------------------
def upload_image(driver, image_path, timeout=20):
    abs_path = os.path.abspath(image_path)
    driver.get("https://lens.google.com/upload")
    upload_input = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
    )
    upload_input.send_keys(abs_path)
    WebDriverWait(driver, timeout).until(
        lambda d: "search" in d.current_url or "results" in d.current_url or "lens" in d.current_url
    )
    time.sleep(4)  # Short wait for results

def get_image_urls(driver, max_images=40):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    img_tags = soup.find_all("img")
    urls = []
    for img in img_tags:
        src = img.get("src") or img.get("data-src")
        if src and src.startswith("http"):
            urls.append(src)
    return list(set(urls))[:max_images]

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
    for extension in supported_extensions:
        image_files.extend(glob.glob(str(folder_path / extension)))
        image_files.extend(glob.glob(str(folder_path / extension.upper())))
    return sorted(image_files)

# -----------------------
# ORB + FLANN Matching (fast and robust)
# -----------------------
def calculate_orb_similarity(img1_path, img2_url, min_matches=10, timeout=10):
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        resp = requests.get(img2_url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0.0, 0
        orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0, 0
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=12,
            key_size=20,
            multi_probe_level=2
        )
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        num_good_matches = len(good_matches)
        if num_good_matches >= min_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = mask.ravel().tolist()
                num_inliers = sum(inliers)
                similarity_score = num_inliers / max(len(kp1), len(kp2))
                return similarity_score, num_inliers
        similarity_score = num_good_matches / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        return similarity_score, num_good_matches
    except Exception:
        return 0.0, 0

# -----------------------
# Similarity Check (ORB only)
# -----------------------
def is_similar_image(img1_path, img2_url, orb_threshold=0.01, min_matches=10):
    orb_score, num_matches = calculate_orb_similarity(img1_path, img2_url, min_matches=min_matches)
    is_match = (orb_score >= orb_threshold and num_matches >= min_matches)
    return is_match, orb_score, num_matches

# -----------------------
# Process Single Image (Parallelized)
# -----------------------
def process_image(driver, image_path, max_images=40, max_workers=8, orb_threshold=0.01, min_matches=10):
    image_name = os.path.basename(image_path)
    logging.info(f"Processing: {image_name}")
    upload_image(driver, image_path)
    urls = get_image_urls(driver, max_images=max_images)
    if not urls:
        logging.warning(f"No URLs found for {image_name}")
        return []
    matched_urls = []
    def check_url(url):
        is_match, orb_score, num_matches = is_similar_image(
            image_path, url, orb_threshold=orb_threshold, min_matches=min_matches
        )
        if is_match:
            return {
                'source_image': image_name,
                'matched_url': url,
                'orb_score': orb_score,
                'num_matches': num_matches
            }
        return None
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
def save_results(all_matches, filename="google_matches.csv"):
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
    parser = argparse.ArgumentParser(description="Google Lens ORB Matcher")
    parser.add_argument('--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('--output_file', type=str, default='matched_images.csv', help='CSV file for output')
    parser.add_argument('--max_images', type=int, default=40, help='Max Google result images to check')
    parser.add_argument('--max_workers', type=int, default=8, help='Max concurrent workers for matching')
    parser.add_argument('--batch_delay', type=float, default=2.0, help='Delay between images (seconds)')
    parser.add_argument('--orb_threshold', type=float, default=0.01, help='ORB similarity score threshold')
    parser.add_argument('--min_matches', type=int, default=10, help='Min ORB matches to consider a match')
    args = parser.parse_args()

    logging.info("Starting Google Lens ORB Matcher (Production)")
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
                max_workers=args.max_workers,
                orb_threshold=args.orb_threshold,
                min_matches=args.min_matches
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