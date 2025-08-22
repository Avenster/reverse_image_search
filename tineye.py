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
from selenium.common.exceptions import TimeoutException
from collections import defaultdict
import argparse
import logging

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

DEFAULT_MAX_WORKERS = 8
DEFAULT_WAIT_TIMEOUT = 15
DEFAULT_ORB_FEATURES = 2000
DEFAULT_ORB_MIN_MATCHES = 10
DEFAULT_ORB_SIMILARITY_THRESHOLD = 0.01
DEFAULT_MAX_IMAGES = 50
DEFAULT_BATCH_DELAY = 2.0

def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--window-size=1920,1080")
    # options.add_argument("--headless=new")
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
    for extension in supported_extensions:
        image_files.extend(glob.glob(str(folder_path / extension)))
        image_files.extend(glob.glob(str(folder_path / extension.upper())))
    return sorted(set(image_files))

def upload_to_tineye(driver, image_path):
    try:
        driver.get("https://tineye.com/")
        time.sleep(3)
        file_input = WebDriverWait(driver, DEFAULT_WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file'][name='image'], #upload-box"))
        )
        abs_path = os.path.abspath(image_path)
        file_input.send_keys(abs_path)
        logging.info("Waiting for TinEye to process image...")
        time.sleep(8)
        match_count = get_match_count(driver)
        return True, match_count
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
        if "no matches" in page_text or "0 matches" in page_text:
            logging.info("TinEye found 0 matches")
            return 0
        similar_matches = driver.find_elements(By.CSS_SELECTOR, ".similar-match, div[class*='match']")
        if similar_matches:
            logging.info(f"Found {len(similar_matches)} result elements")
            return len(similar_matches)
        logging.info("Could not determine match count, assuming 0")
        return 0
    except Exception as e:
        logging.error(f"Error getting match count: {str(e)}")
        return 0

def get_tineye_image_urls(driver, max_images=50):
    urls = []
    try:
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        similar_matches = driver.find_elements(By.CSS_SELECTOR, ".similar-match")
        logging.info(f"Found {len(similar_matches)} similar match elements")
        for match in similar_matches[:max_images]:
            try:
                img = match.find_element(By.TAG_NAME, "img")
                img_src = img.get_attribute("src")
                if img_src and img_src.startswith("http"):
                    clean_url = re.sub(r'-\d+nw-', '-', img_src)
                    clean_url = re.sub(r'/260nw/', '/', clean_url)
                    clean_url = re.sub(r'_260nw', '', clean_url)
                    if 'shutterstock' in clean_url:
                        clean_url = clean_url.replace('/image-photo/', '/image-photo/')
                        clean_url = re.sub(r'-260nw-', '-1500w-', clean_url)
                        if '-1500w-' not in clean_url and 'image-photo' in clean_url:
                            clean_url = clean_url.replace('.jpg', '-1500w.jpg')
                    urls.append(clean_url)
                    if img_src not in urls:
                        urls.append(img_src)
            except Exception:
                continue
        urls = list(dict.fromkeys(urls))
        logging.info(f"Extracted {len(urls)} unique image URLs")
        if urls:
            logging.info("Sample URLs found:")
            for url in urls[:3]:
                logging.info(f"  - {url[:100]}...")
        return urls[:max_images]
    except Exception as e:
        logging.error(f"Error extracting URLs: {str(e)}")
    return urls

def calculate_orb_similarity(img1_path, img2_url, orb_features=DEFAULT_ORB_FEATURES, min_matches=DEFAULT_ORB_MIN_MATCHES):
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        resp = requests.get(img2_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0.0, 0
        orb = cv2.ORB_create(nfeatures=orb_features, scaleFactor=1.2, nlevels=8)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0, 0
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)
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
            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = mask.ravel().tolist()
                    num_inliers = sum(inliers)
                    similarity_score = num_inliers / max(len(kp1), len(kp2))
                    return similarity_score, num_inliers
            except Exception:
                pass
        similarity_score = num_good_matches / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        return similarity_score, num_good_matches
    except Exception:
        return 0.0, 0

def is_similar_image(img1_path, img2_url, orb_threshold=DEFAULT_ORB_SIMILARITY_THRESHOLD, min_matches=DEFAULT_ORB_MIN_MATCHES, orb_features=DEFAULT_ORB_FEATURES):
    orb_score, num_matches = calculate_orb_similarity(img1_path, img2_url, orb_features=orb_features, min_matches=min_matches)
    is_match = (orb_score >= orb_threshold and num_matches >= min_matches)
    return is_match, orb_score, num_matches

def check_similarity_parallel(image_path, urls, orb_threshold, min_matches, orb_features, max_workers):
    matched_urls = []
    image_name = os.path.basename(image_path)
    logging.info(f"Checking {len(urls)} URLs for matches...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(is_similar_image, image_path, url, orb_threshold, min_matches, orb_features): url 
            for url in urls
        }
        completed = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            completed += 1
            try:
                is_match, orb_score, num_matches = future.result(timeout=20)
                logging.info(f"[{completed}/{len(urls)}] Score: {orb_score:.3f}, Matches: {num_matches}")
                if is_match:
                    matched_urls.append({
                        'source_image': image_name,
                        'matched_url': url,
                        'orb_score': orb_score,
                        'num_matches': num_matches
                    })
                    logging.info(f"✓ Match found! Score: {orb_score:.3f}, URL: {url[:60]}...")
            except Exception:
                continue
    logging.info("Completed similarity checking")
    return matched_urls

def process_image(driver, image_path, orb_threshold, min_matches, orb_features, max_images, max_workers):
    image_name = os.path.basename(image_path)
    logging.info(f"[Processing] {image_name}")
    success, match_count = upload_to_tineye(driver, image_path)
    if not success:
        logging.error("Failed to upload image")
        return []
    if match_count == 0:
        logging.info("✓ TinEye reports 0 matches - skipping URL extraction")
        return []
    logging.info(f"✓ TinEye found {match_count} potential matches")
    urls = get_tineye_image_urls(driver, max_images=min(match_count, max_images))
    if not urls:
        logging.error("No URLs could be extracted")
        return []
    logging.info(f"Extracted {len(urls)} URLs to verify")
    matches = check_similarity_parallel(
        image_path, urls, orb_threshold, min_matches, orb_features, max_workers
    )
    logging.info(f"{len(matches)} matches found for {image_name}")
    return matches

def save_results(all_matches, filename="tineye_matched_images.csv"):
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

def main():
    parser = argparse.ArgumentParser(description="TinEye Image ORB Matcher")
    parser.add_argument('--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('--output_file', type=str, default='tineye_matched_images.csv', help='CSV file for output')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS, help='Max concurrent workers for matching')
    parser.add_argument('--max_images', type=int, default=DEFAULT_MAX_IMAGES, help='Max TinEye result images to check')
    parser.add_argument('--orb_features', type=int, default=DEFAULT_ORB_FEATURES, help='ORB nfeatures parameter')
    parser.add_argument('--orb_min_matches', type=int, default=DEFAULT_ORB_MIN_MATCHES, help='Min ORB matches to consider a match')
    parser.add_argument('--orb_similarity_threshold', type=float, default=DEFAULT_ORB_SIMILARITY_THRESHOLD, help='ORB similarity score threshold')
    parser.add_argument('--batch_delay', type=float, default=DEFAULT_BATCH_DELAY, help='Delay between images (seconds)')
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("TINEYE IMAGE ORB MATCHER - Production Version")
    logging.info("=" * 60)
    image_files = get_image_files(args.images_folder)
    if not image_files:
        logging.error(f"No images found in '{args.images_folder}' folder!")
        return
    logging.info(f"Found {len(image_files)} images")
    driver = init_driver()
    all_matches = []
    try:
        for idx, image_path in enumerate(image_files, 1):
            logging.info(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            matches = process_image(
                driver, image_path,
                args.orb_similarity_threshold, args.orb_min_matches,
                args.orb_features, args.max_images, args.max_workers
            )
            all_matches.extend(matches)
            if idx < len(image_files):
                time.sleep(args.batch_delay)
        save_results(all_matches, args.output_file)
        logging.info(f"Images processed: {len(image_files)}")
        logging.info(f"Total matches found: {len(all_matches)}")
        matches_by_source = defaultdict(list)
        for match in all_matches:
            matches_by_source[match['source_image']].append(match)
        if matches_by_source:
            for source, matches in matches_by_source.items():
                avg_score = sum(m['orb_score'] for m in matches) / len(matches)
                logging.info(f"{source}: {len(matches)} matches (avg score: {avg_score:.3f})")
        logging.info("Processing complete!")
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
    finally:
        driver.quit()
        logging.info("Done!")

if __name__ == "__main__":
    main()