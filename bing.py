#!/usr/bin/env python3
"""
Bing Visual Search FLANN Matcher (Optimized Production Variant)

Purpose:
 - Perform visual search on Bing for each local image.
 - Match best similar image via FLANN-based ORB feature matching.
 - Produce stitched comparison images and CSV logs.

Performance-Oriented Adjustments (Logic preserved):
 - Iterative short polling for Bing results (faster acquisition).
 - Reuse ORB detector instance.
 - Centralized configurable timing constants.
 - Reduced unnecessary logging chatter; adjustable log level.
 - Optimized Selenium Chrome options (no headless per requirement).
 - More resilient but faster waits with early exits.

NOTE: Core matching, threshold logic, file moves, CSV outputs, and side effects remain unchanged.
"""

import time
import csv
import cv2
import numpy as np
import requests
import os
import json
import re
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import glob
from urllib.parse import unquote
import argparse
import logging
import shutil
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import random
from typing import List, Optional, Tuple, Dict, Any

# -----------------------
# CONFIG (Tweakable)
# -----------------------
BING_RESULT_MAX_WAIT_SECONDS = 4.0      # Maximum total time to poll for Bing results
BING_RESULT_POLL_INTERVAL = 0.35        # Interval between polls
BING_SCROLL_EACH_POLL = True            # Scroll a bit each poll to encourage loading
IMAGE_DOWNLOAD_TIMEOUT = 4              # Seconds per remote image fetch
UPLOAD_WAIT_TIMEOUT = 18                # Seconds to wait for upload completion (was 20)
BING_FIRST_ELEMENT_TIMEOUT = 6          # Initial wait to click visual search button
ORB_NFEATURES = 2000                    # Keep same to preserve logic
REQUESTS_MAX_POOL = 80
REQUESTS_MAX_RETRIES = 1
REQUESTS_BACKOFF = 0.05

# -----------------------
# USER AGENTS POOL
# -----------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.1; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edg/120.0.2210.91",
]

# Module-level session & detector
requests_session: Optional[requests.Session] = None
orb_detector = cv2.ORB_create(nfeatures=ORB_NFEATURES)

# -----------------------
# Logging Setup (basic; level overridden via CLI)
# -----------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

def get_random_user_agent() -> str:
    return random.choice(USER_AGENTS)

def init_requests_session(max_pool_connections=REQUESTS_MAX_POOL,
                          max_retries=REQUESTS_MAX_RETRIES,
                          backoff_factor=REQUESTS_BACKOFF) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(
        pool_connections=max_pool_connections,
        pool_maxsize=max_pool_connections,
        max_retries=retries
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': get_random_user_agent()})
    import urllib3  # local import to avoid noise
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    # Performance / stability flags
    options.page_load_strategy = "eager"
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    options.add_argument(f"--user-agent={get_random_user_agent()}")

    driver = uc.Chrome(options=options)
    try:
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    except Exception:
        pass
    return driver

# -----------------------
# Create Required Folders
# -----------------------
def create_folders():
    for folder in ['done', 'similar_images']:
        Path(folder).mkdir(exist_ok=True)

def move_image_to_done(image_path: str):
    try:
        done_folder = Path('done')
        done_folder.mkdir(exist_ok=True)
        source = Path(image_path)
        destination = done_folder / source.name
        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = done_folder / f"{source.stem}_{timestamp}{source.suffix}"
        shutil.move(str(source), str(destination))
    except Exception as e:
        logging.error(f"Move failed ({image_path}): {e}")

def create_comparison_image(source_path: str,
                            similar_img_bytes: bytes,
                            score: float,
                            matches: int,
                            output_folder: str = 'similar_images') -> Optional[str]:
    try:
        original_img = cv2.imread(source_path)
        if original_img is None:
            return None
        sim_np = np.frombuffer(similar_img_bytes, np.uint8)
        similar_img = cv2.imdecode(sim_np, cv2.IMREAD_COLOR)
        if similar_img is None:
            return None

        target_height = 600
        def resize_keep_h(img):
            h, w = img.shape[:2]
            ratio = target_height / h
            return cv2.resize(img, (int(w * ratio), target_height), interpolation=cv2.INTER_AREA)

        original_resized = resize_keep_h(original_img)
        similar_resized = resize_keep_h(similar_img)

        text_area_height = 80
        total_width = original_resized.shape[1] + similar_resized.shape[1]
        total_height = target_height + text_area_height
        canvas = np.full((total_height, total_width, 3), 255, dtype=np.uint8)
        ow = original_resized.shape[1]
        canvas[text_area_height:, 0:ow] = original_resized
        canvas[text_area_height:, ow:] = similar_resized

        text1 = "Original Image"
        text2 = f"Bing | Score: {score:.4f} | Matches: {matches}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 0, 0)
        t1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        t2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        t1_x = (ow - t1_size[0]) // 2
        t1_y = (text_area_height - t1_size[1]) // 2 + t1_size[1]
        sw = similar_resized.shape[1]
        t2_x = ow + (sw - t2_size[0]) // 2
        t2_y = (text_area_height - t2_size[1]) // 2 + t2_size[1]
        cv2.putText(canvas, text1, (t1_x, t1_y), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(canvas, text2, (t2_x, t2_y), font, font_scale, color, thickness, cv2.LINE_AA)

        save_folder = Path(output_folder)
        save_folder.mkdir(exist_ok=True)
        source_name = Path(source_path).stem
        filename = f"{source_name}_comparison_{score:.4f}.jpg"
        filepath = save_folder / filename
        if filepath.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = save_folder / f"{source_name}_comparison_{score:.4f}_{ts}.jpg"
        cv2.imwrite(str(filepath), canvas)
        return str(filepath)
    except Exception as e:
        logging.error(f"Comparison image failed ({source_path}): {e}")
        return None

def download_and_create_comparison(url: str,
                                   source_image_path: str,
                                   flann_score: float,
                                   num_matches: int,
                                   timeout: int = IMAGE_DOWNLOAD_TIMEOUT) -> Optional[str]:
    try:
        global requests_session
        session = requests_session or requests
        headers = {
            'User-Agent': get_random_user_agent(),
            'Referer': 'https://www.bing.com/'
        }
        try:
            resp = session.get(url, timeout=timeout, headers=headers, verify=True)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
            resp = session.get(url, timeout=timeout, headers=headers, verify=False)
        resp.raise_for_status()
        return create_comparison_image(
            source_path=source_image_path,
            similar_img_bytes=resp.content,
            score=flann_score,
            matches=num_matches
        )
    except Exception as e:
        logging.warning(f"Download comparison failed ({url}): {e}")
        return None

def get_image_files(folder_path: str) -> List[str]:
    supported = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    folder = Path(folder_path)
    if not folder.exists():
        logging.error(f"Images folder does not exist: {folder_path}")
        return []
    files: List[str] = []
    for ext in supported:
        files.extend(glob.glob(str(folder / ext)))
        files.extend(glob.glob(str(folder / ext.upper())))
    return sorted(files)

def upload_to_bing(driver, image_path: str, timeout: int = UPLOAD_WAIT_TIMEOUT) -> bool:
    try:
        driver.get("https://www.bing.com/images")
        WebDriverWait(driver, BING_FIRST_ELEMENT_TIMEOUT).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#sbi_b, .sbi_b_prtl, [aria-label*='Visual Search'], [aria-label*='visual search']"))
        ).click()
        file_input = WebDriverWait(driver, 6).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        file_input.send_keys(os.path.abspath(image_path))

        WebDriverWait(driver, timeout).until(
            lambda d: ("search?" in d.current_url) or ("detailV2" in d.current_url)
        )
        return True
    except Exception as e:
        logging.warning(f"Upload failed ({image_path}): {e}")
        return False

def get_bing_image_urls(driver, max_images: int = 20) -> List[str]:
    """
    Optimized: poll for a short, bounded time; attempt mild scrolling
    and early exit once enough URLs collected.
    """
    collected: List[str] = []
    start = time.time()
    seen = set()

    def extract_now():
        links = driver.find_elements(By.CSS_SELECTOR, "a.richImgLnk")
        for link in links[:max_images * 2]:  # safety margin
            m_data = link.get_attribute('data-m')
            if not m_data:
                continue
            try:
                data = json.loads(m_data)
            except Exception:
                data = {}
            url = data.get('murl') or data.get('purl')
            if not url:
                continue
            url_dec = unquote(url)
            if (url_dec.startswith('http') and
                'bing.com/th' not in url_dec and
                'mm.bing.net/th' not in url_dec and
                url_dec not in seen):
                seen.add(url_dec)
                collected.append(url_dec)

    while time.time() - start < BING_RESULT_MAX_WAIT_SECONDS and len(collected) < max_images:
        extract_now()
        if len(collected) >= max_images:
            break
        if BING_SCROLL_EACH_POLL:
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight * 0.25);")
        time.sleep(BING_RESULT_POLL_INTERVAL)

    # Fallback: page source parse if insufficient
    if len(collected) < max_images:
        src = driver.page_source
        extra = re.findall(r'"murl":"(https?://[^"]+)"', src)
        extra += re.findall(r'"purl":"(https?://[^"]+)"', src)
        for u in extra:
            u = unquote(u)
            if (u.startswith('http') and
                'bing.com/th' not in u and
                'mm.bing.net/th' not in u and
                u not in seen):
                seen.add(u)
                collected.append(u)
            if len(collected) >= max_images:
                break

    return collected[:max_images]

def get_image_from_url_or_base64(img_url: str,
                                 timeout: int = IMAGE_DOWNLOAD_TIMEOUT):
    global requests_session
    session = requests_session or requests
    if img_url.startswith("data:image"):
        try:
            _, b64data = img_url.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            img_np = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
    try:
        headers = {
            'User-Agent': get_random_user_agent(),
            'Referer': 'https://www.bing.com/'
        }
        try:
            resp = session.get(img_url, timeout=timeout, headers=headers, verify=True)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
            resp = session.get(img_url, timeout=timeout, headers=headers, verify=False)
        resp.raise_for_status()
        img_np = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None

def calculate_flann_similarity(img1_cv,
                               img2_url: str,
                               min_matches: int = 8,
                               timeout: int = IMAGE_DOWNLOAD_TIMEOUT) -> Tuple[float, int]:
    try:
        if img1_cv is None:
            return 0.0, 0
        img2 = get_image_from_url_or_base64(img2_url, timeout=timeout)
        if img2 is None:
            return 0.0, 0

        kp1, des1 = orb_detector.detectAndCompute(img1_cv, None)
        kp2, des2 = orb_detector.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0, 0

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,
                            key_size=12, multi_probe_level=1)
        search_params = dict(checks=40)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        num_good = len(good)
        if num_good > min_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(np.sum(mask))
                score = inliers / num_good
                return min(score, 1.0), inliers
        return 0.0, 0
    except Exception:
        return 0.0, 0

def process_image(driver,
                  image_path: str,
                  max_images: int = 20,
                  max_workers: int = 8,
                  log_writer=None,
                  links_writer=None,
                  threshold: float = 0.15):
    image_name = os.path.basename(image_path)
    start_time = time.time()

    if not upload_to_bing(driver, image_path):
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, "Bing upload failed"])
        return None

    urls = get_bing_image_urls(driver, max_images=max_images)
    if not urls:
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, "No URLs found"])
        return None

    source_img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img_cv is None:
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, "Failed to read source image"])
        return None

    best_match = None
    best_score = 0.0

    def check_url(url: str):
        score, matches = calculate_flann_similarity(source_img_cv, url)
        if links_writer:
            links_writer.writerow([image_name, url])
        return url, score, matches

    worker_count = min(max_workers, max(1, len(urls)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(check_url, u): u for u in urls}
        for future in as_completed(futures):
            try:
                url, score, matches = future.result()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = {
                        'source_image': image_name,
                        'matched_url': url,
                        'flann_score': score,
                        'num_matches': matches
                    }
            except Exception:
                continue

    elapsed = time.time() - start_time
    if best_match:
        comparison_path = download_and_create_comparison(
            url=best_match['matched_url'],
            source_image_path=image_path,
            flann_score=best_match['flann_score'],
            num_matches=best_match['num_matches']
        )
        best_match['comparison_path'] = comparison_path
        move_image_to_done(image_path)
        if log_writer:
            log_writer.writerow([image_name, f"{elapsed:.2f}", 1, "Match found and saved"])
        return best_match
    else:
        if log_writer:
            log_writer.writerow([image_name, f"{elapsed:.2f}", 0, "No match above threshold"])
        return None

def save_results(all_matches: List[Dict[str, Any]], filename: str = "bing_best_matches.csv"):
    matches_filtered = [m for m in all_matches if m]
    if not matches_filtered:
        logging.info("No matches to save.")
        return
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Source Image', 'Matched URL', 'FLANN Score', 'Feature Matches', 'Comparison Image Path'])
        for m in matches_filtered:
            w.writerow([
                m['source_image'],
                m['matched_url'],
                f"{m['flann_score']:.4f}",
                m['num_matches'],
                m.get('comparison_path', 'N/A')
            ])

def main():
    parser = argparse.ArgumentParser(description="Bing Visual Search FLANN Matcher (Optimized)")
    parser.add_argument('-i', '--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('-o', '--output_file', type=str, default='bing_best_matches.csv', help='CSV file for output')
    parser.add_argument('-u', '--max_images', type=int, default=20, help='Max Bing result images to check')
    parser.add_argument('-w', '--max_workers', type=int, default=8, help='Max concurrent workers for matching')
    parser.add_argument('-d', '--batch_delay', type=float, default=1.5, help='Delay between images (seconds)')
    parser.add_argument('-t', '--threshold', type=float, default=0.9, help='FLANN similarity threshold')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    logging.info("Starting Bing Visual Search FLANN Matcher")

    create_folders()

    global requests_session
    requests_session = init_requests_session()

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"bing_processing_log_{run_timestamp}.csv"
    links_filename = f"bing_image_links_{run_timestamp}.csv"

    image_files = get_image_files(args.images_folder)
    if not image_files:
        logging.warning(f"No images found in '{args.images_folder}'. Exiting.")
        return

    driver = init_driver()
    if not driver:
        logging.critical("Web driver initialization failed.")
        return

    all_matches: List[Dict[str, Any]] = []
    total_images = len(image_files)

    try:
        with open(log_filename, 'w', newline='', encoding='utf-8') as log_file, \
             open(links_filename, 'w', newline='', encoding='utf-8') as links_file:

            log_writer = csv.writer(log_file)
            log_writer.writerow(['Image Name', 'Processing Time (s)', 'Match Found', 'Notes'])
            links_writer = csv.writer(links_file)
            links_writer.writerow(['Source Image', 'Scraped URL'])

            for idx, image_path in enumerate(image_files, 1):
                logging.info(f"[{idx}/{total_images}] Processing {os.path.basename(image_path)}")
                result = process_image(
                    driver,
                    image_path,
                    max_images=args.max_images,
                    max_workers=args.max_workers,
                    log_writer=log_writer,
                    links_writer=links_writer,
                    threshold=args.threshold
                )
                if result:
                    all_matches.append(result)
                if idx < total_images:
                    time.sleep(args.batch_delay)

        save_results(all_matches, args.output_file)
        logging.info("Finished")
        logging.info(f"Processed {total_images} images. Matches found: {len(all_matches)}")

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    except Exception as e:
        logging.critical(f"Unexpected error: {e}", exc_info=True)
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        logging.info("Cleanup complete.")

if __name__ == "__main__":
    main()