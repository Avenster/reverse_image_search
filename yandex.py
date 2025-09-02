#!/usr/bin/env python3
"""
Yandex Visual Search & FLANN Matcher (Optimized Fast Variant)

Goal:
 - Keep the original functional logic (workflow, thresholding, outputs, folder moves)
 - Improve speed, robustness, and configurability for QC team use
 - Reduce unnecessary sleeps and blocking waits
 - Add early-exit style polling for result URLs
 - Centralize tunable performance constants
 - Reuse ORB detector and a tuned requests.Session
 - Avoid broad bare except blocks
 - Allow adjustable logging level

Core Logic Preserved:
 - Upload -> navigate to Similar tab -> scrape candidate image URLs
 - FLANN ORB-based similarity scoring
 - Select best match above threshold
 - Create side-by-side comparison image
 - Move matched source image to 'done'
 - Log CSV outputs (processing + scraped URLs)
"""

from __future__ import annotations
import time
import csv
import cv2
import numpy as np
import requests
import os
import base64
from pathlib import Path
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import argparse
import logging
import shutil
from datetime import datetime
import tempfile
import random
import re
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from typing import List, Optional, Tuple, Dict

# -----------------------
# CONFIG (Adjust as needed)
# -----------------------
MAX_YANDEX_URLS_POLL_SECONDS = 4.0     # Total time to attempt collecting result URLs
YANDEX_URLS_POLL_INTERVAL = 0.35       # Poll interval for scraping URLs
SCROLL_DURING_POLL = True              # Slight scrolling each poll
INITIAL_POST_UPLOAD_WAIT = 0.4         # Short stabilization delay after upload
SIMILAR_TAB_WAIT_TIMEOUT = 8           # Max seconds to locate Similar tab
UPLOAD_NAV_TIMEOUT = 18                # Max seconds for initial navigation & result readiness
IMAGE_FETCH_TIMEOUT = 3                # Per-image HTTP timeout (seconds)
REQUESTS_POOL_SIZE = 80
REQUESTS_RETRY_TOTAL = 1
REQUESTS_RETRY_BACKOFF = 0.05
ORB_NFEATURES = 2000
FLANN_MIN_MATCHES = 8                  # Preserve original matching logic
LOWE_RATIO = 0.75
THREADPOOL_OVERHEAD_CAP = 24           # Prevent too many workers overhead

# -----------------------
# Logging Setup (overridden by --log-level)
# -----------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Disable SSL warnings for speed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------
# Global reusable objects
# -----------------------
requests_session: Optional[requests.Session] = None
orb_detector = cv2.ORB_create(nfeatures=ORB_NFEATURES)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/122.0.2365.66"
]

def random_user_agent() -> str:
    return random.choice(USER_AGENTS)

def init_requests_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=REQUESTS_RETRY_TOTAL,
        backoff_factor=REQUESTS_RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(
        pool_connections=REQUESTS_POOL_SIZE,
        pool_maxsize=REQUESTS_POOL_SIZE,
        max_retries=retries
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({
        "User-Agent": random_user_agent(),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8"
    })
    return sess

# -----------------------
# Browser Initialization
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    options.page_load_strategy = "eager"
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"--user-agent={random_user_agent()}")
    # Dedicated temp user data dir per run
    temp_profile_dir = tempfile.mkdtemp(prefix="yandex_vs_profile_")
    options.add_argument(f"--user-data-dir={temp_profile_dir}")

    try:
        driver = uc.Chrome(options=options)
        try:
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception:
            pass
        driver._custom_profile_dir = temp_profile_dir  # attach for cleanup
        return driver
    except Exception as e:
        logging.error(f"Selenium driver init error: {e}")
        raise

# -----------------------
# Folders
# -----------------------
def create_folders():
    for folder in ['done', 'similar_images']:
        Path(folder).mkdir(exist_ok=True)

def move_image_to_done(image_path: str):
    try:
        source = Path(image_path)
        dest_dir = Path("done")
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / source.name
        if dest.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = dest_dir / f"{source.stem}_{ts}{source.suffix}"
        shutil.move(str(source), str(dest))
    except Exception as e:
        logging.error(f"Move failed ({image_path}): {e}")

# -----------------------
# Comparison Image
# -----------------------
def create_comparison_image(source_path: str,
                            similar_img_bytes: bytes,
                            score: float,
                            matches: int,
                            output_folder: str = 'similar_images') -> Optional[str]:
    try:
        original = cv2.imread(source_path)
        if original is None:
            return None
        sim_np = np.frombuffer(similar_img_bytes, np.uint8)
        similar = cv2.imdecode(sim_np, cv2.IMREAD_COLOR)
        if similar is None:
            return None

        target_h = 600

        def resize_h(img):
            h, w = img.shape[:2]
            scale = target_h / h
            return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)

        o_r = resize_h(original)
        s_r = resize_h(similar)

        header_h = 72
        canvas_w = o_r.shape[1] + s_r.shape[1]
        canvas_h = target_h + header_h
        canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
        canvas[header_h:, :o_r.shape[1]] = o_r
        canvas[header_h:, o_r.shape[1]:] = s_r

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.8
        thick = 2
        t1 = "Original Image"
        t2 = f"Yandex | Score: {score:.4f} | Matches: {matches}"

        def put_centered(text, x_start, width):
            size = cv2.getTextSize(text, font, fs, thick)[0]
            x = x_start + (width - size[0]) // 2
            y = (header_h - size[1]) // 2 + size[1]
            cv2.putText(canvas, text, (x, y), font, fs, (0, 0, 0), thick, cv2.LINE_AA)

        put_centered(t1, 0, o_r.shape[1])
        put_centered(t2, o_r.shape[1], s_r.shape[1])

        out_dir = Path(output_folder)
        out_dir.mkdir(exist_ok=True)
        stem = Path(source_path).stem
        fname = out_dir / f"{stem}_comparison_{score:.4f}.jpg"
        if fname.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = out_dir / f"{stem}_comparison_{score:.4f}_{ts}.jpg"
        cv2.imwrite(str(fname), canvas)
        return str(fname)
    except Exception as e:
        logging.error(f"Comparison image error ({source_path}): {e}")
        return None

# -----------------------
# Networking Helpers
# -----------------------
def fetch_image_bytes(url: str, timeout: int = IMAGE_FETCH_TIMEOUT) -> Optional[bytes]:
    global requests_session
    sess = requests_session or requests
    try:
        resp = sess.get(url, timeout=timeout, verify=False)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        return None
    return None

def download_and_create_comparison(url: str,
                                   source_image_path: str,
                                   flann_score: float,
                                   num_matches: int,
                                   timeout: int = IMAGE_FETCH_TIMEOUT) -> Optional[str]:
    try:
        data = fetch_image_bytes(url, timeout=timeout)
        if not data:
            return None
        return create_comparison_image(
            source_path=source_image_path,
            similar_img_bytes=data,
            score=flann_score,
            matches=num_matches
        )
    except Exception:
        return None

def get_image_files(folder_path: str) -> List[str]:
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    p = Path(folder_path)
    if not p.is_dir():
        logging.error(f"Images folder missing: {folder_path}")
        return []
    files: List[str] = []
    for e in exts:
        files.extend(p.glob(e))
        files.extend(p.glob(e.upper()))
    return sorted(str(f) for f in files)

# -----------------------
# Image Decode Helper
# -----------------------
def get_image_from_url_or_base64(img_url: str, timeout: int = IMAGE_FETCH_TIMEOUT):
    if img_url.startswith("data:image"):
        try:
            _, b64data = img_url.split(",", 1)
            raw = base64.b64decode(b64data)
            arr = np.frombuffer(raw, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
    img_bytes = fetch_image_bytes(img_url, timeout=timeout)
    if not img_bytes:
        return None
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

# -----------------------
# FLANN Similarity
# -----------------------
def calculate_flann_similarity(img1_cv,
                               img2_url: str,
                               min_matches: int = FLANN_MIN_MATCHES,
                               timeout: int = IMAGE_FETCH_TIMEOUT) -> Tuple[float, int]:
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
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=40)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < LOWE_RATIO * n.distance:
                    good.append(m)

        if len(good) > min_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())
                score = inliers / len(good)
                return min(score, 1.0), inliers
        return 0.0, 0
    except Exception:
        return 0.0, 0

# -----------------------
# Yandex Interaction
# -----------------------
def upload_to_yandex_and_navigate(driver, image_path: str) -> bool:
    try:
        driver.get("https://yandex.com/images/")
        # Ensure base structure loaded
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )

        # Find file input (multiple strategies)
        selectors = [
            "input[type='file']",
            "input[accept*='image']",
            ".CbirSearchForm-FileInput input",
            ".cbir-panel__file-input input",
        ]
        file_input = None
        for sel in selectors:
            try:
                file_input = WebDriverWait(driver, 4).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
                if file_input:
                    break
            except TimeoutException:
                continue

        if not file_input:
            logging.error("File input not found.")
            return False

        # Force visibility & upload
        driver.execute_script("""
            arguments[0].style.display='block';
            arguments[0].style.visibility='visible';
            arguments[0].style.opacity='1';
            arguments[0].removeAttribute('hidden');
        """, file_input)
        file_input.send_keys(os.path.abspath(image_path))
        time.sleep(INITIAL_POST_UPLOAD_WAIT)

        # Wait for some indication of processed image
        try:
            WebDriverWait(driver, UPLOAD_NAV_TIMEOUT).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".CbirNavigation")),
                    EC.url_contains("cbir_id")
                )
            )
        except TimeoutException:
            logging.warning("Upload result state not clearly detected.")
            return False

        # Attempt to locate Similar tab
        similar_selectors_css = [
            "a[data-cbir-page-type='similar']",
            ".CbirNavigation-TabsItem_name_similar-page",
            "a.CbirNavigation-TabsItem_name_similar-page"
        ]
        similar_tab = None
        for sel in similar_selectors_css:
            try:
                similar_tab = WebDriverWait(driver, SIMILAR_TAB_WAIT_TIMEOUT).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                if similar_tab:
                    break
            except TimeoutException:
                continue

        if not similar_tab:
            # Fallback XPath (English or Russian)
            try:
                similar_tab = WebDriverWait(driver, 4).until(
                    EC.element_to_be_clickable((By.XPATH,
                        "//a[contains(text(), 'Similar') or contains(text(), 'Похожие') or contains(text(), 'похожие')]"))
                )
            except TimeoutException:
                logging.error("Similar tab not found.")
                return False

        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", similar_tab)
        time.sleep(0.15)
        try:
            similar_tab.click()
        except Exception:
            driver.execute_script("arguments[0].click();", similar_tab)

        # Optional brief wait to stabilize similar images view
        time.sleep(0.5)
        return True
    except Exception as e:
        logging.error(f"Upload/navigation failed ({image_path}): {e}")
        return False

def get_yandex_image_urls(driver, max_images: int = 20) -> List[str]:
    """
    Polls the page for candidate image URLs for a bounded time.
    Multiple methods:
      - href param img_url
      - data-bem JSON fragments
      - <img src>
    """
    collected: Dict[str, None] = {}
    start = time.time()

    def harvest():
        # Method 1: Links with img_url
        for link in driver.find_elements(By.CSS_SELECTOR, "a[href*='img_url=']"):
            href = link.get_attribute("href")
            if not href:
                continue
            if "img_url=" in href:
                part = href.split("img_url=")[1].split("&")[0]
                u = unquote(part)
                if u.startswith("http") and "yandex" not in u and "yastatic" not in u:
                    collected.setdefault(u, None)

        # Method 2: data-bem attributes
        candidates = driver.find_elements(By.CSS_SELECTOR, "[data-bem*='http']")
        for el in candidates:
            data_bem = el.get_attribute("data-bem")
            if not data_bem:
                continue
            # Simple URL pattern
            for u in re.findall(r'https?://[^"\'\\s,}]+', data_bem):
                if "yandex" in u or "yastatic" in u:
                    continue
                collected.setdefault(u, None)

        # Method 3: Direct <img src>
        for img in driver.find_elements(By.CSS_SELECTOR, "img[src]"):
            src = img.get_attribute("src")
            if not src:
                continue
            if src.startswith("http") and "yandex" not in src and "yastatic" not in src:
                collected.setdefault(src, None)

    while time.time() - start < MAX_YANDEX_URLS_POLL_SECONDS and len(collected) < max_images:
        harvest()
        if len(collected) >= max_images:
            break
        if SCROLL_DURING_POLL:
            driver.execute_script("window.scrollBy(0, document.body.scrollHeight * 0.25);")
        time.sleep(YANDEX_URLS_POLL_INTERVAL)

    # Final harvest (in case incremental loaded late)
    if len(collected) < max_images:
        harvest()

    urls = list(collected.keys())[:max_images]
    logging.info(f"Collected {len(urls)} candidate image URLs.")
    return urls

# -----------------------
# Processing Single Image
# -----------------------
def process_image(driver,
                  image_path: str,
                  max_urls: int,
                  max_workers: int,
                  log_writer,
                  links_writer,
                  threshold: float):
    image_name = Path(image_path).name
    start = time.time()
    logging.info(f"Processing {image_name}")

    if not upload_to_yandex_and_navigate(driver, image_path):
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start:.2f}", 0, "Upload/navigation failed"])
        return None

    urls = get_yandex_image_urls(driver, max_images=max_urls)
    if not urls:
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start:.2f}", 0, "No URLs"])
        return None

    source_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img is None:
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start:.2f}", 0, "Source load fail"])
        return None

    best_match = None
    best_score = 0.0

    worker_count = min(max_workers, len(urls), THREADPOOL_OVERHEAD_CAP)

    def check(u: str):
        score, matches = calculate_flann_similarity(source_img, u)
        if links_writer:
            links_writer.writerow([image_name, u])
        return u, score, matches

    with ThreadPoolExecutor(max_workers=worker_count) as exe:
        futures = {exe.submit(check, u): u for u in urls}
        for fut in as_completed(futures):
            try:
                u, score, matches = fut.result()
            except Exception:
                continue
            if score > best_score and score >= threshold:
                best_score = score
                best_match = {
                    "source_image": image_name,
                    "matched_url": u,
                    "flann_score": score,
                    "num_matches": matches
                }

    elapsed = time.time() - start
    if best_match:
        if best_match['matched_url'].startswith("data:image"):
            try:
                _, b64data = best_match['matched_url'].split(",", 1)
                img_bytes = base64.b64decode(b64data)
                comp_path = create_comparison_image(
                    source_path=image_path,
                    similar_img_bytes=img_bytes,
                    score=best_match['flann_score'],
                    matches=best_match['num_matches']
                )
            except Exception:
                comp_path = None
        else:
            comp_path = download_and_create_comparison(
                url=best_match['matched_url'],
                source_image_path=image_path,
                flann_score=best_match['flann_score'],
                num_matches=best_match['num_matches']
            )
        best_match['comparison_path'] = comp_path
        move_image_to_done(image_path)
        if log_writer:
            log_writer.writerow([image_name, f"{elapsed:.2f}", 1, "Match saved"])
        logging.info(f"Match: {image_name} score={best_match['flann_score']:.4f}")
        return best_match
    else:
        if log_writer:
            log_writer.writerow([image_name, f"{elapsed:.2f}", 0, "No match >= threshold"])
        logging.info(f"No acceptable match for {image_name}")
        return None

# -----------------------
# Save Results
# -----------------------
def save_results(matches, filename: str):
    filtered = [m for m in matches if m]
    if not filtered:
        logging.info("No matches to write.")
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source Image", "Matched URL", "FLANN Score", "Feature Matches", "Comparison Image Path"])
        for m in filtered:
            w.writerow([
                m['source_image'],
                m['matched_url'],
                f"{m['flann_score']:.4f}",
                m['num_matches'],
                m.get('comparison_path', 'N/A')
            ])
    logging.info(f"Saved {len(filtered)} matches -> {filename}")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Optimized Yandex Visual Search & FLANN Matcher")
    parser.add_argument('-i', '--images_folder', default='images', help='Folder containing source images')
    parser.add_argument('-o', '--output_file', default='yandex_best_matches.csv', help='Output CSV file')
    parser.add_argument('-u', '--max_urls', type=int, default=20, help='Max result URLs to test per image')
    parser.add_argument('-w', '--max_workers', type=int, default=10, help='Max parallel workers for scoring')
    parser.add_argument('-d', '--delay', type=float, default=1.5, help='Delay (seconds) between images')
    parser.add_argument('-t', '--threshold', type=float, default=0.2, help='Minimum similarity threshold')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    logging.info("=== Yandex FLANN Matcher (Optimized) Start ===")
    create_folders()

    global requests_session
    requests_session = init_requests_session()

    images = get_image_files(args.images_folder)
    if not images:
        logging.warning(f"No images in {args.images_folder}. Exiting.")
        return

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"yandex_processing_log_{run_ts}.csv"
    links_file_name = f"yandex_scraped_links_{run_ts}.csv"

    try:
        driver = init_driver()
    except Exception:
        logging.critical("Driver initialization failed.")
        return

    all_matches = []
    total = len(images)

    try:
        with open(log_file_name, "w", newline="", encoding="utf-8") as lf, \
             open(links_file_name, "w", newline="", encoding="utf-8") as linkf:

            log_writer = csv.writer(lf)
            log_writer.writerow(["Image Name", "Processing Time (s)", "Match Found", "Notes"])

            links_writer = csv.writer(linkf)
            links_writer.writerow(["Source Image", "Scraped URL"])

            for idx, img_path in enumerate(images, 1):
                logging.info(f"[{idx}/{total}] {Path(img_path).name}")
                result = process_image(
                    driver=driver,
                    image_path=img_path,
                    max_urls=args.max_urls,
                    max_workers=args.max_workers,
                    log_writer=log_writer,
                    links_writer=links_writer,
                    threshold=args.threshold
                )
                if result:
                    all_matches.append(result)
                if idx < total:
                    time.sleep(args.delay)

        save_results(all_matches, args.output_file)
        logging.info(f"Processed {total} images. Matches: {len([m for m in all_matches if m])}")
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
    except Exception as e:
        logging.critical(f"Unexpected error: {e}", exc_info=True)
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        # Clean temp profile folder
        profile_dir = getattr(driver, "_custom_profile_dir", None)
        if profile_dir and os.path.isdir(profile_dir):
            shutil.rmtree(profile_dir, ignore_errors=True)
        logging.info("Cleanup complete.")
        logging.info("=== Finished ===")

if __name__ == "__main__":
    main()