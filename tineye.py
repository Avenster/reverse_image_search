from __future__ import annotations
import os
import time
import csv
import base64
import shutil
import argparse
import logging
import random
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------ CONFIG DEFAULTS ------------------
TIN_EYE_URL = "https://tineye.com/"
SUPPORTED_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

DEFAULT_THRESHOLD = 0.2
DEFAULT_MAX_URLS = 30
DEFAULT_MAX_WORKERS = 10
DEFAULT_DELAY_BETWEEN = 1.5
DEFAULT_UPLOAD_TIMEOUT = 40
DEFAULT_POST_UPLOAD_WAIT = 8.0
DEFAULT_ORB_FEATURES = 2000
DEFAULT_MIN_MATCHES = 8
LOWE_RATIO = 0.75
THREADPOOL_LIMIT = 24
IMAGE_FETCH_TIMEOUT = 5
REQUESTS_POOL = 80
REQUESTS_RETRIES = 1
REQUESTS_BACKOFF = 0.05

# NEW extraction behavior constants
RESULT_IMG_SELECTOR = "img[data-test='result-image'][src*='img.tineye.com/result/']"
RESULT_IMG_WAIT_TIMEOUT = 25             
RESULT_SCROLL_STEPS_MAX = 12             
RESULT_SCROLL_INCREMENT = 0.75         
RESULT_STABLE_ITERATIONS = 3            
RESULT_LOOP_SLEEP = (0.35, 0.6)         
REGEX_RESULT_URL = re.compile(
    r"https://img\.tineye\.com/result/[0-9a-f]{32,128}(?:-\d+)?\?size=\d+",
    re.IGNORECASE
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) Chrome/122.0.0.0 Safari/537.36"
]

# ------------------ LOGGING ------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ------------------ GLOBALS ------------------
requests_session: Optional[requests.Session] = None
orb_detector = cv2.ORB_create(nfeatures=DEFAULT_ORB_FEATURES)

# ------------------ SESSION / DRIVER ------------------
def random_user_agent() -> str:
    return random.choice(USER_AGENTS)

def init_requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=REQUESTS_RETRIES,
        backoff_factor=REQUESTS_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(pool_connections=REQUESTS_POOL, pool_maxsize=REQUESTS_POOL, max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": random_user_agent(),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8"
    })
    return s

def init_driver(profile_dir: Optional[str]):
    options = uc.ChromeOptions()
    options.page_load_strategy = "eager"
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--window-size=1920,1080")
    # persist profile
    if profile_dir:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        options.add_argument(f"--user-data-dir={profile_dir}")
        options.add_argument("--profile-directory=Default")
    driver = uc.Chrome(options=options)
    try:
        driver.execute_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    except Exception:
        pass
    driver.set_page_load_timeout(60)
    return driver

# ------------------ FOLDERS ------------------
def create_folders():
    for d in ("done", "similar_images"):
        Path(d).mkdir(exist_ok=True)

def move_image_to_done(image_path: str):
    try:
        target = Path("done")
        target.mkdir(exist_ok=True)
        src = Path(image_path)
        dst = target / src.name
        if dst.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = target / f"{src.stem}_{ts}{src.suffix}"
        shutil.move(str(src), str(dst))
    except Exception as e:
        logging.error(f"Move failed ({image_path}): {e}")

# ------------------ IMAGE LIST ------------------
def get_image_files(folder: str) -> List[str]:
    p = Path(folder)
    if not p.is_dir():
        logging.error(f"Images folder not found: {folder}")
        return []
    files: List[str] = []
    for pattern in SUPPORTED_EXTS:
        files.extend(p.glob(pattern))
        files.extend(p.glob(pattern.upper()))
    return sorted({str(f) for f in files})

# ------------------ NETWORK HELPERS ------------------
def fetch_image_bytes(url: str, timeout: int = IMAGE_FETCH_TIMEOUT) -> Optional[bytes]:
    global requests_session
    sess = requests_session or requests
    try:
        r = sess.get(url, timeout=timeout, verify=False)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        return None
    return None

def get_image_from_url_or_base64(ref: str):
    if ref.startswith("data:image"):
        try:
            _, b64d = ref.split(",", 1)
            raw = base64.b64decode(b64d)
            arr = np.frombuffer(raw, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
    raw = fetch_image_bytes(ref)
    if not raw:
        return None
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

# ------------------ FLANN SIMILARITY ------------------
def calculate_flann_similarity(src_img, candidate_url: str) -> Tuple[float, int]:
    try:
        if src_img is None:
            return 0.0, 0
        img2 = get_image_from_url_or_base64(candidate_url)
        if img2 is None:
            return 0.0, 0

        kp1, des1 = orb_detector.detectAndCompute(src_img, None)
        kp2, des2 = orb_detector.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0, 0

        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=40)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < LOWE_RATIO * n.distance:
                    good.append(m)
        if len(good) > DEFAULT_MIN_MATCHES:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = int(mask.sum())
                score = inliers / len(good)
                return min(score, 1.0), inliers
        return 0.0, 0
    except Exception:
        return 0.0, 0

# ------------------ COMPARISON IMAGE ------------------
def create_comparison_image(source_path: str,
                            similar_bytes: bytes,
                            score: float,
                            matches: int,
                            out_dir: str = "similar_images") -> Optional[str]:
    try:
        original = cv2.imread(source_path)
        if original is None:
            return None
        arr = np.frombuffer(similar_bytes, np.uint8)
        similar = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if similar is None:
            return None

        target_h = 600
        def resize_h(img):
            h, w = img.shape[:2]
            scale = target_h / h
            return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)
        o_r = resize_h(original)
        s_r = resize_h(similar)

        header_h = 70
        canvas = np.full((target_h + header_h, o_r.shape[1] + s_r.shape[1], 3), 255, dtype=np.uint8)
        canvas[header_h:, :o_r.shape[1]] = o_r
        canvas[header_h:, o_r.shape[1]:] = s_r

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.8
        th = 2
        def center_text(txt, x0, width):
            (tw, th_txt), _ = cv2.getTextSize(txt, font, fs, th)
            x = x0 + (width - tw)//2
            y = (header_h - th_txt)//2 + th_txt
            cv2.putText(canvas, txt, (x, y), font, fs, (0,0,0), th, cv2.LINE_AA)

        center_text("Original", 0, o_r.shape[1])
        center_text(f"TinEye | Score: {score:.4f} | Matches: {matches}", o_r.shape[1], s_r.shape[1])

        Path(out_dir).mkdir(exist_ok=True)
        stem = Path(source_path).stem
        out_path = Path(out_dir) / f"{stem}_comparison_{score:.4f}.jpg"
        if out_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(out_dir) / f"{stem}_comparison_{score:.4f}_{ts}.jpg"
        cv2.imwrite(str(out_path), canvas)
        return str(out_path)
    except Exception as e:
        logging.error(f"Comparison image error: {e}")
        return None

# ------------------ UPLOAD & PAGE PARSING ------------------
TOO_SIMPLE_PHRASE = "your image is too simple to find matches"

def upload_image(driver, image_path: str, manual_confirm: bool, upload_timeout: int, post_wait: float) -> bool:
    try:
        driver.get(TIN_EYE_URL)
    except Exception:
        return False

    if manual_confirm:
        logging.info("If a CAPTCHA is present, solve it now. Press ENTER to continue...")
        try:
            input()
        except EOFError:
            pass

    selectors = [
        "input#upload-box[type='file']",
        "input[type='file'][name='image']",
        "input[type='file']"
    ]
    file_input = None
    for sel in selectors:
        try:
            file_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, sel))
            )
            if file_input:
                break
        except TimeoutException:
            continue
    if not file_input:
        logging.error("File input not found.")
        return False

    try:
        file_input.send_keys(os.path.abspath(image_path))
    except Exception as e:
        logging.error(f"Failed to send file path: {e}")
        return False

    logging.info(f"Waiting {post_wait}s for TinEye to process...")
    time.sleep(post_wait)
    return True

def parse_match_count(driver) -> Tuple[int, bool]:
    try:
        page_text = driver.page_source.lower()
    except Exception:
        return 0, False

    if TOO_SIMPLE_PHRASE in page_text:
        return 0, True

    m = re.search(r'(\d+)\s+match(?:es)?', page_text)
    if m:
        return int(m.group(1)), False
    return 0, False

# --------------- UPDATED EXTRACTION FUNCTION ---------------
def extract_result_image_urls(driver, max_urls: int) -> List[str]:
    """
    Robustly gather up to max_urls result thumbnail URLs.

    Strategy:
      1. Wait (up to RESULT_IMG_WAIT_TIMEOUT) for at least one result image OR exit early if none.
      2. Iteratively scroll & collect:
         - CSS selection: RESULT_IMG_SELECTOR
         - Keep order, normalize duplicates (strip query string for uniqueness).
      3. If still fewer than requested, run a regex scan on the HTML.
      4. Return list limited to max_urls in first-seen order.
    """
    start_wait = time.time()
    first_found = False
    while time.time() - start_wait < RESULT_IMG_WAIT_TIMEOUT:
        imgs = driver.find_elements(By.CSS_SELECTOR, RESULT_IMG_SELECTOR)
        if imgs:
            first_found = True
            break
        time.sleep(0.4)

    if not first_found:
        logging.info("No result images detected within initial wait window.")
        return []

    collected_order: List[str] = []
    seen_base = set()

    def add_urls_from_elements(elements):
        for el in elements:
            if len(collected_order) >= max_urls:
                break
            try:
                src = el.get_attribute("src")
                if not src or "img.tineye.com/result" not in src:
                    continue
                base = src.split("?")[0]  # normalization key
                if base not in seen_base:
                    seen_base.add(base)
                    collected_order.append(src)
            except Exception:
                continue

    stable_count = 0
    prev_len = 0
    scroll_steps = 0

    while len(collected_order) < max_urls and scroll_steps < RESULT_SCROLL_STEPS_MAX and stable_count < RESULT_STABLE_ITERATIONS:
        elems = driver.find_elements(By.CSS_SELECTOR, RESULT_IMG_SELECTOR)
        add_urls_from_elements(elems)

        current_len = len(collected_order)
        if current_len == prev_len:
            stable_count += 1
        else:
            stable_count = 0
            prev_len = current_len

        if len(collected_order) >= max_urls:
            break

        # scroll
        driver.execute_script(
            "window.scrollBy(0, Math.max(200, window.innerHeight * arguments[0]));",
            RESULT_SCROLL_INCREMENT
        )
        scroll_steps += 1
        time.sleep(random.uniform(*RESULT_LOOP_SLEEP))

    if len(collected_order) < max_urls:
        # Regex fallback for any missed images in raw HTML
        html = driver.page_source
        regex_hits = REGEX_RESULT_URL.findall(html)
        for url in regex_hits:
            if len(collected_order) >= max_urls:
                break
            base = url.split("?")[0]
            if base not in seen_base:
                seen_base.add(base)
                collected_order.append(url)

    logging.info(f"Collected {len(collected_order)} result image URLs (requested max {max_urls}).")
    return collected_order[:max_urls]

# ------------------ PROCESS SINGLE IMAGE ------------------
def process_image(driver,
                  image_path: str,
                  threshold: float,
                  max_urls: int,
                  max_workers: int,
                  manual_confirm: bool,
                  upload_timeout: int,
                  post_wait: float,
                  log_writer,
                  links_writer):

    image_name = Path(image_path).name
    start = time.time()
    logging.info(f"Processing {image_name}")

    if not upload_image(driver, image_path, manual_confirm, upload_timeout, post_wait):
        log_writer.writerow([image_name, f"{time.time()-start:.2f}", 0, "Upload failed"])
        return None

    match_count, too_simple = parse_match_count(driver)
    if too_simple:
        logging.info("Skipped (too simple).")
        log_writer.writerow([image_name, f"{time.time()-start:.2f}", 0, "Too simple"])
        return None

    if match_count == 0:
        logging.info("Skipped (0 matches).")
        log_writer.writerow([image_name, f"{time.time()-start:.2f}", 0, "0 matches"])
        return None

    urls = extract_result_image_urls(driver, max_urls=min(match_count, max_urls))
    if not urls:
        log_writer.writerow([image_name, f"{time.time()-start:.2f}", 0, "No URLs"])
        return None

    for u in urls:
        links_writer.writerow([image_name, u])

    src_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        log_writer.writerow([image_name, f"{time.time()-start:.2f}", 0, "Source read fail"])
        return None

    best_match = None
    best_score = 0.0
    worker_count = min(max_workers, len(urls), THREADPOOL_LIMIT)

    def score(u: str):
        s, m = calculate_flann_similarity(src_img, u)
        return u, s, m

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(score, u): u for u in urls}
        for fut in as_completed(futures):
            try:
                u, s, m = fut.result()
            except Exception:
                continue
            if s > best_score and s >= threshold:
                best_score = s
                best_match = dict(
                    source_image=image_name,
                    matched_url=u,
                    flann_score=s,
                    num_matches=m
                )

    elapsed = time.time() - start
    if best_match:
        data = fetch_image_bytes(best_match['matched_url'])
        comp_path = None
        if data:
            comp_path = create_comparison_image(
                source_path=image_path,
                similar_bytes=data,
                score=best_match['flann_score'],
                matches=best_match['num_matches']
            )
        best_match['comparison_path'] = comp_path
        move_image_to_done(image_path)
        log_writer.writerow([image_name, f"{elapsed:.2f}", 1, "Match saved"])
        logging.info(f"Match found: {image_name} score={best_match['flann_score']:.4f}")
        return best_match
    else:
        log_writer.writerow([image_name, f"{elapsed:.2f}", 0, "No match >= threshold"])
        logging.info(f"No match above threshold for {image_name}")
        return None

# ------------------ SAVE RESULTS ------------------
def save_results(matches: List[Dict[str, Any]], outfile: str):
    valid = [m for m in matches if m]
    if not valid:
        logging.info("No matches to save.")
        return
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source Image", "Matched URL", "FLANN Score", "Feature Matches", "Comparison Image Path"])
        for m in valid:
            w.writerow([
                m['source_image'],
                m['matched_url'],
                f"{m['flann_score']:.4f}",
                m['num_matches'],
                m.get('comparison_path', 'N/A')
            ])
    logging.info(f"Saved {len(valid)} matches -> {outfile}")

# ------------------ MAIN ------------------
def main():
    ap = argparse.ArgumentParser(description="TinEye Visual Search + FLANN (Hybrid Improved URL Extraction)")
    ap.add_argument("-i", "--images-folder", default="images", help="Source images folder")
    ap.add_argument("-o", "--output-file", default="tineye_best_matches.csv", help="Results CSV")
    ap.add_argument("-u", "--max-urls", type=int, default=DEFAULT_MAX_URLS, help="Max candidate URLs per image")
    ap.add_argument("-w", "--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel similarity workers")
    ap.add_argument("-d", "--delay", type=float, default=DEFAULT_DELAY_BETWEEN, help="Delay between images (s)")
    ap.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD, help="Similarity threshold (0-1)")
    ap.add_argument("--profile-dir", default=os.path.join(os.path.expanduser("~"), ".tineye_uc_profile"),
                    help="Persistent Chrome user profile directory")
    ap.add_argument("--no-persistent-profile", action="store_true",
                    help="Use ephemeral session instead of persistent profile")
    ap.add_argument("--manual-confirm", action="store_true",
                    help="Pause for manual CAPTCHA confirmation after loading TinEye home each image")
    ap.add_argument("--post-upload-wait", type=float, default=DEFAULT_POST_UPLOAD_WAIT,
                    help="Seconds to wait after upload before parsing matches (coarse wait)")
    ap.add_argument("--upload-timeout", type=int, default=DEFAULT_UPLOAD_TIMEOUT,
                    help="Max seconds allowed for general upload step (not strict)")
    ap.add_argument("--log-level", default="INFO", help="Logging level")
    args = ap.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    logging.info("=== TinEye Hybrid Matcher (Improved URL Extraction) Start ===")
    create_folders()

    global requests_session
    requests_session = init_requests_session()

    images = get_image_files(args.images_folder)
    if not images:
        logging.warning(f"No images in {args.images_folder}. Exiting.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"tineye_processing_log_{ts}.csv"
    links_file = f"tineye_scraped_links_{ts}.csv"

    profile_dir = None if args.no_persistent_profile else args.profile_dir

    try:
        driver = init_driver(profile_dir=profile_dir)
    except WebDriverException as e:
        logging.critical(f"Driver init failed: {e}")
        return

    matches: List[Dict[str, Any]] = []
    total = len(images)

    try:
        with open(log_file, "w", newline="", encoding="utf-8") as lf, \
             open(links_file, "w", newline="", encoding="utf-8") as linkf:

            log_writer = csv.writer(lf)
            log_writer.writerow(["Image Name", "Processing Time (s)", "Match Found", "Notes"])

            links_writer = csv.writer(linkf)
            links_writer.writerow(["Source Image", "Result Image URL"])

            for idx, img in enumerate(images, 1):
                logging.info(f"[{idx}/{total}] {Path(img).name}")
                result = process_image(
                    driver=driver,
                    image_path=img,
                    threshold=args.threshold,
                    max_urls=args.max_urls,
                    max_workers=args.max_workers,
                    manual_confirm=args.manual_confirm,
                    upload_timeout=args.upload_timeout,
                    post_wait=args.post_upload_wait,
                    log_writer=log_writer,
                    links_writer=links_writer
                )
                if result:
                    matches.append(result)
                if idx < total:
                    time.sleep(args.delay)

        save_results(matches, args.output_file)
        logging.info(f"Processed {total} images. Matches: {len([m for m in matches if m])}")
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
        logging.info("=== Finished ===")

if __name__ == "__main__":
    main()