import os
import csv
import time
import re
import sys
import json
import random
import argparse
import threading
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --------------------------------------------------------------------
# Default Configuration (overridable by CLI flags)
# --------------------------------------------------------------------
CSV_FILE = "data.csv"
OUTPUT_ROOT = "downloaded_images"
PAGE_LOAD_TIMEOUT = 18
REQUEST_TIMEOUT = 25
RETRY_COUNT = 2
SAVE_METADATA_JSON = True
MANIFEST_FILE = "manifest.json"

NUM_DRIVERS = 2
INITIAL_MANUAL_WAIT = 4.0          # seconds when manual wait triggers
MANUAL_WAIT_MODE = "first"         # every | first | none
DYNAMIC_WAIT_TIMEOUT = 1.0
POLITE_DELAY = 0.15
USE_RANDOM_DELAYS = True
LOG_VERBOSE = True
HEADLESS = False
FAST_MODE = False

MAX_DOWNLOAD_WORKERS = 6
SHOW_QUEUE_PROGRESS = True

# Flickr extra logic
FLICKR_TRY_ALT_SIZES_ON_FAIL = True
FLICKR_UPSIZE_ORDER = ["_o","_k","_h","_b","_c","_z",""]  # biggest to smaller ('' = no suffix)
UPSIZE_FLICKR_ALWAYS = False   # If True, attempt larger candidates immediately (may cause many 404s)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def jitter(base):
    if not USE_RANDOM_DELAYS:
        return base
    return base + random.uniform(0.05, 0.25)

def sanitize(text: str) -> str:
    if text is None:
        return "unknown"
    text = text.strip()
    text = re.sub(r'[\\/:*?"<>|]+', '_', text)
    text = re.sub(r'\s+', ' ', text)
    return text[:180]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def init_driver(driver_id: int):
    opts = uc.ChromeOptions()
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,1600")
    opts.add_argument("--lang=en-US,en")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    driver = uc.Chrome(options=opts)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    try:
        driver.get("https://www.instagram.com/")
        driver.add_cookie({"name": "ig_nrcb", "value": "1", "domain": ".instagram.com"})
    except Exception:
        pass
    if LOG_VERBOSE:
        print(f"[Driver {driver_id}] Initialized.")
    return driver

# --------------------------------------------------------------------
# Instagram extraction (STRICT DOM first, then fallback JSON approach)
# --------------------------------------------------------------------
def pick_highest_from_srcset(srcset: str):
    best_url, best_w = None, -1
    for part in srcset.split(','):
        part = part.strip()
        if ' ' in part:
            url_part, size_part = part.rsplit(' ', 1)
            try:
                w = int(size_part.rstrip('w'))
            except ValueError:
                w = 0
        else:
            url_part = part
            w = 0
        if w > best_w:
            best_w = w
            best_url = url_part
    return best_url

def is_cropping_segment(seg: str):
    return bool(re.fullmatch(r'c\d+\.\d+\.\d+\.\d+a', seg))

def is_size_segment(seg: str):
    return bool(re.fullmatch(r's\d+x\d+', seg))

def try_uncrop_url(url: str):
    try:
        parts = urlparse(url)
        path_parts = [p for p in parts.path.split('/') if p]
        cleaned_parts = [p for p in path_parts if not is_cropping_segment(p) and not is_size_segment(p)]
        if cleaned_parts == path_parts:
            return url
        new_path = '/' + '/'.join(cleaned_parts)
        return parts._replace(path=new_path).geturl()
    except Exception:
        return url

def extract_og_image(driver):
    try:
        metas = driver.find_elements(By.CSS_SELECTOR, "meta[property='og:image']")
        for m in metas:
            c = m.get_attribute("content")
            if c and "instagram" in c:
                return c
    except Exception:
        pass
    return None

def extract_instagram_candidates(driver):
    try:
        html = driver.page_source
    except Exception:
        return []
    candidates = []

    # "display_resources"
    for block in re.finditer(r'"display_resources":\s*\[(.+?)\]', html, re.DOTALL):
        inner = block.group(1)
        for m in re.finditer(r'"src":"([^"]+)".+?"config_width":(\d+)', inner):
            url = (m.group(1).encode()
                   .decode('unicode_escape')
                   .replace("\\u0026", "&").replace("\\/", "/"))
            width = int(m.group(2))
            hmatch = re.search(r'"config_height":(\d+)', inner)
            height = int(hmatch.group(1)) if hmatch else None
            candidates.append((url, width, height, "display_resources"))

    # "image_versions2"
    for block in re.finditer(r'"image_versions2":\s*\{.*?"candidates":\s*\[(.+?)\]', html, re.DOTALL):
        inner = block.group(1)
        for m in re.finditer(r'"url":"([^"]+)".+?"width":(\d+),"height":(\d+)', inner):
            url = (m.group(1).encode()
                   .decode('unicode_escape')
                   .replace("\\u0026", "&").replace("\\/", "/"))
            width = int(m.group(2)); height = int(m.group(3))
            candidates.append((url, width, height, "image_versions2"))

    # generic "candidates"
    for block in re.finditer(r'"candidates":\s*\[(.+?)\]', html, re.DOTALL):
        inner = block.group(1)
        for m in re.finditer(r'"url":"([^"]+)".+?"width":(\d+),"height":(\d+)', inner):
            url = (m.group(1).encode()
                   .decode('unicode_escape')
                   .replace("\\u0026", "&").replace("\\/", "/"))
            width = int(m.group(2)); height = int(m.group(3))
            candidates.append((url, width, height, "candidates_generic"))

    seen = set()
    uniq = []
    for url, w, h, tag in candidates:
        tup = (url, w, h)
        if tup not in seen:
            seen.add(tup)
            uniq.append((url, w, h, tag))
    candidates = uniq
    if not candidates:
        return []

    def is_cropped_url(u):
        path_parts = [p for p in urlparse(u).path.split('/') if p]
        return any(is_cropping_segment(p) for p in path_parts)

    non_cropped = [c for c in candidates if not is_cropped_url(c[0])]
    pool = non_cropped if non_cropped else candidates

    def area_key(c):
        _, w, h, _ = c
        if w and h:
            return w * h
        return w or 0

    pool.sort(key=area_key, reverse=True)
    return pool

def fetch_instagram_dom_strict(driver, row_num):
    """
    ONLY look for images inside div._aagv (as you requested).
    If multiple (carousel), pick the one with largest width (from srcset or natural attributes)
    """
    try:
        wrappers = driver.find_elements(By.CSS_SELECTOR, "div._aagv img")
        best = None
        best_w = -1
        for img in wrappers:
            try:
                srcset = img.get_attribute("srcset")
                if srcset:
                    url = pick_highest_from_srcset(srcset)
                    # Heuristic get width from largest descriptor
                    width_candidates = re.findall(r'\s(\d+)w', srcset)
                    if width_candidates:
                        w = max(int(x) for x in width_candidates)
                    else:
                        w = int(img.get_attribute("width") or 0)
                else:
                    url = img.get_attribute("src")
                    w = int(img.get_attribute("width") or 0)
                if url and w >= best_w:
                    best_w = w
                    best = (url, w, None, "insta_dom_strict")
            except Exception:
                continue
        if best:
            if LOG_VERBOSE:
                print(f"[Row {row_num}] STRICT div._aagv selected {best[0]} (w={best[1]})")
            return best
    except Exception:
        pass
    return None

def fetch_instagram_best(driver, row_num):
    # First: strict DOM inside div._aagv
    strict = fetch_instagram_dom_strict(driver, row_num)
    if strict:
        return strict

    # Fallback: JSON candidates
    candidates = extract_instagram_candidates(driver)
    if candidates:
        best_url, w, h, tag = candidates[0]
        if LOG_VERBOSE:
            print(f"[Row {row_num}] fallback JSON via {tag}: {w}x{h}")
        return best_url, w, h, tag

    # DOM generic fallback
    imgs = driver.find_elements(By.XPATH, "//img[contains(@src,'fbcdn') or contains(@src,'instagram')]")
    dom_best = None
    best_len = -1
    for img in imgs:
        try:
            ss = img.get_attribute("srcset")
            u = pick_highest_from_srcset(ss) if ss else img.get_attribute("src")
            if u and len(u) > best_len:
                best_len = len(u)
                dom_best = u
        except Exception:
            continue
    if dom_best:
        if LOG_VERBOSE:
            print(f"[Row {row_num}] generic DOM fallback.")
        return dom_best, None, None, "dom"

    # OG fallback (often square)
    og = extract_og_image(driver)
    if og:
        if LOG_VERBOSE:
            print(f"[Row {row_num}] og:image fallback.")
        return og, None, None, "og"

    return None, None, None, None

def infer_extension_from_url(url: str):
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if not ext or ext in ('.php', '.ashx'):
        return '.jpg'
    if len(ext) > 6:
        return '.jpg'
    return ext

# --------------------------------------------------------------------
# Flickr handling
# --------------------------------------------------------------------
def fetch_flickr_best(driver, row_num):
    """
    Extract main-photo from Flickr DOM.
    Example snippet you provided has: img.main-photo with //live.staticflickr.com/...
    Return (url,width,height,source_tag)
    """
    try:
        # wait quickly (optional): already have general dynamic wait outside; keep minimal here
        img_el = driver.find_element(By.CSS_SELECTOR, ".photo-well-media-scrappy-view img.main-photo")
        src = img_el.get_attribute("src")
        if src.startswith("//"):
            src = "https:" + src
        width = img_el.get_attribute("width")
        height = img_el.get_attribute("height")
        try:
            width = int(width) if width else None
            height = int(height) if height else None
        except Exception:
            width = None; height = None
        if LOG_VERBOSE:
            print(f"[Row {row_num}] Flickr main-photo DOM found ({width}x{height})")
        return src, width, height, "flickr_dom"
    except Exception:
        return None, None, None, None

def expand_flickr_alternates(url: str):
    """
    Given a flickr static image URL like:
      https://live.staticflickr.com/server/id_secret_b.jpg
    Return alternate size URLs in preference order (excluding original current size).
    Only manipulates final '_{size}.EXT' part.
    """
    try:
        path = urlparse(url).path
        base, ext = os.path.splitext(path)
        # Pattern ends with _<sizecode> maybe
        m = re.match(r'(.+_[0-9a-fA-F]+)_([a-z])$', base)  # e.g., .../25956002883_409e9bec1a_b
        # Some Flickr sizes are two chars like _sq, _q, but main large ones are single letters.
        # We'll do a more lenient pattern:
        if not m:
            m2 = re.match(r'(.+_[0-9a-fA-F]+)_([a-z]{1,2})$', base)
        else:
            m2 = None
        size_code = None
        core = None
        if m:
            core = m.group(1)
            size_code = "_" + m.group(2)
        elif m2:
            core = m2.group(1)
            size_code = "_" + m2.group(2)

        if core is None:
            # maybe no size suffix; try adding bigger suffix
            core = base
            size_code = ""

        current = size_code

        alts = []
        for candidate in FLICKR_UPSIZE_ORDER:
            if candidate != current:
                alts.append(core + candidate + ext)
        # Rebuild full URLs
        prefix = url[:url.index(path)]
        return [prefix + p for p in alts]
    except Exception:
        return []

# --------------------------------------------------------------------
# Download Manager (async) with Flickr logic
# --------------------------------------------------------------------
class DownloadManager:
    def __init__(self, max_workers, manifest, manifest_lock):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.manifest = manifest
        self.lock = manifest_lock
        self.futures = []
        self.session_local = threading.local()

    def _get_session(self):
        if not hasattr(self.session_local, "s"):
            s = requests.Session()
            s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            self.session_local.s = s
        return self.session_local.s

    def submit(self, job):
        fut = self.executor.submit(self._download_job, job)
        self.futures.append(fut)

    def _try_download_once(self, session, url, headers, dest_file):
        try:
            r = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)
            ctype = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "image" in ctype:
                with open(dest_file, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                return True
        except Exception:
            pass
        return False

    def _download_job(self, job):
        row_num = job["row_num"]
        page_url = job["page_url"]
        platform = job["platform"]
        dest_file = Path(job["dest_file"]) if job["dest_file"] else None
        image_url = job["image_url"]
        width = job.get("width")
        height = job.get("height")
        source_tag = job.get("source_tag")
        status = "failed"
        final_url = image_url.replace("&amp;", "&") if image_url else ""

        if not image_url:
            status = "no_image"
        elif dest_file and dest_file.exists():
            status = "exists"
        else:
            if dest_file:
                ensure_dir(dest_file.parent)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": page_url,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9"
            }
            session = self._get_session()

            primary_attempts = []
            if platform == "flickr" and UPSIZE_FLICKR_ALWAYS:
                # Try larger sizes first based on alternates (largest -> ... -> original)
                alternates = expand_flickr_alternates(final_url)
                # Put current final_url at END so we start bigger
                primary_attempts = alternates + [final_url]
            else:
                primary_attempts = [final_url]

            success = False
            for url_candidate in primary_attempts:
                for attempt in range(1, RETRY_COUNT + 1):
                    if self._try_download_once(session, url_candidate, headers, dest_file):
                        status = "downloaded"
                        final_url = url_candidate
                        success = True
                        break
                if success:
                    break

            # Instagram uncrop fallback
            if not success and platform == "instagram":
                alt_uncrop = try_uncrop_url(final_url)
                if alt_uncrop != final_url:
                    if self._try_download_once(session, alt_uncrop, headers, dest_file):
                        status = "downloaded"
                        final_url = alt_uncrop
                        success = True

            # Flickr alternate fallback on failure (only if not already tried all upsizes always)
            if not success and platform == "flickr" and FLICKR_TRY_ALT_SIZES_ON_FAIL and not UPSIZE_FLICKR_ALWAYS:
                alternates = expand_flickr_alternates(final_url)
                for aurl in alternates:
                    if self._try_download_once(session, aurl, headers, dest_file):
                        status = "downloaded"
                        final_url = aurl
                        success = True
                        break

            if not success and status != "exists":
                status = "failed"

        if LOG_VERBOSE:
            print(f"[DL | Row {row_num}] {status.upper()} -> {dest_file if dest_file else 'N/A'}")

        entry = {
            "row": row_num,
            "platform": platform,
            "page_url": page_url,
            "image_url": final_url,
            "status": status,
            "file": str(dest_file) if dest_file else "",
            "width": width,
            "height": height,
            "source_tag": source_tag
        }
        with self.lock:
            self.manifest.append(entry)
        return entry

    def wait_finish(self):
        for fut in as_completed(self.futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[DownloadManager] Error: {e}")
        self.executor.shutdown(wait=True)

# --------------------------------------------------------------------
# Row Processing Flow
# --------------------------------------------------------------------
def should_manual_wait(driver_context):
    if MANUAL_WAIT_MODE == "none":
        return False
    if MANUAL_WAIT_MODE == "every":
        return True
    if MANUAL_WAIT_MODE == "first":
        return driver_context["pages_seen"] == 0
    return False

def process_row_fast(driver, row, row_num, driver_id, driver_context, download_manager):
    l1 = sanitize(row.get("L1", "Unknown"))
    l2 = sanitize(row.get("L2", "Unknown"))
    raw_name = row.get("Image_Names") or row.get("Image_Name") or f"image_{row_num}"
    image_name = sanitize(raw_name)
    page_url = row.get("Images_Link") or row.get("Image_Link") or row.get("link")

    if not page_url:
        print(f"[Driver {driver_id} | Row {row_num}] Missing URL.")
        return

    print(f"[Driver {driver_id} | Row {row_num}] Visiting {page_url}")
    t_nav_start = time.time()
    try:
        driver.get(page_url)
    except Exception as e:
        print(f"[Driver {driver_id} | Row {row_num}] Navigation error: {e}")
        return
    nav_time = time.time() - t_nav_start

    if should_manual_wait(driver_context):
        if LOG_VERBOSE:
            print(f"[Driver {driver_id} | Row {row_num}] Manual wait {INITIAL_MANUAL_WAIT:.2f}s.")
        time.sleep(INITIAL_MANUAL_WAIT)

    # Wait quickly for either Instagram or Flickr image target
    try:
        WebDriverWait(driver, DYNAMIC_WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.XPATH,
                 "//img[contains(@src,'fbcdn') or contains(@src,'instagram') or contains(@class,'main-photo')]")
            )
        )
    except Exception:
        pass

    lower = page_url.lower()
    platform = None
    best_url = None
    w = h = None
    tag = None

    if "instagram.com" in lower:
        platform = "instagram"
        best_url, w, h, tag = fetch_instagram_best(driver, row_num)
    elif "flickr.com" in lower:
        platform = "flickr"
        best_url, w, h, tag = fetch_flickr_best(driver, row_num)
    else:
        platform = "unknown"

    if platform in ("instagram", "flickr") and best_url:
        ext = infer_extension_from_url(best_url)
        dest_dir = Path(OUTPUT_ROOT) / l1 / l2
        ensure_dir(dest_dir)
        dest_file = dest_dir / f"{image_name}{ext}"
        print(f"[Driver {driver_id} | Row {row_num}] QUEUED ({platform}:{tag}) -> {best_url} (nav {nav_time:.2f}s)")
        download_manager.submit({
            "row_num": row_num,
            "page_url": page_url,
            "platform": platform,
            "image_url": best_url,
            "dest_file": str(dest_file),
            "width": w,
            "height": h,
            "source_tag": tag
        })
    else:
        print(f"[Driver {driver_id} | Row {row_num}] No image or unsupported platform.")
        download_manager.submit({
            "row_num": row_num,
            "page_url": page_url,
            "platform": platform,
            "image_url": "",
            "dest_file": "",
            "width": None,
            "height": None,
            "source_tag": "none"
        })

    driver_context["pages_seen"] += 1

# --------------------------------------------------------------------
# Driver Worker
# --------------------------------------------------------------------
def driver_worker(driver_id, rows_slice, index_slice, download_manager):
    driver = init_driver(driver_id)
    driver_context = {"pages_seen": 0}
    try:
        for local_idx, row in enumerate(rows_slice, start=1):
            global_row_number = index_slice[local_idx - 1]
            process_row_fast(driver, row, global_row_number, driver_id, driver_context, download_manager)
            if POLITE_DELAY > 0:
                time.sleep(jitter(POLITE_DELAY))
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        print(f"[Driver {driver_id}] Closed.")

# --------------------------------------------------------------------
# CSV Reading
# --------------------------------------------------------------------
def read_csv_rows(csv_file):
    with open(csv_file, newline='', encoding='utf-8') as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
        except csv.Error:
            f.seek(0)
            reader = csv.DictReader(f)
        for r in reader:
            yield r

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    global CSV_FILE, NUM_DRIVERS, HEADLESS, INITIAL_MANUAL_WAIT, MANUAL_WAIT_MODE
    global DYNAMIC_WAIT_TIMEOUT, POLITE_DELAY, USE_RANDOM_DELAYS, FAST_MODE
    global MAX_DOWNLOAD_WORKERS, UPSIZE_FLICKR_ALWAYS, FLICKR_TRY_ALT_SIZES_ON_FAIL

    parser = argparse.ArgumentParser(description="Ultra-fast parallel Instagram + Flickr image downloader.")
    parser.add_argument("--csv", default=CSV_FILE, help="Path to CSV input.")
    parser.add_argument("--drivers", type=int, default=NUM_DRIVERS, help="Number of Selenium drivers.")
    parser.add_argument("--headless", action="store_true", help="Run headless.")
    parser.add_argument("--manual-wait", type=float, default=INITIAL_MANUAL_WAIT,
                        help="Seconds to wait for manual popup closing (if mode triggers).")
    parser.add_argument("--manual-wait-mode", choices=["every", "first", "none"], default=MANUAL_WAIT_MODE,
                        help="When to apply manual wait.")
    parser.add_argument("--dynamic-wait", type=float, default=DYNAMIC_WAIT_TIMEOUT,
                        help="Max seconds waiting for first target image element.")
    parser.add_argument("--polite-delay", type=float, default=POLITE_DELAY,
                        help="Delay between rows per driver.")
    parser.add_argument("--no-random", action="store_true", help="Disable random jitter.")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (reduces waits).")
    parser.add_argument("--max-download-workers", type=int, default=MAX_DOWNLOAD_WORKERS,
                        help="Parallel download threads.")
    parser.add_argument("--flickr-upsize-always", action="store_true",
                        help="Try bigger Flickr sizes before the given one.")
    parser.add_argument("--flickr-no-alt-on-fail", action="store_true",
                        help="Disable Flickr alternate size fallback on failure.")
    args = parser.parse_args()

    # Apply CLI
    CSV_FILE = args.csv
    NUM_DRIVERS = max(1, args.drivers)
    HEADLESS = args.headless
    INITIAL_MANUAL_WAIT = args.manual_wait
    MANUAL_WAIT_MODE = args.manual_wait_mode
    DYNAMIC_WAIT_TIMEOUT = args.dynamic_wait
    POLITE_DELAY = args.polite_delay
    USE_RANDOM_DELAYS = not args.no_random
    FAST_MODE = args.fast
    MAX_DOWNLOAD_WORKERS = max(1, args.max_download_workers)
    UPSIZE_FLICKR_ALWAYS = args.flickr_upsize_always
    if args.flickr_no_alt_on_fail:
        FLICKR_TRY_ALT_SIZES_ON_FAIL = False

    if FAST_MODE:
        DYNAMIC_WAIT_TIMEOUT = min(DYNAMIC_WAIT_TIMEOUT, 0.9)
        POLITE_DELAY = min(POLITE_DELAY, 0.05)
        USE_RANDOM_DELAYS = False

    if not os.path.exists(CSV_FILE):
        print(f"CSV not found: {CSV_FILE}")
        sys.exit(1)

    rows = list(read_csv_rows(CSV_FILE))
    if not rows:
        print("No rows in CSV.")
        return

    # Round-robin distribution
    assignments = [[] for _ in range(NUM_DRIVERS)]
    index_assignments = [[] for _ in range(NUM_DRIVERS)]
    for idx, row in enumerate(rows, start=1):
        bucket = (idx - 1) % NUM_DRIVERS
        assignments[bucket].append(row)
        index_assignments[bucket].append(idx)

    print(f"Total rows: {len(rows)} | Drivers: {NUM_DRIVERS} | FastMode={FAST_MODE} | ManualWaitMode={MANUAL_WAIT_MODE}")
    print(f"Flickr Upsize Always={UPSIZE_FLICKR_ALWAYS} | Flickr Alt-on-fail={FLICKR_TRY_ALT_SIZES_ON_FAIL}")

    manifest = []
    manifest_lock = threading.Lock()
    download_manager = DownloadManager(MAX_DOWNLOAD_WORKERS, manifest, manifest_lock)

    with ThreadPoolExecutor(max_workers=NUM_DRIVERS) as executor:
        futures = []
        for driver_id in range(NUM_DRIVERS):
            futures.append(
                executor.submit(
                    driver_worker,
                    driver_id + 1,
                    assignments[driver_id],
                    index_assignments[driver_id],
                    download_manager
                )
            )
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[Main] Worker error: {e}")

    print("[Main] Waiting for pending downloads...")
    download_manager.wait_finish()

    if SAVE_METADATA_JSON:
        try:
            with open(MANIFEST_FILE, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)
            print(f"Manifest saved -> {MANIFEST_FILE}")
        except Exception as e:
            print(f"Manifest save error: {e}")

    print("Done.")

if __name__ == "__main__":
    main()