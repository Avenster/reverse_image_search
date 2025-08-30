import streamlit as st
import time
import csv
import cv2
import numpy as np
import requests
import os
import base64
import re
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote
import logging
import shutil
from datetime import datetime, timedelta
import subprocess
import platform
import tempfile
import io
from PIL import Image, ExifTags
import zipfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import urllib3

# --- Page Configuration ---
st.set_page_config(
    page_title="Dual Engine Visual QC Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Session State Initialization ---
def init_session_state():
    defaults = {
        'processing': False,
        'paused': False,
        'results': [],
        'uploaded_files': [],
        'temp_dir': None,
        'processing_queue': [],
        'current_index': 0,
        'start_time': None,
        'performance_metrics': [],
        'all_processed_images': [],
        'matched_images': [],
        'unmatched_images': [],
        'google_driver': None,
        'bing_driver': None,
        'session': None,
        'logs': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Suppress Warnings & Configure Logging ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class EnhancedStreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.max_logs = 150

    def emit(self, record):
        msg = self.format(record)
        self.logs.append({
            'timestamp': datetime.now(),
            'level': record.levelname,
            'message': msg
        })
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

log_handler = EnhancedStreamlitLogHandler()
logging.getLogger().addHandler(log_handler)

# --- Utility & Helper Functions ---

def get_image_metadata(image_path):
    """Extract comprehensive image metadata"""
    try:
        with Image.open(image_path) as img:
            metadata = {
                'filename': os.path.basename(image_path),
                'format': img.format,
                'size': img.size,
                'mode': img.mode,
                'file_size_kb': round(os.path.getsize(image_path) / 1024, 2)
            }
            img_array = np.array(img.convert('RGB'))
            metadata['hash'] = hashlib.md5(img_array.tobytes()).hexdigest()
            return metadata
    except Exception as e:
        logging.error(f"Failed to extract metadata from {image_path}: {e}")
        return {'filename': os.path.basename(image_path), 'error': str(e)}

def kill_chrome_processes():
    """Kill existing Chrome/Chromium processes to prevent conflicts."""
    system = platform.system()
    if system in ["Darwin", "Linux"]:
        processes = ["Google Chrome", "Chromium", "chrome"]
        for process in processes:
            try:
                subprocess.run(["pkill", "-f", process], check=False, capture_output=True)
            except FileNotFoundError:
                pass
    elif system == "Windows":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], check=False, capture_output=True)
        except FileNotFoundError:
            pass
    time.sleep(1)

@st.cache_resource(ttl=3600)
def get_requests_session():
    """Initializes and caches a requests session with connection pooling and retries."""
    session = requests.Session()
    retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
    return session

def init_driver(headless=False):
    """Initializes a single undetected_chromedriver instance with robust options."""
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    if headless:
        options.add_argument("--headless=new")

    try:
        driver = uc.Chrome(options=options, version_main=None, use_subprocess=True)
        driver.set_page_load_timeout(40)
        driver.implicitly_wait(10)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize Chrome driver: {e}")
        st.error(f"Failed to initialize a browser instance: {e}. Please ensure Chrome is installed.")
        return None

def initialize_drivers(headless=False):
    """Initializes and stores both Google and Bing drivers in the session state."""
    kill_chrome_processes()
    with st.spinner("🌐 Initializing browsers for Google and Bing..."):
        st.session_state.google_driver = init_driver(headless)
        st.session_state.bing_driver = init_driver(headless)

    if not st.session_state.google_driver or not st.session_state.bing_driver:
        st.error("❌ Failed to initialize one or more browser instances. Cannot start processing.")
        return False
    return True

def quit_drivers():
    """Quits both Selenium drivers if they exist."""
    for driver_key in ['google_driver', 'bing_driver']:
        if hasattr(st.session_state, driver_key) and st.session_state[driver_key]:
            try:
                st.session_state[driver_key].quit()
            except Exception as e:
                logging.warning(f"Error while quitting {driver_key}: {e}")
            st.session_state[driver_key] = None

def create_temp_folders(temp_dir):
    """Create all necessary subfolders in the temporary directory."""
    folders = ['images', 'done', 'similar_images', 'reports', 'unmatched', 'thumbnails']
    for folder in folders:
        os.makedirs(os.path.join(temp_dir, folder), exist_ok=True)

def create_thumbnail(image_path, temp_dir, size=(150, 150)):
    """Create thumbnail for image preview"""
    try:
        thumb_folder = Path(os.path.join(temp_dir, 'thumbnails'))
        source = Path(image_path)
        thumb_path = thumb_folder / f"thumb_{source.name}"
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumb_path)
        return str(thumb_path)
    except Exception as e:
        logging.error(f"Failed to create thumbnail for {image_path}: {e}")
        return None

def save_uploaded_files(uploaded_files, temp_dir):
    """Save uploaded files and detect duplicates"""
    image_paths = []
    duplicates = []
    hashes = set()

    images_dir = os.path.join(temp_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(images_dir, uploaded_file.name)
        file_bytes = uploaded_file.getbuffer()
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        metadata = get_image_metadata(file_path)
        file_hash = metadata.get('hash')
        if file_hash in hashes:
            duplicates.append(uploaded_file.name)
            os.remove(file_path)
        else:
            if file_hash:
                hashes.add(file_hash)
            image_paths.append(file_path)

    return image_paths, duplicates

def copy_unmatched_image(image_path, temp_dir):
    """Copy unmatched image to unmatched folder for tracking"""
    try:
        unmatched_folder = Path(os.path.join(temp_dir, 'unmatched'))
        source = Path(image_path)
        destination = unmatched_folder / source.name
        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = unmatched_folder / f"{source.stem}_{timestamp}{source.suffix}"
        shutil.copy2(str(source), str(destination))
        logging.info(f"Copied unmatched '{source.name}' to 'unmatched' folder.")
        return str(destination)
    except Exception as e:
        logging.error(f"Failed to copy unmatched '{image_path}': {e}")
        return None

def move_image_to_done(image_path, temp_dir):
    """Move processed image to done folder"""
    try:
        done_folder = Path(os.path.join(temp_dir, 'done'))
        source = Path(image_path)
        destination = done_folder / source.name
        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = done_folder / f"{source.stem}_{timestamp}{source.suffix}"
        shutil.move(str(source), str(destination))
        logging.info(f"Moved '{source.name}' to 'done' folder.")
    except Exception as e:
        logging.error(f"Failed to move '{image_path}': {e}")

def get_image_from_url_or_base64(img_url, session, timeout=10):
    """Download and decode image from URL or base64 - using reliable logic"""
    if img_url.startswith("data:image"):
        try:
            _, b64data = img_url.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            img_np = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
    else:
        try:
            headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://lens.google.com/'}
            resp = session.get(img_url, timeout=timeout, headers=headers, verify=False)
            resp.raise_for_status()
            img_np = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None

def calculate_flann_similarity(img1_cv, img2_url, session, min_matches=8, timeout=15):
    """Calculate FLANN similarity score - using reliable logic"""
    try:
        if img1_cv is None:
            return 0.0, 0
        img2 = get_image_from_url_or_base64(img2_url, session, timeout=timeout)
        if img2 is None:
            return 0.0, 0
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img1_cv, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0, 0
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        num_good_matches = len(good)
        if num_good_matches > min_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                num_inliers = np.sum(mask)
                similarity_score = num_inliers / num_good_matches
                return min(similarity_score, 1.0), num_inliers
        return 0.0, 0
    except Exception:
        return 0.0, 0

def create_enhanced_comparison_image(source_path, matched_url, score, matches, engine_name, temp_dir, session, metadata):
    """Enhanced comparison image with more information"""
    try:
        original_img = cv2.imread(source_path)
        if original_img is None: return None

        if matched_url.startswith("data:image"):
            _, base64_data = matched_url.split(',', 1)
            img_bytes = base64.b64decode(base64_data)
        else:
            headers = {'User-Agent': 'Mozilla/5.0', 'Referer': f'https://www.{engine_name.lower()}.com/'}
            resp = session.get(matched_url, headers=headers, verify=False)
            img_bytes = resp.content

        similar_img_np = np.frombuffer(img_bytes, np.uint8)
        similar_img = cv2.imdecode(similar_img_np, cv2.IMREAD_COLOR)
        if similar_img is None: return None

        target_height = 500
        text_area_height = 120

        original_h, original_w = original_img.shape[:2]
        similar_h, similar_w = similar_img.shape[:2]
        original_ratio = target_height / original_h
        original_resized = cv2.resize(original_img, (int(original_w * original_ratio), target_height))
        similar_ratio = target_height / similar_h
        similar_resized = cv2.resize(similar_img, (int(similar_w * similar_ratio), target_height))

        total_width = original_resized.shape[1] + similar_resized.shape[1]
        total_height = target_height + text_area_height
        canvas = np.full((total_height, total_width, 3), 255, dtype=np.uint8)

        canvas[text_area_height:total_height, :original_resized.shape[1]] = original_resized
        canvas[text_area_height:total_height, original_resized.shape[1]:] = similar_resized

        font = cv2.FONT_HERSHEY_SIMPLEX
        small_font_scale = 0.6
        large_font_scale = 0.8
        color = (0, 0, 0)

        cv2.putText(canvas, f"Original: {metadata.get('filename', Path(source_path).name)}", (10, 30), font, small_font_scale, color, 1)
        cv2.putText(canvas, f"Size: {metadata.get('size', ['?','?'])[0]}x{metadata.get('size', ['?','?'])[1]} | {metadata.get('file_size_kb', 0)} KB", (10, 55), font, small_font_scale, color, 1)

        right_x = original_resized.shape[1] + 10
        cv2.putText(canvas, f"{engine_name} Match", (right_x, 30), font, large_font_scale, color, 2)
        cv2.putText(canvas, f"FLANN Score: {score:.4f}", (right_x, 60), font, small_font_scale, color, 1)
        cv2.putText(canvas, f"Feature Matches: {matches}", (right_x, 85), font, small_font_scale, color, 1)

        source_name = Path(source_path).stem
        filename = f"{source_name}_{engine_name}_comparison_{score:.4f}.jpg"
        filepath = os.path.join(temp_dir, 'similar_images', filename)
        if os.path.exists(filepath):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(temp_dir, 'similar_images', f"{source_name}_{engine_name}_comparison_{score:.4f}_{timestamp}.jpg")

        cv2.imwrite(filepath, canvas)
        return filepath
    except Exception as e:
        logging.error(f"Failed to create enhanced comparison: {e}")
        return None

# --- Google Lens Specific Functions ---

def upload_to_google_lens(driver, image_path, timeout=30):
    try:
        driver.get("https://lens.google.com/")
        upload_input = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        upload_input.send_keys(os.path.abspath(image_path))
        WebDriverWait(driver, timeout).until(
            lambda drv: drv.find_elements(By.CSS_SELECTOR, "[data-photo-id]") or EC.url_contains("search")(drv)
        )
        try:
            exact_btn = driver.find_element(By.XPATH, "//span[contains(text(), 'Exact matches')]/ancestor::a")
            driver.execute_script("arguments[0].click();", exact_btn)
            time.sleep(1)
        except NoSuchElementException:
            pass # No exact match button, proceed anyway
        return True
    except Exception as e:
        logging.error(f"Google Lens upload failed for {Path(image_path).name}: {e}")
        return False

def get_google_lens_urls(driver, max_images=50):
    urls = set()
    try:
        # Scroll to load more images
        for _ in range(2):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

        # Primary method: Extract from standard image elements
        normal_imgs = driver.find_elements(By.CSS_SELECTOR, "div.qR29te img[src]")
        for img in normal_imgs:
            src = img.get_attribute("src")
            if src and (src.startswith("http") or src.startswith("data:image")):
                urls.add(src)

        # Fallback method: Regex on page source for more complex URLs
        if len(urls) < max_images:
            page_source = driver.page_source
            url_patterns = [
                r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)"',
                r'"ou":"(https?://[^"]+)"', r'imgurl=(https?://[^&]+)'
            ]
            for pattern in url_patterns:
                for match in re.findall(pattern, page_source, re.IGNORECASE):
                    urls.add(unquote(match))
        return list(urls)[:max_images]
    except Exception as e:
        logging.error(f"Error extracting Google URLs: {e}")
        return []

# --- Bing Visual Search Specific Functions ---

def upload_to_bing(driver, image_path, timeout=30):
    try:
        driver.get("https://www.bing.com/images")
        time.sleep(2)
        # Click the visual search icon
        visual_search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#sbi_b, .sbi_b_prtl, [data-tooltip='Search by image']"))
        )
        visual_search_button.click()
        time.sleep(1)
        # Send image path to the file input
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        file_input.send_keys(os.path.abspath(image_path))
        # Wait for results page to load
        WebDriverWait(driver, timeout).until(lambda d: "images/search" in d.current_url)
        return True
    except Exception as e:
        logging.error(f"Bing upload failed for {Path(image_path).name}: {e}")
        return False

def get_bing_urls(driver, max_images=50):
    urls = set()
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        # Extract URLs from 'data-m' attribute
        rich_links = driver.find_elements(By.CSS_SELECTOR, "a.richImgLnk")
        for link in rich_links:
            m_data = link.get_attribute('data-m')
            if m_data:
                try:
                    data = json.loads(m_data)
                    if 'murl' in data: urls.add(data['murl'])
                except json.JSONDecodeError:
                    continue
        return list(urls)[:max_images]
    except Exception as e:
        logging.error(f"Error extracting Bing URLs: {e}")
        return []

def calculate_eta(current_index, total_images, start_time, avg_time_per_image):
    if current_index == 0 or avg_time_per_image <= 0:
        return "Calculating..."
    remaining_images = total_images - current_index
    estimated_remaining_seconds = remaining_images * avg_time_per_image
    eta_datetime = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
    return eta_datetime.strftime("%H:%M:%S")

def process_image_on_engine(engine_name, driver, upload_func, url_extract_func, image_path, session, max_urls, threshold):
    """Generic function to process an image on a given search engine."""
    logging.info(f"[{engine_name}] Processing {Path(image_path).name}")
    if not upload_func(driver, image_path):
        return {'score': 0.0, 'match_info': None, 'error': f'{engine_name} upload failed'}

    urls = url_extract_func(driver, max_images=max_urls)
    if not urls:
        return {'score': 0.0, 'match_info': None, 'error': f'No URLs found on {engine_name}'}

    source_img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img_cv is None:
        return {'score': 0.0, 'match_info': None, 'error': 'Could not read source image'}

    best_score = 0.0
    best_match_info = None

    def check_url(url):
        score, matches = calculate_flann_similarity(source_img_cv, url, session)
        return url, score, matches

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(check_url, url): url for url in urls}
        for future in as_completed(futures):
            try:
                url, score, matches = future.result()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match_info = {'url': url, 'matches': matches}
            except Exception:
                continue

    return {'score': best_score, 'match_info': best_match_info}

def process_single_image_dual(image_path, temp_dir, config):
    """Orchestrates processing a single image on both Google and Bing concurrently."""
    start_time = time.time()
    image_name = Path(image_path).name

    metadata = get_image_metadata(image_path)
    thumbnail_path = create_thumbnail(image_path, temp_dir)

    base_result = {
        'source_image': image_name,
        'source_path': image_path,
        'thumbnail_path': thumbnail_path,
        'metadata': metadata,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'Processing',
        'best_engine': None,
        'best_score': 0.0
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_google = executor.submit(process_image_on_engine, "Google", st.session_state.google_driver, upload_to_google_lens, get_google_lens_urls, image_path, st.session_state.session, config['google_max_urls'], config['google_threshold'])
        future_bing = executor.submit(process_image_on_engine, "Bing", st.session_state.bing_driver, upload_to_bing, get_bing_urls, image_path, st.session_state.session, config['bing_max_urls'], config['bing_threshold'])

        google_result = future_google.result()
        bing_result = future_bing.result()

    google_score = google_result.get('score', 0.0)
    bing_score = bing_result.get('score', 0.0)

    base_result['google_score'] = google_score
    base_result['bing_score'] = bing_score

    # Determine the best overall match
    if google_score >= config['google_threshold'] or bing_score >= config['bing_threshold']:
        best_engine = "Google" if google_score >= bing_score else "Bing"
        best_score = google_score if best_engine == "Google" else bing_score
        best_match_info = google_result['match_info'] if best_engine == "Google" else bing_result['match_info']

        base_result.update({
            'status': 'Match Found',
            'best_engine': best_engine,
            'best_score': best_score,
            'matched_url': best_match_info['url'],
            'num_matches': best_match_info['matches']
        })

        base_result['comparison_path'] = create_enhanced_comparison_image(
            source_path=image_path, matched_url=best_match_info['url'], score=best_score,
            matches=best_match_info['matches'], engine_name=best_engine,
            temp_dir=temp_dir, session=st.session_state.session, metadata=metadata
        )
        st.session_state.matched_images.append(base_result)
    else:
        base_result.update({
            'status': 'No Match Found',
            'unmatched_path': copy_unmatched_image(image_path, temp_dir)
        })
        st.session_state.unmatched_images.append(base_result)

    move_image_to_done(image_path, temp_dir)
    processing_time = time.time() - start_time
    base_result['processing_time'] = processing_time
    return base_result, processing_time, metadata

# --- UI Components ---

def create_analytics_dashboard(results):
    if not results: return

    st.header("📊 QC Analytics Dashboard")

    matched_results = [r for r in results if r and r.get('status') == 'Match Found']
    unmatched_results = [r for r in results if r and r.get('status') == 'No Match Found']

    col1, col2, col3, col4, col5 = st.columns(5)
    total_processed = len(results)
    total_matches = len(matched_results)
    total_unmatched = len(unmatched_results)
    match_rate = (total_matches / total_processed * 100) if total_processed > 0 else 0

    col1.metric("Total Processed", total_processed)
    col2.metric("✅ Matches Found", total_matches)
    col3.metric("❌ No Matches", total_unmatched)
    col4.metric("Match Rate", f"{match_rate:.1f}%")
    col5.metric("Avg Score", f"{(sum(r['best_score'] for r in matched_results) / total_matches):.3f}" if total_matches > 0 else "N/A")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if matched_results:
            engine_counts = pd.DataFrame([r['best_engine'] for r in matched_results], columns=['Engine']).value_counts().reset_index()
            engine_counts.columns = ['Engine', 'Count']
            fig2 = px.pie(engine_counts, values='Count', names='Engine', 
                          title="Match Distribution by Search Engine",
                          color_discrete_map={'Google': '#4285F4', 'Bing': '#0078D4'})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No matched images to display engine distribution.")

    with chart_col2:
        if matched_results:
            scores = [r['best_score'] for r in matched_results]
            fig1 = px.histogram(x=scores, nbins=20, title="FLANN Score Distribution (Matched Images)", 
                               labels={'x': 'FLANN Score', 'y': 'Count'})
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No matched images to display score distribution")

    if results:
        st.subheader("⏱️ Processing Time Analysis")
        time_col1, time_col2 = st.columns(2)

        with time_col1:
            processing_times = [r['processing_time'] for r in results if 'processing_time' in r]
            if processing_times:
                fig3 = px.line(y=processing_times, title="Processing Time per Image",
                              labels={'index': 'Image Index', 'y': 'Time (seconds)'})
                st.plotly_chart(fig3, use_container_width=True)

        with time_col2:
            if matched_results:
                stage_times = {
                    'Google Score': sum(r.get('google_score', 0) for r in matched_results) / total_matches,
                    'Bing Score': sum(r.get('bing_score', 0) for r in matched_results) / total_matches,
                }
                fig4 = px.bar(x=list(stage_times.keys()), y=list(stage_times.values()),
                            title="Average Match Scores per Engine",
                            labels={'x': 'Engine', 'y': 'Score'})
                st.plotly_chart(fig4, use_container_width=True)

def create_image_preview_section(results):
    """Create preview sections for matched and unmatched images"""
    if not results:
        st.info("No images processed yet")
        return

    st.header("🖼️ Image Preview")

    matched_results = [r for r in results if r and r.get('status') == 'Match Found']
    unmatched_results = [r for r in results if r and r.get('status') == 'No Match Found']

    preview_tab1, preview_tab2 = st.tabs([f"✅ Matched Images ({len(matched_results)})", 
                                          f"❌ Unmatched Images ({len(unmatched_results)})"])

    with preview_tab1:
        if matched_results:
            st.success(f"Found {len(matched_results)} matched images")
            cols_per_row = 3
            for i in range(0, len(matched_results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(matched_results):
                        result = matched_results[i + j]
                        with col:
                            st.markdown(f"**{result['source_image']}**")
                            st.caption(f"Score: {result['best_score']:.4f} via {result.get('best_engine', '')}")
                            if result.get('comparison_path') and os.path.exists(result['comparison_path']):
                                st.image(result['comparison_path'], use_column_width=True)
                            st.caption(f"Matches: {result.get('num_matches', 0)}")
                            with st.expander("Details"):
                                st.write(f"Processing Time: {result.get('processing_time', 0):.2f}s")
                                st.write(f"Google Score: {result.get('google_score', 0):.4f}")
                                st.write(f"Bing Score: {result.get('bing_score', 0):.4f}")
                                st.write(f"Size: {result['metadata'].get('size', ['?', '?'])[0]}x{result['metadata'].get('size', ['?', '?'])[1]}")
                                st.write(f"File Size: {result['metadata'].get('file_size_kb', 0)} KB")
        else:
            st.info("No matched images found")

    with preview_tab2:
        if unmatched_results:
            st.warning(f"Found {len(unmatched_results)} unmatched images")
            cols_per_row = 4
            for i in range(0, len(unmatched_results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(unmatched_results):
                        result = unmatched_results[i + j]
                        with col:
                            st.markdown(f"**{result['source_image']}**")
                            if result.get('thumbnail_path') and os.path.exists(result['thumbnail_path']):
                                st.image(result['thumbnail_path'], use_column_width=True)
                            st.caption("No match found")
                            with st.expander("Details"):
                                st.write(f"Processing Time: {result.get('processing_time', 0):.2f}s")
                                st.write(f"Google Score: {result.get('google_score', 0):.4f}")
                                st.write(f"Bing Score: {result.get('bing_score', 0):.4f}")
                                st.write(f"Size: {result['metadata'].get('size', ['?', '?'])[0]}x{result['metadata'].get('size', ['?', '?'])[1]}")
                                st.write(f"File Size: {result['metadata'].get('file_size_kb', 0)} KB")
        else:
            st.info("No unmatched images found")

def create_detailed_results_table(results):
    """Create comprehensive table including both matched and unmatched images"""
    if not results:
        st.info("No results to display")
        return None

    st.subheader("📋 Complete Results Table")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        show_filter = st.selectbox("Show:", ["All", "Matched Only", "Unmatched Only"])
    with filter_col2:
        sort_by = st.selectbox("Sort by:", ["Image Name", "Status", "Score", "Processing Time"])
    with filter_col3:
        sort_order = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)

    table_data = []
    for r in results:
        if r:
            metadata = r.get('metadata', {})
            table_data.append({
                'Image': r['source_image'],
                'Status': r.get('status', 'Unknown'),
                'Engine': r.get('best_engine', 'N/A') if r.get('status') == 'Match Found' else 'N/A',
                'Score': r.get('best_score', 0.0),
                'Google Score': r.get('google_score', 0.0),
                'Bing Score': r.get('bing_score', 0.0),
                'Matches': r.get('num_matches', 0),
                'Time (s)': round(r.get('processing_time', 0), 2),
                'Size': f"{metadata.get('size', ['?', '?'])[0]}x{metadata.get('size', ['?', '?'])[1]}",
                'File Size (KB)': metadata.get('file_size_kb', 0),
                'Timestamp': r.get('timestamp', 'N/A'),
                'Match URL': r.get('matched_url', 'No match') if r.get('matched_url') else 'No match'
            })

    df = pd.DataFrame(table_data)

    # Apply filters
    if show_filter == "Matched Only":
        df = df[df['Status'] == 'Match Found']
    elif show_filter == "Unmatched Only":
        df = df[df['Status'] == 'No Match Found']

    # Apply sorting
    ascending = sort_order == "Ascending"
    if sort_by == "Image Name":
        df = df.sort_values('Image', ascending=ascending)
    elif sort_by == "Status":
        df = df.sort_values('Status', ascending=ascending)
    elif sort_by == "Score":
        df = df.sort_values('Score', ascending=ascending)
    elif sort_by == "Processing Time":
        df = df.sort_values('Time (s)', ascending=ascending)

    def highlight_status(row):
        if row['Status'] == 'Match Found':
            return ['background-color: #d4f4dd'] * len(row)
        elif row['Status'] == 'No Match Found':
            return ['background-color: #ffd4d4'] * len(row)
        else:
            return [''] * len(row)

    styled_df = df.style.apply(highlight_status, axis=1).format({
        'Score': '{:.4f}',
        'Google Score': '{:.4f}',
        'Bing Score': '{:.4f}',
        'Time (s)': '{:.2f}'
    })

    st.dataframe(styled_df, use_container_width=True, height=400)
    st.caption(f"**Total:** {len(df)} | **Matched:** {len(df[df['Status'] == 'Match Found'])} | **Unmatched:** {len(df[df['Status'] == 'No Match Found'])}")

    return df

def generate_comprehensive_report(results):
    """Generate comprehensive QC report including all processed images"""
    if not results:
        return None

    matched_results = [r for r in results if r and r.get('status') == 'Match Found']
    unmatched_results = [r for r in results if r and r.get('status') == 'No Match Found']

    report = {
        'summary': {
            'total_images': len(results),
            'successful_matches': len(matched_results),
            'failed_matches': len(unmatched_results),
            'match_rate_percent': round(len(matched_results) / len(results) * 100, 2) if results else 0,
            'average_score': round(sum(r['best_score'] for r in matched_results) / len(matched_results), 4) if matched_results else 0,
            'average_processing_time': round(sum(r.get('processing_time', 0) for r in results) / len(results), 2) if results else 0,
            'report_generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'matched_images': matched_results,
        'unmatched_images': unmatched_results,
        'all_results': results
    }
    return report

def create_reports_section():
    st.header("📄 QC Reports & Export")
    if not st.session_state.results:
        st.info("No results available for report generation")
        return

    report = generate_comprehensive_report(st.session_state.results)
    if not report:
        return

    # Display report summary
    st.subheader("📊 Report Summary")
    summary = report['summary']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", summary['total_images'])
        st.metric("Match Rate", f"{summary['match_rate_percent']:.1f}%")
    with col2:
        st.metric("Successful Matches", summary['successful_matches'])
        st.metric("Average Score", f"{summary['average_score']:.3f}")
    with col3:
        st.metric("Failed Matches", summary['failed_matches'])
        st.metric("Avg Processing Time", f"{summary['average_processing_time']:.1f}s")

    st.caption(f"Report generated: {summary['report_generated']}")
    st.divider()

    # Export options
    st.subheader("📥 Export Options")
    export_col1, export_col2, export_col3, export_col4 = st.columns(4)

    with export_col1:
        if st.button("📊 Generate Full CSV Report", use_container_width=True):
            csv_buffer = io.StringIO()
            all_data = []
            for r in st.session_state.results:
                if r:
                    metadata = r.get('metadata', {})
                    all_data.append({
                        'Image': r['source_image'],
                        'Status': r.get('status', 'Unknown'),
                        'Engine': r.get('best_engine', 'N/A') if r.get('status') == 'Match Found' else 'N/A',
                        'FLANN Score': r.get('best_score', 0),
                        'Google Score': r.get('google_score', 0),
                        'Bing Score': r.get('bing_score', 0),
                        'Feature Matches': r.get('num_matches', 0),
                        'Processing Time (s)': round(r.get('processing_time', 0), 2),
                        'Image Size': f"{metadata.get('size', ['?', '?'])[0]}x{metadata.get('size', ['?', '?'])[1]}",
                        'File Size (KB)': metadata.get('file_size_kb', 0),
                        'Timestamp': r.get('timestamp', 'N/A'),
                        'Match URL': r.get('matched_url', 'No match') if r.get('matched_url') else 'No match'
                    })
            df = pd.DataFrame(all_data)
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📥 Download Full CSV Report",
                csv_buffer.getvalue(),
                f"qc_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

        with export_col2:
            # JSON Export
            if st.button("📄 Generate JSON Report", use_container_width=True):
                json_str = json.dumps(report, indent=2, default=str)
                st.download_button(
                    "📥 Download JSON Report",
                    json_str,
                    f"qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True
                )
        with export_col3:
            # Images ZIP Export
            matched_with_comparison = [r for r in st.session_state.results if r and r.get('comparison_path')]
            if matched_with_comparison:
                if st.button("🖼️ Export Comparison Images", use_container_width=True):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_f:
                        # Add matched comparison images
                        for r in matched_with_comparison:
                            if os.path.exists(r['comparison_path']):
                                zip_f.write(r['comparison_path'], f"matched/{os.path.basename(r['comparison_path'])}")
                        # Add unmatched thumbnails
                        unmatched = [r for r in st.session_state.results if r and r.get('status') == 'No Match Found']
                        for r in unmatched:
                            if r.get('thumbnail_path') and os.path.exists(r['thumbnail_path']):
                                zip_f.write(r['thumbnail_path'], f"unmatched/{os.path.basename(r['thumbnail_path'])}")
                    st.download_button(
                        "📥 Download Images ZIP",
                        zip_buffer.getvalue(),
                        f"qc_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        "application/zip",
                        use_container_width=True
                    )
            else:
                st.info("No comparison images available")

            with export_col4:
                # Reset Session
                if st.button("🔄 New Session", use_container_width=True, type="secondary"):
                    quit_drivers()
                    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                        shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
                    # Reset session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()

def main_ui():
    """Defines the main UI structure and layout."""
    st.markdown("""
        <style> .main-header { text-align: center; padding-bottom: 2rem; } </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🤖 Dual Engine Visual QC Tool (Google & Bing)</h1>', unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ General Configuration")
        batch_delay = st.slider("Delay between images (s)", 0.0, 5.0, 1.0, 0.1)
        headless_mode = st.checkbox("Run in Headless Mode", value=True, help="Run browsers in the background.")

        st.divider()
        st.header("🔵 Google Lens Configuration")
        google_max_urls = st.slider("Max Google Results", 10, 100, 50)
        google_threshold = st.slider("Google Similarity Threshold", 0.0, 1.0, 0.20, 0.01)

        st.divider()
        st.header("🔷 Bing Configuration")
        bing_max_urls = st.slider("Max Bing Results", 10, 100, 40)
        bing_threshold = st.slider("Bing Similarity Threshold", 0.0, 1.0, 0.15, 0.01)

        st.divider()
        if st.button("🔄 New Session / Clear All", use_container_width=True, type="primary"):
            quit_drivers()
            if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.divider()
        # Quick stats
        if st.session_state.results:
            matched = len([r for r in st.session_state.results if r and r.get('status') == 'Match Found'])
            unmatched = len([r for r in st.session_state.results if r and r.get('status') == 'No Match Found'])
            st.metric("Total Processed", len(st.session_state.results))
            st.metric("Matched", matched, delta=f"{matched/(len(st.session_state.results))*100:.1f}%")
            st.metric("Unmatched", unmatched)
        st.divider()
        show_logs = st.checkbox("Show Processing Logs")

    config = {
        'batch_delay': batch_delay, 'headless_mode': headless_mode,
        'google_max_urls': google_max_urls, 'google_threshold': google_threshold,
        'bing_max_urls': bing_max_urls, 'bing_threshold': bing_threshold,
    }

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 UPLOAD & PROCESS",
        "🖼️ IMAGE PREVIEWS",
        "📊 ANALYTICS",
        "📋 RESULTS TABLE",
        "📄 REPORTS & EXPORT",
        "📝 LOGS"
    ])

    with tab1:
        st.header("Image Upload")
        uploaded_files = st.file_uploader(
            "Choose image files", type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

        if st.session_state.uploaded_files:
            process_col1, process_col2, process_col3, process_col4 = st.columns(4)
            with process_col1:
                if not st.session_state.processing and st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    st.session_state.processing = True
                    st.session_state.paused = False
                    st.session_state.results = []
                    st.session_state.current_index = 0
                    st.session_state.start_time = time.time()
                    st.session_state.performance_metrics = []
                    st.session_state.matched_images = []
                    st.session_state.unmatched_images = []
                    st.rerun()
            with process_col2:
                if st.session_state.processing:
                    if st.session_state.paused:
                        if st.button("▶️ Resume", use_container_width=True):
                            st.session_state.paused = False
                            st.rerun()
                    else:
                        if st.button("⏸️ Pause", use_container_width=True):
                            st.session_state.paused = True
                            st.rerun()
            with process_col3:
                if st.session_state.processing and st.button("⏹️ Stop", use_container_width=True):
                    st.session_state.processing = False
                    st.session_state.paused = False
                    quit_drivers()
                    st.rerun()
            with process_col4:
                if st.session_state.results and st.button("🔄 Clear Results", use_container_width=True):
                    st.session_state.results = []
                    st.session_state.matched_images = []
                    st.session_state.unmatched_images = []
                    st.rerun()

        if st.session_state.processing and st.session_state.uploaded_files and not st.session_state.paused:
            run_processing_loop(config)
        elif st.session_state.paused:
            st.warning("⏸️ Processing paused. Click Resume to continue.")

    with tab2:
        create_image_preview_section(st.session_state.results)

    with tab3:
        create_analytics_dashboard(st.session_state.results)

    with tab4:
        df = create_detailed_results_table(st.session_state.results)
        if df is not None and not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Table as CSV",
                csv,
                f"qc_results_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

    with tab5:
        create_reports_section()

    with tab6:
        st.header("📝 Processing Logs")
        if show_logs and log_handler.logs:
            log_container = st.container()
            with log_container:
                for log in reversed(log_handler.logs[-50:]):
                    level_icon = "🔵" if log['level'] == "INFO" else "🔴" if log['level'] == "ERROR" else "🟡"
                    st.text(f"{level_icon} [{log['timestamp'].strftime('%H:%M:%S')}] {log['message']}")
        else:
            st.info("Enable 'Show Processing Logs' in the sidebar to view logs")

def run_processing_loop(config):
    """The main loop that controls the image processing flow."""
    if st.session_state.current_index == 0:
        st.session_state.temp_dir = tempfile.mkdtemp()
        create_temp_folders(st.session_state.temp_dir)
        with st.spinner("Saving uploaded files..."):
            image_paths, duplicates = save_uploaded_files(st.session_state.uploaded_files, st.session_state.temp_dir)
            st.session_state.processing_queue = image_paths
        if duplicates:
            st.warning(f"⚠️ Skipped {len(duplicates)} duplicate files: {', '.join(duplicates)}")
        if not initialize_drivers(headless=config['headless_mode']):
            st.session_state.processing = False
            return
        st.session_state.session = get_requests_session()

    total_images = len(st.session_state.processing_queue)

    progress_container = st.container()
    results_stream = st.container()

    while st.session_state.current_index < total_images and st.session_state.processing and not st.session_state.paused:
        current_image = st.session_state.processing_queue[st.session_state.current_index]

        with progress_container:
            progress = (st.session_state.current_index + 1) / total_images
            progress_text = f"Processing {os.path.basename(current_image)} ({st.session_state.current_index + 1}/{total_images})"
            st.progress(progress, text=progress_text)

            col1, col2, col3 = st.columns(3)
            with col1:
                avg_time = sum(st.session_state.performance_metrics) / len(st.session_state.performance_metrics) if st.session_state.performance_metrics else 0
                st.metric("Avg Time/Image", f"{avg_time:.1f}s")
            with col2:
                eta = calculate_eta(st.session_state.current_index, total_images, st.session_state.start_time, avg_time)
                st.metric("ETA", eta)
            with col3:
                elapsed = time.time() - st.session_state.start_time if st.session_state.start_time else 0
                st.metric("Elapsed", f"{elapsed:.0f}s")

        result, processing_time, metadata = process_single_image_dual(
            current_image, st.session_state.temp_dir, config
        )

        st.session_state.results.append(result)
        st.session_state.performance_metrics.append(processing_time)
        st.session_state.all_processed_images.append(result)

        st.session_state.current_index += 1

        with results_stream:
            if result:
                if result.get('status') == 'Match Found':
                    st.success(f"✅ Match found for {result['source_image']} (Score: {result['best_score']:.3f}) via {result.get('best_engine','')}")
                    if result.get('comparison_path') and os.path.exists(result['comparison_path']):
                        with st.expander("View Comparison"):
                            st.image(result['comparison_path'])
                else:
                    st.warning(f"❌ No match found for {result['source_image']}")
                    if result.get('thumbnail_path') and os.path.exists(result['thumbnail_path']):
                        with st.expander("View Image"):
                            st.image(result['thumbnail_path'])

        if config['batch_delay'] > 0 and st.session_state.current_index < total_images:
            time.sleep(config['batch_delay'])

        st.rerun()

    if st.session_state.current_index >= total_images and st.session_state.processing:
        st.session_state.processing = False
        quit_drivers()
        st.success("🎉 Processing Complete!")
        st.balloons()
        matched_count = len([r for r in st.session_state.results if r and r.get('status') == 'Match Found'])
        unmatched_count = len([r for r in st.session_state.results if r and r.get('status') == 'No Match Found'])
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Total Processed", len(st.session_state.results))
        with summary_col2:
            st.metric("Matched", matched_count)
        with summary_col3:
            st.metric("Unmatched", unmatched_count)

if __name__ == "__main__":
    main_ui()