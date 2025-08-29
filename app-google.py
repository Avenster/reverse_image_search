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

# Page config
st.set_page_config(
    page_title="Enhanced Google Lens QC Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced session state initialization
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = []
if 'all_processed_images' not in st.session_state:
    st.session_state.all_processed_images = []
if 'matched_images' not in st.session_state:
    st.session_state.matched_images = []
if 'unmatched_images' not in st.session_state:
    st.session_state.unmatched_images = []

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Enhanced logging configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class EnhancedStreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.max_logs = 100

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

def init_requests_session(max_pool_connections=50, max_retries=1, backoff_factor=0.1):
    """Initialize requests session with connection pooling"""
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=backoff_factor,
                    status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(pool_connections=max_pool_connections, pool_maxsize=max_pool_connections, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    return session

@st.cache_resource(ttl=3600)
def get_requests_session():
    return init_requests_session()

def kill_chrome_processes():
    """Kill existing Chrome processes"""
    if platform.system() in ["Darwin", "Linux"]:
        for process_name in ["Google Chrome", "Chromium", "chrome"]:
            try:
                subprocess.run(["pkill", "-f", process_name], check=False, capture_output=True)
            except FileNotFoundError:
                pass
    elif platform.system() == "Windows":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], check=False, capture_output=True)
        except FileNotFoundError:
            pass
    time.sleep(1)

def init_driver(headless=False):
    """Initialize Chrome driver using the reliable logic from the original script."""
    kill_chrome_processes()
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-plugins-discovery")
    user_data_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", f"chrome_data_{os.getpid()}")
    options.add_argument(f"--user-data-dir={user_data_dir}")

    try:
        logging.info("Attempting to initialize Chrome driver...")
        driver = uc.Chrome(options=options, version_main=None)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except WebDriverException as e:
        logging.error(f"Failed to initialize Chrome driver: {e}")
        logging.info("Trying with a simpler configuration...")
        driver = uc.Chrome(options=options, use_subprocess=True)
        return driver

def create_temp_folders(temp_dir):
    """Create required folders"""
    folders = ['done', 'similar_images', 'reports', 'unmatched', 'thumbnails']
    for folder in folders:
        os.makedirs(os.path.join(temp_dir, folder), exist_ok=True)

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
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Check for duplicates
        metadata = get_image_metadata(file_path)
        if metadata.get('hash') and metadata.get('hash') in hashes:
            duplicates.append(uploaded_file.name)
            os.remove(file_path)
        else:
            if metadata.get('hash'):
                hashes.add(metadata['hash'])
            image_paths.append(file_path)

    return image_paths, duplicates

def upload_to_google_lens_and_click_exact_match(driver, image_path, timeout=30):
    """Upload image to Google Lens - using the reliable original logic"""
    try:
        abs_path = os.path.abspath(image_path)
        driver.get("https://lens.google.com/")

        upload_input = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        upload_input.send_keys(abs_path)

        WebDriverWait(driver, timeout).until(
            lambda drv: drv.find_elements(By.CSS_SELECTOR, "[data-photo-id]") or
                        drv.find_elements(By.CSS_SELECTOR, "a[href*='imgurl=']") or
                        drv.find_elements(By.CSS_SELECTOR, ".fk90Z") or
                        EC.url_contains("search")(drv)
        )

        try:
            no_match = driver.find_element(By.CSS_SELECTOR, ".fk90Z[role='heading']")
            if "No matches for your search" in no_match.text:
                logging.info("No matches found for image, skipping.")
                return False
        except NoSuchElementException:
            pass

        try:
            exact_btn = driver.find_element(
                By.XPATH,
                "//span[contains(text(), 'Exact matches')]/ancestor::a"
            )
            if exact_btn:
                driver.execute_script("arguments[0].click();", exact_btn)
                logging.info("Clicked 'Exact matches' button.")
                time.sleep(1)
        except NoSuchElementException:
            logging.info("Exact matches button not found, continuing.")

        return True
    except TimeoutException:
        logging.error(f"Timed out waiting for Google Lens elements for {image_path}.")
        return False
    except Exception as e:
        logging.error(f"Upload to Google Lens failed for {image_path}: {e}")
        return False

def get_exact_match_image_urls(driver, max_images=50):
    """Extract exact match image URLs - using the reliable original logic"""
    urls = set()
    try:
        normal_imgs = driver.find_elements(By.CSS_SELECTOR, "div.qR29te img[src]")
        for img in normal_imgs:
            src = img.get_attribute("src")
            if src:
                if src.startswith("http") or (src.startswith("data:image") and not src.startswith("data:image/svg")):
                    urls.add(src)
        logging.info(f"Extracted {len(urls)} normal/base64 image URLs in 'Exact matches'.")
        if urls:
            return list(urls)[:max_images]
    except Exception as e:
        logging.warning(f"Normal image extraction failed: {e}")

    try:
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
        page_source = driver.page_source
        url_patterns = [
            r'"(https?://[^"]+\.(?:jpg|jpeg|png|gif|bmp|webp)(?:\?[^"]*)?)"',
            r"'(https?://[^']+\.(?:jpg|jpeg|png|gif|bmp|webp)(?:\?[^']*)?)'",
            r'"ou":"(https?://[^"]+)"',
            r'imgurl=(https?://[^&]+)'
        ]
        for pattern in url_patterns:
            matches = re.findall(pattern, page_source, re.IGNORECASE)
            for match in matches:
                decoded_url = unquote(match)
                if decoded_url.startswith('http'):
                    urls.add(decoded_url)
        logging.info(f"Extracted {len(urls)} fallback visually similar image URLs.")
        return list(urls)[:max_images]
    except Exception as e:
        logging.error(f"Error extracting fallback Google URLs: {e}")
        return []

def get_image_from_url_or_base64(img_url, session, timeout=10):
    """Download and decode image from URL or base64 - using reliable original logic"""
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
    """Calculate FLANN similarity score - using the reliable original logic"""
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

def create_enhanced_comparison_image(source_path, matched_url, score, matches, metadata, temp_dir, session):
    """Enhanced comparison image with more information"""
    try:
        original_img = cv2.imread(source_path)
        if original_img is None: return None

        if matched_url.startswith("data:image"):
            _, base64_data = matched_url.split(',', 1)
            img_bytes = base64.b64decode(base64_data)
        else:
            headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://lens.google.com/'}
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

        cv2.putText(canvas, f"Original: {metadata['filename']}", (10, 30), font, small_font_scale, color, 1)
        cv2.putText(canvas, f"Size: {metadata['size'][0]}x{metadata['size'][1]} | {metadata['file_size_kb']} KB", (10, 55), font, small_font_scale, color, 1)

        right_x = original_resized.shape[1] + 10
        cv2.putText(canvas, "Google Lens Match", (right_x, 30), font, large_font_scale, color, 2)
        cv2.putText(canvas, f"FLANN Score: {score:.4f}", (right_x, 60), font, small_font_scale, color, 1)
        cv2.putText(canvas, f"Feature Matches: {matches}", (right_x, 85), font, small_font_scale, color, 1)

        source_name = Path(source_path).stem
        filename = f"{source_name}_comparison_{score:.4f}.jpg"
        filepath = os.path.join(temp_dir, 'similar_images', filename)
        if os.path.exists(filepath):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(temp_dir, 'similar_images', f"{source_name}_comparison_{score:.4f}_{timestamp}.jpg")
        
        cv2.imwrite(filepath, canvas)
        return filepath
    except Exception as e:
        logging.error(f"Failed to create enhanced comparison: {e}")
        return None

def process_single_image(driver, image_path, temp_dir, session, max_urls=50, max_workers=10, threshold=0.2):
    """Process a single image using the reliable original logic."""
    start_time = time.time()
    image_name = Path(image_path).name
    logging.info(f"Processing: {image_name}")

    metadata = get_image_metadata(image_path)
    thumbnail_path = create_thumbnail(image_path, temp_dir)
    
    # Create a result object regardless of match outcome
    base_result = {
        'source_image': image_name,
        'source_path': image_path,
        'thumbnail_path': thumbnail_path,
        'metadata': metadata,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'Processing'
    }
    
    upload_success = upload_to_google_lens_and_click_exact_match(driver, image_path)
    if not upload_success:
        base_result['status'] = 'No Match Found'
        base_result['processing_time'] = time.time() - start_time
        base_result['matched_url'] = None
        base_result['flann_score'] = 0.0
        base_result['num_matches'] = 0
        unmatched_path = copy_unmatched_image(image_path, temp_dir)
        base_result['unmatched_path'] = unmatched_path
        move_image_to_done(image_path, temp_dir)
        return base_result, time.time() - start_time, metadata

    upload_time = time.time()

    urls = get_exact_match_image_urls(driver, max_images=max_urls)
    if not urls:
        base_result['status'] = 'No Match Found'
        base_result['processing_time'] = time.time() - start_time
        base_result['matched_url'] = None
        base_result['flann_score'] = 0.0
        base_result['num_matches'] = 0
        base_result['upload_time'] = upload_time - start_time
        unmatched_path = copy_unmatched_image(image_path, temp_dir)
        base_result['unmatched_path'] = unmatched_path
        move_image_to_done(image_path, temp_dir)
        return base_result, time.time() - start_time, metadata
    
    extraction_time = time.time()

    source_img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img_cv is None:
        logging.error(f"Could not read source image {image_name} for processing.")
        base_result['status'] = 'Error - Could not read image'
        base_result['processing_time'] = time.time() - start_time
        return base_result, time.time() - start_time, metadata

    best_match = None
    best_score = 0.0

    def check_url(url):
        score, matches = calculate_flann_similarity(source_img_cv, url, session)
        return url, score, matches

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_url, url): url for url in urls}
        for future in as_completed(futures):
            try:
                url, score, matches = future.result()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = {
                        'source_image': image_name,
                        'source_path': image_path,
                        'thumbnail_path': thumbnail_path,
                        'matched_url': url,
                        'flann_score': score,
                        'num_matches': matches,
                        'metadata': metadata,
                        'processing_time': time.time() - start_time,
                        'upload_time': upload_time - start_time,
                        'extraction_time': extraction_time - upload_time,
                        'similarity_time': time.time() - extraction_time,
                        'total_urls_checked': len(urls),
                        'status': 'Match Found',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            except Exception:
                continue

    total_time = time.time() - start_time

    if best_match:
        logging.info(f"Best match for {image_name} with score {best_match['flann_score']:.4f}")
        comparison_path = create_enhanced_comparison_image(
            source_path=image_path,
            matched_url=best_match['matched_url'],
            score=best_match['flann_score'],
            matches=best_match['num_matches'],
            metadata=metadata,
            temp_dir=temp_dir,
            session=session
        )
        best_match['comparison_path'] = comparison_path
        move_image_to_done(image_path, temp_dir)
        return best_match, total_time, metadata
    else:
        # No match found above threshold
        base_result['status'] = 'No Match Found'
        base_result['processing_time'] = total_time
        base_result['matched_url'] = None
        base_result['flann_score'] = 0.0
        base_result['num_matches'] = 0
        base_result['upload_time'] = upload_time - start_time
        base_result['extraction_time'] = extraction_time - upload_time
        base_result['similarity_time'] = time.time() - extraction_time
        base_result['total_urls_checked'] = len(urls)
        unmatched_path = copy_unmatched_image(image_path, temp_dir)
        base_result['unmatched_path'] = unmatched_path
        move_image_to_done(image_path, temp_dir)
        return base_result, total_time, metadata

def calculate_eta(current_index, total_images, start_time, avg_time_per_image):
    if current_index == 0 or avg_time_per_image <= 0:
        return "Calculating..."
    remaining_images = total_images - current_index
    estimated_remaining_seconds = remaining_images * avg_time_per_image
    eta_datetime = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
    return eta_datetime.strftime("%H:%M:%S")

def create_analytics_dashboard(results):
    if not results: return
    
    st.header("📊 QC Analytics Dashboard")
    
    # Separate matched and unmatched
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
    
    if matched_results:
        avg_score = sum(r['flann_score'] for r in matched_results) / len(matched_results)
        col5.metric("Avg Score", f"{avg_score:.3f}")
    else:
        col5.metric("Avg Score", "N/A")

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if matched_results:
            scores = [r['flann_score'] for r in matched_results]
            fig1 = px.histogram(x=scores, nbins=20, title="FLANN Score Distribution (Matched Images)", 
                               labels={'x': 'FLANN Score', 'y': 'Count'})
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No matched images to display score distribution")

    with chart_col2:
        # Pie chart showing match vs no match
        pie_data = pd.DataFrame({
            'Status': ['Matched', 'Unmatched'],
            'Count': [total_matches, total_unmatched]
        })
        fig2 = px.pie(pie_data, values='Count', names='Status', 
                     title="Match Distribution",
                     color_discrete_map={'Matched': '#00CC88', 'Unmatched': '#FF6666'})
        st.plotly_chart(fig2, use_container_width=True)

    # Processing time analysis
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
            if matched_results and any('upload_time' in r for r in matched_results):
                stage_times = {
                    'Upload': sum(r.get('upload_time', 0) for r in matched_results) / len(matched_results),
                    'URL Extraction': sum(r.get('extraction_time', 0) for r in matched_results) / len(matched_results),
                    'Similarity Calc': sum(r.get('similarity_time', 0) for r in matched_results) / len(matched_results)
                }
                fig4 = px.bar(x=list(stage_times.keys()), y=list(stage_times.values()),
                            title="Average Time per Processing Stage",
                            labels={'x': 'Stage', 'y': 'Time (seconds)'})
                st.plotly_chart(fig4, use_container_width=True)

def create_image_preview_section(results):
    """Create preview sections for matched and unmatched images"""
    if not results:
        st.info("No images processed yet")
        return
    
    st.header("🖼️ Image Preview")
    
    # Separate matched and unmatched
    matched_results = [r for r in results if r and r.get('status') == 'Match Found']
    unmatched_results = [r for r in results if r and r.get('status') == 'No Match Found']
    
    # Create tabs for matched and unmatched
    preview_tab1, preview_tab2 = st.tabs([f"✅ Matched Images ({len(matched_results)})", 
                                          f"❌ Unmatched Images ({len(unmatched_results)})"])
    
    with preview_tab1:
        if matched_results:
            st.success(f"Found {len(matched_results)} matched images")
            
            # Display in grid
            cols_per_row = 3
            for i in range(0, len(matched_results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(matched_results):
                        result = matched_results[i + j]
                        with col:
                            st.markdown(f"**{result['source_image']}**")
                            st.caption(f"Score: {result['flann_score']:.4f}")
                            
                            # Show comparison image if available
                            if result.get('comparison_path') and os.path.exists(result['comparison_path']):
                                st.image(result['comparison_path'], use_column_width=True)
                            elif result.get('thumbnail_path') and os.path.exists(result['thumbnail_path']):
                                st.image(result['thumbnail_path'], use_column_width=True)
                            
                            st.caption(f"Matches: {result.get('num_matches', 0)}")
                            with st.expander("Details"):
                                st.write(f"Processing Time: {result.get('processing_time', 0):.2f}s")
                                st.write(f"URLs Checked: {result.get('total_urls_checked', 0)}")
        else:
            st.info("No matched images found")
    
    with preview_tab2:
        if unmatched_results:
            st.warning(f"Found {len(unmatched_results)} unmatched images")
            
            # Display in grid
            cols_per_row = 4
            for i in range(0, len(unmatched_results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(unmatched_results):
                        result = unmatched_results[i + j]
                        with col:
                            st.markdown(f"**{result['source_image']}**")
                            
                            # Show thumbnail if available
                            if result.get('thumbnail_path') and os.path.exists(result['thumbnail_path']):
                                st.image(result['thumbnail_path'], use_column_width=True)
                            
                            st.caption("No match found")
                            with st.expander("Details"):
                                st.write(f"Processing Time: {result.get('processing_time', 0):.2f}s")
                                metadata = result.get('metadata', {})
                                st.write(f"Size: {metadata.get('size', ['?', '?'])[0]}x{metadata.get('size', ['?', '?'])[1]}")
                                st.write(f"File Size: {metadata.get('file_size_kb', 0)} KB")
        else:
            st.info("No unmatched images found")

def create_detailed_results_table(results):
    """Create comprehensive table including both matched and unmatched images"""
    if not results:
        st.info("No results to display")
        return None
    
    st.subheader("📋 Complete Results Table")
    
    # Add filter options
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        show_filter = st.selectbox("Show:", ["All", "Matched Only", "Unmatched Only"])
    with filter_col2:
        sort_by = st.selectbox("Sort by:", ["Image Name", "Status", "Score", "Processing Time"])
    with filter_col3:
        sort_order = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)
    
    # Prepare table data
    table_data = []
    for r in results:
        if r:
            metadata = r.get('metadata', {})
            table_data.append({
                'Image': r['source_image'],
                'Status': r.get('status', 'Unknown'),
                'Score': r.get('flann_score', 0.0),
                'Matches': r.get('num_matches', 0),
                'Time (s)': round(r.get('processing_time', 0), 2),
                'Size': f"{metadata.get('size', ['?', '?'])[0]}x{metadata.get('size', ['?', '?'])[1]}",
                'File Size (KB)': metadata.get('file_size_kb', 0),
                'URLs Checked': r.get('total_urls_checked', 0),
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
    
    # Display table with conditional formatting
    def highlight_status(row):
        if row['Status'] == 'Match Found':
            return ['background-color: #d4f4dd'] * len(row)
        elif row['Status'] == 'No Match Found':
            return ['background-color: #ffd4d4'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = df.style.apply(highlight_status, axis=1).format({
        'Score': '{:.4f}',
        'Time (s)': '{:.2f}'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Summary statistics
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
            'average_score': round(sum(r['flann_score'] for r in matched_results) / len(matched_results), 4) if matched_results else 0,
            'average_processing_time': round(sum(r.get('processing_time', 0) for r in results) / len(results), 2) if results else 0,
            'report_generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'matched_images': matched_results,
        'unmatched_images': unmatched_results,
        'all_results': results
    }
    return report

def main_enhanced():
    st.markdown("""
    <style> 
    .main-header { 
        text-align: center; 
        padding: 1rem 0; 
        background: linear-gradient(90deg, #1f77b4, #17a2b8); 
        color: white; 
        margin: -1rem -1rem 2rem -1rem; 
        border-radius: 0 0 10px 10px; 
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-header"><h1>🔍 Enhanced Google Lens QC Tool</h1></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🛠️ Processing Configuration")
        max_urls = st.slider("Max results per image", 10, 100, 50)
        max_workers = st.slider("Concurrent workers", 5, 25, 10)
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.2, 0.05)
        batch_delay = st.slider("Delay between images (s)", 0.0, 3.0, 1.0, 0.1)
        headless_mode = st.checkbox("Headless browser", value=True)
        show_logs = st.checkbox("Show processing logs")
        
        st.divider()
        st.header("📊 Quick Stats")
        if st.session_state.results:
            matched = len([r for r in st.session_state.results if r and r.get('status') == 'Match Found'])
            unmatched = len([r for r in st.session_state.results if r and r.get('status') == 'No Match Found'])
            st.metric("Total Processed", len(st.session_state.results))
            st.metric("Matched", matched, delta=f"{matched/(len(st.session_state.results))*100:.1f}%")
            st.metric("Unmatched", unmatched)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Upload & Process", 
        "🖼️ Image Preview", 
        "📊 Analytics", 
        "📋 Results Table", 
        "📄 Reports",
        "📝 Logs"
    ])

    with tab1:
        st.header("📁 Image Upload")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"✅ {len(uploaded_files)} files uploaded")

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
                    st.rerun()
            with process_col4:
                if st.session_state.results and st.button("🔄 Clear Results", use_container_width=True):
                    st.session_state.results = []
                    st.session_state.matched_images = []
                    st.session_state.unmatched_images = []
                    st.rerun()

        if st.session_state.processing and st.session_state.uploaded_files and not st.session_state.paused:
            process_images_enhanced(max_urls, max_workers, threshold, batch_delay, headless_mode)
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
            st.info("Enable 'Show processing logs' in the sidebar to view logs")

def process_images_enhanced(max_urls, max_workers, threshold, batch_delay, headless_mode):
    st.header("🔄 Processing Images")

    if st.session_state.temp_dir is None:
        st.session_state.temp_dir = tempfile.mkdtemp()
        create_temp_folders(st.session_state.temp_dir)

    if st.session_state.current_index == 0:
        with st.spinner("Preparing files..."):
            image_paths, duplicates = save_uploaded_files(st.session_state.uploaded_files, st.session_state.temp_dir)
            st.session_state.processing_queue = image_paths
            if duplicates: 
                st.warning(f"⚠️ Skipped {len(duplicates)} duplicate files: {', '.join(duplicates)}")

    total_images = len(st.session_state.processing_queue)

    if st.session_state.current_index == 0:
        with st.spinner("🌐 Initializing browser..."):
            driver = init_driver(headless=headless_mode)
            session = get_requests_session()
            st.session_state.driver = driver
            st.session_state.session = session
    else:
        driver = st.session_state.driver
        session = st.session_state.session

    if driver is None:
        st.error("❌ Failed to initialize browser. Please try again.")
        st.session_state.processing = False
        return

    progress_container = st.container()
    results_stream = st.container()

    while st.session_state.current_index < total_images and st.session_state.processing and not st.session_state.paused:
        current_image = st.session_state.processing_queue[st.session_state.current_index]
        
        with progress_container:
            progress = (st.session_state.current_index + 1) / total_images
            progress_text = f"Processing {os.path.basename(current_image)} ({st.session_state.current_index + 1}/{total_images})"
            st.progress(progress, text=progress_text)
            
            # Display metrics
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

        result, processing_time, metadata = process_single_image(
            driver, current_image, st.session_state.temp_dir, session,
            max_urls=max_urls, max_workers=max_workers, threshold=threshold
        )
        
        st.session_state.results.append(result)
        st.session_state.performance_metrics.append(processing_time)
        
        # Track matched and unmatched separately
        if result and result.get('status') == 'Match Found':
            st.session_state.matched_images.append(result)
        elif result and result.get('status') == 'No Match Found':
            st.session_state.unmatched_images.append(result)
        
        st.session_state.current_index += 1

        with results_stream:
            if result:
                if result.get('status') == 'Match Found':
                    st.success(f"✅ Match found for {result['source_image']} (Score: {result['flann_score']:.3f})")
                    if result.get('comparison_path') and os.path.exists(result['comparison_path']):
                        with st.expander("View Comparison"):
                            st.image(result['comparison_path'])
                else:
                    st.warning(f"❌ No match found for {result['source_image']}")
                    if result.get('thumbnail_path') and os.path.exists(result['thumbnail_path']):
                        with st.expander("View Image"):
                            st.image(result['thumbnail_path'])
        
        if batch_delay > 0 and st.session_state.current_index < total_images:
            time.sleep(batch_delay)
        
        st.rerun()

    if st.session_state.current_index >= total_images and st.session_state.processing:
        st.session_state.processing = False
        if hasattr(st.session_state, 'driver') and st.session_state.driver:
            st.session_state.driver.quit()
        
        st.success("🎉 Processing Complete!")
        st.balloons()
        
        # Show final summary
        matched_count = len([r for r in st.session_state.results if r and r.get('status') == 'Match Found'])
        unmatched_count = len([r for r in st.session_state.results if r and r.get('status') == 'No Match Found'])
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Total Processed", len(st.session_state.results))
        with summary_col2:
            st.metric("Matched", matched_count)
        with summary_col3:
            st.metric("Unmatched", unmatched_count)

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
        # CSV Export with all results
        if st.button("📊 Generate Full CSV Report", use_container_width=True):
            csv_buffer = io.StringIO()
            all_data = []
            for r in st.session_state.results:
                if r:
                    metadata = r.get('metadata', {})
                    all_data.append({
                        'Image': r['source_image'],
                        'Status': r.get('status', 'Unknown'),
                        'FLANN Score': r.get('flann_score', 0),
                        'Feature Matches': r.get('num_matches', 0),
                        'Processing Time (s)': round(r.get('processing_time', 0), 2),
                        'Image Size': f"{metadata.get('size', ['?', '?'])[0]}x{metadata.get('size', ['?', '?'])[1]}",
                        'File Size (KB)': metadata.get('file_size_kb', 0),
                        'URLs Checked': r.get('total_urls_checked', 0),
                        'Timestamp': r.get('timestamp', 'N/A')
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
            if hasattr(st.session_state, 'driver') and st.session_state.driver:
                st.session_state.driver.quit()
            if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
            
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main_enhanced()