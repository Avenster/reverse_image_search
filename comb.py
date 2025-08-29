# main_app.py

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
from PIL import Image
import zipfile
import pandas as pd
import plotly.express as px
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

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'processing': False,
        'results': [],
        'uploaded_files': [],
        'temp_dir': None,
        'processing_queue': [],
        'current_index': 0,
        'start_time': None,
        'performance_metrics': [],
        'google_driver': None,
        'bing_driver': None,
        'session': None
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

class StreamlitLogHandler(logging.Handler):
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

log_handler = StreamlitLogHandler()
logging.getLogger().addHandler(log_handler)

# --- Utility & Helper Functions ---

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

def create_thumbnail(image_path, temp_dir, size=(200, 200)):
    """Creates a thumbnail for a given image."""
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
    """Saves uploaded files, checks for duplicates using hashes, and returns unique file paths."""
    image_paths, duplicates, hashes = [], [], set()
    images_dir = os.path.join(temp_dir, 'images')
    for uploaded_file in uploaded_files:
        file_path = os.path.join(images_dir, uploaded_file.name)
        file_bytes = uploaded_file.getbuffer()
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        
        file_hash = hashlib.md5(file_bytes).hexdigest()
        if file_hash in hashes:
            duplicates.append(uploaded_file.name)
            os.remove(file_path)
        else:
            hashes.add(file_hash)
            image_paths.append(file_path)
    return image_paths, duplicates

def copy_unmatched_image(image_path, temp_dir):
    """Copies an image that had no matches to the 'unmatched' folder for later export."""
    try:
        unmatched_folder = Path(os.path.join(temp_dir, 'unmatched'))
        source = Path(image_path)
        destination = unmatched_folder / source.name
        if not destination.exists():
            shutil.copy2(str(source), str(destination))
            logging.info(f"Copied unmatched '{source.name}' to 'unmatched' folder.")
            return str(destination)
        return str(destination) # Return path even if it already exists
    except Exception as e:
        logging.error(f"Failed to copy unmatched '{image_path}': {e}")
        return None

# --- Image Processing & Similarity Calculation ---

def get_image_from_url_or_base64(img_url, session, timeout=10):
    """Downloads image from a URL or decodes a base64 string."""
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
            headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.google.com/'}
            resp = session.get(img_url, timeout=timeout, headers=headers, verify=False)
            resp.raise_for_status()
            img_np = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None

def calculate_flann_similarity(img1_cv, img2_url, session, min_matches=8, timeout=15):
    """Calculates image similarity score using ORB and FLANN."""
    try:
        if img1_cv is None: return 0.0, 0
        img2 = get_image_from_url_or_base64(img2_url, session, timeout=timeout)
        if img2 is None: return 0.0, 0

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
        if len(good) > min_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                return min(np.sum(mask) / len(good), 1.0), np.sum(mask)
        return 0.0, 0
    except Exception:
        return 0.0, 0

def create_enhanced_comparison_image(source_path, matched_url, score, matches, engine_name, temp_dir, session):
    """Creates a side-by-side comparison image with metadata."""
    try:
        original_img = cv2.imread(source_path)
        if matched_url.startswith("data:image"):
            _, base64_data = matched_url.split(',', 1)
            img_bytes = base64.b64decode(base64_data)
        else:
            headers = {'User-Agent': 'Mozilla/5.0', 'Referer': f'https://www.{engine_name.lower()}.com/'}
            img_bytes = session.get(matched_url, headers=headers, verify=False).content
        
        similar_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if original_img is None or similar_img is None: return None

        target_height, text_area_height = 500, 80
        h1, w1 = original_img.shape[:2]
        h2, w2 = similar_img.shape[:2]
        r1, r2 = target_height / h1, target_height / h2
        
        original_resized = cv2.resize(original_img, (int(w1 * r1), target_height))
        similar_resized = cv2.resize(similar_img, (int(w2 * r2), target_height))

        total_width = original_resized.shape[1] + similar_resized.shape[1]
        canvas = np.full((target_height + text_area_height, total_width, 3), 255, dtype=np.uint8)
        canvas[text_area_height:, :original_resized.shape[1]] = original_resized
        canvas[text_area_height:, original_resized.shape[1]:] = similar_resized

        font, scale, color, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
        cv2.putText(canvas, f"Original: {Path(source_path).name}", (10, 30), font, scale, color, thick)
        right_x = original_resized.shape[1] + 10
        cv2.putText(canvas, f"Best Match via {engine_name}", (right_x, 30), font, scale, color, thick)
        cv2.putText(canvas, f"FLANN Score: {score:.4f} ({matches} matches)", (right_x, 60), font, scale, color, thick)

        source_name = Path(source_path).stem
        filename = f"{source_name}_{engine_name}_comparison.jpg"
        filepath = os.path.join(temp_dir, 'similar_images', filename)
        cv2.imwrite(filepath, canvas)
        return filepath
    except Exception as e:
        logging.error(f"Failed to create comparison image: {e}")
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
        # Scroll to load more images
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

# --- Core Processing Logic ---

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
    
    thumbnail_path = create_thumbnail(image_path, temp_dir)

    base_result = {
        'source_image': image_name,
        'source_path': image_path,
        'thumbnail_path': thumbnail_path,
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
            temp_dir=temp_dir, session=st.session_state.session
        )
    else:
        base_result.update({
            'status': 'No Match Found',
            'unmatched_path': copy_unmatched_image(image_path, temp_dir)
        })
        
    # Move original image to 'done' folder after processing
    try:
        done_folder = os.path.join(temp_dir, 'done')
        if not os.path.exists(os.path.join(done_folder, image_name)):
            shutil.move(image_path, done_folder)
    except Exception as e:
        logging.warning(f"Could not move {image_name} to done folder: {e}")
        
    processing_time = time.time() - start_time
    base_result['processing_time'] = processing_time
    return base_result, processing_time

# --- UI Components ---

def create_analytics_dashboard(results):
    st.header("📊 QC Analytics Dashboard")
    if not results:
        st.info("Process images to see analytics.")
        return

    df = pd.DataFrame(results)
    
    total_processed = len(df)
    matches = df[df['status'] == 'Match Found']
    total_matches = len(matches)
    total_unmatched = total_processed - total_matches
    match_rate = (total_matches / total_processed * 100) if total_processed > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Processed", total_processed)
    col2.metric("✅ Matches Found", total_matches)
    col3.metric("❌ No Matches", total_unmatched)
    col4.metric("Match Rate", f"{match_rate:.1f}%")
    col5.metric("Avg Score (Matches)", f"{matches['best_score'].mean():.3f}" if total_matches > 0 else "N/A")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if total_matches > 0:
            engine_counts = matches['best_engine'].value_counts().reset_index()
            engine_counts.columns = ['Engine', 'Count']
            fig = px.pie(engine_counts, values='Count', names='Engine', title="Match Distribution by Search Engine",
                         color_discrete_map={'Google': '#4285F4', 'Bing': '#0078D4'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No matches found to display engine distribution.")
            
    with chart_col2:
        if total_matches > 0:
            fig = px.histogram(matches, x='best_score', nbins=20, title="FLANN Score Distribution (Matched Images)",
                               labels={'best_score': 'FLANN Score'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scores to display.")

def create_image_preview_section(results):
    st.header("🖼️ Image Previews")
    if not results:
        st.info("Process images to see previews.")
        return

    cols_per_row = 4
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, result in enumerate(results[i : i + cols_per_row]):
            with cols[j]:
                st.markdown(f"**{result['source_image']}**")
                if result['status'] == 'Match Found':
                    st.image(result['comparison_path'], use_column_width=True)
                    st.success(f"Match via {result['best_engine']} (Score: {result['best_score']:.3f})")
                else:
                    st.image(result['thumbnail_path'], use_column_width=True)
                    st.warning("No match found")
                with st.expander("Details"):
                    st.metric("Time", f"{result['processing_time']:.2f}s")
                    st.metric("Google Score", f"{result['google_score']:.4f}")
                    st.metric("Bing Score", f"{result['bing_score']:.4f}")

def create_detailed_results_table(results):
    st.header("📋 Detailed Results Table")
    if not results:
        st.info("No results to display.")
        return None

    df = pd.DataFrame(results)
    display_cols = {
        'source_image': 'Image', 'status': 'Status', 'best_engine': 'Engine',
        'best_score': 'Score', 'google_score': 'Google Score', 'bing_score': 'Bing Score',
        'processing_time': 'Time (s)', 'matched_url': 'Match URL'
    }
    # Ensure all columns exist before selection
    cols_to_select = [k for k in display_cols.keys() if k in df.columns]
    df_display = df[cols_to_select].copy()
    df_display.rename(columns=display_cols, inplace=True)
    
    st.dataframe(df_display.style.format({
        'Score': '{:.4f}', 'Google Score': '{:.4f}', 'Bing Score': '{:.4f}', 'Time (s)': '{:.2f}'
    }), use_container_width=True, height=500)
    
    return df_display

def create_reports_section():
    st.header("📄 Reports & Export")
    if not st.session_state.results:
        st.info("Process images to generate reports.")
        return

    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        df = pd.DataFrame(st.session_state.results)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📊 Download Full Report (CSV)", csv_data, 
            f"qc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            "text/csv", use_container_width=True
        )

    with export_col2:
        matched_results = [r for r in st.session_state.results if r.get('comparison_path')]
        if matched_results:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_f:
                for r in matched_results:
                    if os.path.exists(r['comparison_path']):
                        zip_f.write(r['comparison_path'], os.path.basename(r['comparison_path']))
            st.download_button(
                "🖼️ Download Comparison Images (ZIP)", zip_buffer.getvalue(),
                f"comparison_images_{datetime.now().strftime('%Y%m%d')}.zip",
                "application/zip", use_container_width=True, type="secondary"
            )
        else:
            st.button("🖼️ No Comparison Images", disabled=True, use_container_width=True)

    with export_col3:
        unmatched_folder = os.path.join(st.session_state.temp_dir, 'unmatched')
        if os.path.exists(unmatched_folder) and any(os.scandir(unmatched_folder)):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_f:
                for root, _, files in os.walk(unmatched_folder):
                    for file in files:
                        zip_f.write(os.path.join(root, file), file)
            st.download_button(
                "❌ Download Unmatched Images (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"unmatched_images_{datetime.now().strftime('%Y%m%d')}.zip",
                mime="application/zip", use_container_width=True, type="secondary"
            )
        else:
            st.button("❌ No Unmatched Images", disabled=True, use_container_width=True)


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
        headless_mode = st.checkbox("Run in Headless Mode", value=False, help="Run browsers in the background.")
        
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
    
    config = {
        'batch_delay': batch_delay, 'headless_mode': headless_mode,
        'google_max_urls': google_max_urls, 'google_threshold': google_threshold,
        'bing_max_urls': bing_max_urls, 'bing_threshold': bing_threshold,
    }

    # --- Main Content Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 UPLOAD & PROCESS", "🖼️ IMAGE PREVIEWS", "📊 ANALYTICS", "📋 RESULTS TABLE", "📄 REPORTS & EXPORT"
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
            process_col1, process_col2, _ = st.columns([1,1,2])
            with process_col1:
                if not st.session_state.processing and st.button("🚀 Start Processing", type="primary", use_container_width=True):
                    st.session_state.processing = True
                    st.session_state.results = []
                    st.session_state.current_index = 0
                    st.session_state.start_time = time.time()
                    st.session_state.performance_metrics = []
                    st.rerun()
            with process_col2:
                if st.session_state.processing and st.button("⏹️ Stop Processing", use_container_width=True):
                    st.session_state.processing = False
                    quit_drivers()
                    st.rerun()
        
        if st.session_state.processing and st.session_state.uploaded_files:
            run_processing_loop(config)

    with tab2:
        create_image_preview_section(st.session_state.results)

    with tab3:
        create_analytics_dashboard(st.session_state.results)
    
    with tab4:
        create_detailed_results_table(st.session_state.results)

    with tab5:
        create_reports_section()

def run_processing_loop(config):
    """The main loop that controls the image processing flow."""
    if st.session_state.current_index == 0:
        st.session_state.temp_dir = tempfile.mkdtemp()
        create_temp_folders(st.session_state.temp_dir)
        with st.spinner("Saving uploaded files..."):
            image_paths, duplicates = save_uploaded_files(st.session_state.uploaded_files, st.session_state.temp_dir)
            st.session_state.processing_queue = image_paths
        if duplicates:
            st.warning(f"⚠️ Skipped {len(duplicates)} duplicate files.")
        
        if not initialize_drivers(headless=config['headless_mode']):
            st.session_state.processing = False
            return
        st.session_state.session = get_requests_session()
    
    total_images = len(st.session_state.processing_queue)
    
    if not st.session_state.processing or st.session_state.current_index >= total_images:
        st.success("🎉 Processing Complete!")
        st.balloons()
        quit_drivers()
        st.session_state.processing = False
        st.rerun()
        return

    progress_container = st.container()
    results_stream = st.container()

    current_image = st.session_state.processing_queue[st.session_state.current_index]

    with progress_container:
        progress = (st.session_state.current_index) / total_images
        progress_text = f"Processing {Path(current_image).name} ({st.session_state.current_index + 1}/{total_images})"
        st.progress(progress, text=progress_text)
    
    result, p_time = process_single_image_dual(current_image, st.session_state.temp_dir, config)
    st.session_state.results.append(result)
    st.session_state.performance_metrics.append(p_time)

    with results_stream:
        if result['status'] == 'Match Found':
            st.success(f"✅ Match found for **{result['source_image']}** via **{result['best_engine']}** (Score: {result['best_score']:.3f})")
        else:
            st.warning(f"❌ No match found for **{result['source_image']}**")

    st.session_state.current_index += 1
    if config['batch_delay'] > 0 and st.session_state.current_index < total_images:
        time.sleep(config['batch_delay'])
    
    st.rerun()

# --- Main Execution ---
if __name__ == "__main__":
    main_ui()