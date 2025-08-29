import streamlit as st
import time
import csv
import cv2
import numpy as np
import requests
import os
import base64
import re
from pathlib import Path
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
import logging
import shutil
from datetime import datetime
import subprocess
import platform
import urllib3
import tempfile
import io
from PIL import Image
import zipfile
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Page config
st.set_page_config(
    page_title="Yandex Visual Search FLANN Matcher",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)

# Initialize log handler
log_handler = StreamlitLogHandler()
logging.getLogger().addHandler(log_handler)

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

@st.cache_resource
def get_requests_session():
    return init_requests_session()

def kill_chrome_processes():
    """Kill existing Chrome processes to avoid conflicts - keeping original Yandex logic"""
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
    time.sleep(0.5)

def init_driver(headless=False):
    """Initialize undetected Chrome driver - keeping original Yandex logic"""
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
    user_data_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", f"chrome_data_{os.getpid()}")
    options.add_argument(f"--user-data-dir={user_data_dir}")
    
    try:
        logging.info("Initializing Chrome driver...")
        driver = uc.Chrome(options=options, version_main=None)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize Chrome driver: {e}")
        driver = uc.Chrome(options=options, use_subprocess=True)
        return driver

def create_temp_folders(temp_dir):
    """Create required folders in temp directory"""
    folders = ['done', 'similar_images']
    for folder in folders:
        os.makedirs(os.path.join(temp_dir, folder), exist_ok=True)

def save_uploaded_files(uploaded_files, temp_dir):
    """Save uploaded files to temporary directory"""
    image_paths = []
    images_dir = os.path.join(temp_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(images_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(file_path)
    
    return image_paths

def move_image_to_done(image_path, temp_dir):
    """Move processed image to done folder - keeping original Yandex logic"""
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

def create_comparison_image(source_path, similar_img_bytes, score, matches, temp_dir):
    """Create a side-by-side comparison image - keeping original Yandex logic"""
    try:
        original_img = cv2.imread(source_path)
        if original_img is None:
            logging.error(f"Could not read source image: {source_path}")
            return None
        
        similar_img_np = np.frombuffer(similar_img_bytes, np.uint8)
        similar_img = cv2.imdecode(similar_img_np, cv2.IMREAD_COLOR)
        if similar_img is None:
            logging.error("Could not decode downloaded similar image.")
            return None
        
        # Resize images to target height
        target_height = 600
        original_h, original_w = original_img.shape[:2]
        similar_h, similar_w = similar_img.shape[:2]
        
        original_ratio = target_height / original_h
        original_resized = cv2.resize(original_img, (int(original_w * original_ratio), target_height))
        
        similar_ratio = target_height / similar_h
        similar_resized = cv2.resize(similar_img, (int(similar_w * similar_ratio), target_height))
        
        # Create stitched image with text area
        text_area_height = 80
        total_width = original_resized.shape[1] + similar_resized.shape[1]
        total_height = target_height + text_area_height
        
        stitched_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)
        stitched_image[text_area_height:total_height, :original_resized.shape[1]] = original_resized
        stitched_image[text_area_height:total_height, original_resized.shape[1]:] = similar_resized
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 0)
        thickness = 2
        
        text1 = "Original Image"
        text2 = f"Yandex | Score: {score:.4f} | Matches: {matches}"
        
        text1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text1_x = (original_resized.shape[1] - text1_size[0]) // 2
        text1_y = (text_area_height - text1_size[1]) // 2 + text1_size[1]
        
        text2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        text2_x = original_resized.shape[1] + (similar_resized.shape[1] - text2_size[0]) // 2
        text2_y = (text_area_height - text2_size[1]) // 2 + text2_size[1]
        
        cv2.putText(stitched_image, text1, (text1_x, text1_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(stitched_image, text2, (text2_x, text2_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        # Save the comparison image
        source_name = Path(source_path).stem
        save_folder = os.path.join(temp_dir, 'similar_images')
        filename = f"{source_name}_comparison_{score:.4f}.jpg"
        filepath = os.path.join(save_folder, filename)
        
        if os.path.exists(filepath):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(save_folder, f"{source_name}_comparison_{score:.4f}_{timestamp}.jpg")
        
        cv2.imwrite(filepath, stitched_image)
        logging.info(f"Saved comparison image: {os.path.basename(filepath)}")
        return filepath
    except Exception as e:
        logging.error(f"Failed to create comparison image for {source_path}: {e}")
        return None

def download_and_create_comparison(url, source_image_path, flann_score, num_matches, temp_dir, session, timeout=2):
    """Download image and create comparison - keeping original Yandex logic (ULTRA FAST VERSION)"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://yandex.com/'
        }
        # Single fast request - skip SSL verification, no retries
        response = session.get(url, timeout=timeout, headers=headers, verify=False)
        if response.status_code != 200:
            return None
            
        return create_comparison_image(
            source_path=source_image_path,
            similar_img_bytes=response.content,
            score=flann_score,
            matches=num_matches,
            temp_dir=temp_dir
        )
    except:
        # Skip any problematic URL immediately
        return None

def get_image_from_url_or_base64(img_url, session, timeout=2):
    """Get image from URL or base64 string - keeping original Yandex logic (ULTRA FAST VERSION)"""
    try:
        if img_url.startswith("data:image"):
            header, b64data = img_url.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
            return img_cv
        else:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://yandex.com/'
            }
            # Single fast request - skip SSL verification
            resp = session.get(img_url, timeout=timeout, headers=headers, verify=False)
            if resp.status_code != 200:
                return None
            img_np = np.frombuffer(resp.content, np.uint8)
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
            return img_cv
    except:
        # Skip any problematic URL immediately
        return None

def calculate_flann_similarity(img1_cv, img2_url, session, min_matches=8, timeout=2):
    """Calculate FLANN-based similarity between two images - keeping original Yandex logic (ULTRA FAST VERSION)"""
    try:
        if img1_cv is None:
            return 0.0, 0
        
        img2 = get_image_from_url_or_base64(img2_url, session, timeout=timeout)
        if img2 is None:
            return 0.0, 0
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img1_cv, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0.0, 0
        
        # FLANN parameters for ORB
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
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
    except:
        return 0.0, 0

def upload_to_yandex_and_navigate(driver, image_path, timeout=30, debug=False):
    """Upload image to Yandex and navigate to similar images - keeping original logic"""
    try:
        if debug:
            st.write(f"🔍 Navigating to Yandex Images...")
        driver.get("https://yandex.com/images/")
        wait = WebDriverWait(driver, timeout)
        
        # Wait for page to load
        time.sleep(1)
        
        if debug:
            st.write(f"📁 Looking for file input...")
        # Find file input with multiple selectors
        file_input_selectors = [
            "input[type='file']",
            "input[accept*='image']",
            ".input_type_file input",
            ".CbirSearchForm-FileInput input"
        ]
        
        file_input = None
        for selector in file_input_selectors:
            try:
                file_input = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if debug:
                    st.write(f"✅ Found file input with selector: {selector}")
                break
            except TimeoutException:
                continue
        
        if not file_input:
            logging.error("Could not find file input")
            if debug:
                st.error("Could not find file input")
            return False
        
        # Make file input visible and upload
        abs_path = os.path.abspath(image_path)
        driver.execute_script("""
            arguments[0].style.display = 'block';
            arguments[0].style.visibility = 'visible';
            arguments[0].style.opacity = '1';
            arguments[0].style.position = 'static';
        """, file_input)
        
        if debug:
            st.write(f"📤 Uploading image...")
        file_input.send_keys(abs_path)
        logging.info(f"Uploading: {os.path.basename(image_path)}")
        time.sleep(0.5)
        
        if debug:
            st.write(f"⏳ Waiting for results to load...")
        # Wait for results to load
        try:
            wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".CbirNavigation-TabsItem")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".SerpItem")),
                    EC.url_contains("cbir_id")
                )
            )
        except TimeoutException:
            logging.warning("Results page didn't load properly")
            if debug:
                st.warning("Results page didn't load properly")
            return False
        
        # Find and click Similar images tab with multiple fallback selectors
        similar_tab_selectors = [
            "a[data-cbir-page-type='similar']",
            ".CbirNavigation-TabsItem_name_similar-page",
            "a.CbirNavigation-TabsItem_name_similar-page",
            "//a[contains(text(), 'Similar') or contains(text(), 'Похожие')]"
        ]
        
        similar_tab = None
        for selector in similar_tab_selectors:
            try:
                if selector.startswith("//"):
                    similar_tab = wait.until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                else:
                    similar_tab = wait.until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                logging.info(f"Found similar tab with selector: {selector}")
                if debug:
                    st.write(f"✅ Found similar images tab")
                break
            except TimeoutException:
                continue
        
        # Additional fallback: Find by class and text content
        if not similar_tab:
            try:
                similar_tab = wait.until(
                    EC.element_to_be_clickable((By.XPATH, 
                        "//a[contains(@class, 'CbirNavigation-TabsItem') and (contains(text(), 'Similar') or contains(text(), 'Похожие') or contains(text(), 'похожие'))]"))
                )
                logging.info("Found similar tab by text content")
                if debug:
                    st.write(f"✅ Found similar images tab by text content")
            except TimeoutException:
                logging.error("Could not find Similar Images tab")
                if debug:
                    st.error("Could not find Similar Images tab")
                return False
        
        # Click the similar tab
        try:
            driver.execute_script("arguments[0].scrollIntoView(true);", similar_tab)
            time.sleep(0.5)
            try:
                similar_tab.click()
            except:
                driver.execute_script("arguments[0].click();", similar_tab)
            
            time.sleep(1)
            
            current_url = driver.current_url
            if "cbir_page=similar" in current_url or "similar" in current_url.lower():
                logging.info("Successfully navigated to similar images")
                if debug:
                    st.write("✅ Successfully navigated to similar images")
            else:
                logging.info(f"URL doesn't contain similar page indicator: {current_url}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error clicking similar tab: {str(e)}")
            if debug:
                st.error(f"Error clicking similar tab: {str(e)}")
            return False
        
    except TimeoutException:
        logging.error(f"Timed out waiting for Yandex elements for {image_path}.")
        if debug:
            st.error(f"Timeout waiting for Yandex elements")
        return False
    except Exception as e:
        logging.error(f"Upload to Yandex failed for {image_path}: {e}")
        if debug:
            st.error(f"Upload to Yandex failed: {str(e)}")
        return False

def get_yandex_image_urls(driver, max_images=20):
    """Extract image URLs from Yandex results - keeping original logic"""
    urls = set()
    
    # Scroll to load more images (commented out in original for speed)
    # for _ in range(3):
    #     driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    #     time.sleep(1.5)
    
    # Try multiple extraction methods
    try:
        # Method 1: Extract from img_url parameters
        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='img_url=']")
        for link in links:
            href = link.get_attribute("href")
            if href and "img_url=" in href:
                url_match = href.split("img_url=")[1].split("&")[0]
                decoded_url = unquote(url_match)
                if decoded_url.startswith("http"):
                    urls.add(decoded_url)
        
        # Method 2: Extract from data attributes
        elements = driver.find_elements(By.CSS_SELECTOR, "[data-bem*='http']")
        for elem in elements:
            data_bem = elem.get_attribute("data-bem")
            if data_bem:
                url_pattern = r'https?://[^"\'\\s,}]+'
                matches = re.findall(url_pattern, data_bem)
                for url in matches:
                    if not any(x in url for x in ['yandex', 'yastatic']):
                        urls.add(url)
        
        # Method 3: Direct image sources
        images = driver.find_elements(By.CSS_SELECTOR, "img[src]")
        for img in images:
            src = img.get_attribute("src")
            if src and src.startswith("http") and not any(x in src for x in ['yandex', 'yastatic']):
                urls.add(src)
                
    except Exception as e:
        logging.warning(f"Error extracting Yandex URLs: {e}")
    
    logging.info(f"Extracted {len(urls)} image URLs from Yandex.")
    return list(urls)[:max_images]

def process_single_image(driver, image_path, temp_dir, session, max_urls=20, max_workers=10, threshold=0.9, debug=False):
    """Process a single image - keeping original Yandex logic"""
    image_name = Path(image_path).name
    start_time = time.time()
    logging.info(f"Processing: {image_name}")
    
    if debug:
        st.write(f"🔍 Processing {image_name}")
    
    # Upload to Yandex
    if not upload_to_yandex_and_navigate(driver, image_path, debug=debug):
        if debug:
            st.warning("Upload failed")
        return None
    
    # Get image URLs
    urls = get_yandex_image_urls(driver, max_images=max_urls)
    if not urls:
        if debug:
            st.warning("No image URLs found")
        return None
    
    if debug:
        st.write(f"Found {len(urls)} URLs to check")
    
    # Load source image
    source_img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img_cv is None:
        logging.error(f"Could not read source image {image_name}")
        return None
    
    # Find best match using FLANN
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
                        'matched_url': url,
                        'flann_score': score,
                        'num_matches': matches
                    }
            except:
                continue
    
    processing_time = time.time() - start_time
    
    if best_match:
        logging.info(f"Best match found for {image_name} with score {best_match['flann_score']:.4f}")
        
        # Download and create comparison image
        if best_match['matched_url'].startswith("data:image"):
            header, base64_data = best_match['matched_url'].split(',', 1)
            img_bytes = base64.b64decode(base64_data)
            comparison_path = create_comparison_image(
                source_path=image_path,
                similar_img_bytes=img_bytes,
                score=best_match['flann_score'],
                matches=best_match['num_matches'],
                temp_dir=temp_dir
            )
        else:
            comparison_path = download_and_create_comparison(
                url=best_match['matched_url'],
                source_image_path=image_path,
                flann_score=best_match['flann_score'],
                num_matches=best_match['num_matches'],
                temp_dir=temp_dir,
                session=session
            )
        
        best_match['comparison_path'] = comparison_path
        
        # Move image to done folder only if match found
        move_image_to_done(image_path, temp_dir)
        
        return best_match
    else:
        logging.info(f"No match found for {image_name} above threshold {threshold}.")
        if debug:
            st.info(f"No match found above threshold {threshold}")
        return None

def main_streamlit():
    st.title("🔍 Yandex Visual Search FLANN Matcher")
    st.markdown("Upload images to find visually similar matches using Yandex Images and FLANN feature matching.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_urls = st.slider("Max Yandex results per image", 5, 50, 20)
        max_workers = st.slider("Concurrent workers", 1, 12, 10)
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.9, 0.05)
        batch_delay = st.slider("Delay between images (seconds)", 0.5, 5.0, 0.5, 0.1)
        
        st.header("🐛 Debug Options")
        debug_mode = st.checkbox("Enable debug mode", help="Show detailed processing steps")
        headless_mode = st.checkbox("Run browser in headless mode", help="Hide browser window (may cause issues)")
        
        st.header("📊 Processing Stats")
        if st.session_state.results:
            st.metric("Images Processed", len(st.session_state.results))
            matches = [r for r in st.session_state.results if r is not None]
            st.metric("Matches Found", len(matches))
            if matches:
                avg_score = sum(m['flann_score'] for m in matches) / len(matches)
                st.metric("Average Score", f"{avg_score:.3f}")
    
    # File upload
    st.header("📁 Upload Images")
    uploaded_files = st.file_uploader(
        "Choose image files", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], 
        accept_multiple_files=True,
        help="Upload multiple images to process with Yandex Images"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Preview uploaded images
        if st.checkbox("Preview uploaded images"):
            cols = st.columns(min(4, len(uploaded_files)))
            for i, file in enumerate(uploaded_files[:8]):  # Show max 8 previews
                with cols[i % 4]:
                    image = Image.open(file)
                    st.image(image, caption=file.name, use_container_width=True)
        
        st.session_state.uploaded_files = uploaded_files
    
    # Start processing button
    if uploaded_files and not st.session_state.processing:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            st.session_state.processing = True
            st.session_state.results = []
            st.rerun()
    
    # Processing section
    if st.session_state.processing and st.session_state.uploaded_files:
        st.header("🔄 Processing Images")
        
        # Create temporary directory
        if st.session_state.temp_dir is None:
            st.session_state.temp_dir = tempfile.mkdtemp()
            create_temp_folders(st.session_state.temp_dir)
        
        # Save uploaded files
        image_paths = save_uploaded_files(st.session_state.uploaded_files, st.session_state.temp_dir)
        
        # Initialize driver and session
        with st.spinner("Initializing browser..."):
            driver = init_driver(headless=headless_mode)
            session = get_requests_session()
        
        if driver is None:
            st.error("Failed to initialize browser. Please try again.")
            st.session_state.processing = False
            st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        # Debug information container
        debug_container = st.container() if debug_mode else None
        
        total_images = len(image_paths)
        
        try:
            for idx, image_path in enumerate(image_paths):
                progress = idx / total_images
                progress_bar.progress(progress)
                status_text.text(f"Processing {os.path.basename(image_path)} ({idx + 1}/{total_images})")
                
                # Show debug info in expandable section
                if debug_mode:
                    with debug_container:
                        with st.expander(f"🐛 Debug: {os.path.basename(image_path)}", expanded=False):
                            # Process image with debug output
                            result = process_single_image(
                                driver, image_path, st.session_state.temp_dir, session,
                                max_urls=max_urls, max_workers=max_workers, 
                                threshold=threshold, debug=True
                            )
                else:
                    # Process image without debug output
                    result = process_single_image(
                        driver, image_path, st.session_state.temp_dir, session,
                        max_urls=max_urls, max_workers=max_workers, 
                        threshold=threshold, debug=False
                    )
                
                st.session_state.results.append(result)
                
                # Show result immediately
                with results_container:
                    if result:
                        st.success(f"✅ Match found for {result['source_image']} (Score: {result['flann_score']:.3f})")
                        if result.get('comparison_path') and os.path.exists(result['comparison_path']):
                            st.image(result['comparison_path'], caption=f"Comparison for {result['source_image']}")
                    else:
                        st.warning(f"❌ No match found for {os.path.basename(image_path)}")
                
                # Delay between images
                if idx < total_images - 1:
                    time.sleep(batch_delay)
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            st.session_state.processing = False
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.session_state.processing = False
        finally:
            if driver:
                driver.quit()
            
            # Cleanup temp Chrome data
            user_data_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", f"chrome_data_{os.getpid()}")
            if os.path.exists(user_data_dir):
                shutil.rmtree(user_data_dir, ignore_errors=True)
    
    # Results section
    if st.session_state.results and not st.session_state.processing:
        st.header("📋 Results Summary")
        
        matches = [r for r in st.session_state.results if r is not None]
        
        if matches:
            # Create results table
            results_data = []
            for match in matches:
                results_data.append({
                    'Image': match['source_image'],
                    'Score': f"{match['flann_score']:.4f}",
                    'Matches': match['num_matches'],
                    'URL': match['matched_url'][:100] + '...' if len(match['matched_url']) > 100 else match['matched_url']
                })
            
            st.dataframe(results_data, use_container_width=True)
            
            # Download results as CSV
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(['Source Image', 'Matched URL', 'FLANN Score', 'Feature Matches'])
            for match in matches:
                writer.writerow([
                    match['source_image'],
                    match['matched_url'],
                    f"{match['flann_score']:.4f}",
                    match['num_matches']
                ])
            
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_buffer.getvalue(),
                file_name=f"yandex_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Download comparison images as ZIP
            if any(m.get('comparison_path') for m in matches):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for match in matches:
                        if match.get('comparison_path') and os.path.exists(match['comparison_path']):
                            zip_file.write(match['comparison_path'], os.path.basename(match['comparison_path']))
                
                st.download_button(
                    label="📥 Download Comparison Images ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"yandex_comparison_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        else:
            st.info("No matches found above the similarity threshold.")
        
        # Reset button
        if st.button("🔄 Process New Images", use_container_width=True):
            st.session_state.processing = False
            st.session_state.results = []
            st.session_state.uploaded_files = []
            if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
            st.session_state.temp_dir = None
            st.rerun()

if __name__ == "__main__":
    main_streamlit()