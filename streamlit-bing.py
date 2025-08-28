import streamlit as st
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
import logging
import shutil
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import urllib3
import tempfile
import io
from PIL import Image
import zipfile

# Page config
st.set_page_config(
    page_title="Bing Visual Search FLANN Matcher",
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
logging.basicConfig(level=logging.INFO)

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

def init_driver(headless=False):
    """Initialize Chrome driver with optimized settings"""
    options = uc.ChromeOptions()
    
    # Use headless mode based on parameter
    if headless:
        options.add_argument("--headless=new")
    
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")  # Disable image loading for faster performance
    options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    try:
        driver = uc.Chrome(options=options, version_main=None)
        
        # Set longer timeouts
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)
        
        # Anti-detection measures
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                window.chrome = {
                    runtime: {}
                };
            '''
        })
        
        return driver
    except Exception as e:
        st.error(f"Failed to initialize Chrome driver: {e}")
        return None

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

def upload_to_bing(driver, image_path, timeout=30, debug=False):
    """Upload image to Bing visual search with improved error handling"""
    try:
        if debug:
            st.write(f"🔍 Navigating to Bing Images...")
        driver.get("https://www.bing.com/images")
        time.sleep(3)  # Give page time to load
        
        if debug:
            st.write(f"🎯 Looking for visual search button...")
        # Try multiple selectors for the visual search button
        visual_search_selectors = [
            "#sbi_b",
            ".sbi_b_prtl", 
            "[aria-label*='Visual Search']",
            "[aria-label*='visual search']",
            "[data-tooltip='Search by image']",
            ".camera_icon"
        ]
        
        visual_search_button = None
        for selector in visual_search_selectors:
            try:
                visual_search_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                if debug:
                    st.write(f"✅ Found visual search button with selector: {selector}")
                break
            except Exception:
                continue
        
        if not visual_search_button:
            st.error("❌ Could not find visual search button")
            return False
        
        visual_search_button.click()
        time.sleep(2)
        
        if debug:
            st.write(f"📁 Looking for file input...")
        # Find file input with multiple attempts
        file_input = None
        file_input_selectors = [
            "input[type='file']",
            "#sb_fileinput",
            "[accept*='image']"
        ]
        
        for selector in file_input_selectors:
            try:
                file_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if debug:
                    st.write(f"✅ Found file input with selector: {selector}")
                break
            except Exception:
                continue
        
        if not file_input:
            st.error("❌ Could not find file input")
            return False
        
        if debug:
            st.write(f"📤 Uploading image...")
        abs_path = os.path.abspath(image_path)
        file_input.send_keys(abs_path)
        
        if debug:
            st.write(f"⏳ Waiting for results page...")
        # Wait for results with multiple conditions
        WebDriverWait(driver, timeout).until(
            lambda d: any([
                "search?" in d.current_url,
                "detailV2" in d.current_url,
                "images/search" in d.current_url
            ])
        )
        
        if debug:
            st.write(f"✅ Upload successful!")
        return True
        
    except Exception as e:
        st.error(f"❌ Upload to Bing failed for {os.path.basename(image_path)}: {str(e)}")
        # Take screenshot for debugging
        if debug:
            try:
                screenshot_path = f"/tmp/bing_error_{int(time.time())}.png"
                driver.save_screenshot(screenshot_path)
                st.write(f"📸 Screenshot saved to: {screenshot_path}")
            except:
                pass
        return False

def get_bing_image_urls(driver, max_images=20):
    """Extract image URLs from Bing results"""
    urls = []
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        try:
            WebDriverWait(driver, 5).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.richImgLnk")))
        except Exception:
            time.sleep(0.6)
        
        rich_links = driver.find_elements(By.CSS_SELECTOR, "a.richImgLnk")
        for link in rich_links[:max_images]:
            m_data = link.get_attribute('data-m')
            if m_data:
                try:
                    data = json.loads(m_data)
                    if 'murl' in data: 
                        urls.append(data['murl'])
                    elif 'purl' in data: 
                        urls.append(data['purl'])
                except Exception:
                    continue
        
        if len(urls) < 10:
            page_source = driver.page_source
            urls.extend(re.findall(r'"murl":"(https?://[^"]+)"', page_source))
            urls.extend(re.findall(r'"purl":"(https?://[^"]+)"', page_source))
        
        cleaned = []
        seen = set()
        for url in urls:
            url = unquote(url)
            if url.startswith('http') and not ('bing.com/th' in url or 'mm.bing.net/th' in url):
                if url not in seen:
                    cleaned.append(url)
                    seen.add(url)
        
        return cleaned[:max_images]
    except Exception as e:
        logging.warning(f"Error extracting Bing URLs: {e}")
        return []

def get_image_from_url_or_base64(img_url, session, timeout=5):
    """Download and decode image from URL or base64"""
    if img_url.startswith("data:image"):
        try:
            header, b64data = img_url.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
            return img_cv
        except Exception:
            return None
    else:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.bing.com/'
            }
            try:
                resp = session.get(img_url, timeout=timeout, headers=headers, verify=True)
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
                resp = session.get(img_url, timeout=timeout, headers=headers, verify=False)
            
            resp.raise_for_status()
            img_np = np.frombuffer(resp.content, np.uint8)
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
            return img_cv
        except Exception:
            return None

def calculate_flann_similarity(img1_cv, img2_url, session, min_matches=8, timeout=5):
    """Calculate FLANN similarity score between two images"""
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

def create_comparison_image(source_path, similar_img_bytes, score, matches, temp_dir):
    """Create side-by-side comparison image"""
    try:
        original_img = cv2.imread(source_path)
        if original_img is None:
            return None
        
        similar_img_np = np.frombuffer(similar_img_bytes, np.uint8)
        similar_img = cv2.imdecode(similar_img_np, cv2.IMREAD_COLOR)
        if similar_img is None:
            return None
        
        target_height = 400
        original_h, original_w = original_img.shape[:2]
        similar_h, similar_w = similar_img.shape[:2]
        
        original_ratio = target_height / original_h
        original_new_w = int(original_w * original_ratio)
        original_resized = cv2.resize(original_img, (original_new_w, target_height), interpolation=cv2.INTER_AREA)
        
        similar_ratio = target_height / similar_h
        similar_new_w = int(similar_w * similar_ratio)
        similar_resized = cv2.resize(similar_img, (similar_new_w, target_height), interpolation=cv2.INTER_AREA)
        
        text_area_height = 60
        total_width = original_resized.shape[1] + similar_resized.shape[1]
        total_height = target_height + text_area_height
        
        stitched_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)
        
        stitched_image[text_area_height:total_height, 0:original_new_w] = original_resized
        stitched_image[text_area_height:total_height, original_new_w:total_width] = similar_resized
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)
        thickness = 2
        
        text1 = "Original"
        text2 = f"Match | Score: {score:.3f}"
        
        text1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text1_x = (original_new_w - text1_size[0]) // 2
        text1_y = (text_area_height - text1_size[1]) // 2 + text1_size[1]
        
        text2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        text2_x = original_new_w + (similar_new_w - text2_size[0]) // 2
        text2_y = (text_area_height - text2_size[1]) // 2 + text2_size[1]
        
        cv2.putText(stitched_image, text1, (text1_x, text1_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(stitched_image, text2, (text2_x, text2_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        source_name = Path(source_path).stem
        save_folder = os.path.join(temp_dir, 'similar_images')
        filename = f"{source_name}_comparison_{score:.4f}.jpg"
        filepath = os.path.join(save_folder, filename)
        
        cv2.imwrite(filepath, stitched_image)
        return filepath
    except Exception as e:
        logging.error(f"Failed to create comparison image: {e}")
        return None

def process_single_image(driver, image_path, temp_dir, session, max_images=20, max_workers=4, threshold=0.15, debug=False):
    """Process a single image and find matches"""
    image_name = os.path.basename(image_path)
    
    if not upload_to_bing(driver, image_path, debug=debug):
        return None
    
    urls = get_bing_image_urls(driver, max_images=max_images)
    if not urls:
        if debug:
            st.warning(f"No URLs found for {image_name}")
        return None
    
    if debug:
        st.write(f"Found {len(urls)} URLs to check for {image_name}")
    
    source_img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img_cv is None:
        return None
    
    best_match = None
    best_score = 0.0
    
    def check_url(url):
        score, matches = calculate_flann_similarity(source_img_cv, url, session)
        return url, score, matches
    
    worker_count = min(max_workers, max(1, len(urls)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
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
            except Exception:
                continue
    
    if best_match:
        # Download and create comparison
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            try:
                response = session.get(best_match['matched_url'], timeout=10, headers=headers, verify=True)
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
                response = session.get(best_match['matched_url'], timeout=10, headers=headers, verify=False)
            
            response.raise_for_status()
            
            comparison_path = create_comparison_image(
                image_path, response.content, best_match['flann_score'], 
                best_match['num_matches'], temp_dir
            )
            best_match['comparison_path'] = comparison_path
        except Exception as e:
            if debug:
                st.error(f"Failed to create comparison for {image_name}: {e}")
            best_match['comparison_path'] = None
    
    return best_match

def main_streamlit():
    st.title("🔍 Bing Visual Search FLANN Matcher")
    st.markdown("Upload images to find visually similar matches using Bing Visual Search and FLANN feature matching.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_images = st.slider("Max Bing results per image", 5, 50, 20)
        max_workers = st.slider("Concurrent workers", 1, 8, 4)
        threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.15, 0.05)
        batch_delay = st.slider("Delay between images (seconds)", 0.5, 5.0, 2.0, 0.5)
        
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
        type=['jpg', 'jpeg', 'png', 'bmp'], 
        accept_multiple_files=True,
        help="Upload multiple images to process"
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
                                max_images=max_images, max_workers=max_workers, 
                                threshold=threshold, debug=True
                            )
                else:
                    # Process image without debug output
                    result = process_single_image(
                        driver, image_path, st.session_state.temp_dir, session,
                        max_images=max_images, max_workers=max_workers, 
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
                file_name=f"bing_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
                    file_name=f"comparison_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        else:
            st.info("No matches found above the similarity threshold.")
        
        # Reset button
        if st.button("🔄 Process New Images", use_container_width=True):
            st.session_state.processing = False
            st.session_state.results = []
            st.session_state.uploaded_files = []
            st.session_state.temp_dir = None
            st.rerun()

if __name__ == "__main__":
    main_streamlit()