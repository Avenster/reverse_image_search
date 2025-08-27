#!/usr/bin/env python3
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
import urllib3

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# Module-level requests session (initialized in main)
requests_session = None

def init_requests_session(max_pool_connections=100, max_retries=1, backoff_factor=0.1):
    """
    Initialize a requests.Session with a HTTPAdapter that has a larger connection pool
    and a small retry policy to reduce overhead and transient failures.
    """
    session = requests.Session()
    # Reduce retries and backoff for faster processing
    retries = Retry(total=max_retries, backoff_factor=backoff_factor,
                    status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(pool_connections=max_pool_connections, pool_maxsize=max_pool_connections, max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    # Disable SSL verification warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless=new")  # Uncomment for headless
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    try:
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    except Exception:
        # If script injection fails, proceed anyway
        pass
    return driver

# -----------------------
# Create Required Folders
# -----------------------
def create_folders():
    folders = ['done', 'similar_images']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

# -----------------------
# Move Image to Done Folder
# -----------------------
def move_image_to_done(image_path):
    try:
        done_folder = Path('done')
        done_folder.mkdir(exist_ok=True)
        
        source = Path(image_path)
        destination = done_folder / source.name
        
        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = done_folder / f"{source.stem}_{timestamp}{source.suffix}"
        
        shutil.move(str(source), str(destination))
        logging.info(f"Moved {source.name} to done folder")
    except Exception as e:
        logging.error(f"Failed to move {image_path} to done folder: {e}")

# -----------------------
# Create and Save Stitched Comparison Image
# -----------------------
def create_comparison_image(source_path, similar_img_bytes, score, matches, output_folder='similar_images'):
    try:
        # Load original image from path
        original_img = cv2.imread(source_path)
        if original_img is None:
            logging.error(f"Could not read source image: {source_path}")
            return None
        
        # Load similar image from downloaded bytes
        similar_img_np = np.frombuffer(similar_img_bytes, np.uint8)
        similar_img = cv2.imdecode(similar_img_np, cv2.IMREAD_COLOR)
        if similar_img is None:
            logging.error("Could not decode downloaded similar image.")
            return None
        
        # --- Image Resizing to a common height ---
        target_height = 600
        original_h, original_w = original_img.shape[:2]
        similar_h, similar_w = similar_img.shape[:2]
        
        original_ratio = target_height / original_h
        original_new_w = int(original_w * original_ratio)
        original_resized = cv2.resize(original_img, (original_new_w, target_height), interpolation=cv2.INTER_AREA)
        
        similar_ratio = target_height / similar_h
        similar_new_w = int(similar_w * similar_ratio)
        similar_resized = cv2.resize(similar_img, (similar_new_w, target_height), interpolation=cv2.INTER_AREA)
        
        # --- Create Canvas for stitching and text ---
        text_area_height = 80
        total_width = original_resized.shape[1] + similar_resized.shape[1]
        total_height = target_height + text_area_height
        
        stitched_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8) # White canvas
        
        # Place resized images side-by-side
        stitched_image[text_area_height:total_height, 0:original_new_w] = original_resized
        stitched_image[text_area_height:total_height, original_new_w:total_width] = similar_resized
        
        # --- Add text to the top of the canvas ---
        text1 = "Original Image"
        text2 = f"Bing | Score: {score:.4f} | Matches: {matches}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 0) # Black
        thickness = 2
        
        # Center text above each respective image
        text1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text1_x = (original_new_w - text1_size[0]) // 2
        text1_y = (text_area_height - text1_size[1]) // 2 + text1_size[1]
        
        text2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        text2_x = original_new_w + (similar_new_w - text2_size[0]) // 2
        text2_y = (text_area_height - text2_size[1]) // 2 + text2_size[1]
        
        cv2.putText(stitched_image, text1, (text1_x, text1_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(stitched_image, text2, (text2_x, text2_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        # --- Save the final composite image ---
        source_name = Path(source_path).stem
        save_folder = Path(output_folder)
        save_folder.mkdir(exist_ok=True)
        filename = f"{source_name}_comparison_{score:.4f}.jpg"
        filepath = save_folder / filename
        
        if filepath.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = save_folder / f"{source_name}_comparison_{score:.4f}_{timestamp}.jpg"
            
        cv2.imwrite(str(filepath), stitched_image)
        logging.info(f"Saved comparison image: {filepath.name}")
        return str(filepath)
    except Exception as e:
        logging.error(f"Failed to create comparison image for {source_path}: {e}")
        return None

# -----------------------
# Download and Create Comparison Image - IMPROVED ERROR HANDLING
# -----------------------
def download_and_create_comparison(url, source_image_path, flann_score, num_matches, timeout=5):
    try:
        global requests_session
        session = requests_session or requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bing.com/'
        }
        
        # Try with SSL verification first, then without if it fails
        try:
            response = session.get(url, timeout=timeout, headers=headers, verify=True)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
            # Retry without SSL verification for problematic sites
            response = session.get(url, timeout=timeout, headers=headers, verify=False)
        
        response.raise_for_status()
        
        # Create the stitched comparison image
        stitched_image_path = create_comparison_image(
            source_path=source_image_path,
            similar_img_bytes=response.content,
            score=flann_score,
            matches=num_matches
        )
        
        return stitched_image_path
    except Exception as e:
        logging.error(f"Failed to download image from {url} for comparison: {e}")
        return None

# -----------------------
# Get Images from Folder
# -----------------------
def get_image_files(folder_path):
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    folder_path = Path(folder_path)
    if not folder_path.exists():
        logging.error(f"Folder '{folder_path}' does not exist!")
        return []
    for ext in supported_extensions:
        image_files.extend(glob.glob(str(folder_path / ext)))
        image_files.extend(glob.glob(str(folder_path / ext.upper())))
    return sorted(image_files)

# -----------------------
# Upload to Bing Visual Search
# -----------------------
def upload_to_bing(driver, image_path, timeout=20):
    try:
        driver.get("https://www.bing.com/images")
        # Wait for the visual search button
        WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#sbi_b, .sbi_b_prtl, [aria-label*='Visual Search'], [aria-label*='visual search']"))
        ).click()
        
        file_input = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        
        abs_path = os.path.abspath(image_path)
        file_input.send_keys(abs_path)
        
        # wait for the results page
        WebDriverWait(driver, timeout).until(
            lambda d: "search?" in d.current_url or "detailV2" in d.current_url
        )
        return True
    except Exception as e:
        logging.warning(f"Upload to Bing failed for {image_path}: {e}")
        return False

# -----------------------
# Extract Image URLs from Bing Results (LIMITED TO 20)
# -----------------------
def get_bing_image_urls(driver, max_images=20):
    urls = []
    try:
        # Scroll once and wait briefly for results to populate
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
                except Exception:
                    data = {}
                if 'murl' in data: urls.append(data['murl'])
                elif 'purl' in data: urls.append(data['purl'])
        
        # If not enough results collected, try extracting from page source
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

# -----------------------
# Helper function to get image from URL (from Google script) - IMPROVED ERROR HANDLING
# -----------------------
def get_image_from_url_or_base64(img_url, timeout=5):
    global requests_session
    session = requests_session or requests
    
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
            # Try with SSL verification first, then without if it fails
            try:
                resp = session.get(img_url, timeout=timeout, headers=headers, verify=True)
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
                # Retry without SSL verification for problematic sites
                resp = session.get(img_url, timeout=timeout, headers=headers, verify=False)
            
            resp.raise_for_status()
            img_np = np.frombuffer(resp.content, np.uint8)
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
            return img_cv
        except Exception:
            return None

# -----------------------
# FLANN Feature Matching (COPIED FROM GOOGLE SCRIPT) - IMPROVED TIMEOUT HANDLING
# -----------------------
def calculate_flann_similarity(img1_cv, img2_url, min_matches=8, timeout=5):
    try:
        if img1_cv is None: 
            return 0.0, 0
        
        img2 = get_image_from_url_or_base64(img2_url, timeout=timeout)
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

# -----------------------
# Process Single Image (MODIFIED TO MATCH GOOGLE SCRIPT LOGIC)
# -----------------------
def process_image(driver, image_path, max_images=20, max_workers=8, log_writer=None, links_writer=None, threshold=0.15):
    image_name = os.path.basename(image_path)
    start_time = time.time()
    logging.info(f"Processing: {image_name}")
    
    if not upload_to_bing(driver, image_path):
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, "Bing upload failed"])
        return None
    
    urls = get_bing_image_urls(driver, max_images=max_images)
    if not urls:
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, "No URLs found"])
        return None
    
    # Load source image for comparison
    source_img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if source_img_cv is None:
        logging.error(f"Could not read source image {image_name} for processing.")
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, "Failed to read source image file"])
        return None
    
    best_match = None
    best_score = 0.0
    
    def check_url(url):
        score, matches = calculate_flann_similarity(source_img_cv, url)
        if links_writer:
            links_writer.writerow([image_name, url])
        return url, score, matches
    
    # Limit worker count to avoid overhead if fewer urls
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
    
    processing_time = time.time() - start_time
    
    if best_match:
        logging.info(f"Best match found for {image_name} with score {best_match['flann_score']:.4f}")
        
        # Download and create comparison image
        comparison_path = download_and_create_comparison(
            url=best_match['matched_url'],
            source_image_path=image_path,
            flann_score=best_match['flann_score'],
            num_matches=best_match['num_matches']
        )
        best_match['comparison_path'] = comparison_path
        
        # Move image to done folder (like Google script)
        move_image_to_done(image_path)
        
        if log_writer:
            log_writer.writerow([image_name, f"{processing_time:.2f}", 1, "Match found and saved"])
        
        return best_match
    else:
        logging.info(f"No match found for {image_name} above threshold {threshold}.")
        if log_writer:
            log_writer.writerow([image_name, f"{processing_time:.2f}", 0, f"No match found above threshold"])
        return None

# -----------------------
# Save Results
# -----------------------
def save_results(all_matches, filename="bing_best_matches.csv"):
    if not all_matches:
        logging.info("No matches were found to save.")
        return
    
    all_matches = [m for m in all_matches if m is not None]
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Image', 'Matched URL', 'FLANN Score', 'Feature Matches', 'Comparison Image Path'])
        for match in all_matches:
            writer.writerow([
                match['source_image'],
                match['matched_url'],
                f"{match['flann_score']:.4f}",
                match['num_matches'],
                match.get('comparison_path', 'N/A')
            ])
    logging.info(f"Results for {len(all_matches)} matches saved to '{filename}'.")

# -----------------------
# Main Function
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Bing Visual Search FLANN Matcher")
    parser.add_argument('-i', '--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('-o', '--output_file', type=str, default='bing_best_matches.csv', help='CSV file for output')
    parser.add_argument('-u', '--max_images', type=int, default=20, help='Max Bing result images to check')
    parser.add_argument('-w', '--max_workers', type=int, default=8, help='Max concurrent workers for matching')
    parser.add_argument('-d', '--batch_delay', type=float, default=2.0, help='Delay between images (seconds)')
    parser.add_argument('-t', '--threshold', type=float, default=0.9, help='FLANN similarity threshold')
    args = parser.parse_args()
    
    logging.info("--- Starting Bing Visual Search FLANN Matcher ---")
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
        logging.critical("Failed to initialize web driver. Cannot continue.")
        return
    
    all_matches = []
    
    try:
        with open(log_filename, 'w', newline='', encoding='utf-8') as log_file, \
             open(links_filename, 'w', newline='', encoding='utf-8') as links_file:
            
            log_writer = csv.writer(log_file)
            log_writer.writerow(['Image Name', 'Processing Time (s)', 'Match Found', 'Notes'])
            
            links_writer = csv.writer(links_file)
            links_writer.writerow(['Source Image', 'Scraped URL'])
            
            total_images = len(image_files)
            for idx, image_path in enumerate(image_files, 1):
                logging.info(f"--- [Image {idx}/{total_images}] ---")
                match_result = process_image(
                    driver, image_path,
                    max_images=args.max_images,
                    max_workers=args.max_workers,
                    log_writer=log_writer,
                    links_writer=links_writer,
                    threshold=args.threshold
                )
                
                if match_result:
                    all_matches.append(match_result)
                
                if idx < total_images:
                    logging.info(f"Waiting for {args.batch_delay} seconds before next image...")
                    time.sleep(args.batch_delay)
            
        save_results(all_matches, args.output_file)
        logging.info(f"--- Finished ---")
        logging.info(f"Processed {total_images} images. Found {len(all_matches)} matches.")
        
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
    finally:
        if driver:
            driver.quit()
        logging.info("Browser closed and cleanup complete.")

if __name__ == "__main__":
    main()