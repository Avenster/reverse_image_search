import time
import csv
import cv2
import numpy as np
import requests
import os
import json
import re
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

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless=new")  # Production: run headless
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
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
        
        # If file already exists in done folder, add timestamp
        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = source.stem, timestamp, source.suffix
            destination = done_folder / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
        
        shutil.move(str(source), str(destination))
        logging.info(f"Moved {source.name} to done folder")
    except Exception as e:
        logging.error(f"Failed to move {image_path} to done folder: {e}")

# -----------------------
# Download Similar Image
# -----------------------
def download_similar_image(url, source_image_name, orb_score, timeout=10):
    try:
        similar_folder = Path('similar_images')
        similar_folder.mkdir(exist_ok=True)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bing.com/'
        }
        
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # Extract file extension from URL or use jpg as default
        url_parts = url.split('.')
        extension = url_parts[-1].split('?')[0] if len(url_parts) > 1 else 'jpg'
        if extension not in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            extension = 'jpg'
        
        # Create filename with naming convention
        source_name = Path(source_image_name).stem
        filename = f"{source_name}_{orb_score:.4f}.{extension}"
        filepath = similar_folder / filename
        
        # If file exists, add timestamp
        if filepath.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{source_name}_{orb_score:.4f}_{timestamp}.{extension}"
            filepath = similar_folder / filename
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        logging.info(f"Downloaded similar image: {filename}")
        return str(filepath)
    except Exception as e:
        logging.error(f"Failed to download image from {url}: {e}")
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
        WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#sbi_b, .sbi_b_prtl, [aria-label*='Visual Search'], [aria-label*='visual search']"))
        ).click()
        file_input = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        abs_path = os.path.abspath(image_path)
        file_input.send_keys(abs_path)
        WebDriverWait(driver, timeout).until(
            lambda d: "search?" in d.current_url or "detailV2" in d.current_url
        )
        return True
    except Exception as e:
        logging.warning(f"Upload to Bing failed for {image_path}: {e}")
        return False

# -----------------------
# Extract Image URLs from Bing Results
# -----------------------
def get_bing_image_urls(driver, max_images=40):
    urls = []
    try:
        # Scroll to load more images
        for _ in range(2):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
        # Try richImgLnk method
        rich_links = driver.find_elements(By.CSS_SELECTOR, "a.richImgLnk")
        for link in rich_links[:max_images]:
            m_data = link.get_attribute('data-m')
            if m_data:
                data = json.loads(m_data)
                if 'murl' in data:
                    urls.append(data['murl'])
                elif 'purl' in data:
                    urls.append(data['purl'])
        # Fallback: parse page source if few found
        if len(urls) < 10:
            page_source = driver.page_source
            murl_pattern = r'"murl":"(https?://[^"]+)"'
            purl_pattern = r'"purl":"(https?://[^"]+)"'
            urls.extend(re.findall(murl_pattern, page_source))
            urls.extend(re.findall(purl_pattern, page_source))
        # Clean URLs
        cleaned = []
        for url in urls:
            url = unquote(url)
            if url.startswith('http') and not ('bing.com/th' in url or 'mm.bing.net/th' in url):
                cleaned.append(url)
        return cleaned[:max_images]
    except Exception as e:
        logging.warning(f"Error extracting Bing URLs: {e}")
        return []

# -----------------------
# ORB Feature Matching
# -----------------------
def calculate_orb_similarity(img1_path, img2_url, min_matches=8, timeout=15):
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None: return 0.0, 0
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.bing.com/'
        }
        resp = requests.get(img2_url, timeout=timeout, headers=headers)
        if resp.status_code != 200: return 0.0, 0
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img2 is None: return 0.0, 0
        # Resize images for speed
        max_dim = 1024
        for img, shape in [(img1, img1.shape), (img2, img2.shape)]:
            h, w = shape
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return 0.0, 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
        num_good_matches = len(good_matches)
        if num_good_matches >= min_matches:
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = mask.ravel().tolist()
                    num_inliers = sum(inliers)
                    if num_inliers >= min_matches:
                        similarity_score = num_inliers / min(len(kp1), len(kp2))
                        return min(similarity_score, 1.0), num_inliers
            except Exception:
                pass
        if num_good_matches > 0:
            similarity_score = num_good_matches / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
            return min(similarity_score, 1.0), num_good_matches
        return 0.0, 0
    except Exception:
        return 0.0, 0

# -----------------------
# Check Similarity (ORB only)
# -----------------------
def is_similar_image(img1_path, img2_url):
    orb_score, num_matches = calculate_orb_similarity(img1_path, img2_url, min_matches=8)
    is_match = (orb_score >= 0.008 and num_matches >= 8)
    return is_match, orb_score, num_matches

# -----------------------
# Process Single Image (Parallelized URL checks)
# -----------------------
def process_image(driver, image_path, max_images=40, max_workers=8, log_writer=None, links_writer=None):
    image_name = os.path.basename(image_path)
    start_time = time.time()
    logging.info(f"Processing: {image_name}")
    
    if not upload_to_bing(driver, image_path):
        end_time = time.time()
        processing_time = end_time - start_time
        if log_writer:
            log_writer.writerow([image_name, f"{processing_time:.2f}", 0, "Bing upload failed"])
        logging.warning(f"Bing Visual Search upload failed for {image_name}")
        return []
    
    urls = get_bing_image_urls(driver, max_images=max_images)
    if not urls:
        end_time = time.time()
        processing_time = end_time - start_time
        if log_writer:
            log_writer.writerow([image_name, f"{processing_time:.2f}", 0, "No URLs found"])
        logging.warning(f"No URLs found for {image_name}")
        return []
    
    matched_urls = []
    def check_url(url):
        is_match, orb_score, num_matches = is_similar_image(image_path, url)
        # Save the image link as soon as processed
        if links_writer:
            links_writer.writerow([image_name, url])
        if is_match:
            # Download the similar image
            downloaded_path = download_similar_image(url, image_name, orb_score)
            return {
                'source_image': image_name,
                'matched_url': url,
                'orb_score': orb_score,
                'num_matches': num_matches,
                'downloaded_path': downloaded_path
            }
        return None
    
    # Parallelize URL checks
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(check_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            result = future.result()
            if result: matched_urls.append(result)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Log the processing time
    if log_writer:
        log_writer.writerow([image_name, f"{processing_time:.2f}", len(matched_urls), f"Found {len(matched_urls)} matches"])
    
    logging.info(f"Found {len(matched_urls)} matches for {image_name} (took {processing_time:.2f}s)")
    
    # Move processed image to done folder concurrently
    move_executor = ThreadPoolExecutor(max_workers=1)
    move_executor.submit(move_image_to_done, image_path)
    move_executor.shutdown(wait=False)
    
    return matched_urls

# -----------------------
# Save Results
# -----------------------
def save_results(all_matches, filename="bing_matched_images.csv"):
    if not all_matches:
        logging.info("No matches to save")
        return
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Image', 'Matched URL', 'ORB Score', 'Feature Matches', 'Downloaded Path'])
        for match in all_matches:
            writer.writerow([
                match['source_image'],
                match['matched_url'],
                f"{match['orb_score']:.4f}",
                match['num_matches'],
                match.get('downloaded_path', 'Failed to download')
            ])
    logging.info(f"Results saved to {filename}")

# -----------------------
# Main Function (Production CLI)
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Bing Visual Search ORB Matcher")
    parser.add_argument('--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('--output_file', type=str, default='bing_matches.csv', help='CSV file for output')
    parser.add_argument('--max_images', type=int, default=40, help='Max Bing result images to check')
    parser.add_argument('--max_workers', type=int, default=8, help='Max concurrent workers for matching')
    parser.add_argument('--batch_delay', type=float, default=2.0, help='Delay between images (seconds)')
    args = parser.parse_args()

    logging.info("Starting Bing Visual Search ORB Matcher (Production)")
    
    # Create required folders
    create_folders()
    
    # Create log file for processing times
    log_filename = f"bing_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    links_filename = f"bing_image_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    image_files = get_image_files(args.images_folder)
    total_images = len(image_files)
    processed_count = 0
    if not image_files:
        logging.error(f"No images found in '{args.images_folder}'!")
        return
    
    driver = init_driver()
    all_matches = []
    
    try:
        with open(log_filename, 'w', newline='', encoding='utf-8') as log_file, open(links_filename, 'w', newline='', encoding='utf-8') as links_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['Image Name', 'Processing Time (seconds)', 'Matches Found', 'Notes'])
            links_writer = csv.writer(links_file)
            links_writer.writerow(['Source Image', 'Matched URL'])
            
            for idx, image_path in enumerate(image_files, 1):
                logging.info(f"[{idx}/{total_images}] Processing: {os.path.basename(image_path)}")
                matches = process_image(
                    driver, image_path,
                    max_images=args.max_images,
                    max_workers=args.max_workers,
                    log_writer=log_writer,
                    links_writer=links_writer
                )
                all_matches.extend(matches)
                processed_count += 1
                logging.info(f"Processed {processed_count} of {total_images} images.")
                if idx < total_images:
                    time.sleep(args.batch_delay)
            
        save_results(all_matches, args.output_file)
        logging.info(f"Processed {total_images} images, total matches: {len(all_matches)}")
        logging.info(f"Processing log saved to {log_filename}")
        logging.info(f"Image links saved to {links_filename}")
        
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()