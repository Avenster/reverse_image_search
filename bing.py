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
    # options.add_argument("--headless=new")  # Uncomment for headless
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
        
        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = done_folder / f"{source.stem}_{timestamp}{source.suffix}"
        
        shutil.move(str(source), str(destination))
        logging.info(f"Moved {source.name} to done folder")
    except Exception as e:
        logging.error(f"Failed to move {image_path} to done folder: {e}")

# -----------------------
# Create and Save Stitched Comparison Image (NEW)
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
        text2 = f"Best Match | Score: {score:.4f} | Matches: {matches}"
        
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
# Download and Create Comparison Image (MODIFIED)
# -----------------------
def download_and_create_comparison(url, source_image_path, flann_score, num_matches, timeout=10):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bing.com/'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # Instead of saving the raw file, create the stitched comparison image
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
        for _ in range(2):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
        rich_links = driver.find_elements(By.CSS_SELECTOR, "a.richImgLnk")
        for link in rich_links[:max_images]:
            m_data = link.get_attribute('data-m')
            if m_data:
                data = json.loads(m_data)
                if 'murl' in data: urls.append(data['murl'])
                elif 'purl' in data: urls.append(data['purl'])
        if len(urls) < 10:
            page_source = driver.page_source
            urls.extend(re.findall(r'"murl":"(https?://[^"]+)"', page_source))
            urls.extend(re.findall(r'"purl":"(https?://[^"]+)"', page_source))
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
# FLANN Feature Matching
# -----------------------
def calculate_flann_similarity(img1_path, img2_url, min_matches=8, timeout=15):
    try:
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None: return 0.0, 0
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(img2_url, timeout=timeout, headers=headers)
        if resp.status_code != 200: return 0.0, 0
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img2 is None: return 0.0, 0

        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None: return 0.0, 0

        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        num_good_matches = len(good)

        if num_good_matches >= min_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers = mask.ravel().tolist()
                num_inliers = sum(inliers)
                similarity_score = num_inliers / min(len(kp1), len(kp2))
                return min(similarity_score, 1.0), num_inliers
        return 0.0, 0
    except Exception:
        return 0.0, 0

# -----------------------
# Process Single Image (Save only best match)
# -----------------------
def process_image(driver, image_path, max_images=40, max_workers=8, log_writer=None, links_writer=None, threshold=0.02):
    image_name = os.path.basename(image_path)
    start_time = time.time()
    logging.info(f"Processing: {image_name}")
    
    if not upload_to_bing(driver, image_path):
        if log_writer:
            log_writer.writerow([image_name, "0", 0, "Bing upload failed"])
        return []
    
    urls = get_bing_image_urls(driver, max_images=max_images)
    if not urls:
        if log_writer:
            log_writer.writerow([image_name, "0", 0, "No URLs found"])
        return []
    
    best_match = None
    best_score = 0.0
    
    def check_url(url):
        score, matches = calculate_flann_similarity(image_path, url)
        if links_writer:
            links_writer.writerow([image_name, url])
        return (url, score, matches)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_url, url) for url in urls]
        for future in as_completed(futures):
            url, score, matches = future.result()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = {
                    'source_image': image_name,
                    'matched_url': url,
                    'flann_score': score,
                    'num_matches': matches
                }
    
    processing_time = time.time() - start_time
    if log_writer:
        log_writer.writerow([image_name, f"{processing_time:.2f}", 1 if best_match else 0, "Best match saved"])
    
    if best_match:
        # Call the new function to download and create the comparison image (MODIFIED)
        best_match['downloaded_path'] = download_and_create_comparison(
            url=best_match['matched_url'],
            source_image_path=image_path,
            flann_score=best_match['flann_score'],
            num_matches=best_match['num_matches']
        )
    
    move_executor = ThreadPoolExecutor(max_workers=1)
    move_executor.submit(move_image_to_done, image_path)
    move_executor.shutdown(wait=False)
    
    return [best_match] if best_match else []

# -----------------------
# Save Results
# -----------------------
def save_results(all_matches, filename="bing_best_matches.csv"):
    if not all_matches:
        logging.info("No matches to save")
        return
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Image', 'Matched URL', 'FLANN Score', 'Feature Matches', 'Downloaded Path'])
        for match in all_matches:
            writer.writerow([
                match['source_image'],
                match['matched_url'],
                f"{match['flann_score']:.4f}",
                match['num_matches'],
                match.get('downloaded_path', 'Failed to download or create comparison')
            ])
    logging.info(f"Results saved to {filename}")

# -----------------------
# Main Function
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Bing Visual Search FLANN Matcher")
    parser.add_argument('--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('--output_file', type=str, default='bing_best_matches.csv', help='CSV file for output')
    parser.add_argument('--max_images', type=int, default=40, help='Max Bing result images to check')
    parser.add_argument('--max_workers', type=int, default=8, help='Max concurrent workers for matching')
    parser.add_argument('--batch_delay', type=float, default=2.0, help='Delay between images (seconds)')
    parser.add_argument('--threshold', type=float, default=0.02, help='FLANN similarity threshold')
    args = parser.parse_args()

    logging.info("Starting Bing Visual Search FLANN Matcher")
    create_folders()
    
    log_filename = f"bing_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    links_filename = f"bing_image_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    image_files = get_image_files(args.images_folder)
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
                logging.info(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
                matches = process_image(
                    driver, image_path,
                    max_images=args.max_images,
                    max_workers=args.max_workers,
                    log_writer=log_writer,
                    links_writer=links_writer,
                    threshold=args.threshold
                )
                all_matches.extend(matches)
                if idx < len(image_files):
                    time.sleep(args.batch_delay)
            
        save_results(all_matches, args.output_file)
        logging.info(f"Processed {len(image_files)} images, best matches: {len(all_matches)}")
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()