#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Google Lens Visual Search and FLANN Matcher

This script automates reverse image searches using Google Lens. For each local image,
it uploads it to Google Lens, scrapes the URLs of visually similar images,
and then uses the FLANN (Fast Library for Approximate Nearest Neighbors) algorithm
to find the best match among the results.

Key Features:
- Uses undetected_chromedriver to avoid bot detection.
- Processes a batch of images from a specified folder.
- Finds the single best match for each source image based on feature similarity.
- Creates a stitched comparison image showing the original and the best match side-by-side.
- Moves ONLY the successfully matched original images to a 'done' folder.
- Generates CSV logs for processing details, found URLs, and final matches.

Optimizations in this version:
1.  Speed: Replaced static 'time.sleep' calls with dynamic WebDriverWait for faster execution.
2.  Efficiency: Loads the source image into memory once per run, avoiding repeated disk reads during comparison.
3.  Logic: Only moves source images to the 'done' folder if a match is found.
"""

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
import subprocess
import platform

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# -----------------------
# Kill Chrome Processes (for macOS/Linux issues)
# -----------------------
def kill_chrome_processes():
    """Kill existing Chrome/Chromium processes to prevent conflicts."""
    if platform.system() in ["Darwin", "Linux"]:
        for process_name in ["Google Chrome", "Chromium", "chrome"]:
            try:
                # Use pkill to forcefully terminate processes by name
                subprocess.run(["pkill", "-f", process_name], check=False, capture_output=True)
            except FileNotFoundError:
                # pkill might not be available on all systems
                pass
    elif platform.system() == "Windows":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], check=False, capture_output=True)
        except FileNotFoundError:
            pass
    time.sleep(2) # Give a moment for processes to terminate

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    """Initializes the undetected_chromedriver with optimized options."""
    kill_chrome_processes()
    
    options = uc.ChromeOptions()
    
    # --- Performance & Stealth Options ---
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless=new")  # Uncomment for headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-plugins-discovery")
    
    # Using a temporary user data directory for a clean session
    user_data_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", f"chrome_data_{os.getpid()}")
    options.add_argument(f"--user-data-dir={user_data_dir}")
    
    try:
        logging.info("Attempting to initialize Chrome driver...")
        driver = uc.Chrome(options=options, version_main=None)
        # Hides the "navigator.webdriver" property that signals automation
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except WebDriverException as e:
        logging.error(f"Failed to initialize Chrome driver: {e}")
        logging.info("Trying with a simpler configuration...")
        driver = uc.Chrome(options=options, use_subprocess=True)
        return driver

# -----------------------
# Create Required Folders
# -----------------------
def create_folders():
    """Creates necessary output folders if they don't exist."""
    folders = ['done', 'similar_images']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

# -----------------------
# Move Image to Done Folder
# -----------------------
def move_image_to_done(image_path):
    """Moves a successfully processed image to the 'done' folder."""
    try:
        done_folder = Path('done')
        source = Path(image_path)
        destination = done_folder / source.name
        
        # Avoid overwriting by adding a timestamp if the file already exists
        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = done_folder / f"{source.stem}_{timestamp}{source.suffix}"
        
        shutil.move(str(source), str(destination))
        logging.info(f"Moved '{source.name}' to 'done' folder.")
    except Exception as e:
        logging.error(f"Failed to move '{image_path}': {e}")

# -----------------------
# Create and Save Stitched Comparison Image
# -----------------------
def create_comparison_image(source_path, similar_img_bytes, score, matches, output_folder='similar_images'):
    """Creates a side-by-side comparison image of the original and the best match."""
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

        # --- Image Resizing to a common height for consistent comparison ---
        target_height = 600
        original_h, original_w = original_img.shape[:2]
        similar_h, similar_w = similar_img.shape[:2]

        original_ratio = target_height / original_h
        original_resized = cv2.resize(original_img, (int(original_w * original_ratio), target_height))

        similar_ratio = target_height / similar_h
        similar_resized = cv2.resize(similar_img, (int(similar_w * similar_ratio), target_height))

        # --- Create Canvas for stitching images and adding text ---
        text_area_height = 80
        total_width = original_resized.shape[1] + similar_resized.shape[1]
        total_height = target_height + text_area_height
        
        stitched_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8) # White canvas

        # Place resized images side-by-side
        stitched_image[text_area_height:total_height, :original_resized.shape[1]] = original_resized
        stitched_image[text_area_height:total_height, original_resized.shape[1]:] = similar_resized

        # --- Add text labels to the top of the canvas ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 0) # Black
        thickness = 2
        
        text1 = "Original Image"
        text2 = f"Google | Score: {score:.4f} | Matches: {matches}"
        
        # Center text above each respective image
        text1_size = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text1_x = (original_resized.shape[1] - text1_size[0]) // 2
        text1_y = (text_area_height - text1_size[1]) // 2 + text1_size[1]

        text2_size = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        text2_x = original_resized.shape[1] + (similar_resized.shape[1] - text2_size[0]) // 2
        text2_y = (text_area_height - text2_size[1]) // 2 + text2_size[1]
        
        cv2.putText(stitched_image, text1, (text1_x, text1_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(stitched_image, text2, (text2_x, text2_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # --- Save the final composite image ---
        source_name = Path(source_path).stem
        save_folder = Path(output_folder)
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
# Download Image and Create Comparison
# -----------------------
def download_and_create_comparison(url, source_image_path, flann_score, num_matches, timeout=10):
    """Downloads the best match and triggers the comparison image creation."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://lens.google.com/'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        return create_comparison_image(
            source_path=source_image_path,
            similar_img_bytes=response.content,
            score=flann_score,
            matches=num_matches
        )
    except requests.RequestException as e:
        logging.error(f"Failed to download image from {url}: {e}")
        return None

# -----------------------
# Get Images from Folder
# -----------------------
def get_image_files(folder_path):
    """Gathers all supported image files from a given folder."""
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        logging.error(f"Folder '{folder_path}' does not exist!")
        return []
    for ext in supported_extensions:
        image_files.extend(folder_path.glob(ext))
        image_files.extend(folder_path.glob(ext.upper()))
    return sorted([str(f) for f in image_files])

# -----------------------
# Upload to Google Lens (Optimized)
# -----------------------
def upload_to_google_lens(driver, image_path, timeout=30):
    """Uploads an image to Google Lens, with optimized waits."""
    try:
        abs_path = os.path.abspath(image_path)
        driver.get("https://lens.google.com/")
        
        # OPTIMIZATION: Wait for the file input element to be present instead of a static sleep.
        upload_input = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        
        upload_input.send_keys(abs_path)
        
        # OPTIMIZATION: Wait for results to load instead of a long static sleep.
        # This checks for multiple possible indicators of a successful upload.
        WebDriverWait(driver, timeout).until(
            EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-photo-id]")),
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='imgurl=']")),
                EC.url_contains("search")
            )
        )
        logging.info("Upload successful - results detected.")
        return True
        
    except TimeoutException:
        logging.error(f"Timed out waiting for Google Lens elements for {image_path}.")
        return False
    except Exception as e:
        logging.error(f"Upload to Google Lens failed for {image_path}: {e}")
        return False

# -----------------------
# Image URL Extraction
# -----------------------
def get_google_image_urls(driver, max_images=50):
    """Extracts visually similar image URLs from the Google Lens results page."""
    urls = set()
    try:
        # Scroll to load more results
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5) # Wait for content to load after scroll
        
        page_source = driver.page_source
        
        # Use regex to find URLs in common patterns within the page source.
        # This is often faster than finding and iterating through Selenium elements.
        url_patterns = [
            r'"(https?://[^"]+\.(?:jpg|jpeg|png|gif|bmp|webp)(?:\?[^"]*)?)"',
            r"'(https?://[^']+\.(?:jpg|jpeg|png|gif|bmp|webp)(?:\?[^']*)?)'",
            r'"ou":"(https?://[^"]+)"',
            r'imgurl=(https?://[^&]+)'
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, page_source, re.IGNORECASE)
            for match in matches:
                # Decode URL encoding (e.g., %2F -> /) and filter out unwanted domains
                decoded_url = unquote(match)
                if decoded_url.startswith('http') and not any(skip in decoded_url for skip in [
                    'google.com', 'gstatic.com', 'googleusercontent.com', 'data:image'
                ]):
                    urls.add(decoded_url)
        
        logging.info(f"Extracted {len(urls)} potential image URLs.")
        return list(urls)[:max_images]
        
    except Exception as e:
        logging.error(f"Error extracting Google URLs: {e}")
        return []

# -----------------------
# FLANN Feature Matching (Optimized)
# -----------------------
def calculate_flann_similarity(img1_cv, img2_url, min_matches=8, timeout=15):
    """
    Calculates similarity between a pre-loaded local image and a remote image from a URL.
    
    Args:
        img1_cv (numpy.ndarray): The source image pre-loaded as a grayscale numpy array.
        img2_url (str): The URL of the image to compare against.
    
    Returns:
        tuple: (similarity_score, number_of_inlier_matches)
    """
    try:
        # OPTIMIZATION: The source image (img1_cv) is passed directly, avoiding repeated disk reads.
        if img1_cv is None: 
            return 0.0, 0
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(img2_url, timeout=timeout, headers=headers)
        if resp.status_code != 200: 
            return 0.0, 0
        
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
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
        # Catch-all for any errors during processing (e.g., network, OpenCV issues)
        return 0.0, 0

# -----------------------
# Process Single Image (Updated Logic)
# -----------------------
def process_image(driver, image_path, max_urls=50, max_workers=10, log_writer=None, links_writer=None, threshold=0.1):
    image_name = Path(image_path).name
    start_time = time.time()
    logging.info(f"Processing: {image_name}")
    
    if not upload_to_google_lens(driver, image_path):
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, "Google Lens upload failed"])
        return None
    
    urls = get_google_image_urls(driver, max_images=max_urls)
    if not urls:
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, "No visually similar image URLs found"])
        return None
    
    # OPTIMIZATION: Load the source image once into memory before starting comparisons.
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
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_url, url): url for url in urls}
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
    
    # LOGIC CHANGE: Only create comparison and move file if a good match was found.
    if best_match:
        logging.info(f"Best match found for {image_name} with score {best_match['flann_score']:.4f}")
        comparison_path = download_and_create_comparison(
            url=best_match['matched_url'],
            source_image_path=image_path,
            flann_score=best_match['flann_score'],
            num_matches=best_match['num_matches']
        )
        best_match['comparison_path'] = comparison_path
        
        # Move the original image to 'done' folder upon successful match
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
def save_results(all_matches, filename="google_best_matches.csv"):
    if not all_matches:
        logging.info("No matches were found to save.")
        return
    
    # Filter out any None results that may have slipped through
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
    parser = argparse.ArgumentParser(description="Google Lens Visual Search & FLANN Matcher")
    parser.add_argument('-i', '--images_folder', type=str, default='images', help='Folder containing source images.')
    parser.add_argument('-o', '--output_file', type=str, default='google_best_matches.csv', help='Output CSV file for best matches.')
    parser.add_argument('-u', '--max_urls', type=int, default=50, help='Max Google result URLs to check per image.')
    parser.add_argument('-w', '--max_workers', type=int, default=10, help='Max concurrent workers for downloading and matching.')
    parser.add_argument('-d', '--delay', type=float, default=2.0, help='Delay (seconds) between processing each image.')
    parser.add_argument('-t', '--threshold', type=float, default=0.8, help='Minimum FLANN similarity score to consider a match.')
    args = parser.parse_args()

    logging.info("--- Starting Google Lens Matcher ---")
    create_folders()
    
    # Create timestamped log files for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"log_processing_{run_timestamp}.csv"
    links_filename = f"log_urls_scraped_{run_timestamp}.csv"

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
                    max_urls=args.max_urls,
                    max_workers=args.max_workers,
                    log_writer=log_writer,
                    links_writer=links_writer,
                    threshold=args.threshold
                )
                if match_result:
                    all_matches.append(match_result)
                
                if idx < total_images:
                    logging.info(f"Waiting for {args.delay} seconds before next image...")
                    time.sleep(args.delay)
            
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
        # Clean up temporary chrome user data directory
        user_data_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", f"chrome_data_{os.getpid()}")
        if os.path.exists(user_data_dir):
            shutil.rmtree(user_data_dir, ignore_errors=True)
        logging.info("Browser closed and cleanup complete.")

if __name__ == "__main__":
    main()