import time
import csv
import cv2
import numpy as np
import requests
import os
import logging
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import shutil
from datetime import datetime
import torch
import kornia as K
import kornia.feature as KF

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# -----------------------
# LoFTR Setup
# -----------------------
class LoFTRMatcher:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matcher = KF.LoFTR(pretrained='outdoor').to(self.device).eval()
        logging.info(f"LoFTR initialized on {self.device}")
    
    def preprocess_image(self, img, target_size=640):
        """Preprocess image for LoFTR"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize while maintaining aspect ratio
        h, w = img.shape
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).float()[None, None] / 255.0
        return img_tensor.to(self.device)
    
    def match_images(self, img1_path, img2_array):
        """Match two images using LoFTR"""
        try:
            # Read first image
            img1 = cv2.imread(img1_path)
            if img1 is None:
                return 0.0, 0
            
            # Second image is already numpy array
            if img2_array is None:
                return 0.0, 0
            
            # Preprocess images
            img1_tensor = self.preprocess_image(img1)
            img2_tensor = self.preprocess_image(img2_array)
            
            # Prepare input
            input_dict = {
                'image0': img1_tensor,
                'image1': img2_tensor
            }
            
            # Run LoFTR
            with torch.no_grad():
                correspondences = self.matcher(input_dict)
            
            # Extract matches
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()
            
            # Filter high-confidence matches
            high_conf_mask = confidence > 0.8
            num_high_conf = high_conf_mask.sum()
            
            # Calculate similarity score
            num_matches = len(mkpts0)
            if num_matches > 0:
                avg_confidence = confidence.mean()
                # Weighted score: number of matches and average confidence
                similarity_score = (num_high_conf / 100.0) * avg_confidence
                similarity_score = min(similarity_score, 1.0)  # Cap at 1.0
            else:
                similarity_score = 0.0
            
            return similarity_score, num_high_conf
            
        except Exception as e:
            logging.error(f"LoFTR matching error: {e}")
            return 0.0, 0

# Global LoFTR instance
loftr_matcher = None

def init_loftr():
    global loftr_matcher
    if loftr_matcher is None:
        loftr_matcher = LoFTRMatcher()

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    # options.add_argument("--no-sandbox")
    options.add_argument("--headless=new")
    options.binary_location = "/usr/bin/google-chrome"
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
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
def download_similar_image(url, source_image_name, loftr_score, timeout=10):
    try:
        similar_folder = Path('similar_images')
        similar_folder.mkdir(exist_ok=True)
        
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Extract file extension from URL or use jpg as default
        url_parts = url.split('.')
        extension = url_parts[-1].split('?')[0] if len(url_parts) > 1 else 'jpg'
        if extension not in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            extension = 'jpg'
        
        # Create filename with naming convention
        source_name = Path(source_image_name).stem
        filename = f"{source_name}_{loftr_score:.4f}.{extension}"
        filepath = similar_folder / filename
        
        # If file exists, add timestamp
        if filepath.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{source_name}_{loftr_score:.4f}_{timestamp}.{extension}"
            filepath = similar_folder / filename
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        logging.info(f"Downloaded similar image: {filename}")
        return str(filepath)
    except Exception as e:
        logging.error(f"Failed to download image from {url}: {e}")
        return None

# -----------------------
# Upload and Extract URLs with Full Frame Selection
# -----------------------
def upload_image(driver, image_path, timeout=30):
    """Upload image and handle full frame selection"""
    abs_path = os.path.abspath(image_path)
    
    # Navigate to Google Lens
    driver.get("https://lens.google.com")
    time.sleep(2)
    
    # Find and click the camera/upload button
    try:
        # Try different selectors for the upload button
        upload_selectors = [
            "svg[class*='camera']",
            "button[aria-label*='Search by image']",
            "div[aria-label*='Search by image']",
            "[data-promo-id*='camera']",
            "div[role='button'] svg"
        ]
        
        for selector in upload_selectors:
            try:
                upload_btn = driver.find_element(By.CSS_SELECTOR, selector)
                if upload_btn:
                    upload_btn.click()
                    time.sleep(1)
                    break
            except:
                continue
    except:
        pass
    
    # Upload the file
    try:
        upload_input = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        upload_input.send_keys(abs_path)
        logging.info(f"Uploaded image: {os.path.basename(image_path)}")
    except Exception as e:
        logging.error(f"Failed to upload image: {e}")
        return False
    
    # Wait for image to process
    time.sleep(3)
    
    # Try to select full frame
    try:
        # Look for selection interface
        selection_found = False
        
        # Try to find crop/selection interface
        crop_selectors = [
            "div[role='button'][aria-label*='Done']",
            "button[aria-label*='Done']",
            "div[aria-label*='Select all']",
            "button[aria-label*='Use full image']",
            "span:contains('Done')",
            "button:contains('Done')"
        ]
        
        for selector in crop_selectors:
            try:
                element = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                element.click()
                selection_found = True
                logging.info("Clicked selection/done button")
                break
            except:
                continue
        
        # If no selection interface, image might be automatically processed
        if not selection_found:
            logging.info("No selection interface found, proceeding with auto-detection")
            
    except Exception as e:
        logging.info(f"Selection interface handling: {e}")
    
    # Wait for results to load
    time.sleep(5)
    
    # Check if we're on a results page
    current_url = driver.current_url
    if "search" in current_url or "lens" in current_url:
        logging.info(f"Successfully navigated to results page")
        return True
    
    return False

def get_image_urls(driver, max_images=40):
    """Extract image URLs from Google Lens results with improved logic"""
    urls = set()
    
    # Wait for initial results to load
    time.sleep(3)
    
    # Scroll to load more images
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0
    max_scrolls = 5
    
    while scroll_attempts < max_scrolls:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Check for "Show more results" button and click if exists
        try:
            show_more_selectors = [
                "button[aria-label*='Show more']",
                "div[role='button'][aria-label*='More']",
                "span:contains('Show more')",
                "button:contains('More results')"
            ]
            for selector in show_more_selectors:
                try:
                    show_more = driver.find_element(By.CSS_SELECTOR, selector)
                    if show_more and show_more.is_displayed():
                        show_more.click()
                        time.sleep(2)
                        break
                except:
                    continue
        except:
            pass
        
        # Calculate new scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            scroll_attempts += 1
        else:
            scroll_attempts = 0
        last_height = new_height
    
    # Parse the page source
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # Find all image elements with multiple strategies
    # Strategy 1: Direct img tags
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-iurl")
        if src and src.startswith("http") and "google" not in src.lower():
            urls.add(src)
    
    # Strategy 2: Links with image parameters
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        if href and ("imgurl=" in href or "img_url=" in href):
            # Extract image URL from parameters
            import urllib.parse
            try:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                if "imgurl" in parsed:
                    urls.add(parsed["imgurl"][0])
                elif "img_url" in parsed:
                    urls.add(parsed["img_url"][0])
            except:
                pass
    
    # Strategy 3: Divs with background images
    for div in soup.find_all("div", style=True):
        style = div.get("style")
        if style and "background-image" in style and "url(" in style:
            try:
                url_start = style.index("url(") + 4
                url_end = style.index(")", url_start)
                url = style[url_start:url_end].strip("'\"")
                if url.startswith("http"):
                    urls.add(url)
            except:
                pass
    
    # Strategy 4: Data attributes
    for elem in soup.find_all(attrs={"data-thumbnail-url": True}):
        url = elem.get("data-thumbnail-url")
        if url and url.startswith("http"):
            urls.add(url)
    
    # Convert to list and limit
    url_list = list(urls)[:max_images]
    
    logging.info(f"Found {len(url_list)} unique image URLs")
    
    # If no URLs found, try alternative extraction
    if not url_list:
        logging.warning("No URLs found with primary methods, trying JavaScript extraction")
        try:
            # Try to extract URLs via JavaScript
            js_urls = driver.execute_script("""
                var urls = [];
                var imgs = document.querySelectorAll('img[src*="http"], img[data-src*="http"]');
                imgs.forEach(function(img) {
                    var src = img.src || img.getAttribute('data-src');
                    if (src && src.startsWith('http') && !src.includes('google')) {
                        urls.push(src);
                    }
                });
                return urls;
            """)
            if js_urls:
                url_list = list(set(js_urls))[:max_images]
                logging.info(f"Found {len(url_list)} URLs via JavaScript")
        except Exception as e:
            logging.error(f"JavaScript extraction failed: {e}")
    
    return url_list

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
    for extension in supported_extensions:
        image_files.extend(glob.glob(str(folder_path / extension)))
        image_files.extend(glob.glob(str(folder_path / extension.upper())))
    return sorted(image_files)

# -----------------------
# LoFTR Similarity Check
# -----------------------
def calculate_loftr_similarity(img1_path, img2_url, timeout=10):
    try:
        # Download image from URL
        resp = requests.get(img2_url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img2 is None:
            return 0.0, 0
        
        # Use LoFTR matcher
        similarity_score, num_matches = loftr_matcher.match_images(img1_path, img2)
        return similarity_score, num_matches
        
    except Exception as e:
        return 0.0, 0

# -----------------------
# Similarity Check with LoFTR
# -----------------------
def is_similar_image(img1_path, img2_url, loftr_threshold=0.3, min_matches=20):
    loftr_score, num_matches = calculate_loftr_similarity(img1_path, img2_url)
    is_match = (loftr_score >= loftr_threshold and num_matches >= min_matches)
    return is_match, loftr_score, num_matches

# -----------------------
# Process Single Image (Parallelized)
# -----------------------
def process_image(driver, image_path, max_images=40, max_workers=8, loftr_threshold=0.3, min_matches=20, log_writer=None, links_writer=None):
    image_name = os.path.basename(image_path)
    start_time = time.time()
    logging.info(f"Processing: {image_name}")
    
    # Upload image and get to results page
    success = upload_image(driver, image_path)
    
    if not success:
        logging.warning(f"Failed to upload {image_name}, retrying...")
        time.sleep(2)
        success = upload_image(driver, image_path)
    
    # Extract URLs
    urls = get_image_urls(driver, max_images=max_images)
    
    if not urls:
        end_time = time.time()
        processing_time = end_time - start_time
        if log_writer:
            log_writer.writerow([image_name, f"{processing_time:.2f}", 0, "No URLs found"])
        logging.warning(f"No URLs found for {image_name}")
        return []
    
    matched_urls = []
    def check_url(url):
        is_match, loftr_score, num_matches = is_similar_image(
            image_path, url, loftr_threshold=loftr_threshold, min_matches=min_matches
        )
        # Save image link as soon as processed
        if links_writer:
            links_writer.writerow([image_name, url])
        if is_match:
            # Download the similar image
            downloaded_path = download_similar_image(url, image_name, loftr_score)
            return {
                'source_image': image_name,
                'matched_url': url,
                'loftr_score': loftr_score,
                'num_matches': num_matches,
                'downloaded_path': downloaded_path
            }
        return None
    
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
def save_results(all_matches, filename="google_matches.csv"):
    if not all_matches:
        logging.info("No matches to save")
        return
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Image', 'Matched URL', 'LoFTR Score', 'Feature Matches', 'Downloaded Path'])
        for match in all_matches:
            writer.writerow([
                match['source_image'],
                match['matched_url'],
                f"{match['loftr_score']:.4f}",
                match['num_matches'],
                match.get('downloaded_path', 'Failed to download')
            ])
    logging.info(f"Results saved to {filename}")

# -----------------------
# Main Function (Production CLI)
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Google Lens LoFTR Matcher")
    parser.add_argument('--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('--output_file', type=str, default='matched_images.csv', help='CSV file for output')
    parser.add_argument('--max_images', type=int, default=40, help='Max Google result images to check')
    parser.add_argument('--max_workers', type=int, default=8, help='Max concurrent workers for matching')
    parser.add_argument('--batch_delay', type=float, default=2.0, help='Delay between images (seconds)')
    parser.add_argument('--loftr_threshold', type=float, default=0.3, help='LoFTR similarity score threshold')
    parser.add_argument('--min_matches', type=int, default=20, help='Min LoFTR matches to consider a match')
    args = parser.parse_args()

    logging.info("Starting Google Lens LoFTR Matcher (Production)")
    
    # Initialize LoFTR
    init_loftr()
    
    # Create required folders
    create_folders()
    
    # Create log file for processing times
    log_filename = f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    links_filename = f"google_image_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
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
                    loftr_threshold=args.loftr_threshold,
                    min_matches=args.min_matches,
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