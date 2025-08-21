import time
import csv
import cv2
import numpy as np
import requests
import os
from pathlib import Path
import glob
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from collections import defaultdict
import argparse
import logging
import shutil
from datetime import datetime

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# -----------------------
# Configuration
# -----------------------
DEFAULT_MAX_WORKERS = 8
DEFAULT_WAIT_TIMEOUT = 15
DEFAULT_ORB_FEATURES = 2000
DEFAULT_ORB_MIN_MATCHES = 10
DEFAULT_ORB_SIMILARITY_THRESHOLD = 0.01
DEFAULT_MAX_IMAGES = 50
DEFAULT_BATCH_DELAY = 2.0

class YandexImageMatcher:
    def __init__(
        self,
        images_folder='images',
        output_file='yandex_matches.csv',
        max_workers=DEFAULT_MAX_WORKERS,
        max_images=DEFAULT_MAX_IMAGES,
        orb_features=DEFAULT_ORB_FEATURES,
        orb_min_matches=DEFAULT_ORB_MIN_MATCHES,
        orb_similarity_threshold=DEFAULT_ORB_SIMILARITY_THRESHOLD,
        batch_delay=DEFAULT_BATCH_DELAY
    ):
        self.images_folder = Path(images_folder)
        self.output_file = output_file
        self.driver = None
        self.orb = cv2.ORB_create(nfeatures=orb_features, scaleFactor=1.2, nlevels=8)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://yandex.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        self.max_workers = max_workers
        self.max_images = max_images
        self.orb_min_matches = orb_min_matches
        self.orb_similarity_threshold = orb_similarity_threshold
        self.batch_delay = batch_delay
        
        # Create required folders
        self.create_folders()
        
        # Create log file for processing times
        self.log_filename = f"yandex_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.log_writer = None

        # For saving image links concurrently
        self.links_filename = f"yandex_image_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.links_file = open(self.links_filename, 'w', newline='', encoding='utf-8')
        self.links_writer = csv.writer(self.links_file)
        self.links_writer.writerow(['Source Image', 'Matched URL'])

        # Counter for processed images
        self.processed_count = 0
        self.total_images = 0

    def create_folders(self):
        """Create required folders"""
        folders = ['done', 'similar_images']
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def move_image_to_done(self, image_path):
        """Move processed image to done folder"""
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

    def download_similar_image(self, url, source_image_name, orb_score, timeout=10):
        """Download similar image with naming convention"""
        try:
            similar_folder = Path('similar_images')
            similar_folder.mkdir(exist_ok=True)
            
            session = requests.Session()
            session.headers.update(self.headers)
            
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Extract file extension from URL or use jpg as default
            url_parts = url.split('.')
            extension = url_parts[-1].split('?')[0] if len(url_parts) > 1 else 'jpg'
            if extension not in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']:
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

    def init_driver(self):
        """Initialize undetected Chrome driver in headless mode for production"""
        options = uc.ChromeOptions()
        options.add_argument("--user-data-dir=/tmp/chrome-user-data")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--window-size=1920,1080")
        # options.add_argument("--headless=new")
        driver = uc.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.set_page_load_timeout(30)
        return driver

    def get_image_files(self):
        """Get all image files from folder"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        files = []
        if not self.images_folder.exists():
            raise FileNotFoundError(f"Folder '{self.images_folder}' not found")
        for ext in extensions:
            files.extend(glob.glob(str(self.images_folder / ext)))
            files.extend(glob.glob(str(self.images_folder / ext.upper())))
        return sorted(set(files))

    def wait_for_page_load(self):
        """Wait for page to fully load"""
        try:
            WebDriverWait(self.driver, DEFAULT_WAIT_TIMEOUT).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            time.sleep(1)
        except TimeoutException:
            logging.warning("Page load timeout, continuing...")

    def upload_to_yandex(self, image_path):
        """Upload image to Yandex and navigate to similar images"""
        try:
            logging.info(f"Navigating to Yandex Images...")
            self.driver.get("https://yandex.com/images/")
            self.wait_for_page_load()
            wait = WebDriverWait(self.driver, DEFAULT_WAIT_TIMEOUT)
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
                    break
                except TimeoutException:
                    continue
            if not file_input:
                logging.error("Could not find file input")
                return False
            abs_path = os.path.abspath(image_path)
            self.driver.execute_script("""
                arguments[0].style.display = 'block';
                arguments[0].style.visibility = 'visible';
                arguments[0].style.opacity = '1';
                arguments[0].style.position = 'static';
            """, file_input)
            file_input.send_keys(abs_path)
            logging.info(f"Uploading: {os.path.basename(image_path)}")
            time.sleep(3)
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
                return False
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
                    break
                except TimeoutException:
                    continue
            if not similar_tab:
                try:
                    similar_tab = wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'CbirNavigation-TabsItem') and (contains(text(), 'Similar') or contains(text(), 'Похожие') or contains(text(), 'похожие'))]"))
                    )
                    logging.info("Found similar tab by text content")
                except TimeoutException:
                    logging.error("Could not find Similar Images tab")
                    return False
            try:
                self.driver.execute_script("arguments[0].scrollIntoView(true);", similar_tab)
                time.sleep(1)
                try:
                    similar_tab.click()
                except ElementClickInterceptedException:
                    self.driver.execute_script("arguments[0].click();", similar_tab)
                time.sleep(3)
                current_url = self.driver.current_url
                if "cbir_page=similar" in current_url or "similar" in current_url.lower():
                    logging.info("Successfully navigated to similar images")
                    return True
                else:
                    logging.info(f"URL doesn't contain similar page indicator: {current_url}")
                    return True
            except Exception as e:
                logging.error(f"Error clicking similar tab: {str(e)}")
                return False
        except Exception as e:
            logging.error(f"Upload error: {str(e)}")
            return False

    def extract_image_urls(self, max_images):
        """Extract image URLs from results page"""
        urls = set()
        logging.info("Extracting image URLs...")
        for i in range(3):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
        time.sleep(2)
        extracted_urls = self.driver.execute_script("""
            var urls = new Set();
            document.querySelectorAll('a[href*="img_url="]').forEach(link => {
                try {
                    var urlMatch = link.href.match(/img_url=([^&]+)/);
                    if (urlMatch) {
                        var decoded = decodeURIComponent(urlMatch[1]);
                        if (decoded.startsWith('http') && !decoded.includes('yandex') && !decoded.includes('yastatic')) {
                            urls.add(decoded);
                        }
                    }
                } catch(e) {}
            });
            document.querySelectorAll('[data-bem*="http"]').forEach(elem => {
                try {
                    var bem = elem.getAttribute('data-bem') || '';
                    var matches = bem.match(/https?:\\/\\/[^"'\\s,}]+/g) || [];
                    matches.forEach(url => {
                        url = url.replace(/\\\\u[\da-f]{4}/gi, '');
                        if (!url.includes('yandex') && !url.includes('yastatic') && !url.includes('avatars.mds')) {
                            urls.add(url);
                        }
                    });
                } catch(e) {}
            });
            document.querySelectorAll('img[src*="http"], [data-src*="http"]').forEach(img => {
                try {
                    var src = img.src || img.getAttribute('data-src') || '';
                    if (src.startsWith('http') && !src.includes('yandex') && !src.includes('yastatic')) {
                        urls.add(src);
                    }
                } catch(e) {}
            });
            document.querySelectorAll('script, [data-state]').forEach(elem => {
                try {
                    var content = elem.textContent || elem.getAttribute('data-state') || '';
                    var matches = content.match(/https?:\\/\\/[^"'\\s,}\\]]+\\.(jpg|jpeg|png|webp|gif)/gi) || [];
                    matches.forEach(url => {
                        url = url.replace(/\\\\?/g, '');
                        if (!url.includes('yandex') && !url.includes('yastatic')) {
                            urls.add(url);
                        }
                    });
                } catch(e) {}
            });
            return Array.from(urls).slice(0, arguments[0]);
        """, max_images)
        for url in extracted_urls:
            try:
                url = unquote(url).strip()
                if (url.startswith('http') and 
                    not any(x in url.lower() for x in ['yandex', 'yastatic', 'data:', 'blob:', 'avatars.mds']) and
                    any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif'])):
                    urls.add(url)
            except:
                continue
        logging.info(f"Found {len(urls)} unique image URLs")
        return list(urls)

    def calculate_orb_similarity(self, img1_path, img2_url):
        """ORB similarity calculation, production-ready"""
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None:
                return 0.0, 0
            img1 = self._resize_image(img1, 800)
            session = requests.Session()
            session.headers.update(self.headers)
            resp = session.get(img2_url, timeout=15, stream=True)
            if resp.status_code != 200:
                return 0.0, 0
            content = b''
            for chunk in resp.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 10 * 1024 * 1024:
                    break
            img_array = np.frombuffer(content, np.uint8)
            img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img2 is None:
                return 0.0, 0
            img2 = self._resize_image(img2, 800)
            kp1, des1 = self.orb.detectAndCompute(img1, None)
            kp2, des2 = self.orb.detectAndCompute(img2, None)
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return 0.0, 0
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            num_matches = len(good_matches)
            if num_matches >= self.orb_min_matches:
                try:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inliers = sum(mask.ravel().tolist())
                        if inliers >= self.orb_min_matches:
                            score = inliers / min(len(kp1), len(kp2))
                            return min(score, 1.0), inliers
                except:
                    pass
            if num_matches > 0:
                score = num_matches / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
                return min(score, 1.0), num_matches
            return 0.0, 0
        except Exception:
            return 0.0, 0

    def _resize_image(self, img, max_dim):
        """Helper to resize image if needed"""
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def check_similarity_parallel(self, image_path, urls):
        """Check multiple URLs in parallel using ThreadPoolExecutor"""
        matched_urls = []
        image_name = os.path.basename(image_path)
        logging.info(f"Checking {len(urls)} URLs for matches...")
        
        def check_url_and_download(url):
            orb_score, num_matches = self.calculate_orb_similarity(image_path, url)
            logging.info(f"Score: {orb_score:.3f}, Matches: {num_matches}")
            # Save the image link as soon as processed
            self.links_writer.writerow([image_name, url])
            self.links_file.flush()
            if orb_score >= self.orb_similarity_threshold and num_matches >= self.orb_min_matches:
                # Download the similar image
                downloaded_path = self.download_similar_image(url, image_name, orb_score)
                logging.info(f"✓ Match found! Score: {orb_score:.3f}, URL: {url[:60]}...")
                return {
                    'source_image': image_name,
                    'matched_url': url,
                    'orb_score': orb_score,
                    'num_matches': num_matches,
                    'downloaded_path': downloaded_path
                }
            return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(check_url_and_download, url): url 
                for url in urls
            }
            completed = 0
            for future in as_completed(future_to_url):
                completed += 1
                try:
                    result = future.result(timeout=30)
                    if result:
                        matched_urls.append(result)
                except Exception:
                    continue
        logging.info("Completed similarity checking")
        return matched_urls

    def process_image(self, image_path):
        """Process single image"""
        image_name = os.path.basename(image_path)
        start_time = time.time()
        logging.info(f"[Processing] {image_name}")
        
        if not self.upload_to_yandex(image_path):
            end_time = time.time()
            processing_time = end_time - start_time
            if self.log_writer:
                self.log_writer.writerow([image_name, f"{processing_time:.2f}", 0, "Upload failed"])
            logging.error("Upload failed")
            return []
        
        urls = self.extract_image_urls(self.max_images)
        if not urls:
            end_time = time.time()
            processing_time = end_time - start_time
            if self.log_writer:
                self.log_writer.writerow([image_name, f"{processing_time:.2f}", 0, "No URLs found"])
            logging.error("No URLs found")
            return []
        
        logging.info(f"Found {len(urls)} URLs to check")
        matches = self.check_similarity_parallel(image_path, urls)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Log the processing time
        if self.log_writer:
            self.log_writer.writerow([image_name, f"{processing_time:.2f}", len(matches), f"Found {len(matches)} matches"])
        
        logging.info(f"{len(matches)} matches found for {image_name} (took {processing_time:.2f}s)")
        
        # Move processed image to done folder concurrently
        move_executor = ThreadPoolExecutor(max_workers=1)
        move_executor.submit(self.move_image_to_done, image_path)
        move_executor.shutdown(wait=False)

        # Counter for processed images
        self.processed_count += 1
        logging.info(f"Processed {self.processed_count} of {self.total_images} images.")

        return matches

    def save_results(self, all_matches):
        """Save results to CSV"""
        if not all_matches:
            logging.info("No matches to save")
            return
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
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
        logging.info(f"Results saved to {self.output_file}")

    def run(self):
        """Main execution, CLI ready"""
        logging.info("=" * 80)
        logging.info("YANDEX IMAGE ORB MATCHER - Production Version")
        logging.info("=" * 80)
        image_files = self.get_image_files()
        self.total_images = len(image_files)
        if not image_files:
            logging.error("No images found in the folder!")
            return
        logging.info(f"Found {self.total_images} images to process")
        self.driver = self.init_driver()
        all_matches = []
        
        try:
            with open(self.log_filename, 'w', newline='', encoding='utf-8') as log_file:
                self.log_writer = csv.writer(log_file)
                self.log_writer.writerow(['Image Name', 'Processing Time (seconds)', 'Matches Found', 'Notes'])
                
                for idx, image_path in enumerate(image_files, 1):
                    logging.info(f"[{idx}/{self.total_images}] Processing: {os.path.basename(image_path)}")
                    matches = self.process_image(image_path)
                    all_matches.extend(matches)
                    if idx < self.total_images:
                        time.sleep(self.batch_delay)
                        
            logging.info("=" * 80)
            logging.info("RESULTS SUMMARY")
            logging.info("=" * 80)
            self.save_results(all_matches)
            logging.info(f"Images processed: {self.total_images}")
            logging.info(f"Total matches found: {len(all_matches)}")
            logging.info(f"Processing log saved to {self.log_filename}")
            logging.info(f"Image links saved to {self.links_filename}")
            
            matches_by_source = defaultdict(list)
            for match in all_matches:
                matches_by_source[match['source_image']].append(match)
            if matches_by_source:
                for source, matches in matches_by_source.items():
                    avg_score = sum(m['orb_score'] for m in matches) / len(matches)
                    logging.info(f"{source}: {len(matches)} matches (avg score: {avg_score:.3f})")
            logging.info("Processing complete!")
            
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error(f"Error: {str(e)}")
        finally:
            self.links_file.close()
            if self.driver:
                logging.info("Closing browser...")
                self.driver.quit()
            logging.info("Done!")

def main():
    parser = argparse.ArgumentParser(description="Yandex Image ORB Matcher")
    parser.add_argument('--images_folder', type=str, default='images', help='Folder containing images')
    parser.add_argument('--output_file', type=str, default='yandex_matches.csv', help='CSV file for output')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS, help='Max concurrent workers for matching')
    parser.add_argument('--max_images', type=int, default=DEFAULT_MAX_IMAGES, help='Max Yandex result images to check')
    parser.add_argument('--orb_features', type=int, default=DEFAULT_ORB_FEATURES, help='ORB nfeatures parameter')
    parser.add_argument('--orb_min_matches', type=int, default=DEFAULT_ORB_MIN_MATCHES, help='Min ORB matches to consider a match')
    parser.add_argument('--orb_similarity_threshold', type=float, default=DEFAULT_ORB_SIMILARITY_THRESHOLD, help='ORB similarity score threshold')
    parser.add_argument('--batch_delay', type=float, default=DEFAULT_BATCH_DELAY, help='Delay between images (seconds)')
    args = parser.parse_args()

    matcher = YandexImageMatcher(
        images_folder=args.images_folder,
        output_file=args.output_file,
        max_workers=args.max_workers,
        max_images=args.max_images,
        orb_features=args.orb_features,
        orb_min_matches=args.orb_min_matches,
        orb_similarity_threshold=args.orb_similarity_threshold,
        batch_delay=args.batch_delay
    )
    matcher.run()

if __name__ == "__main__":
    main()