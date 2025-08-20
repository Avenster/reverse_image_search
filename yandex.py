import time
import csv
import cv2
import numpy as np
import requests
import os
from pathlib import Path
import glob
from urllib.parse import unquote, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from collections import defaultdict

# Configuration
MAX_WORKERS = 5  # Parallel image checking
WAIT_TIMEOUT = 15  # Increased timeout
SHORT_WAIT = 1
MEDIUM_WAIT = 3
LONG_WAIT = 5
ORB_FEATURES = 2000  # Reduced from 3000 for speed
ORB_MIN_MATCHES = 10
ORB_SIMILARITY_THRESHOLD = 0.01

class YandexImageMatcher:
    def __init__(self, images_folder='images', output_file='yandex_matched_images.csv'):
        self.images_folder = Path(images_folder)
        self.output_file = output_file
        self.driver = None
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=1.2, nlevels=8)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://yandex.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
    def init_driver(self):
        """Initialize undetected Chrome driver"""
        options = uc.ChromeOptions()
        options.add_argument("--user-data-dir=/tmp/chrome-user-data")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--window-size=1920,1080")
        
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
            WebDriverWait(self.driver, WAIT_TIMEOUT).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            time.sleep(SHORT_WAIT)
        except TimeoutException:
            print("  Page load timeout, continuing...")
    
    def upload_to_yandex(self, image_path):
        """Upload image to Yandex and navigate to similar images"""
        try:
            print(f"  Navigating to Yandex Images...")
            # Navigate to Yandex Images
            self.driver.get("https://yandex.com/images/")
            self.wait_for_page_load()
            
            # Find and use the file input
            wait = WebDriverWait(self.driver, WAIT_TIMEOUT)
            
            # Try multiple selectors for file input
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
                print("  Could not find file input")
                return False
            
            # Make file input visible and upload
            abs_path = os.path.abspath(image_path)
            print(f"  Uploading: {os.path.basename(image_path)}")
            
            self.driver.execute_script("""
                arguments[0].style.display = 'block';
                arguments[0].style.visibility = 'visible';
                arguments[0].style.opacity = '1';
                arguments[0].style.position = 'static';
            """, file_input)
            
            file_input.send_keys(abs_path)
            
            # Wait for upload and redirect
            print("  Waiting for upload to complete...")
            time.sleep(MEDIUM_WAIT)
            
            # Wait for the results page to load
            try:
                wait.until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".CbirNavigation-TabsItem")),
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".SerpItem")),
                        EC.url_contains("cbir_id")
                    )
                )
            except TimeoutException:
                print("  Results page didn't load properly")
                return False
            
            print("  Upload completed, looking for Similar Images tab...")
            
            # Now find and click the "Similar images" tab
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
                        # XPath selector
                        similar_tab = wait.until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                    else:
                        # CSS selector
                        similar_tab = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                    print(f"  Found similar tab with selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not similar_tab:
                # Try to find by text content as fallback
                try:
                    similar_tab = wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//a[contains(@class, 'CbirNavigation-TabsItem') and (contains(text(), 'Similar') or contains(text(), 'Похожие') or contains(text(), 'похожие'))]"))
                    )
                    print("  Found similar tab by text content")
                except TimeoutException:
                    print("  Could not find Similar Images tab")
                    return False
            
            # Click the similar images tab
            try:
                print("  Clicking Similar Images tab...")
                
                # Scroll tab into view
                self.driver.execute_script("arguments[0].scrollIntoView(true);", similar_tab)
                time.sleep(SHORT_WAIT)
                
                # Try clicking
                try:
                    similar_tab.click()
                except ElementClickInterceptedException:
                    # If regular click fails, try JavaScript click
                    self.driver.execute_script("arguments[0].click();", similar_tab)
                
                # Wait for similar images page to load
                time.sleep(MEDIUM_WAIT)
                
                # Verify we're on the similar images page
                current_url = self.driver.current_url
                if "cbir_page=similar" in current_url or "similar" in current_url.lower():
                    print("  Successfully navigated to similar images")
                    return True
                else:
                    print(f"  URL doesn't contain similar page indicator: {current_url}")
                    # Still try to continue as Yandex might have changed URL structure
                    return True
                    
            except Exception as e:
                print(f"  Error clicking similar tab: {str(e)}")
                return False
            
        except Exception as e:
            print(f"  Upload error: {str(e)}")
            return False
    
    def extract_image_urls(self, max_images=50):
        """Extract image URLs from results page"""
        urls = set()
        
        print("  Extracting image URLs...")
        
        # Scroll to load more images
        for i in range(5):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(SHORT_WAIT)
        
        # Wait a bit for images to load
        time.sleep(MEDIUM_WAIT)
        
        # Extract URLs using multiple methods
        extracted_urls = self.driver.execute_script("""
            var urls = new Set();
            
            // Method 1: From img_url parameters in links
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
            
            // Method 2: From data-bem attributes
            document.querySelectorAll('[data-bem*="http"]').forEach(elem => {
                try {
                    var bem = elem.getAttribute('data-bem') || '';
                    var matches = bem.match(/https?:\\/\\/[^"'\\s,}]+/g) || [];
                    matches.forEach(url => {
                        url = url.replace(/\\\\u[\da-f]{4}/gi, ''); // Remove unicode escapes
                        if (!url.includes('yandex') && !url.includes('yastatic') && !url.includes('avatars.mds')) {
                            urls.add(url);
                        }
                    });
                } catch(e) {}
            });
            
            // Method 3: From image sources and data attributes
            document.querySelectorAll('img[src*="http"], [data-src*="http"]').forEach(img => {
                try {
                    var src = img.src || img.getAttribute('data-src') || '';
                    if (src.startsWith('http') && !src.includes('yandex') && !src.includes('yastatic')) {
                        urls.add(src);
                    }
                } catch(e) {}
            });
            
            // Method 4: Look for JSON data in script tags or data attributes
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
        
        # Clean and filter URLs
        for url in extracted_urls:
            try:
                url = unquote(url).strip()
                if (url.startswith('http') and 
                    not any(x in url.lower() for x in ['yandex', 'yastatic', 'data:', 'blob:', 'avatars.mds']) and
                    any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif'])):
                    urls.add(url)
            except:
                continue
        
        print(f"  Found {len(urls)} unique image URLs")
        return list(urls)
    
    def calculate_orb_similarity(self, img1_path, img2_url):
        """Optimized ORB similarity calculation"""
        try:
            # Load local image
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None:
                return 0.0, 0
            
            # Resize if too large
            img1 = self._resize_image(img1, 800)
            
            # Download remote image with proper headers
            session = requests.Session()
            session.headers.update(self.headers)
            
            resp = session.get(img2_url, timeout=15, stream=True)
            if resp.status_code != 200:
                return 0.0, 0
            
            # Limit download size
            content = b''
            for chunk in resp.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 10 * 1024 * 1024:  # 10MB limit
                    break
            
            img_array = np.frombuffer(content, np.uint8)
            img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img2 is None:
                return 0.0, 0
            
            img2 = self._resize_image(img2, 800)
            
            # Compute ORB features
            kp1, des1 = self.orb.detectAndCompute(img1, None)
            kp2, des2 = self.orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return 0.0, 0
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            num_matches = len(good_matches)
            
            # Geometric verification for high-confidence matches
            if num_matches >= ORB_MIN_MATCHES:
                try:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        inliers = sum(mask.ravel().tolist())
                        if inliers >= ORB_MIN_MATCHES:
                            score = inliers / min(len(kp1), len(kp2))
                            return min(score, 1.0), inliers
                except:
                    pass
            
            # Fallback score based on raw matches
            if num_matches > 0:
                score = num_matches / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
                return min(score, 1.0), num_matches
            
            return 0.0, 0
            
        except Exception as e:
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
        """Check multiple URLs in parallel"""
        matched_urls = []
        image_name = os.path.basename(image_path)
        
        print(f"  Checking {len(urls)} URLs for matches...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(self.calculate_orb_similarity, image_path, url): url 
                for url in urls
            }
            
            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                completed += 1
                
                try:
                    orb_score, num_matches = future.result(timeout=20)
                    
                    print(f"  [{completed}/{len(urls)}] Score: {orb_score:.3f}, Matches: {num_matches}", end='\r')
                    
                    if orb_score >= ORB_SIMILARITY_THRESHOLD and num_matches >= ORB_MIN_MATCHES:
                        matched_urls.append({
                            'source_image': image_name,
                            'matched_url': url,
                            'orb_score': orb_score,
                            'num_matches': num_matches
                        })
                        print(f"\n  ✓ Match found! Score: {orb_score:.3f}, URL: {url[:60]}...")
                        
                except Exception as e:
                    continue
        
        print(f"\n  Completed similarity checking")
        return matched_urls
    
    def process_image(self, image_path):
        """Process single image"""
        image_name = os.path.basename(image_path)
        print(f"\n[Processing] {image_name}")
        print("-" * 50)
        
        # Upload and navigate to similar images
        if not self.upload_to_yandex(image_path):
            print("  ❌ Upload failed")
            return []
        
        # Extract image URLs
        urls = self.extract_image_urls(max_images=50)
        
        if not urls:
            print("  ❌ No URLs found")
            return []
        
        print(f"  Found {len(urls)} URLs to check")
        
        # Check similarity in parallel
        matches = self.check_similarity_parallel(image_path, urls)
        
        print(f"  ✅ {len(matches)} matches found for {image_name}")
        return matches
    
    def save_results(self, all_matches):
        """Save results to CSV"""
        if not all_matches:
            print("No matches to save")
            return
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Source Image', 'Matched URL', 'ORB Score', 'Feature Matches'])
            
            for match in all_matches:
                writer.writerow([
                    match['source_image'],
                    match['matched_url'],
                    f"{match['orb_score']:.4f}",
                    match['num_matches']
                ])
        
        print(f"\n📄 Results saved to {self.output_file}")
    
    def run(self):
        """Main execution"""
        print("=" * 80)
        print("🔍 YANDEX IMAGE MATCHER - Fixed Version")
        print("=" * 80)
        
        # Get images
        image_files = self.get_image_files()
        
        if not image_files:
            print("❌ No images found in the folder!")
            return
        
        print(f"📁 Found {len(image_files)} images to process")
        
        # Initialize driver
        print("🚀 Initializing browser...")
        self.driver = self.init_driver()
        
        all_matches = []
        
        try:
            # Process each image
            for idx, image_path in enumerate(image_files, 1):
                print(f"\n🖼️  [{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
                matches = self.process_image(image_path)
                all_matches.extend(matches)
                
                # Delay between images to avoid rate limiting
                if idx < len(image_files):
                    print("  ⏳ Waiting before next image...")
                    time.sleep(2)
            
            # Results summary
            print("\n" + "=" * 80)
            print("📊 RESULTS SUMMARY")
            print("=" * 80)
            
            self.save_results(all_matches)
            
            # Statistics
            print(f"\n📈 Statistics:")
            print(f"   • Images processed: {len(image_files)}")
            print(f"   • Total matches found: {len(all_matches)}")
            
            # Per-image breakdown
            matches_by_source = defaultdict(list)
            for match in all_matches:
                matches_by_source[match['source_image']].append(match)
            
            if matches_by_source:
                print(f"\n📋 Matches per image:")
                for source, matches in matches_by_source.items():
                    avg_score = sum(m['orb_score'] for m in matches) / len(matches)
                    print(f"   • {source}: {len(matches)} matches (avg score: {avg_score:.3f})")
            
            print(f"\n✅ Processing complete!")
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
        finally:
            if self.driver:
                print("🔚 Closing browser...")
                self.driver.quit()
            print("👋 Done!")

def main():
    # You can customize these parameters
    matcher = YandexImageMatcher(
        images_folder='images',           # Folder containing your images
        output_file='yandex_matches.csv' # Output CSV file
    )
    matcher.run()

if __name__ == "__main__":
    main()