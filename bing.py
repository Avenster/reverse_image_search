import time
import csv
import cv2
import numpy as np
import requests
import os
import json
import re
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from pathlib import Path
import glob
from urllib.parse import unquote

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = uc.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

# -----------------------
# Get Images from Folder
# -----------------------
def get_image_files(folder_path):
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist!")
        return []
    
    for extension in supported_extensions:
        image_files.extend(glob.glob(str(folder_path / extension)))
        image_files.extend(glob.glob(str(folder_path / extension.upper())))
    
    return sorted(image_files)

# -----------------------
# Upload to Bing Visual Search
# -----------------------
def upload_to_bing(driver, image_path):
    """
    Upload image to Bing Visual Search
    """
    try:
        # Go to Bing images
        driver.get("https://www.bing.com/images")
        time.sleep(3)
        
        # Find the visual search button (camera icon)
        visual_search_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "#sbi_b, .sbi_b_prtl, [aria-label*='Visual Search'], [aria-label*='visual search']"))
        )
        visual_search_btn.click()
        time.sleep(2)
        
        # Find file input
        file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        abs_path = os.path.abspath(image_path)
        file_input.send_keys(abs_path)
        
        # Wait for results to load
        time.sleep(10)
        
        # Check if we're on the results page
        if "search?" in driver.current_url or "detailV2" in driver.current_url:
            return True
            
    except Exception as e:
        print(f"Error during upload: {str(e)}")
    
    return False

# -----------------------
# Extract Image URLs from Bing Results
# -----------------------
def get_bing_image_urls(driver, max_images=50):
    """
    Extract image URLs from Bing visual search results - FIXED VERSION
    """
    urls = []
    
    try:
        # Wait for images to load
        time.sleep(3)
        
        # Scroll to load more images
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Method 1: Find all richImgLnk elements - FIXED
        rich_links = driver.find_elements(By.CSS_SELECTOR, "a.richImgLnk")
        print(f"Found {len(rich_links)} richImgLnk elements")
        
        for link in rich_links[:max_images]:
            try:
                # FIXED: Get the 'data-m' attribute (not just 'm')
                m_data = link.get_attribute('data-m')
                if m_data:
                    data = json.loads(m_data)
                    # Extract the actual image URL
                    if 'murl' in data:
                        urls.append(data['murl'])
                    elif 'purl' in data:
                        urls.append(data['purl'])
            except Exception as e:
                print(f"Error parsing richImgLnk data: {e}")
                pass
        
        print(f"Extracted {len(urls)} URLs from richImgLnk elements")
        
        # Method 2: Try to find more URLs from other elements if needed
        if len(urls) < 10:
            # Look for data-m attributes in other divs
            divs_with_data = driver.find_elements(By.CSS_SELECTOR, "div[data-m], li[data-m]")
            print(f"Found {len(divs_with_data)} additional elements with data-m")
            
            for div in divs_with_data[:max_images]:
                try:
                    data_m = div.get_attribute('data-m')
                    if data_m:
                        data = json.loads(data_m)
                        if 'murl' in data:
                            urls.append(data['murl'])
                        elif 'purl' in data:
                            urls.append(data['purl'])
                except:
                    pass
        
        # Method 3: Extract from page source as fallback
        if len(urls) < 10:
            page_source = driver.page_source
            
            # Look for URLs in JavaScript
            murl_pattern = r'"murl":"(https?://[^"]+)"'
            found_urls = re.findall(murl_pattern, page_source)
            urls.extend(found_urls)
            
            # Also look for purl
            purl_pattern = r'"purl":"(https?://[^"]+)"'
            found_purls = re.findall(purl_pattern, page_source)
            urls.extend(found_purls)
        
        # Clean URLs but keep ALL valid ones (removed aggressive filtering)
        cleaned_urls = []
        
        for url in urls:
            # Unescape URL if needed
            url = unquote(url)
            
            # Only skip obvious non-image URLs
            if url.startswith('data:') or not url.startswith('http'):
                continue
            
            # Skip only Bing thumbnail URLs that are clearly not the original images
            if 'bing.com/th' in url or 'mm.bing.net/th' in url:
                continue
            
            # Add ALL valid URLs (removed duplicate checking as requested)
            cleaned_urls.append(url)
        
        print(f"Final cleaned URLs count: {len(cleaned_urls)}")
        
        # Debug: print first few URLs
        if cleaned_urls:
            print("Sample URLs found:")
            for i, url in enumerate(cleaned_urls[:5]):
                print(f"  {i+1}. {url[:80]}...")
        else:
            print("⚠️  No URLs extracted! Debugging page structure...")
            # Debug output
            print(f"Current URL: {driver.current_url}")
            print(f"Page title: {driver.title}")
            
            # Check if we can find any links at all
            all_links = driver.find_elements(By.TAG_NAME, "a")
            print(f"Total links on page: {len(all_links)}")
            
            # Check for any elements with data-m
            data_m_elements = driver.find_elements(By.CSS_SELECTOR, "[data-m]")
            print(f"Elements with data-m: {len(data_m_elements)}")
            
            if data_m_elements:
                print("Sample data-m content:")
                for i, elem in enumerate(data_m_elements[:3]):
                    data_m = elem.get_attribute('data-m')
                    print(f"  Element {i+1}: {data_m[:100]}...")
        
        return cleaned_urls[:max_images]
        
    except Exception as e:
        print(f"Error extracting URLs: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return urls

# -----------------------
# ORB Feature Matching
# -----------------------
def calculate_orb_similarity(img1_path, img2_url, min_matches=8):
    """
    Uses ORB with FLANN matcher for image matching
    """
    try:
        # Load local image
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None:
            return 0.0, 0
        
        # Download remote image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bing.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }
        
        resp = requests.get(img2_url, timeout=15, headers=headers, allow_redirects=True)
        if resp.status_code != 200:
            return 0.0, 0
        
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img2 is None:
            return 0.0, 0
        
        # Resize images if too large (for faster processing)
        max_dim = 1000
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        if max(h1, w1) > max_dim:
            scale = max_dim / max(h1, w1)
            new_w1, new_h1 = int(w1 * scale), int(h1 * scale)
            img1 = cv2.resize(img1, (new_w1, new_h1))
        
        if max(h2, w2) > max_dim:
            scale = max_dim / max(h2, w2)
            new_w2, new_h2 = int(w2 * scale), int(h2 * scale)
            img2 = cv2.resize(img2, (new_w2, new_h2))
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return 0.0, 0
        
        # Use BFMatcher for ORB (faster and more reliable for binary descriptors)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match descriptors
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # Slightly more lenient for Bing
                    good_matches.append(m)
        
        num_good_matches = len(good_matches)
        
        # Calculate similarity score
        if num_good_matches >= min_matches:
            # Try geometric verification
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    inliers = mask.ravel().tolist()
                    num_inliers = sum(inliers)
                    
                    # Calculate score based on inliers
                    if num_inliers >= min_matches:
                        similarity_score = num_inliers / min(len(kp1), len(kp2))
                        return min(similarity_score, 1.0), num_inliers
            except:
                pass
        
        # Fallback score if geometric verification fails
        if num_good_matches > 0:
            similarity_score = num_good_matches / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
            return min(similarity_score, 1.0), num_good_matches
        
        return 0.0, 0
        
    except Exception as e:
        return 0.0, 0

# -----------------------
# Template Matching
# -----------------------
def calculate_template_matching(img1_path, img2_url, threshold=0.7):
    """
    Multi-scale template matching
    """
    try:
        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None:
            return 0.0
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bing.com/'
        }
        
        resp = requests.get(img2_url, timeout=15, headers=headers, allow_redirects=True)
        if resp.status_code != 200:
            return 0.0
        
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img2 is None:
            return 0.0
        
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        # Try multiple scales
        scales = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        max_score = 0
        
        for scale in scales:
            width = int(w1 * scale)
            height = int(h1 * scale)
            
            # Skip if template would be too large or too small
            if width > w2 or height > h2 or width < 20 or height < 20:
                continue
            
            resized = cv2.resize(img1, (width, height))
            
            # Apply template matching
            result = cv2.matchTemplate(img2, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            max_score = max(max_score, max_val)
        
        return max_score
        
    except Exception as e:
        return 0.0

# -----------------------
# Combined Similarity Check
# -----------------------
def is_similar_image(img1_path, img2_url):
    """
    Check if images are similar using multiple methods
    """
    # ORB matching - lowered threshold for better detection
    orb_score, num_matches = calculate_orb_similarity(img1_path, img2_url, min_matches=8)
    
    # Template matching
    template_score = calculate_template_matching(img1_path, img2_url)
    
    # Consider it a match if:
    # 1. ORB finds good matches (lowered threshold)
    # 2. OR template matching finds high correlation
    is_match = (orb_score >= 0.008 and num_matches >= 8) or (template_score >= 0.7)
    
    return is_match, orb_score, template_score, num_matches

# -----------------------
# Process Single Image
# -----------------------
def process_image(driver, image_path):
    """Process a single image through Bing Visual Search"""
    image_name = os.path.basename(image_path)
    print(f"\nProcessing: {image_name}")
    print("-" * 50)
    
    # Upload to Bing
    print("Uploading to Bing Visual Search...")
    success = upload_to_bing(driver, image_path)
    
    if not success:
        print("Failed to upload image")
        return []
    
    # Extract URLs
    print("Extracting image URLs from results...")
    urls = get_bing_image_urls(driver, max_images=50)
    
    if not urls:
        print("No URLs found - checking page structure...")
        # Debug: print page title to verify we're on the right page
        print(f"Current URL: {driver.current_url[:100]}")
        print(f"Page title: {driver.title}")
        return []
    
    print(f"Found {len(urls)} image URLs to check")
    
    # Check similarity with ALL URLs (as requested)
    matched_urls = []
    print("\nChecking for matches...")
    
    for i, url in enumerate(urls, 1):
        print(f"  Checking [{i}/{len(urls)}]: {url[:60]}...", end='')
        
        try:
            is_match, orb_score, template_score, num_matches = is_similar_image(image_path, url)
            
            if is_match:
                print(f"\n  ✓ MATCH FOUND!")
                print(f"    ORB: {orb_score:.4f} ({num_matches} matches), Template: {template_score:.4f}")
                
                matched_urls.append({
                    'source_image': image_name,
                    'matched_url': url,
                    'orb_score': orb_score,
                    'template_score': template_score,
                    'num_matches': num_matches
                })
            else:
                print(f" [ORB: {orb_score:.3f}, Template: {template_score:.3f}]")
                
        except Exception as e:
            print(f"\n  ⚠ Error: {str(e)}")
            continue
    
    print(f"\n✓ Found {len(matched_urls)} matching images for {image_name}")
    return matched_urls

# -----------------------
# Save Results
# -----------------------
def save_results(all_matches, filename="bing_matched_images.csv"):
    """Save matched URLs to CSV"""
    if not all_matches:
        print("No matches to save")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source Image', 'Matched URL', 'ORB Score', 'Template Score', 'Feature Matches'])
        
        for match in all_matches:
            writer.writerow([
                match['source_image'],
                match['matched_url'],
                f"{match['orb_score']:.4f}",
                f"{match['template_score']:.4f}",
                match['num_matches']
            ])
    
    print(f"✓ Results saved to {filename}")

# -----------------------
# Main Function
# -----------------------
def main():
    # Configuration
    IMAGES_FOLDER = 'images'
    OUTPUT_FILE = 'bing_matched_images.csv'
    BATCH_DELAY = 5
    
    print("=" * 60)
    print("BING VISUAL SEARCH IMAGE MATCHER - FIXED VERSION")
    print("Using ORB feature matching + Template matching")
    print("=" * 60)
    
    # Get images
    print(f"\nScanning folder: {IMAGES_FOLDER}")
    image_files = get_image_files(IMAGES_FOLDER)
    
    if not image_files:
        print(f"No images found in '{IMAGES_FOLDER}' folder!")
        return
    
    print(f"Found {len(image_files)} images:")
    for img in image_files:
        print(f"  • {os.path.basename(img)}")
    
    # Initialize browser
    print("\nInitializing browser...")
    driver = init_driver()
    
    all_matches = []
    
    try:
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(image_files)}] Processing image {idx} of {len(image_files)}")
            print(f"{'='*60}")
            
            matches = process_image(driver, image_path)
            all_matches.extend(matches)
            
            # Delay between images
            if idx < len(image_files):
                print(f"\nWaiting {BATCH_DELAY} seconds before next image...")
                time.sleep(BATCH_DELAY)
        
        # Save results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        
        save_results(all_matches, OUTPUT_FILE)
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"  • Images processed: {len(image_files)}")
        print(f"  • Total matches found: {len(all_matches)}")
        
        # Show matches per image
        from collections import defaultdict
        matches_by_source = defaultdict(list)
        for match in all_matches:
            matches_by_source[match['source_image']].append(match)
        
        print(f"\nMatches per image:")
        for source, matches in matches_by_source.items():
            print(f"  • {source}: {len(matches)} matches")
            if matches:
                best_match = max(matches, key=lambda x: x['orb_score'])
                print(f"    Best match: ORB={best_match['orb_score']:.3f}, Template={best_match['template_score']:.3f}")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing browser...")
        driver.quit()
        print("Done!")

if __name__ == "__main__":
    main()