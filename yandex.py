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
from urllib.parse import unquote, urlparse, parse_qs

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=en-US")
    options.add_argument("--disable-extensions")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
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
# Upload to Yandex and Navigate to Similar Images
# -----------------------
def upload_to_yandex(driver, image_path):
    """
    Upload image to Yandex starting from main page and navigate to similar images page
    """
    try:
        # Go to main Yandex page
        print("Loading Yandex main page...")
        driver.get("https://yandex.com/")
        time.sleep(3)
        
        # Look for camera icon - try multiple approaches
        camera_button = None
        
        # Method 1: Look for camera icon in search area
        try:
            camera_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".search3__icon-camera, .search2__icon_camera, [data-testid='image-search'], .image-search__camera, .search__button_type_camera"))
            )
            print("Found camera icon using standard selectors")
        except:
            pass
        
        # Method 2: Try clicking on Images link first, then camera
        if not camera_button:
            try:
                # Click Images link
                images_link = driver.find_element(By.LINK_TEXT, "Images")
                images_link.click()
                time.sleep(2)
                
                # Now look for camera
                camera_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".search3__icon-camera, .search2__icon_camera, .CbirInput-SearchByImageButton, .input__search-by-image"))
                )
                print("Found camera icon after clicking Images link")
            except:
                pass
        
        # Method 3: Direct navigation to images page and find camera
        if not camera_button:
            try:
                driver.get("https://yandex.com/images/")
                time.sleep(3)
                
                camera_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".search3__icon-camera, .CbirInput-SearchByImageButton, .input__search-by-image, [aria-label*='Search by image']"))
                )
                print("Found camera icon on images page")
            except:
                pass
        
        # Method 4: JavaScript approach to find camera button
        if not camera_button:
            try:
                camera_elements = driver.execute_script("""
                    // Look for camera icons by various attributes
                    var selectors = [
                        '[data-testid*="camera"]',
                        '[aria-label*="camera"]',
                        '[aria-label*="image"]',
                        '.search3__icon-camera',
                        '.CbirInput-SearchByImageButton',
                        '.input__search-by-image',
                        '[title*="Search by image"]',
                        '[title*="camera"]'
                    ];
                    
                    for (var selector of selectors) {
                        var elements = document.querySelectorAll(selector);
                        if (elements.length > 0) {
                            return elements[0];
                        }
                    }
                    return null;
                """)
                
                if camera_elements:
                    camera_button = camera_elements
                    print("Found camera icon using JavaScript search")
            except:
                pass
        
        if not camera_button:
            print("ERROR: Could not find camera icon on Yandex")
            return False
        
        # Click the camera button
        try:
            # Scroll to button and click
            driver.execute_script("arguments[0].scrollIntoView(true);", camera_button)
            time.sleep(1)
            camera_button.click()
            print("Clicked camera icon successfully")
            time.sleep(2)
        except:
            # Try JavaScript click
            driver.execute_script("arguments[0].click();", camera_button)
            print("Clicked camera icon using JavaScript")
            time.sleep(2)
        
        # Find file input and upload
        print("Looking for file input...")
        file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
        
        file_input = None
        for inp in file_inputs:
            if inp.is_displayed() or True:  # Sometimes hidden but functional
                file_input = inp
                break
        
        if not file_input:
            print("ERROR: Could not find file input")
            return False
        
        # Upload the file
        abs_path = os.path.abspath(image_path)
        print(f"Uploading file: {abs_path}")
        file_input.send_keys(abs_path)
        
        # Wait for upload and processing
        print("Waiting for upload and initial processing...")
        time.sleep(8)
        
        # Check if we're on a results page
        current_url = driver.current_url
        print(f"Current URL after upload: {current_url[:100]}...")
        
        # Look for "Show more similar images" or similar button
        print("Looking for 'Show more similar images' button...")
        time.sleep
        
        similar_button = None
        
        # Method 1: Look for specific button classes
        try:
            similar_button_selectors = [
                "a.CbirSimilarList-MoreButton",
                "a.Button.CbirSimilarList-MoreButton",
                "a[href*='cbir_page=similar']",
                ".CbirSimilarList-MoreButton",
                ".Button_link.CbirSimilarList-MoreButton"
            ]
            
            for selector in similar_button_selectors:
                try:
                    similar_button = driver.find_element(By.CSS_SELECTOR, selector)
                    if similar_button.is_displayed():
                        print(f"Found similar button with selector: {selector}")
                        break
                except:
                    continue
        except:
            pass
        
        # Method 2: JavaScript search for buttons with relevant text
        if not similar_button:
            try:
                similar_button = driver.execute_script("""
                    var buttons = document.querySelectorAll('a, button');
                    for (var i = 0; i < buttons.length; i++) {
                        var text = buttons[i].textContent || buttons[i].innerText || '';
                        var href = buttons[i].href || '';
                        
                        if (text.toLowerCase().includes('similar') || 
                            text.toLowerCase().includes('show more') ||
                            href.includes('cbir_page=similar')) {
                            return buttons[i];
                        }
                    }
                    return null;
                """)
                
                if similar_button:
                    print("Found similar button using JavaScript text search")
            except:
                pass
        
        # Method 3: Check if already on similar page or navigate manually
        if not similar_button:
            print("Could not find similar button, checking URL...")
            
            # Check if we already have a cbir_id in URL
            if "cbir_id=" in current_url:
                if "cbir_page=similar" not in current_url:
                    # Manually construct similar images URL
                    if "?" in current_url:
                        similar_url = current_url + "&cbir_page=similar"
                    else:
                        similar_url = current_url + "?cbir_page=similar"
                    
                    print(f"Navigating to similar images page: {similar_url[:100]}...")
                    driver.get(similar_url)
                    time.sleep(5)
                    return True
                else:
                    print("Already on similar images page")
                    return True
            else:
                print("ERROR: No cbir_id found in URL, upload may have failed")
                return False
        
        # Click the similar button if found
        
        if similar_button:
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", similar_button)
                time.sleep(1)
                similar_button.click()
                print("Clicked 'Show more similar images' button")
                time.sleep(5)
                return True
            except Exception as e:
                print(f"Error clicking similar button: {str(e)}")
                # Try JavaScript click
                try:
                    driver.execute_script("arguments[0].click();", similar_button)
                    print("Clicked similar button using JavaScript")
                    time.sleep(5)
                    return True
                except:
                    print("Failed to click similar button")
                    return False
        
        return False
            
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# -----------------------
# Extract Image URLs from Yandex Similar Images Page
# -----------------------
def get_yandex_image_urls(driver, max_images=50):
    """
    Extract image URLs from Yandex similar images results
    """
    urls = []
    
    try:
        # Wait for content to load
        time.sleep(3)
        
        print("Current page URL:", driver.current_url[:100])
        
        # Scroll to load more images
        print("Scrolling to load more images...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        for scroll_attempt in range(8):  # More scrolls for better coverage
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            print(f"Scroll {scroll_attempt + 1}: New height = {new_height}")
        
        # Method 1: Extract from img_url parameters in links
        print("Extracting URLs from link href attributes...")
        
        # Look for links with img_url parameter
        links_with_img_url = driver.execute_script("""
            var links = document.querySelectorAll('a[href*="img_url="]');
            var urls = [];
            
            for (var i = 0; i < links.length; i++) {
                var href = links[i].href;
                var match = href.match(/img_url=([^&]+)/);
                if (match) {
                    try {
                        var decoded = decodeURIComponent(match[1]);
                        if (decoded.startsWith('http')) {
                            urls.push(decoded);
                        }
                    } catch (e) {
                        // Skip invalid URLs
                    }
                }
            }
            return urls;
        """)
        
        if links_with_img_url:
            urls.extend(links_with_img_url)
            print(f"Found {len(links_with_img_url)} URLs from img_url parameters")
        
        # Method 2: Look for image elements and their parent links
        print("Looking for image elements...")
        
        image_selectors = [
            "img.ImagesContentImage-Image",
            ".SerpItem img",
            ".JustifierRowLayout-Item img",
            ".ImagesView-Content img",
            ".serp-item img",
            ".thumb img"
        ]
        
        for selector in image_selectors:
            try:
                img_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                print(f"Found {len(img_elements)} images with selector: {selector}")
                
                for img in img_elements[:max_images]:
                    try:
                        # Try to get parent link with img_url
                        parent_link = img.find_element(By.XPATH, "./ancestor::a[@href]")
                        if parent_link:
                            href = parent_link.get_attribute('href')
                            if 'img_url=' in href:
                                parsed = urlparse(href)
                                params = parse_qs(parsed.query)
                                if 'img_url' in params:
                                    img_url = unquote(params['img_url'][0])
                                    if img_url.startswith('http') and img_url not in urls:
                                        urls.append(img_url)
                    except:
                        # Try getting direct src if no parent link
                        try:
                            src = img.get_attribute('src')
                            if src and src.startswith('http') and 'yandex' not in src and src not in urls:
                                urls.append(src)
                        except:
                            pass
            except:
                continue
        
        # Method 3: Look for data attributes and onclick handlers
        print("Looking for data attributes...")
        
        data_urls = driver.execute_script("""
            var urls = [];
            var elements = document.querySelectorAll('[data-bem], [onclick], [data-url]');
            
            for (var i = 0; i < elements.length; i++) {
                var elem = elements[i];
                
                // Check data attributes
                for (var attr of elem.attributes) {
                    if (attr.value && attr.value.includes('http') && 
                        !attr.value.includes('yandex.net') && 
                        !attr.value.includes('yastatic')) {
                        
                        try {
                            var decoded = decodeURIComponent(attr.value);
                            if (decoded.startsWith('http') && 
                                !decoded.includes('yandex.net') &&
                                !decoded.includes('yastatic')) {
                                urls.push(decoded);
                            }
                        } catch (e) {}
                    }
                }
                
                // Check onclick handlers
                var onclick = elem.getAttribute('onclick') || '';
                var urlMatch = onclick.match(/https?:\/\/[^'"\\s]+/g);
                if (urlMatch) {
                    for (var url of urlMatch) {
                        if (!url.includes('yandex.net') && !url.includes('yastatic')) {
                            urls.push(url);
                        }
                    }
                }
            }
            
            return [...new Set(urls)]; // Remove duplicates
        """)
        
        if data_urls:
            urls.extend(data_urls)
            print(f"Found {len(data_urls)} URLs from data attributes")
        
        # Clean and filter URLs
        cleaned_urls = []
        seen = set()
        
        for url in urls:
            # Clean URL
            url = unquote(url)
            
            # Skip Yandex internal URLs and invalid URLs
            skip_patterns = [
                'yandex.net/i?id=',
                'avatars.mds.yandex',
                'yastatic.net',
                'data:',
                'blob:',
                'javascript:',
                'yandex.com'
            ]
            
            if any(pattern in url for pattern in skip_patterns):
                continue
            
            # Must be valid HTTP URL
            if not url.startswith('http'):
                continue
            
            # Add to list if not seen
            if url not in seen:
                seen.add(url)
                cleaned_urls.append(url)
        
        print(f"Extracted {len(cleaned_urls)} unique image URLs after filtering")
        
        # Debug: print first few URLs
        if cleaned_urls:
            print("Sample URLs found:")
            for i, url in enumerate(cleaned_urls[:5]):
                print(f"  {i+1}. {url[:80]}...")
        else:
            print("No valid URLs extracted. Debugging page structure...")
            # Debug page content
            page_text = driver.execute_script("return document.body.innerText;")[:500]
            print(f"Page content sample: {page_text}")
            
            # Check for any img elements
            all_imgs = driver.find_elements(By.TAG_NAME, "img")
            print(f"Total img elements found: {len(all_imgs)}")
            
            if all_imgs:
                sample_src = all_imgs[0].get_attribute('src') if all_imgs else None
                print(f"Sample img src: {sample_src}")
        
        return cleaned_urls[:max_images]
        
    except Exception as e:
        print(f"Error extracting URLs: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return urls

# -----------------------
# ORB Feature Matching (Same as before)
# -----------------------
def calculate_orb_similarity(img1_path, img2_url, min_matches=8):
    """
    Uses ORB with BFMatcher for image matching
    """
    try:
        # Load local image
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None:
            return 0.0, 0
        
        # Download remote image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://yandex.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }
        
        resp = requests.get(img2_url, timeout=15, headers=headers, allow_redirects=True)
        if resp.status_code != 200:
            return 0.0, 0
        
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img2 is None:
            return 0.0, 0
        
        # Resize if too large
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
        
        # BFMatcher for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Match descriptors
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        num_good_matches = len(good_matches)
        
        # Geometric verification with RANSAC
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
            except:
                pass
        
        # Fallback score
        if num_good_matches > 0:
            similarity_score = num_good_matches / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0
            return min(similarity_score, 1.0), num_good_matches
        
        return 0.0, 0
        
    except Exception as e:
        return 0.0, 0

# -----------------------
# Template Matching
# -----------------------
def calculate_template_matching(img1_path, img2_url):
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
            'Referer': 'https://yandex.com/'
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
            
            if width > w2 or height > h2 or width < 20 or height < 20:
                continue
            
            resized = cv2.resize(img1, (width, height))
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
    Check if images are similar using ORB and template matching
    """
    orb_score, num_matches = calculate_orb_similarity(img1_path, img2_url, min_matches=8)
    template_score = calculate_template_matching(img1_path, img2_url)
    
    # Relaxed thresholds for Yandex
    is_match = (orb_score >= 0.008 and num_matches >= 8) or (template_score >= 0.7)
    
    return is_match, orb_score, template_score, num_matches

# -----------------------
# Process Single Image
# -----------------------
def process_image(driver, image_path):
    """Process a single image through Yandex Image Search"""
    image_name = os.path.basename(image_path)
    print(f"\nProcessing: {image_name}")
    print("-" * 50)
    
    # Upload to Yandex and navigate to similar images
    print("Uploading to Yandex Image Search...")
    success = upload_to_yandex(driver, image_path)
    
    if not success:
        print("Failed to navigate to similar images page")
        return []
    
    # Extract URLs from similar images page
    print("Extracting image URLs from similar images results...")
    urls = get_yandex_image_urls(driver, max_images=50)
    
    if not urls:
        print("No URLs found - please check if page loaded correctly")
        print(f"Current URL: {driver.current_url[:100]}")
        return []
    
    print(f"Found {len(urls)} valid image URLs to check")
    
    # Check similarity
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
def save_results(all_matches, filename="yandex_matched_images.csv"):
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
    OUTPUT_FILE = 'yandex_matched_images.csv'
    BATCH_DELAY = 5
    
    print("=" * 60)
    print("YANDEX IMAGE SEARCH MATCHER")
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