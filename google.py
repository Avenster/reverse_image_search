import time
import csv
import cv2
import numpy as np
import requests
import os
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
import glob

# -----------------------
# Browser Setup
# -----------------------
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--user-data-dir=/tmp/chrome-user-data")
    options.add_argument("--profile-directory=Default")
    driver = uc.Chrome(options=options)
    return driver

# -----------------------
# Upload and Extract URLs
# -----------------------
def upload_image(driver, image_path):
    abs_path = os.path.abspath(image_path)
    driver.get("https://lens.google.com/upload")
    upload_input = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
    )
    upload_input.send_keys(abs_path)
    time.sleep(8)  # Wait for results

def get_image_urls(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    img_tags = soup.find_all("img")
    urls = []
    for img in img_tags:
        src = img.get("src") or img.get("data-src")
        if src and src.startswith("http"):
            urls.append(src)
    return list(set(urls))

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
# PROVEN CV TECHNIQUE: ORB + FLANN Matching
# This is one of the most reliable methods for image matching
# -----------------------
def calculate_orb_similarity(img1_path, img2_url, min_matches=10):
    """
    Uses ORB (Oriented FAST and Rotated BRIEF) with FLANN matcher.
    ORB is rotation invariant, scale invariant, and very fast.
    """
    try:
        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        
        # Download and load second image
        resp = requests.get(img2_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0.0, 0
        
        # Create ORB detector with increased features
        orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0.0, 0
        
        # FLANN matcher for ORB (using LSH)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=12,
                           key_size=20,
                           multi_probe_level=2)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match descriptors
        if len(des1) < 2 or len(des2) < 2:
            return 0.0, 0
            
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Calculate similarity score
        num_good_matches = len(good_matches)
        
        # Geometric verification using homography
        if num_good_matches >= min_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                inliers = mask.ravel().tolist()
                num_inliers = sum(inliers)
                
                # Score based on inliers and total matches
                similarity_score = num_inliers / max(len(kp1), len(kp2))
                return similarity_score, num_inliers
        
        # If not enough matches for homography
        similarity_score = num_good_matches / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        return similarity_score, num_good_matches
        
    except Exception as e:
        print(f"Error in ORB matching: {str(e)}")
        return 0.0, 0

# -----------------------
# SECONDARY TECHNIQUE: Template Matching (for exact/near-exact matches)
# -----------------------
def calculate_template_matching(img1_path, img2_url):
    """
    Template matching for detecting exact or near-exact copies.
    Very effective for finding the same image at different scales.
    """
    try:
        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        
        resp = requests.get(img2_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        img_array = np.frombuffer(resp.content, np.uint8)
        img2 = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize template to multiple scales
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        # Try different scales
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        max_score = 0
        
        for scale in scales:
            # Resize template
            width = int(w1 * scale)
            height = int(h1 * scale)
            
            # Skip if template would be larger than search image
            if width > w2 or height > h2:
                continue
                
            resized = cv2.resize(img1, (width, height))
            
            # Apply template matching
            result = cv2.matchTemplate(img2, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            max_score = max(max_score, max_val)
        
        return max_score
        
    except Exception as e:
        print(f"Error in template matching: {str(e)}")
        return 0.0

# -----------------------
# Combined Similarity Check
# -----------------------
def is_similar_image(img1_path, img2_url, orb_threshold=0.01, template_threshold=0.8):
    """
    Combines ORB and template matching for robust similarity detection.
    Returns True if either method indicates a match.
    """
    # ORB matching (primary method)
    orb_score, num_matches = calculate_orb_similarity(img1_path, img2_url)
    
    # Template matching (for exact copies)
    template_score = calculate_template_matching(img1_path, img2_url)
    
    # Consider it a match if:
    # 1. ORB finds enough good matches with geometric verification
    # 2. OR template matching finds high correlation
    is_match = (orb_score >= orb_threshold and num_matches >= 10) or (template_score >= template_threshold)
    
    return is_match, orb_score, template_score, num_matches

# -----------------------
# Process Single Image
# -----------------------
def process_image(driver, image_path):
    """Process a single image and find similar matches"""
    image_name = os.path.basename(image_path)
    print(f"\nProcessing: {image_name}")
    print("-" * 50)
    
    # Upload to Google Lens
    print("Uploading to Google Lens...")
    upload_image(driver, image_path)
    
    # Get URLs
    print("Extracting image URLs...")
    urls = get_image_urls(driver)
    print(f"Found {len(urls)} images to check")
    
    # Check similarity
    matched_urls = []
    print("\nChecking for matches...")
    
    for i, url in enumerate(urls, 1):
        try:
            is_match, orb_score, template_score, num_matches = is_similar_image(image_path, url)
            
            if is_match:
                print(f"  ✓ MATCH FOUND [{i}/{len(urls)}]")
                print(f"    ORB Score: {orb_score:.4f} ({num_matches} matches)")
                print(f"    Template Score: {template_score:.4f}")
                print(f"    URL: {url[:80]}...")
                
                matched_urls.append({
                    'source_image': image_name,
                    'matched_url': url,
                    'orb_score': orb_score,
                    'template_score': template_score,
                    'num_matches': num_matches
                })
            else:
                print(f"  ✗ No match [{i}/{len(urls)}]", end='\r')
                
        except Exception as e:
            print(f"  ⚠ Error checking URL [{i}/{len(urls)}]: {str(e)}")
            continue
    
    print(f"\nFound {len(matched_urls)} matching images")
    return matched_urls

# -----------------------
# Save Results
# -----------------------
def save_results(all_matches, filename="matched_images.csv"):
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
    
    print(f"\n✓ Results saved to {filename}")

# -----------------------
# Main Function
# -----------------------
def main():
    # Configuration
    IMAGES_FOLDER = 'images'
    OUTPUT_FILE = 'matched_images.csv'
    BATCH_DELAY = 5  # Delay between images to avoid rate limiting
    
    print("=" * 60)
    print("GOOGLE LENS IMAGE MATCHER")
    print("Using ORB feature matching + Template matching")
    print("=" * 60)
    
    # Get images
    print(f"\nScanning folder: {IMAGES_FOLDER}")
    image_files = get_image_files(IMAGES_FOLDER)
    
    if not image_files:
        print(f"No images found in '{IMAGES_FOLDER}' folder!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Initialize browser
    print("\nInitializing browser...")
    driver = init_driver()
    
    all_matches = []
    
    try:
        # Process each image
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] " + "=" * 50)
            
            matches = process_image(driver, image_path)
            all_matches.extend(matches)
            
            # Delay between images
            if idx < len(image_files):
                print(f"\nWaiting {BATCH_DELAY} seconds...")
                time.sleep(BATCH_DELAY)
        
        # Save results
        print("\n" + "=" * 60)
        save_results(all_matches, OUTPUT_FILE)
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"  • Images processed: {len(image_files)}")
        print(f"  • Total matches found: {len(all_matches)}")
        
        # Show top matches by source
        from collections import defaultdict
        matches_by_source = defaultdict(list)
        for match in all_matches:
            matches_by_source[match['source_image']].append(match)
        
        print(f"\nMatches per image:")
        for source, matches in matches_by_source.items():
            print(f"  • {source}: {len(matches)} matches")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("\nClosing browser...")
        driver.quit()
        print("Done!")

if __name__ == "__main__":
    main()