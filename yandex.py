import time
import csv
import cv2
import numpy as np
import requests
import os
import base64
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import argparse
import logging
import shutil
from datetime import datetime
import subprocess
import platform
import urllib3
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import json

# -----------------------
# Enhanced Data Structures
# -----------------------
@dataclass
class MatchResult:
    source_image: str
    matched_url: str
    similarity_score: float
    confidence: float
    method_scores: Dict[str, float]
    feature_matches: int
    geometric_verification: bool
    comparison_path: Optional[str] = None

@dataclass
class FeatureSet:
    keypoints: List
    descriptors: np.ndarray
    method: str

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Disable SSL warnings for speed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class EnhancedImageMatcher:
    """Enhanced image matching with multiple algorithms and fusion"""
    
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.sift = cv2.SIFT_create(nfeatures=2000)  
        self.akaze = cv2.AKAZE_create()
        self.brisk = cv2.BRISK_create()
        
        # FLANN matcher for ORB (LSH)
        self.flann_orb = cv2.FlannBasedMatcher(
            dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
            dict(checks=50)
        )
        
        # FLANN matcher for SIFT (KDTree)
        self.flann_sift = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        
        # BF matcher as fallback
        self.bf_matcher = cv2.BFMatcher()
    
    def extract_features_multi_method(self, image: np.ndarray) -> Dict[str, FeatureSet]:
        """Extract features using multiple methods"""
        features = {}
        
        try:
            # ORB features (binary)
            kp_orb, desc_orb = self.orb.detectAndCompute(image, None)
            if desc_orb is not None and len(desc_orb) > 0:
                features['orb'] = FeatureSet(kp_orb, desc_orb, 'orb')
        except Exception as e:
            logging.debug(f"ORB extraction failed: {e}")
        
        try:
            # SIFT features (float)
            kp_sift, desc_sift = self.sift.detectAndCompute(image, None)
            if desc_sift is not None and len(desc_sift) > 0:
                features['sift'] = FeatureSet(kp_sift, desc_sift, 'sift')
        except Exception as e:
            logging.debug(f"SIFT extraction failed: {e}")
            
        try:
            # AKAZE features (binary)
            kp_akaze, desc_akaze = self.akaze.detectAndCompute(image, None)
            if desc_akaze is not None and len(desc_akaze) > 0:
                features['akaze'] = FeatureSet(kp_akaze, desc_akaze, 'akaze')
        except Exception as e:
            logging.debug(f"AKAZE extraction failed: {e}")
            
        try:
            # BRISK features (binary)  
            kp_brisk, desc_brisk = self.brisk.detectAndCompute(image, None)
            if desc_brisk is not None and len(desc_brisk) > 0:
                features['brisk'] = FeatureSet(kp_brisk, desc_brisk, 'brisk')
        except Exception as e:
            logging.debug(f"BRISK extraction failed: {e}")
        
        return features
    
    def match_features(self, features1: FeatureSet, features2: FeatureSet) -> Tuple[List, float, int]:
        """Match features between two feature sets"""
        try:
            if features1.method != features2.method:
                return [], 0.0, 0
                
            method = features1.method
            desc1, desc2 = features1.descriptors, features2.descriptors
            
            if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
                return [], 0.0, 0
            
            # Choose appropriate matcher
            if method == 'sift':
                matches = self.flann_sift.knnMatch(desc1, desc2, k=2)
            elif method in ['orb', 'akaze', 'brisk']:
                # For binary descriptors, use LSH or BF matcher
                try:
                    matches = self.flann_orb.knnMatch(desc1, desc2, k=2)
                except:
                    # Fallback to brute force
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                    matches = bf.knnMatch(desc1, desc2, k=2)
            else:
                return [], 0.0, 0
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    ratio_threshold = 0.75 if method == 'sift' else 0.8
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # Geometric verification with homography
            num_inliers = 0
            if len(good_matches) >= 8:
                src_pts = np.float32([features1.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([features2.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                try:
                    M, mask = cv2.findHomography(src_pts, dst_pts, 
                                               method=cv2.RANSAC, 
                                               ransacReprojThreshold=5.0,
                                               confidence=0.99)
                    if mask is not None:
                        num_inliers = int(np.sum(mask))
                except:
                    pass
            
            # Calculate similarity score
            if len(good_matches) > 0:
                similarity = min(num_inliers / len(good_matches), 1.0) if num_inliers > 0 else 0.0
                # Boost score based on absolute number of inliers
                inlier_boost = min(num_inliers / 50.0, 1.0)
                final_score = (similarity * 0.7 + inlier_boost * 0.3)
            else:
                final_score = 0.0
            
            return good_matches, final_score, num_inliers
            
        except Exception as e:
            logging.debug(f"Matching failed for {method}: {e}")
            return [], 0.0, 0
    
    def calculate_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate histogram-based similarity"""
        try:
            # Convert to RGB for better color comparison
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB) if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB) if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            # Calculate histograms
            hist1 = cv2.calcHist([img1_rgb], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2_rgb], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            
            # Normalize
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Compare using correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(correlation, 0.0)
            
        except Exception as e:
            logging.debug(f"Histogram similarity calculation failed: {e}")
            return 0.0
    
    def calculate_structural_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM-like structural similarity"""
        try:
            # Resize to same dimensions for comparison
            target_size = (256, 256)
            img1_resized = cv2.resize(img1, target_size)
            img2_resized = cv2.resize(img2, target_size)
            
            # Convert to float
            img1_f = img1_resized.astype(np.float64)
            img2_f = img2_resized.astype(np.float64)
            
            # Calculate means
            mu1 = np.mean(img1_f)
            mu2 = np.mean(img2_f)
            
            # Calculate variances and covariance
            var1 = np.var(img1_f)
            var2 = np.var(img2_f)
            cov = np.mean((img1_f - mu1) * (img2_f - mu2))
            
            # SSIM formula components
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
            
            return max(ssim, 0.0)
            
        except Exception as e:
            logging.debug(f"Structural similarity calculation failed: {e}")
            return 0.0
    
    def calculate_enhanced_similarity(self, img1_path: str, img2_url: str, timeout=4) -> MatchResult:
        """Calculate enhanced similarity using multiple methods and fusion"""
        try:
            # Load source image
            img1 = cv2.imread(img1_path)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            
            # Download comparison image
            img2 = self.get_image_from_url_or_base64(img2_url, timeout)
            if img2 is None:
                return None
            
            # Convert to color for histogram comparison
            img2_color = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            img1_color = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) if len(img2_color.shape) == 3 else img2_color
            
            # Extract features using multiple methods
            features1 = self.extract_features_multi_method(img1_gray)
            features2 = self.extract_features_multi_method(img2_gray)
            
            # Calculate feature-based similarities
            method_scores = {}
            best_geometric_verification = False
            total_matches = 0
            
            for method in ['orb', 'sift', 'akaze', 'brisk']:
                if method in features1 and method in features2:
                    good_matches, score, inliers = self.match_features(features1[method], features2[method])
                    method_scores[f'{method}_score'] = score
                    method_scores[f'{method}_matches'] = len(good_matches)
                    method_scores[f'{method}_inliers'] = inliers
                    total_matches += len(good_matches)
                    
                    if inliers >= 8:  # Sufficient for geometric verification
                        best_geometric_verification = True
            
            # Calculate histogram similarity
            hist_sim = self.calculate_histogram_similarity(img1_color, img2_color)
            method_scores['histogram_similarity'] = hist_sim
            
            # Calculate structural similarity
            struct_sim = self.calculate_structural_similarity(img1_gray, img2_gray)
            method_scores['structural_similarity'] = struct_sim
            
            # Template matching for additional verification
            template_score = self.calculate_template_similarity(img1_gray, img2_gray)
            method_scores['template_similarity'] = template_score
            
            # Multi-method fusion scoring
            final_score, confidence = self.fuse_similarity_scores(method_scores, best_geometric_verification)
            
            return MatchResult(
                source_image=Path(img1_path).name,
                matched_url=img2_url,
                similarity_score=final_score,
                confidence=confidence,
                method_scores=method_scores,
                feature_matches=total_matches,
                geometric_verification=best_geometric_verification
            )
            
        except Exception as e:
            logging.debug(f"Enhanced similarity calculation failed: {e}")
            return None
    
    def calculate_template_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate template matching similarity"""
        try:
            # Resize images to same scale for template matching
            target_size = (200, 200)
            img1_resized = cv2.resize(img1, target_size)
            img2_resized = cv2.resize(img2, target_size)
            
            # Template match in both directions
            result1 = cv2.matchTemplate(img1_resized, img2_resized, cv2.TM_CCOEFF_NORMED)
            result2 = cv2.matchTemplate(img2_resized, img1_resized, cv2.TM_CCOEFF_NORMED)
            
            max_score1 = np.max(result1)
            max_score2 = np.max(result2)
            
            return max(max_score1, max_score2, 0.0)
            
        except Exception:
            return 0.0
    
    def fuse_similarity_scores(self, method_scores: Dict[str, float], geometric_verified: bool) -> Tuple[float, float]:
        """Fuse multiple similarity scores with confidence estimation"""
        
        # Extract individual method scores
        feature_scores = []
        for method in ['orb', 'sift', 'akaze', 'brisk']:
            score_key = f'{method}_score'
            if score_key in method_scores and method_scores[score_key] > 0:
                feature_scores.append(method_scores[score_key])
        
        hist_sim = method_scores.get('histogram_similarity', 0.0)
        struct_sim = method_scores.get('structural_similarity', 0.0)
        template_sim = method_scores.get('template_similarity', 0.0)
        
        # Weighted fusion
        weights = {
            'feature_avg': 0.4,
            'histogram': 0.2,
            'structural': 0.2,
            'template': 0.2
        }
        
        # Calculate weighted average of feature methods
        feature_avg = np.mean(feature_scores) if feature_scores else 0.0
        
        # Geometric verification bonus
        geometric_bonus = 0.1 if geometric_verified else 0.0
        
        # Final fused score
        final_score = (
            weights['feature_avg'] * feature_avg +
            weights['histogram'] * hist_sim +
            weights['structural'] * struct_sim +
            weights['template'] * template_sim +
            geometric_bonus
        )
        
        # Confidence calculation
        num_active_methods = len([s for s in [feature_avg, hist_sim, struct_sim, template_sim] if s > 0])
        agreement_score = 1.0 - np.std([s for s in [feature_avg, hist_sim, struct_sim, template_sim] if s > 0]) if num_active_methods > 1 else 0.5
        
        confidence = min(
            (num_active_methods / 4.0) * 0.6 + 
            agreement_score * 0.3 +
            (0.1 if geometric_verified else 0.0),
            1.0
        )
        
        return min(final_score, 1.0), confidence
    
    def get_image_from_url_or_base64(self, img_url: str, timeout=4) -> Optional[np.ndarray]:
        """Get image from URL or base64 string - enhanced error handling"""
        try:
            if img_url.startswith("data:image"):
                header, b64data = img_url.split(',', 1)
                img_bytes = base64.b64decode(b64data)
                img_np = np.frombuffer(img_bytes, np.uint8)
                img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                return img_cv
            else:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'https://yandex.com/',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
                }
                resp = requests.get(img_url, timeout=timeout, headers=headers, verify=False)
                if resp.status_code != 200:
                    return None
                
                img_np = np.frombuffer(resp.content, np.uint8)
                img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                return img_cv
        except Exception as e:
            logging.debug(f"Failed to get image from {img_url}: {e}")
            return None

# -----------------------
# Enhanced Utility Functions  
# -----------------------

def create_enhanced_comparison_image(source_path, similar_img_bytes, match_result: MatchResult, output_folder='similar_images'):
    """Create enhanced side-by-side comparison with detailed metrics"""
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

        # Resize images
        target_height = 400
        original_h, original_w = original_img.shape[:2]
        similar_h, similar_w = similar_img.shape[:2]

        original_ratio = target_height / original_h
        original_resized = cv2.resize(original_img, (int(original_w * original_ratio), target_height))

        similar_ratio = target_height / similar_h
        similar_resized = cv2.resize(similar_img, (int(similar_w * similar_ratio), target_height))

        # Create enhanced layout with metrics panel
        text_area_height = 150
        total_width = original_resized.shape[1] + similar_resized.shape[1]
        total_height = target_height + text_area_height

        stitched_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)
        stitched_image[text_area_height:total_height, :original_resized.shape[1]] = original_resized
        stitched_image[text_area_height:total_height, original_resized.shape[1]:] = similar_resized

        # Enhanced text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)
        thickness = 1
        line_spacing = 25

        # Title
        title = f"Match Analysis | Score: {match_result.similarity_score:.3f} | Confidence: {match_result.confidence:.3f}"
        title_size = cv2.getTextSize(title, font, 0.8, 2)[0]
        title_x = (total_width - title_size[0]) // 2
        cv2.putText(stitched_image, title, (title_x, 25), font, 0.8, (0, 0, 150), 2, cv2.LINE_AA)

        # Method scores
        y_pos = 50
        method_texts = []
        
        for method in ['orb', 'sift', 'akaze', 'brisk']:
            score_key = f'{method}_score'
            matches_key = f'{method}_matches'
            if score_key in match_result.method_scores:
                score = match_result.method_scores[score_key]
                matches = match_result.method_scores.get(matches_key, 0)
                method_texts.append(f"{method.upper()}: {score:.3f} ({matches} matches)")
        
        # Additional metrics
        if 'histogram_similarity' in match_result.method_scores:
            method_texts.append(f"Histogram: {match_result.method_scores['histogram_similarity']:.3f}")
        if 'structural_similarity' in match_result.method_scores:
            method_texts.append(f"Structure: {match_result.method_scores['structural_similarity']:.3f}")
        if 'template_similarity' in match_result.method_scores:
            method_texts.append(f"Template: {match_result.method_scores['template_similarity']:.3f}")
        
        # Draw method scores in columns
        col_width = total_width // 3
        for i, text in enumerate(method_texts):
            x_pos = 10 + (i % 3) * col_width
            y_offset = y_pos + (i // 3) * line_spacing
            cv2.putText(stitched_image, text, (x_pos, y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Geometric verification indicator
        geo_text = f"Geometric Verification: {'PASSED' if match_result.geometric_verification else 'FAILED'}"
        geo_color = (0, 150, 0) if match_result.geometric_verification else (0, 0, 150)
        cv2.putText(stitched_image, geo_text, (10, y_pos + 75), font, font_scale, geo_color, thickness, cv2.LINE_AA)

        # Save the enhanced comparison image
        source_name = Path(source_path).stem
        save_folder = Path(output_folder)
        os.makedirs(save_folder, exist_ok=True)
        filename = f"{source_name}_enhanced_{match_result.similarity_score:.3f}.jpg"
        filepath = save_folder / filename

        if filepath.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = save_folder / f"{source_name}_enhanced_{match_result.similarity_score:.3f}_{timestamp}.jpg"

        cv2.imwrite(str(filepath), stitched_image)
        logging.info(f"Saved enhanced comparison: {filepath.name}")
        return str(filepath)
        
    except Exception as e:
        logging.error(f"Failed to create enhanced comparison image: {e}")
        return None

def process_image_enhanced(driver, image_path, matcher: EnhancedImageMatcher, max_urls=20, max_workers=8, 
                          log_writer=None, links_writer=None, threshold=0.25, confidence_threshold=0.3):
    """Enhanced image processing with multi-method matching"""
    image_name = Path(image_path).name
    start_time = time.time()
    logging.info(f"Processing with enhanced matching: {image_name}")

    # Upload to Yandex (using existing function)
    if not upload_to_yandex_and_navigate(driver, image_path):
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, 0.0, 0.0, "Upload failed", "{}"])
        return None

    # Get image URLs
    urls = get_yandex_image_urls(driver, max_images=max_urls)
    if not urls:
        if log_writer:
            log_writer.writerow([image_name, f"{time.time() - start_time:.2f}", 0, 0.0, 0.0, "No URLs found", "{}"])
        return None

    # Enhanced matching
    best_match = None
    best_score = 0.0
    
    def check_url_enhanced(url):
        result = matcher.calculate_enhanced_similarity(image_path, url)
        if links_writer and result:
            links_writer.writerow([image_name, url, f"{result.similarity_score:.4f}", f"{result.confidence:.4f}"])
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_url_enhanced, url): url for url in urls}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result.similarity_score > best_score and result.confidence >= confidence_threshold:
                    if result.similarity_score >= threshold:
                        best_score = result.similarity_score
                        best_match = result
            except Exception as e:
                logging.debug(f"URL check failed: {e}")
                continue

    processing_time = time.time() - start_time

    if best_match and best_match.similarity_score >= threshold:
        logging.info(f"Enhanced match for {image_name}: score {best_match.similarity_score:.4f}, confidence {best_match.confidence:.4f}")

        # Create enhanced comparison image
        if best_match.matched_url.startswith("data:image"):
            header, base64_data = best_match.matched_url.split(',', 1)
            img_bytes = base64.b64decode(base64_data)
        else:
            try:
                headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://yandex.com/'}
                resp = requests.get(best_match.matched_url, timeout=4, headers=headers, verify=False)
                img_bytes = resp.content if resp.status_code == 200 else None
            except:
                img_bytes = None

        if img_bytes:
            comparison_path = create_enhanced_comparison_image(
                source_path=image_path,
                similar_img_bytes=img_bytes,
                match_result=best_match
            )
            best_match.comparison_path = comparison_path

            # Move to done folder
            move_image_to_done(image_path)

        if log_writer:
            methods_json = json.dumps(best_match.method_scores, indent=None)
            log_writer.writerow([
                image_name, 
                f"{processing_time:.2f}", 
                1, 
                f"{best_match.similarity_score:.4f}",
                f"{best_match.confidence:.4f}",
                "Enhanced match found", 
                methods_json
            ])

        return best_match
    else:
        logging.info(f"No enhanced match found for {image_name} above threshold.")
        if log_writer:
            log_writer.writerow([image_name, f"{processing_time:.2f}", 0, 0.0, 0.0, "No match above threshold", "{}"])
        return None

def save_enhanced_results(all_matches: List[MatchResult], filename="enhanced_yandex_matches.csv"):
    """Save enhanced results with detailed metrics"""
    if not all_matches:
        logging.info("No enhanced matches to save.")
        return

    valid_matches = [m for m in all_matches if m is not None]

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Enhanced header
        writer.writerow([
            'Source Image', 'Matched URL', 'Final Score', 'Confidence', 
            'Feature Matches', 'Geometric Verification',
            'ORB Score', 'SIFT Score', 'AKAZE Score', 'BRISK Score',
            'Histogram Sim', 'Structural Sim', 'Template Sim',
            'Comparison Image Path'
        ])
        
        for match in valid_matches:
            writer.writerow([
                match.source_image,
                match.matched_url,
                f"{match.similarity_score:.4f}",
                f"{match.confidence:.4f}",
                match.feature_matches,
                match.geometric_verification,
                f"{match.method_scores.get('orb_score', 0.0):.4f}",
                f"{match.method_scores.get('sift_score', 0.0):.4f}",
                f"{match.method_scores.get('akaze_score', 0.0):.4f}",
                f"{match.method_scores.get('brisk_score', 0.0):.4f}",
                f"{match.method_scores.get('histogram_similarity', 0.0):.4f}",
                f"{match.method_scores.get('structural_similarity', 0.0):.4f}",
                f"{match.method_scores.get('template_similarity', 0.0):.4f}",
                match.comparison_path or 'N/A'
            ])

    logging.info(f"Enhanced results for {len(valid_matches)} matches saved to '{filename}'.")

# -----------------------
# Main Enhanced Function
# -----------------------

def main_enhanced():
    parser = argparse.ArgumentParser(description="Enhanced Yandex Visual Search & Multi-Method Matcher")
    parser.add_argument('-i', '--images_folder', type=str, default='images', help='Folder containing source images.')
    parser.add_argument('-o', '--output_file', type=str, default='enhanced_yandex_matches.csv', help='Output CSV file for matches.')
    parser.add_argument('-u', '--max_urls', type=int, default=30, help='Max Yandex URLs to check per image.')
    parser.add_argument('-w', '--max_workers', type=int, default=8, help='Max concurrent workers.')
    parser.add_argument('-d', '--delay', type=float, default=1.5, help='Delay between processing images.')
    parser.add_argument('-t', '--threshold', type=float, default=0.25, help='Minimum similarity score threshold.')
    parser.add_argument('-c', '--confidence', type=float, default=0.3, help='Minimum confidence threshold.')

    args = parser.parse_args()

    logging.info("--- Starting Enhanced Yandex Matcher ---")
    create_folders()

    # Initialize enhanced matcher
    matcher = EnhancedImageMatcher()

    # Create enhanced log files
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"enhanced_processing_log_{run_timestamp}.csv"
    links_filename = f"enhanced_urls_log_{run_timestamp}.csv"

    # Get image files
    image_files = get_image_files(args.images_folder)
    if not image_files:
        logging.warning(f"No images found in '{args.images_folder}'. Exiting.")
        return

    # Initialize driver
    driver = init_driver()
    if not driver:
        logging.critical("Failed to initialize web driver.")
        return

    all_matches = []

    try:
        with open(log_filename, 'w', newline='', encoding='utf-8') as log_file, \
             open(links_filename, 'w', newline='', encoding='utf-8') as links_file:

            log_writer = csv.writer(log_file)
            log_writer.writerow(['Image Name', 'Processing Time (s)', 'Match Found', 'Final Score', 'Confidence', 'Notes', 'Method Scores'])

            links_writer = csv.writer(links_file)
            links_writer.writerow(['Source Image', 'Scraped URL', 'Similarity Score', 'Confidence'])

            total_images = len(image_files)

            for idx, image_path in enumerate(image_files, 1):
                logging.info(f"--- [Enhanced Processing {idx}/{total_images}] ---")

                match_result = process_image_enhanced(
                    driver, image_path, matcher,
                    max_urls=args.max_urls,
                    max_workers=args.max_workers,
                    log_writer=log_writer,
                    links_writer=links_writer,
                    threshold=args.threshold,
                    confidence_threshold=args.confidence
                )

                if match_result:
                    all_matches.append(match_result)

                if idx < total_images:
                    logging.info(f"Waiting {args.delay} seconds before next image...")
                    time.sleep(args.delay)

        # Save enhanced results
        save_enhanced_results(all_matches, args.output_file)

        logging.info(f"--- Enhanced Processing Complete ---")
        logging.info(f"Processed {total_images} images. Found {len(all_matches)} high-quality matches.")

    except KeyboardInterrupt:
        logging.warning("Process interrupted by user.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        
        # Cleanup
        user_data_dir = _tmp_profile_dir()
        if os.path.exists(user_data_dir):
            shutil.rmtree(user_data_dir, ignore_errors=True)

        logging.info("Enhanced processing complete and cleanup finished.")

# -----------------------
# Keep original functions for backward compatibility
# -----------------------

def _tmp_profile_dir():
    pid = os.getpid()
    sysname = platform.system()
    if sysname == "Windows":
        return os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", f"chrome_data_{pid}")
    elif sysname == "Darwin":
        return f"/tmp/chrome_data_{pid}"
    else:
        return f"/tmp/chrome_data_{pid}"

def kill_chrome_processes():
    """Kill existing Chrome processes to avoid conflicts"""
    sysname = platform.system()
    if sysname in ["Darwin", "Linux"]:
        for process_name in ["Google Chrome", "Chromium", "chrome"]:
            try:
                subprocess.run(["pkill", "-f", process_name], check=False, capture_output=True)
            except FileNotFoundError:
                pass
    elif sysname == "Windows":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], check=False, capture_output=True)
        except FileNotFoundError:
            pass
    time.sleep(0.5)

def init_driver():
    """Initialize undetected Chrome driver"""
    kill_chrome_processes()
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless=new")  # Uncomment for headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--ignore-certificate-errors")
    user_data_dir = _tmp_profile_dir()
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument("--lang=en-US,en;q=0.9")

    try:
        logging.info("Initializing Chrome driver...")
        driver = uc.Chrome(options=options, version_main=None)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        logging.error(f"Failed to initialize Chrome driver: {e}")
        driver = uc.Chrome(options=options, use_subprocess=True)
        return driver

def create_folders():
    """Create required folders"""
    folders = ['done', 'similar_images']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def move_image_to_done(image_path):
    """Move processed image to done folder"""
    try:
        done_folder = Path('done')
        source = Path(image_path)
        destination = done_folder / source.name

        if destination.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = done_folder / f"{source.stem}_{timestamp}{source.suffix}"

        shutil.move(str(source), str(destination))
        logging.info(f"Moved '{source.name}' to 'done' folder.")
    except Exception as e:
        logging.error(f"Failed to move '{image_path}': {e}")

def get_image_files(folder_path):
    """Get all image files from folder"""
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

def _extract_urls_from_dom(driver):
    """Harvest candidate image URLs currently present in DOM."""
    urls = set()

    # Primary: tile anchors with img_url param
    anchors = driver.find_elements(By.CSS_SELECTOR, "a.ImagesContentImage-Cover[href*='img_url='], a.Link.ImagesContentImage-Cover[href*='img_url=']")
    for a in anchors:
        href = a.get_attribute("href") or ""
        if not href:
            continue
        try:
            qs = parse_qs(urlsplit(href).query)
            if "img_url" in qs and qs["img_url"]:
                u = unquote(qs["img_url"][0])
                if u.startswith("http"):
                    urls.add(u)
            elif "url" in qs and qs["url"]:
                u = unquote(qs["url"][0])
                if u.startswith("http") and ("yandex" not in u and "yastatic" not in u and "avatars.mds.yandex.net" not in u):
                    urls.add(u)
        except Exception:
            continue

    # Secondary: any hrefs containing img_url=
    links = driver.find_elements(By.CSS_SELECTOR, "a[href*='img_url=']")
    for link in links:
        href = link.get_attribute("href") or ""
        try:
            qs = parse_qs(urlsplit(href).query)
            if "img_url" in qs and qs["img_url"]:
                u = unquote(qs["img_url"][0])
                if u.startswith("http"):
                    urls.add(u)
        except Exception:
            continue

    # Tertiary: JSON in data-bem
    elements = driver.find_elements(By.CSS_SELECTOR, "[data-bem*='http']")
    url_pattern = r'https?://[^"\'\\s,}]+'
    for elem in elements:
        data_bem = elem.get_attribute("data-bem")
        if not data_bem:
            continue
        for u in re.findall(url_pattern, data_bem):
            if u.startswith("http") and ("yandex" not in u and "yastatic" not in u and "avatars.mds.yandex.net" not in u):
                urls.add(u)

    return urls

def get_yandex_image_urls(driver, max_images=20, max_wait_s=20, per_scroll_wait=0.8):
    """Extract image URLs from Yandex 'Similar' results."""
    urls = set()

    wait = WebDriverWait(driver, 15)
    try:
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='list'] .JustifierRowLayout-Item, .SerpItem-Thumb, .SerpItem"))
        )
    except TimeoutException:
        logging.warning("Similar results grid not detected.")
        return []

    start = time.time()
    last_count = -1
    stagnation_rounds = 0

    while len(urls) < max_images and (time.time() - start) < max_wait_s:
        new_urls = _extract_urls_from_dom(driver)
        urls |= new_urls

        if len(urls) >= max_images:
            break

        items = driver.find_elements(By.CSS_SELECTOR, "div[role='list'] .JustifierRowLayout-Item, .SerpItem-Thumb, .SerpItem")
        if items:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block:'end'});", items[-1])
            except Exception:
                driver.execute_script("window.scrollBy(0, Math.max(800, window.innerHeight));")
        else:
            driver.execute_script("window.scrollBy(0, Math.max(800, window.innerHeight));")

        time.sleep(per_scroll_wait)

        if len(urls) == last_count:
            stagnation_rounds += 1
            if stagnation_rounds >= 3:
                break
        else:
            stagnation_rounds = 0
            last_count = len(urls)

    logging.info(f"Extracted {len(urls)} image URLs from Yandex.")
    return list(urls)[:max_images]

def upload_to_yandex_and_navigate(driver, image_path, timeout=30):
    """Upload image to Yandex and navigate to Similar images"""
    try:
        driver.get("https://yandex.com/images/")
        wait = WebDriverWait(driver, timeout)

        # Accept consent if shown
        try:
            consent_btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label*='Accept'], button[aria-label*='agree'], .Button2"))
            )
            consent_btn.click()
            time.sleep(0.5)
        except Exception:
            pass

        # Find file input
        file_input_selectors = [
            "input[type='file']",
            "input[accept*='image']",
            ".CbirSearch-FileInput input",
            ".CbirPanel-FileInput input",
            ".CbirSearchForm-FileInput input",
        ]
        file_input = None
        for selector in file_input_selectors:
            try:
                file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                if file_input:
                    break
            except TimeoutException:
                continue

        if not file_input:
            logging.error("Could not find file input on Yandex Images.")
            return False

        # Upload
        abs_path = os.path.abspath(image_path)
        driver.execute_script("""
            const el = arguments[0];
            el.style.display = 'block';
            el.style.visibility = 'visible';
            el.style.opacity = '1';
            el.style.position = 'static';
            el.removeAttribute('hidden');
        """, file_input)
        file_input.send_keys(abs_path)
        logging.info(f"Uploading: {os.path.basename(image_path)}")

        # Wait for results
        try:
            wait.until(EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[data-cbir-page-type='similar']")),
                EC.presence_of_element_located((By.CSS_SELECTOR, ".CbirNavigation-TabsItem")),
                EC.url_contains("cbir_id")
            ))
        except TimeoutException:
            logging.warning("Results page didn't load properly.")
            return False

        # Click Similar tab
        similar_tab_selectors = [
            "a[data-cbir-page-type='similar']",
            "a.CbirNavigation-TabsItem_name_similar-page",
            "a[href*='cbir_page=similar']",
        ]
        similar_tab = None
        for sel in similar_tab_selectors:
            try:
                similar_tab = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
                )
                break
            except TimeoutException:
                continue

        if not similar_tab:
            try:
                similar_tab = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(., 'Similar') or contains(., 'Похожие') or contains(., 'похожие')]"))
                )
            except TimeoutException:
                logging.error("Could not find 'Similar' tab.")
                return False

        driver.execute_script("arguments[0].scrollIntoView(true);", similar_tab)
        time.sleep(0.3)
        try:
            similar_tab.click()
        except Exception:
            driver.execute_script("arguments[0].click();", similar_tab)

        # Wait for similar grid
        try:
            wait.until(EC.any_of(
                EC.url_contains("cbir_page=similar"),
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='list'] .JustifierRowLayout-Item, .SerpItem-Thumb, .SerpItem"))
            ))
        except TimeoutException:
            logging.warning("Similar images grid not loaded in time.")
            return False

        logging.info("Successfully navigated to Similar images.")
        return True

    except Exception as e:
        logging.error(f"Upload to Yandex failed for {image_path}: {e}")
        return False

if __name__ == "__main__":
    main_enhanced()