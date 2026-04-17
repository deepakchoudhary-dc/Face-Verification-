"""
DOPPELGANGER DETECTION MODULE
=============================
Addresses the ancient saying: "There are 7 people in this world with similar faces"

This module calculates the probability that two faces belong to DIFFERENT people
who happen to look similar (doppelgangers/lookalikes) vs. the SAME person.

Techniques:
1. Micro-Expression Analysis - Subtle facial quirks unique to each person
2. Facial Asymmetry Analysis - Everyone has unique asymmetry patterns
3. Ear Shape Analysis - Ears are as unique as fingerprints
4. Birthmark/Mole Detection - Unique facial markers
5. Skin Texture Fingerprinting - Pore patterns are unique
6. Vein Pattern Analysis - Facial vein mapping

This is CRITICAL for:
- High-security KYC where lookalikes might try to impersonate
- Preventing identity fraud by family members
- Twin verification (identical twins have different fingerprints and ear shapes)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
import math


class DoppelgangerDetector:
    """
    Advanced system to differentiate between true identity match
    and doppelganger (lookalike) scenarios.
    """
    
    def __init__(self):
        self.DOPPELGANGER_THRESHOLD = 0.70  # 70% similarity without unique markers = likely doppelganger
        
    def analyze_identity_uniqueness(self, image: np.ndarray, face_box: Dict, 
                                     landmarks: Dict = None) -> Dict[str, Any]:
        """
        Extract unique identity markers that differentiate even lookalikes.
        """
        result = {
            'uniqueness_score': 0.0,
            'unique_markers': [],
            'asymmetry_signature': None,
            'micro_features': {},
            'anti_doppelganger_confidence': 0.0,
            'phenotypic_signature': {}
        }
        
        if image is None:
            return result
            
        try:
            # 1. FACIAL ASYMMETRY ANALYSIS
            asymmetry = self._analyze_facial_asymmetry(image, face_box, landmarks)
            result['asymmetry_signature'] = asymmetry
            
            # 2. MOLE/BIRTHMARK DETECTION
            moles = self._detect_facial_moles(image, face_box)
            if moles['moles_detected'] > 0:
                result['unique_markers'].extend(moles['mole_positions'])
                result['micro_features']['moles'] = moles
                
            # 3. SKIN TEXTURE FINGERPRINT
            skin_fp = self._extract_skin_fingerprint(image, face_box)
            result['micro_features']['skin_texture'] = skin_fp
            
            # 4. EAR SHAPE ANALYSIS (if visible)
            ear_analysis = self._analyze_ear_shape(image, face_box)
            result['micro_features']['ear_shape'] = ear_analysis
            
            # 5. FACIAL VEIN PATTERN (requires good lighting)
            vein_pattern = self._detect_facial_veins(image, face_box)
            result['micro_features']['vein_pattern'] = vein_pattern
            
            # 6. WRINKLE/CREASE PATTERN
            wrinkles = self._analyze_wrinkle_pattern(image, face_box, landmarks)
            result['micro_features']['wrinkle_pattern'] = wrinkles

            # 7. HEREDITARY PHENOTYPIC SIGNATURE
            phenotypic_signature = self._extract_phenotypic_signature(
                image, face_box, landmarks, ear_analysis
            )
            result['phenotypic_signature'] = phenotypic_signature
            result['micro_features']['hereditary_traits'] = phenotypic_signature
            
            # CALCULATE UNIQUENESS SCORE
            score = 0.0
            
            # Asymmetry contributes to uniqueness
            if asymmetry and asymmetry.get('asymmetry_index', 0) > 0:
                score += 0.15
                
            # Moles are highly unique
            if moles['moles_detected'] > 0:
                score += min(moles['moles_detected'] * 0.1, 0.3)
                
            # Skin texture complexity
            if skin_fp.get('texture_complexity', 0) > 0.5:
                score += 0.2
                
            # Ear shape (very unique)
            if ear_analysis.get('ear_detected'):
                score += 0.25
                
            # Vein pattern
            if vein_pattern.get('pattern_detected'):
                score += 0.15
                
            result['uniqueness_score'] = round(min(score, 1.0) * 100, 2)
            result['anti_doppelganger_confidence'] = result['uniqueness_score']
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def compare_for_doppelganger(self, features1: Dict, features2: Dict, 
                                  face_match_score: float) -> Dict[str, Any]:
        """
        Determine if two similar faces are the same person or doppelgangers.
        
        Args:
            features1: Unique features from face 1
            features2: Unique features from face 2
            face_match_score: Standard face recognition match score (0-100)
        """
        result = {
            'verdict': 'UNKNOWN',
            'is_doppelganger': False,
            'same_person_probability': 0.0,
            'doppelganger_probability': 0.0,
            'evidence': [],
            'recommendation': ''
        }
        
        try:
            evidence_for_same = []
            evidence_for_different = []
            
            # 1. COMPARE ASYMMETRY SIGNATURES
            asym1 = features1.get('asymmetry_signature', {})
            asym2 = features2.get('asymmetry_signature', {})
            
            if asym1 and asym2:
                asym_match = self._compare_asymmetry(asym1, asym2)
                if asym_match > 0.7:
                    evidence_for_same.append(f"Asymmetry pattern match: {asym_match*100:.1f}%")
                else:
                    evidence_for_different.append(f"Asymmetry pattern mismatch: {asym_match*100:.1f}%")
                    
            # 2. COMPARE MOLE POSITIONS
            moles1 = features1.get('micro_features', {}).get('moles', {})
            moles2 = features2.get('micro_features', {}).get('moles', {})
            
            mole_match = self._compare_mole_patterns(moles1, moles2)
            if mole_match['conclusion'] == 'MATCH':
                evidence_for_same.append(f"Mole pattern match: {mole_match['similarity']*100:.1f}%")
            elif mole_match['conclusion'] == 'MISMATCH':
                evidence_for_different.append(f"Mole pattern mismatch: {mole_match['details']}")
                
            # 3. COMPARE SKIN TEXTURE
            skin1 = features1.get('micro_features', {}).get('skin_texture', {})
            skin2 = features2.get('micro_features', {}).get('skin_texture', {})
            
            if skin1 and skin2:
                skin_match = self._compare_skin_texture(skin1, skin2)
                if skin_match > 0.6:
                    evidence_for_same.append(f"Skin texture correlation: {skin_match*100:.1f}%")
                    
            # 4. COMPARE EAR SHAPES
            ear1 = features1.get('micro_features', {}).get('ear_shape', {})
            ear2 = features2.get('micro_features', {}).get('ear_shape', {})
            
            if ear1.get('ear_detected') and ear2.get('ear_detected'):
                ear_match = self._compare_ear_shape(ear1, ear2)
                if ear_match > 0.7:
                    evidence_for_same.append(f"Ear shape match: {ear_match*100:.1f}%")
                else:
                    evidence_for_different.append(f"Ear shape differs: {ear_match*100:.1f}%")
                    
            # CALCULATE FINAL VERDICT
            same_evidence_weight = len(evidence_for_same) * 0.2
            diff_evidence_weight = len(evidence_for_different) * 0.25
            
            # Face match score is the baseline
            base_probability = face_match_score / 100
            
            # Adjust based on unique marker evidence
            same_person_prob = base_probability + same_evidence_weight - diff_evidence_weight
            same_person_prob = max(0, min(1, same_person_prob))
            
            doppelganger_prob = 1 - same_person_prob
            
            result['same_person_probability'] = round(same_person_prob * 100, 2)
            result['doppelganger_probability'] = round(doppelganger_prob * 100, 2)
            result['evidence'] = evidence_for_same + evidence_for_different
            
            # Determine verdict
            if same_person_prob >= 0.75:
                result['verdict'] = 'SAME_PERSON'
                result['is_doppelganger'] = False
                result['recommendation'] = 'High confidence - Same individual'
            elif same_person_prob >= 0.5:
                result['verdict'] = 'LIKELY_SAME'
                result['is_doppelganger'] = False
                result['recommendation'] = 'Probable match - Additional verification recommended'
            elif same_person_prob >= 0.3:
                result['verdict'] = 'UNCERTAIN'
                result['is_doppelganger'] = True
                result['recommendation'] = '[WARNING] POSSIBLE DOPPELGANGER - Manual review required'
            else:
                result['verdict'] = 'LIKELY_DOPPELGANGER'
                result['is_doppelganger'] = True
                result['recommendation'] = '[WARNING] HIGH DOPPELGANGER RISK - Faces look similar but unique markers differ'
                
        except Exception as e:
            result['error'] = str(e)
            
        return result

    def compare_for_kinship(self, features1: Dict, features2: Dict, face_match_score: float) -> Dict[str, Any]:
        """
        Estimate whether two non-identical faces exhibit close hereditary structure.
        This is intentionally additive and does not override direct same-person logic.
        """
        result = {
            'likely_related': False,
            'kinship_probability': 0.0,
            'relationship_hypothesis': 'not_indicated',
            'trait_similarity': {},
            'evidence': [],
            'recommendation': ''
        }

        try:
            sig1 = features1.get('phenotypic_signature', {})
            sig2 = features2.get('phenotypic_signature', {})
            vector1 = sig1.get('signature_vector', [])
            vector2 = sig2.get('signature_vector', [])

            if vector1 and vector2:
                min_len = min(len(vector1), len(vector2))
                vector1 = np.asarray(vector1[:min_len], dtype=np.float32)
                vector2 = np.asarray(vector2[:min_len], dtype=np.float32)
                distance = np.mean(np.abs(vector1 - vector2))
                structure_similarity = float(max(0.0, 1.0 - distance))
            else:
                structure_similarity = 0.0

            trait_names = sig1.get('trait_names', [])
            for idx, name in enumerate(trait_names[:len(vector1)]):
                result['trait_similarity'][name] = round(
                    max(0.0, 1.0 - abs(float(vector1[idx]) - float(vector2[idx]))),
                    4
                )

            if structure_similarity > 0:
                result['evidence'].append(
                    f"Bone-structure phenotype similarity: {structure_similarity * 100:.1f}%"
                )

            ear1 = features1.get('micro_features', {}).get('ear_shape', {})
            ear2 = features2.get('micro_features', {}).get('ear_shape', {})
            ear_similarity = self._compare_ear_shape(ear1, ear2) if ear1 and ear2 else 0.5
            if ear1.get('ear_detected') and ear2.get('ear_detected') and ear_similarity > 0.65:
                result['evidence'].append(f"Ear cartilage morphology alignment: {ear_similarity * 100:.1f}%")

            vein1 = features1.get('micro_features', {}).get('vein_pattern', {})
            vein2 = features2.get('micro_features', {}).get('vein_pattern', {})
            vein_similarity = self._compare_pattern_hash(
                vein1.get('pattern_signature', ''),
                vein2.get('pattern_signature', '')
            )
            if vein_similarity > 0.6:
                result['evidence'].append(f"Facial vascular map affinity: {vein_similarity * 100:.1f}%")

            face_bias = max(0.0, min(1.0, face_match_score / 100.0))
            kinship_prob = (
                0.60 * structure_similarity
                + 0.20 * ear_similarity
                + 0.10 * vein_similarity
                + 0.10 * min(face_bias, 0.65)
            )
            kinship_prob = max(0.0, min(1.0, kinship_prob))

            likely_related = kinship_prob >= 0.72 and face_match_score < 92.0
            if kinship_prob >= 0.82:
                relationship = 'possible_sibling'
            elif kinship_prob >= 0.72:
                relationship = 'possible_close_relative'
            elif kinship_prob >= 0.60:
                relationship = 'weak_familial_signal'
            else:
                relationship = 'not_indicated'

            recommendation = ''
            if likely_related:
                recommendation = (
                    f"Direct same-person match is not dominant, but phenotypic structure suggests {relationship.replace('_', ' ')}"
                )

            result.update({
                'likely_related': likely_related,
                'kinship_probability': round(kinship_prob * 100, 2),
                'relationship_hypothesis': relationship,
                'recommendation': recommendation
            })
        except Exception as e:
            result['error'] = str(e)

        return result
    
    def _analyze_facial_asymmetry(self, image: np.ndarray, face_box: Dict, 
                                   landmarks: Dict = None) -> Dict:
        """
        Analyze facial asymmetry - every face has unique asymmetry.
        """
        asymmetry = {
            'asymmetry_index': 0.0,
            'left_right_difference': {},
            'signature_vector': []
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = gray[y:y+h, x:x+w]
            
            if face.size == 0:
                return asymmetry
                
            # Split face into left and right
            mid = w // 2
            left_half = face[:, :mid]
            right_half = face[:, mid:]
            
            # Flip right half to compare
            right_flipped = cv2.flip(right_half, 1)
            
            # Resize to same dimensions
            min_w = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_w]
            right_flipped = right_flipped[:, :min_w]
            
            # Calculate difference
            diff = cv2.absdiff(left_half, right_flipped)
            asymmetry_index = np.mean(diff) / 255.0
            
            asymmetry['asymmetry_index'] = round(asymmetry_index, 4)
            
            # Create signature vector (regional asymmetry)
            # Divide into grid
            grid_h, grid_w = 4, 4
            cell_h, cell_w = left_half.shape[0] // grid_h, min_w // grid_w
            
            signature = []
            for i in range(grid_h):
                for j in range(grid_w):
                    cell_diff = diff[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                    if cell_diff.size > 0:
                        signature.append(round(np.mean(cell_diff), 2))
                        
            asymmetry['signature_vector'] = signature
            
            # Regional analysis
            # Eye region asymmetry
            eye_region = diff[:h//3, :]
            asymmetry['left_right_difference']['eyes'] = round(np.mean(eye_region), 2)
            
            # Nose region
            nose_region = diff[h//3:2*h//3, min_w//3:2*min_w//3]
            if nose_region.size > 0:
                asymmetry['left_right_difference']['nose'] = round(np.mean(nose_region), 2)
                
            # Mouth region
            mouth_region = diff[2*h//3:, :]
            asymmetry['left_right_difference']['mouth'] = round(np.mean(mouth_region), 2)
            
        except Exception as e:
            pass
            
        return asymmetry
    
    def _detect_facial_moles(self, image: np.ndarray, face_box: Dict) -> Dict:
        """
        Detect moles, birthmarks, and other unique facial markers.
        """
        moles = {
            'moles_detected': 0,
            'mole_positions': [],
            'mole_sizes': []
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            face = image[y:y+h, x:x+w]
            if face.size == 0:
                return moles
                
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            
            # Moles are typically dark brown/black spots
            # Low value (darkness) and specific hue range
            lower_mole = np.array([0, 10, 0])
            upper_mole = np.array([30, 255, 80])
            
            mask = cv2.inRange(hsv, lower_mole, upper_mole)
            
            # Find contours (potential moles)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Moles are small, roughly circular
                if 5 < area < 500:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        
                        # Moles are relatively circular
                        if circularity > 0.5:
                            M = cv2.moments(contour)
                            if M['m00'] > 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                                
                                # Normalize position relative to face
                                rel_x = cx / w
                                rel_y = cy / h
                                
                                moles['mole_positions'].append({
                                    'relative_x': round(rel_x, 3),
                                    'relative_y': round(rel_y, 3),
                                    'size': round(area, 1)
                                })
                                moles['mole_sizes'].append(area)
                                
            moles['moles_detected'] = len(moles['mole_positions'])
            
        except Exception as e:
            pass
            
        return moles
    
    def _extract_skin_fingerprint(self, image: np.ndarray, face_box: Dict) -> Dict:
        """
        Extract unique skin texture pattern (pore pattern fingerprint).
        """
        fingerprint = {
            'texture_complexity': 0.0,
            'texture_hash': '',
            'pore_density': 0.0
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Sample cheek region (best for skin texture)
            cheek_y = y + h // 3
            cheek_h = h // 4
            cheek_x = x + w // 4
            cheek_w = w // 4
            
            cheek = gray[cheek_y:cheek_y+cheek_h, cheek_x:cheek_x+cheek_w]
            
            if cheek.size == 0:
                return fingerprint
                
            # High-pass filter to extract texture
            blurred = cv2.GaussianBlur(cheek, (9, 9), 0)
            texture = cv2.absdiff(cheek, blurred)
            
            # Texture complexity (entropy)
            hist = cv2.calcHist([texture], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            fingerprint['texture_complexity'] = round(entropy / 8.0, 3)
            
            # Pore density
            _, pores = cv2.threshold(texture, 15, 255, cv2.THRESH_BINARY)
            fingerprint['pore_density'] = round(np.sum(pores > 0) / pores.size, 4)
            
            # Create texture hash (simplified perceptual hash)
            resized = cv2.resize(texture, (8, 8))
            mean_val = np.mean(resized)
            hash_bits = (resized > mean_val).flatten()
            hash_str = ''.join('1' if b else '0' for b in hash_bits)
            fingerprint['texture_hash'] = hex(int(hash_str, 2))[2:].zfill(16)
            
        except Exception as e:
            pass
            
        return fingerprint
    
    def _analyze_ear_shape(self, image: np.ndarray, face_box: Dict) -> Dict:
        """
        Analyze ear shape if visible - ears are as unique as fingerprints.
        """
        ear = {
            'ear_detected': False,
            'ear_shape_code': '',
            'ear_area_ratio': 0.0
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            # Ear typically on sides of face box (outside or edge)
            # Check left side of face
            left_ear_x = max(0, x - w // 4)
            left_ear_w = w // 3
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            left_region = gray[y:y+h, left_ear_x:left_ear_x+left_ear_w]
            
            if left_region.size > 100:
                # Ears have distinctive curved edges
                edges = cv2.Canny(left_region, 50, 150)
                
                # Look for ear-shaped contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Significant contour
                        # Check if shape is ear-like (tall, curved)
                        x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                        aspect = h_c / w_c if w_c > 0 else 0
                        
                        if 1.5 < aspect < 3.0:  # Ears are typically taller than wide
                            ear['ear_detected'] = True
                            ear['ear_area_ratio'] = round(area / (left_region.shape[0] * left_region.shape[1]), 4)
                            
                            # Create ear shape code
                            mask = np.zeros(left_region.shape, dtype=np.uint8)
                            cv2.drawContours(mask, [contour], -1, 255, -1)
                            resized = cv2.resize(mask, (8, 8))
                            hash_bits = (resized > 127).flatten()
                            ear['ear_shape_code'] = hex(int(''.join('1' if b else '0' for b in hash_bits), 2))[2:].zfill(16)
                            break
                            
        except Exception as e:
            pass
            
        return ear
    
    def _detect_facial_veins(self, image: np.ndarray, face_box: Dict) -> Dict:
        """
        Detect facial vein patterns (unique biometric).
        """
        veins = {
            'pattern_detected': False,
            'vein_density': 0.0,
            'pattern_signature': ''
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            face = image[y:y+h, x:x+w]
            if face.size == 0:
                return veins
                
            # Veins are visible in the blue channel (blood absorbs red)
            b, g, r = cv2.split(face)
            
            # Enhance vein visibility
            # Green channel minus red often shows veins better
            vein_enhanced = cv2.subtract(g, r)
            
            # Apply CLAHE to enhance
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(vein_enhanced)
            
            # Edge detection for vein patterns
            edges = cv2.Canny(enhanced, 30, 100)
            
            # Calculate vein density
            density = np.sum(edges > 0) / edges.size
            veins['vein_density'] = round(density, 4)
            
            if density > 0.02:  # Some veins visible
                veins['pattern_detected'] = True
                
                # Create pattern signature
                resized = cv2.resize(edges, (8, 8))
                hash_bits = (resized > 0).flatten()
                veins['pattern_signature'] = hex(int(''.join('1' if b else '0' for b in hash_bits), 2))[2:].zfill(16)
                
        except Exception as e:
            pass
            
        return veins
    
    def _analyze_wrinkle_pattern(self, image: np.ndarray, face_box: Dict, 
                                  landmarks: Dict = None) -> Dict:
        """
        Analyze wrinkle/crease patterns - unique even among twins.
        """
        wrinkles = {
            'wrinkle_score': 0.0,
            'forehead_lines': 0,
            'crow_feet_score': 0.0,
            'nasolabial_depth': 0.0
        }
        
        try:
            x, y, w, h = face_box.get('x', 0), face_box.get('y', 0), face_box.get('w', 100), face_box.get('h', 100)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = gray[y:y+h, x:x+w]
            
            if face.size == 0:
                return wrinkles
                
            # FOREHEAD WRINKLES (horizontal lines)
            forehead = face[:h//4, w//4:3*w//4]
            if forehead.size > 0:
                # Detect horizontal lines
                edges = cv2.Canny(forehead, 20, 60)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=w//4, maxLineGap=10)
                
                if lines is not None:
                    h_lines = sum(1 for l in lines if abs(l[0][1] - l[0][3]) < 10)
                    wrinkles['forehead_lines'] = h_lines
                    
            # CROW'S FEET (wrinkles around eyes)
            eye_corner_left = face[h//5:h//3, :w//5]
            eye_corner_right = face[h//5:h//3, 4*w//5:]
            
            crow_feet = 0
            for corner in [eye_corner_left, eye_corner_right]:
                if corner.size > 0:
                    edges = cv2.Canny(corner, 15, 45)
                    crow_feet += np.sum(edges > 0) / edges.size
                    
            wrinkles['crow_feet_score'] = round(crow_feet / 2, 4)
            
            # NASOLABIAL FOLDS (lines from nose to mouth)
            lower_face = face[h//2:, :]
            if lower_face.size > 0:
                edges = cv2.Canny(lower_face, 25, 75)
                # Look for diagonal lines
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 10, minLineLength=h//8, maxLineGap=5)
                
                if lines is not None:
                    diagonal_lines = sum(1 for l in lines 
                                        if 20 < abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]) * 180/np.pi) < 70)
                    wrinkles['nasolabial_depth'] = min(diagonal_lines * 0.1, 1.0)
                    
            # Overall wrinkle score
            wrinkles['wrinkle_score'] = round(
                (wrinkles['forehead_lines'] * 0.1 + 
                 wrinkles['crow_feet_score'] * 5 + 
                 wrinkles['nasolabial_depth']) / 3, 2
            )
            
        except Exception as e:
            pass
            
        return wrinkles

    def _extract_phenotypic_signature(
        self,
        image: np.ndarray,
        face_box: Dict,
        landmarks: Dict = None,
        ear_analysis: Dict = None
    ) -> Dict[str, Any]:
        """
        Encode coarse hereditary morphology that tends to persist across close
        relatives: orbital spacing, nasal bridge proportions, jaw taper, and
        ear geometry when visible.
        """
        signature = {
            'trait_names': [],
            'signature_vector': [],
            'bone_ratios': {},
            'confidence': 0.0
        }

        try:
            x, y, w, h = self._resolve_face_box(image, face_box)
            if w <= 0 or h <= 0:
                return signature

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = gray[y:y+h, x:x+w]
            if face.size == 0:
                return signature

            face_edges = cv2.Canny(face, 40, 120)
            left_profile = self._profile_extent(face_edges[:, :max(1, w // 2)], axis=1)
            right_profile = self._profile_extent(cv2.flip(face_edges[:, max(1, w // 2):], 1), axis=1)
            mid_profile = self._profile_extent(face_edges[:, w // 3:2 * w // 3], axis=0)

            eye_band = face[int(h * 0.22):int(h * 0.42), :]
            lower_band = face[int(h * 0.58):int(h * 0.84), :]
            brow_band = face[int(h * 0.12):int(h * 0.24), :]

            orbital_width_ratio = float(np.mean(mid_profile) / max(w, 1)) if len(mid_profile) else 0.32
            brow_orbit_ratio = float(np.std(eye_band) / max(np.std(brow_band), 1.0)) if eye_band.size and brow_band.size else 1.0
            jaw_taper_ratio = float(
                (np.mean(left_profile[int(len(left_profile) * 0.7):]) + np.mean(right_profile[int(len(right_profile) * 0.7):]))
                / max(np.mean(left_profile[:max(1, int(len(left_profile) * 0.3))]) + np.mean(right_profile[:max(1, int(len(right_profile) * 0.3))]), 1.0)
            ) if len(left_profile) and len(right_profile) else 0.75

            nose_bridge_ratio = float(np.std(face[int(h * 0.25):int(h * 0.60), int(w * 0.42):int(w * 0.58)]) / max(np.std(face), 1.0))
            nose_base_ratio = float(np.mean(face_edges[int(h * 0.52):int(h * 0.72), int(w * 0.30):int(w * 0.70)] > 0))
            cheekbone_ratio = float(np.mean(face_edges[int(h * 0.38):int(h * 0.58), :] > 0))
            mouth_width_ratio = float(np.mean(face_edges[int(h * 0.66):int(h * 0.82), :] > 0))

            ear_ratio = float(ear_analysis.get('ear_area_ratio', 0.0)) if ear_analysis else 0.0

            traits = {
                'orbital_width_ratio': min(max(orbital_width_ratio, 0.0), 1.0),
                'brow_orbit_ratio': min(max(brow_orbit_ratio / 3.0, 0.0), 1.0),
                'nose_bridge_ratio': min(max(nose_bridge_ratio / 2.5, 0.0), 1.0),
                'nose_base_ratio': min(max(nose_base_ratio * 3.5, 0.0), 1.0),
                'jaw_taper_ratio': min(max(jaw_taper_ratio, 0.0), 1.0),
                'cheekbone_ratio': min(max(cheekbone_ratio * 3.0, 0.0), 1.0),
                'mouth_width_ratio': min(max(mouth_width_ratio * 3.0, 0.0), 1.0),
                'ear_cartilage_ratio': min(max(ear_ratio * 6.0, 0.0), 1.0),
            }

            signature['trait_names'] = list(traits.keys())
            signature['signature_vector'] = [round(float(value), 4) for value in traits.values()]
            signature['bone_ratios'] = {name: round(float(value), 4) for name, value in traits.items()}
            signature['confidence'] = float(round(
                min(1.0, 0.45 + 0.1 * bool(ear_analysis and ear_analysis.get('ear_detected')) + 0.45 * min(np.std(face) / 64.0, 1.0)),
                4
            ))
        except Exception:
            pass

        return signature

    @staticmethod
    def _resolve_face_box(image: np.ndarray, face_box: Dict) -> Tuple[int, int, int, int]:
        h, w = image.shape[:2]
        if not face_box:
            return 0, 0, w, h
        x = max(0, int(face_box.get('x', 0)))
        y = max(0, int(face_box.get('y', 0)))
        bw = max(1, int(face_box.get('w', w)))
        bh = max(1, int(face_box.get('h', h)))
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        if x2 - x < max(16, w // 5) or y2 - y < max(16, h // 5):
            return 0, 0, w, h
        return x, y, x2 - x, y2 - y

    @staticmethod
    def _profile_extent(edge_map: np.ndarray, axis: int = 1) -> np.ndarray:
        if edge_map.size == 0:
            return np.asarray([], dtype=np.float32)
        profile = []
        if axis == 1:
            for row in edge_map:
                active = np.where(row > 0)[0]
                profile.append(float(active[-1] - active[0]) if active.size > 1 else 0.0)
        else:
            for col in edge_map.T:
                active = np.where(col > 0)[0]
                profile.append(float(active[-1] - active[0]) if active.size > 1 else 0.0)
        return np.asarray(profile, dtype=np.float32)
    
    def _compare_asymmetry(self, asym1: Dict, asym2: Dict) -> float:
        """Compare asymmetry signatures."""
        vec1 = asym1.get('signature_vector', [])
        vec2 = asym2.get('signature_vector', [])
        
        if not vec1 or not vec2:
            return 0.5
            
        # Ensure same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        # Correlation
        if len(vec1) > 1:
            correlation = np.corrcoef(vec1, vec2)[0, 1]
            return max(0, (correlation + 1) / 2)  # Normalize to 0-1
            
        return 0.5
    
    def _compare_mole_patterns(self, moles1: Dict, moles2: Dict) -> Dict:
        """Compare mole patterns between two faces."""
        result = {
            'similarity': 0.0,
            'conclusion': 'UNKNOWN',
            'details': ''
        }
        
        pos1 = moles1.get('mole_positions', [])
        pos2 = moles2.get('mole_positions', [])
        
        # If both have no moles, inconclusive
        if not pos1 and not pos2:
            result['conclusion'] = 'INCONCLUSIVE'
            result['details'] = 'No moles detected in either image'
            return result
            
        # If only one has moles, mismatch
        if (pos1 and not pos2) or (pos2 and not pos1):
            result['conclusion'] = 'MISMATCH'
            result['details'] = 'Mole presence differs between images'
            return result
            
        # Compare positions
        matched = 0
        tolerance = 0.1  # 10% position tolerance
        
        for m1 in pos1:
            for m2 in pos2:
                dist = math.sqrt(
                    (m1['relative_x'] - m2['relative_x'])**2 + 
                    (m1['relative_y'] - m2['relative_y'])**2
                )
                if dist < tolerance:
                    matched += 1
                    break
                    
        if len(pos1) > 0:
            result['similarity'] = matched / len(pos1)
            
        if result['similarity'] > 0.7:
            result['conclusion'] = 'MATCH'
        elif result['similarity'] > 0.3:
            result['conclusion'] = 'PARTIAL_MATCH'
        else:
            result['conclusion'] = 'MISMATCH'
            result['details'] = 'Mole positions do not correspond'
            
        return result
    
    def _compare_skin_texture(self, skin1: Dict, skin2: Dict) -> float:
        """Compare skin texture fingerprints."""
        hash1 = skin1.get('texture_hash', '')
        hash2 = skin2.get('texture_hash', '')
        
        if not hash1 or not hash2:
            return 0.5
            
        # Hamming distance between hashes
        try:
            bin1 = bin(int(hash1, 16))[2:].zfill(64)
            bin2 = bin(int(hash2, 16))[2:].zfill(64)
            
            diff = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            similarity = 1 - (diff / len(bin1))
            return similarity
        except:
            return 0.5
    
    def _compare_ear_shape(self, ear1: Dict, ear2: Dict) -> float:
        """Compare ear shape codes."""
        code1 = ear1.get('ear_shape_code', '')
        code2 = ear2.get('ear_shape_code', '')
        
        if not code1 or not code2:
            return 0.5
            
        try:
            bin1 = bin(int(code1, 16))[2:].zfill(64)
            bin2 = bin(int(code2, 16))[2:].zfill(64)
            
            diff = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            similarity = 1 - (diff / len(bin1))
            return similarity
        except:
            return 0.5

    @staticmethod
    def _compare_pattern_hash(hash1: str, hash2: str) -> float:
        if not hash1 or not hash2:
            return 0.5
        try:
            bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
            diff = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            return 1 - (diff / max(len(bin1), 1))
        except Exception:
            return 0.5
