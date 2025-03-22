"""
Sample Generator for COPixel Detection System.
This script generates sample images and videos for testing the detection system.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import logging
from PIL import Image, ImageDraw, ImageFont
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for sample data"""
    dirs = [
        "samples",
        "samples/deepfake",
        "samples/document",
        "samples/signature/reference",
        "samples/signature/query",
    ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def generate_sample_video(output_path, is_fake=False):
    """Generate a sample video for deepfake detection testing
    
    Args:
        output_path: Path to save the video
        is_fake: If True, create a "fake" video with visual artifacts
    """
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 3  # seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Generate frames
    for i in range(duration * fps):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw a face-like oval
        cv2.ellipse(frame, (width//2, height//2), (120, 160), 0, 0, 360, (200, 200, 200), -1)
        
        # Draw eyes
        eye_y = height//2 - 30
        left_eye_x = width//2 - 60
        right_eye_x = width//2 + 60
        
        cv2.circle(frame, (left_eye_x, eye_y), 20, (255, 255, 255), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 20, (255, 255, 255), -1)
        cv2.circle(frame, (left_eye_x, eye_y), 8, (0, 0, 0), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 8, (0, 0, 0), -1)
        
        # Draw a smile
        smile_y = height//2 + 40
        cv2.ellipse(frame, (width//2, smile_y), (80, 40), 0, 0, 180, (0, 0, 0), 2)
        
        # If fake, add some artifacts
        if is_fake:
            # Add visual glitches every few frames
            if i % 15 == 0:
                # Random block artifacts
                block_x = np.random.randint(width//4, width*3//4)
                block_y = np.random.randint(height//4, height*3//4)
                block_w = np.random.randint(20, 50)
                block_h = np.random.randint(20, 50)
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.rectangle(frame, (block_x, block_y), (block_x+block_w, block_y+block_h), color, -1)
            
            # Add inconsistent eye movement
            if i % 10 == 0:
                offset_x = np.random.randint(-5, 5)
                cv2.circle(frame, (left_eye_x + offset_x, eye_y), 8, (0, 0, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f'Frame {i}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add fake/real label
        label = "FAKE" if is_fake else "REAL"
        cv2.putText(frame, label, (width-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_fake else (0, 255, 0), 2)
        
        # Write the frame
        out.write(frame)
    
    # Release the video writer
    out.release()
    logger.info(f"Generated {'fake' if is_fake else 'real'} video at {output_path}")

def generate_sample_document(output_path, is_forged=False, forgery_type=None):
    """Generate a sample document for forgery detection testing
    
    Args:
        output_path: Path to save the document image
        is_forged: If True, create a forged document
        forgery_type: Type of forgery to simulate
    """
    # Document properties
    width, height = 2100, 2970  # Roughly A4 size
    
    # Create a blank document with white background
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Try to use a common font
    try:
        # Try to load a font that should be available on most systems
        font_large = ImageFont.truetype("arial.ttf", 60)
        font_medium = ImageFont.truetype("arial.ttf", 40)
        font_small = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw document header
    draw.text((width//2 - 400, 100), "CERTIFICATE OF AUTHENTICATION", fill=(0, 0, 0), font=font_large)
    draw.line([(100, 180), (width-100, 180)], fill=(0, 0, 0), width=2)
    
    # Draw document content
    draw.text((150, 300), "This document certifies that:", fill=(0, 0, 0), font=font_medium)
    draw.text((150, 400), "John Doe", fill=(0, 0, 100), font=font_large)
    
    draw.text((150, 600), "Has successfully completed the requirements for:", fill=(0, 0, 0), font=font_medium)
    draw.text((150, 700), "Advanced AI Detection Systems", fill=(0, 0, 100), font=font_large)
    
    # Draw date and certificate number
    draw.text((150, 900), f"Date: {random.randint(1, 30):02d}/{random.randint(1, 12):02d}/2023", fill=(0, 0, 0), font=font_medium)
    cert_num = f"CERT-{random.randint(10000, 99999)}"
    draw.text((150, 1000), f"Certificate Number: {cert_num}", fill=(0, 0, 0), font=font_medium)
    
    # Draw signature area
    draw.text((150, 1200), "Authorized Signature:", fill=(0, 0, 0), font=font_medium)
    
    # Draw a signature
    sig_x, sig_y = 400, 1300
    sig_width, sig_height = 300, 100
    
    # Draw some lines to simulate a signature
    for _ in range(20):
        x1 = sig_x + random.randint(0, sig_width)
        y1 = sig_y + random.randint(0, sig_height)
        x2 = sig_x + random.randint(0, sig_width)
        y2 = sig_y + random.randint(0, sig_height)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 200), width=2)
    
    # Draw a stamp/seal
    seal_x, seal_y = width - 400, 1300
    draw.ellipse([(seal_x-100, seal_y-100), (seal_x+100, seal_y+100)], outline=(200, 0, 0), width=2)
    draw.text((seal_x-70, seal_y-20), "VERIFIED", fill=(200, 0, 0), font=font_small)
    
    # If forged, add some modifications
    if is_forged:
        if forgery_type == "content":
            # Modify the name - simulate changed text
            draw.rectangle([(150, 390), (450, 470)], fill=(255, 255, 255))
            draw.text((150, 400), "Jane Smith", fill=(0, 0, 100), font=font_large)
            logger.info("Added content manipulation forgery")
            
        elif forgery_type == "signature":
            # Modify the signature - clear and redraw differently
            draw.rectangle([(sig_x, sig_y), (sig_x+sig_width, sig_y+sig_height)], fill=(255, 255, 255))
            for _ in range(15):
                x1 = sig_x + random.randint(0, sig_width)
                y1 = sig_y + random.randint(0, sig_height)
                x2 = sig_x + random.randint(0, sig_width)
                y2 = sig_y + random.randint(0, sig_height)
                draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=3)
            logger.info("Added signature forgery")
            
        elif forgery_type == "splicing":
            # Add a digital splicing effect - insert an additional stamp
            for i in range(60):
                angle = i * 6  # 0 to 360 degrees
                x = seal_x + 300 + int(120 * np.cos(np.radians(angle)))
                y = seal_y + int(120 * np.sin(np.radians(angle)))
                draw.point((x, y), fill=(0, 150, 0))
            draw.text((seal_x+230, seal_y-20), "COPY", fill=(0, 150, 0), font=font_small)
            logger.info("Added digital splicing forgery")
            
        else:  # Default or "ai" type
            # Add subtle pixelation to simulate AI generation
            for _ in range(100):
                px = random.randint(0, width-1)
                py = random.randint(0, height-1)
                box_size = random.randint(5, 15)
                color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
                draw.rectangle([(px, py), (px+box_size, py+box_size)], fill=color)
            logger.info("Added AI generation artifacts")
    
    # Save the document
    img.save(output_path)
    logger.info(f"Generated {'forged' if is_forged else 'authentic'} document at {output_path}")

def generate_sample_signature(reference_path, query_path, is_forged=False):
    """Generate a sample signature pair for forgery detection testing
    
    Args:
        reference_path: Path to save the reference signature
        query_path: Path to save the query signature
        is_forged: If True, create a forged query signature
    """
    # Signature properties
    width, height = 500, 200
    
    # Create a reference signature
    ref_img = Image.new('L', (width, height), color=255)
    ref_draw = ImageDraw.Draw(ref_img)
    
    # Generate a random signature pattern
    points = []
    x, y = width // 4, height // 2
    
    # First part of signature - generate some control points
    for _ in range(10):
        x += random.randint(5, 30)
        y += random.randint(-20, 20)
        points.append((x, y))
    
    # Second part - some loops and curves
    x, y = points[-1]
    for _ in range(5):
        # Small loop
        loop_radius = random.randint(10, 30)
        loop_points = []
        for angle in range(0, 360, 20):
            lx = x + int(loop_radius * np.cos(np.radians(angle)))
            ly = y + int(loop_radius * np.sin(np.radians(angle)))
            loop_points.append((lx, ly))
        
        # Add loop to points
        points.extend(loop_points)
        
        # Move to next position
        x += random.randint(20, 50)
        y += random.randint(-10, 10)
    
    # Draw the signature with smooth curves
    for i in range(len(points)-1):
        ref_draw.line([points[i], points[i+1]], fill=0, width=3)
    
    # Add some random dots and marks
    for _ in range(3):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        ref_draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=0)
    
    # Create query signature - either a genuine copy or a forgery
    if is_forged:
        # Create a different signature
        query_img = Image.new('L', (width, height), color=255)
        query_draw = ImageDraw.Draw(query_img)
        
        # Try to mimic the reference but with differences
        for i in range(len(points)-1):
            # Add random deviations
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            
            # Add small offsets to points to simulate forgery
            x1 += random.randint(-10, 10)
            y1 += random.randint(-10, 10)
            x2 += random.randint(-10, 10)
            y2 += random.randint(-10, 10)
            
            query_draw.line([(x1, y1), (x2, y2)], fill=0, width=random.randint(2, 4))
        
        # Add some different dots and marks
        for _ in range(5):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            query_draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=0)
            
    else:
        # Create an authentic copy with minor variations
        query_img = Image.new('L', (width, height), color=255)
        query_draw = ImageDraw.Draw(query_img)
        
        # Same general pattern but with tiny natural variations
        for i in range(len(points)-1):
            # Add very small random deviations (natural variation)
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            
            # Add tiny offsets to points
            x1 += random.randint(-3, 3)
            y1 += random.randint(-3, 3)
            x2 += random.randint(-3, 3)
            y2 += random.randint(-3, 3)
            
            query_draw.line([(x1, y1), (x2, y2)], fill=0, width=3)
        
        # Add same random dots and marks
        for _ in range(3):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            query_draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=0)
    
    # Save both signatures
    ref_img.save(reference_path)
    query_img.save(query_path)
    logger.info(f"Generated {'forged' if is_forged else 'authentic'} signature pair:")
    logger.info(f"  Reference: {reference_path}")
    logger.info(f"  Query: {query_path}")

def main():
    """Main function to handle command-line arguments and generate samples"""
    parser = argparse.ArgumentParser(description="Generate sample data for COPixel AI Detection System")
    parser.add_argument("--type", choices=["all", "deepfake", "document", "signature"], default="all",
                        help="Type of sample to generate")
    parser.add_argument("--count", type=int, default=2, help="Number of samples to generate for each type")
    args = parser.parse_args()
    
    # Create sample directories
    create_directories()
    
    try:
        # Generate samples based on type
        if args.type in ["all", "deepfake"]:
            # Generate deepfake videos
            for i in range(args.count):
                # Generate a real video
                real_path = f"samples/deepfake/real_video_{i+1}.mp4"
                generate_sample_video(real_path, is_fake=False)
                
                # Generate a fake video
                fake_path = f"samples/deepfake/fake_video_{i+1}.mp4"
                generate_sample_video(fake_path, is_fake=True)
        
        if args.type in ["all", "document"]:
            # Generate document images
            for i in range(args.count):
                # Generate an authentic document
                auth_path = f"samples/document/authentic_doc_{i+1}.jpg"
                generate_sample_document(auth_path, is_forged=False)
                
                # Generate forged documents with different forgery types
                forgery_types = ["content", "signature", "splicing", "ai"]
                for ft in forgery_types:
                    forge_path = f"samples/document/forged_doc_{ft}_{i+1}.jpg"
                    generate_sample_document(forge_path, is_forged=True, forgery_type=ft)
        
        if args.type in ["all", "signature"]:
            # Generate signature pairs
            for i in range(args.count):
                # Generate an authentic signature pair
                ref_path = f"samples/signature/reference/authentic_sig_{i+1}.png"
                query_path = f"samples/signature/query/authentic_sig_{i+1}.png"
                generate_sample_signature(ref_path, query_path, is_forged=False)
                
                # Generate a forged signature pair
                ref_path = f"samples/signature/reference/forged_sig_{i+1}.png"
                query_path = f"samples/signature/query/forged_sig_{i+1}.png"
                generate_sample_signature(ref_path, query_path, is_forged=True)
                
        logger.info("Sample generation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating samples: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 