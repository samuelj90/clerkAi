"""
OCR (Optical Character Recognition) service for ClerkAI.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
from io import BytesIO
import logging

from config.settings import settings
from config.logging import LoggerMixin, log_performance

logger = logging.getLogger(__name__)


class OCRResult:
    """OCR result container."""
    
    def __init__(
        self,
        text: str,
        confidence: float,
        language: str,
        bounding_boxes: Optional[List[Dict]] = None,
        processing_time: Optional[float] = None
    ):
        self.text = text
        self.confidence = confidence
        self.language = language
        self.bounding_boxes = bounding_boxes or []
        self.processing_time = processing_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "bounding_boxes": self.bounding_boxes,
            "processing_time": self.processing_time
        }


class OCRService(LoggerMixin):
    """OCR service for extracting text from images and documents."""
    
    def __init__(self):
        """Initialize OCR service."""
        # Configure Tesseract
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        
        # Supported image formats
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
        self.supported_pdf = {'.pdf'}
        
        # OCR configuration
        self.ocr_config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine Mode, uniform text block
        
        self.logger.info("OCR service initialized")
    
    @log_performance("ocr_extract_text")
    def extract_text(
        self,
        file_path: str,
        language: str = None,
        enhance_image: bool = True,
        get_confidence: bool = True,
        get_bounding_boxes: bool = False
    ) -> OCRResult:
        """
        Extract text from an image or PDF file.
        
        Args:
            file_path: Path to the file
            language: OCR language (default from settings)
            enhance_image: Whether to enhance image before OCR
            get_confidence: Whether to calculate confidence score
            get_bounding_boxes: Whether to extract bounding boxes
            
        Returns:
            OCRResult: Extracted text and metadata
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
            Exception: If OCR processing fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        language = language or settings.ocr_language
        
        self.logger.info(
            "Starting OCR extraction",
            file_path=file_path,
            language=language,
            enhance=enhance_image
        )
        
        try:
            if file_ext in self.supported_formats:
                return self._extract_from_image(
                    file_path, language, enhance_image, 
                    get_confidence, get_bounding_boxes
                )
            elif file_ext in self.supported_pdf:
                return self._extract_from_pdf(
                    file_path, language, enhance_image,
                    get_confidence, get_bounding_boxes
                )
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            self.logger.error(
                "OCR extraction failed",
                file_path=file_path,
                error=str(e)
            )
            raise
    
    def _extract_from_image(
        self,
        image_path: str,
        language: str,
        enhance_image: bool,
        get_confidence: bool,
        get_bounding_boxes: bool
    ) -> OCRResult:
        """Extract text from image file."""
        # Load and preprocess image
        image = Image.open(image_path)
        
        if enhance_image:
            image = self._enhance_image(image)
        
        # Configure OCR
        config = f"-l {language} {self.ocr_config}"
        
        # Extract text
        text = pytesseract.image_to_string(image, config=config).strip()
        
        # Get confidence if requested
        confidence = 0.0
        if get_confidence:
            try:
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
            except Exception as e:
                self.logger.warning(f"Failed to get confidence: {e}")
        
        # Get bounding boxes if requested
        bounding_boxes = []
        if get_bounding_boxes:
            try:
                boxes = pytesseract.image_to_boxes(image, config=config)
                bounding_boxes = self._parse_bounding_boxes(boxes)
            except Exception as e:
                self.logger.warning(f"Failed to get bounding boxes: {e}")
        
        self.logger.info(
            "Image OCR completed",
            text_length=len(text),
            confidence=confidence
        )
        
        return OCRResult(
            text=text,
            confidence=confidence,
            language=language,
            bounding_boxes=bounding_boxes
        )
    
    def _extract_from_pdf(
        self,
        pdf_path: str,
        language: str,
        enhance_image: bool,
        get_confidence: bool,
        get_bounding_boxes: bool
    ) -> OCRResult:
        """Extract text from PDF file."""
        all_text = []
        all_confidences = []
        all_bounding_boxes = []
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=300,  # High DPI for better OCR accuracy
                fmt='PNG'
            )
            
            self.logger.info(f"Converted PDF to {len(images)} images")
            
            # Process each page
            for page_num, image in enumerate(images, 1):
                self.logger.debug(f"Processing page {page_num}")
                
                if enhance_image:
                    image = self._enhance_image(image)
                
                # Configure OCR
                config = f"-l {language} {self.ocr_config}"
                
                # Extract text
                page_text = pytesseract.image_to_string(image, config=config).strip()
                if page_text:
                    all_text.append(f"[Page {page_num}]\n{page_text}")
                
                # Get confidence if requested
                if get_confidence:
                    try:
                        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        if confidences:
                            page_confidence = sum(confidences) / len(confidences)
                            all_confidences.append(page_confidence)
                    except Exception as e:
                        self.logger.warning(f"Failed to get confidence for page {page_num}: {e}")
                
                # Get bounding boxes if requested
                if get_bounding_boxes:
                    try:
                        boxes = pytesseract.image_to_boxes(image, config=config)
                        page_boxes = self._parse_bounding_boxes(boxes, page_num)
                        all_bounding_boxes.extend(page_boxes)
                    except Exception as e:
                        self.logger.warning(f"Failed to get bounding boxes for page {page_num}: {e}")
            
            # Combine results
            combined_text = "\n\n".join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            self.logger.info(
                "PDF OCR completed",
                pages=len(images),
                text_length=len(combined_text),
                confidence=avg_confidence
            )
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                language=language,
                bounding_boxes=all_bounding_boxes
            )
            
        except Exception as e:
            self.logger.error(f"PDF OCR failed: {e}")
            raise
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image for better OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image object
        """
        try:
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Resize if too small (minimum 300 DPI equivalent)
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _parse_bounding_boxes(self, boxes_string: str, page_num: int = 1) -> List[Dict]:
        """
        Parse Tesseract bounding boxes output.
        
        Args:
            boxes_string: Raw bounding boxes string from Tesseract
            page_num: Page number for PDF processing
            
        Returns:
            List of bounding box dictionaries
        """
        bounding_boxes = []
        
        for line in boxes_string.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    char = parts[0]
                    left = int(parts[1])
                    bottom = int(parts[2])
                    right = int(parts[3])
                    top = int(parts[4])
                    page = int(parts[5]) if len(parts) > 5 else page_num
                    
                    bounding_boxes.append({
                        'character': char,
                        'left': left,
                        'bottom': bottom,
                        'right': right,
                        'top': top,
                        'page': page
                    })
        
        return bounding_boxes
    
    def get_available_languages(self) -> List[str]:
        """
        Get list of available OCR languages.
        
        Returns:
            List of language codes
        """
        try:
            languages = pytesseract.get_languages(config='')
            self.logger.info(f"Available OCR languages: {languages}")
            return languages
        except Exception as e:
            self.logger.error(f"Failed to get available languages: {e}")
            return [settings.ocr_language]
    
    def validate_image(self, file_path: str) -> bool:
        """
        Validate if file is a valid image for OCR.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in self.supported_formats:
                # Try to open image
                with Image.open(file_path) as img:
                    img.verify()
                return True
            elif file_ext in self.supported_pdf:
                # Basic PDF validation
                return os.path.getsize(file_path) > 0
            else:
                return False
                
        except Exception as e:
            self.logger.warning(f"Image validation failed: {e}")
            return False
    
    def estimate_processing_time(self, file_path: str) -> float:
        """
        Estimate OCR processing time in seconds.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            file_ext = Path(file_path).suffix.lower()
            
            # Base time estimates (empirical)
            if file_ext in self.supported_formats:
                # ~2-5 seconds per MB for images
                return file_size_mb * 3.5
            elif file_ext in self.supported_pdf:
                # ~5-10 seconds per MB for PDFs (depends on page count)
                return file_size_mb * 7.5
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Time estimation failed: {e}")
            return 60.0  # Default estimate
    
    def health_check(self) -> Dict[str, any]:
        """
        Perform OCR service health check.
        
        Returns:
            Health check results
        """
        health = {
            "service": "OCR",
            "status": "healthy",
            "tesseract_available": False,
            "available_languages": [],
            "version": None,
            "errors": []
        }
        
        try:
            # Check if Tesseract is available
            version = pytesseract.get_tesseract_version()
            health["tesseract_available"] = True
            health["version"] = str(version)
            
            # Get available languages
            languages = self.get_available_languages()
            health["available_languages"] = languages
            
            # Check if configured language is available
            if settings.ocr_language not in languages:
                health["errors"].append(f"Configured language '{settings.ocr_language}' not available")
                health["status"] = "degraded"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["errors"].append(f"Tesseract not available: {str(e)}")
        
        return health


# Global OCR service instance
ocr_service = OCRService()