"""
Natural Language Processing service for ClerkAI.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import spacy
from spacy import displacy
from collections import Counter
import logging

from config.settings import settings
from config.logging import LoggerMixin, log_performance

logger = logging.getLogger(__name__)


class NLPResult:
    """NLP processing result container."""
    
    def __init__(
        self,
        text: str,
        entities: List[Dict],
        keywords: List[str],
        summary: Optional[str] = None,
        sentiment: Optional[str] = None,
        language: Optional[str] = None,
        processing_time: Optional[float] = None
    ):
        self.text = text
        self.entities = entities
        self.keywords = keywords
        self.summary = summary
        self.sentiment = sentiment
        self.language = language
        self.processing_time = processing_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "entities": self.entities,
            "keywords": self.keywords,
            "summary": self.summary,
            "sentiment": self.sentiment,
            "language": self.language,
            "processing_time": self.processing_time
        }


class NLPService(LoggerMixin):
    """Natural Language Processing service for text analysis."""
    
    def __init__(self):
        """Initialize NLP service."""
        self.nlp = None
        self._load_model()
        
        # Common patterns for structured data extraction
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|\$)',
            'percentage': r'\d+(?:\.\d+)?%',
            'number': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        }
        
        self.logger.info("NLP service initialized")
    
    def _load_model(self):
        """Load spaCy NLP model."""
        try:
            # Try to load the large English model first
            model_names = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
            
            for model_name in model_names:
                try:
                    self.nlp = spacy.load(model_name)
                    self.logger.info(f"Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue
            
            if self.nlp is None:
                raise OSError("No spaCy English model found")
                
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
    
    @log_performance("nlp_process_text")
    def process_text(
        self,
        text: str,
        extract_entities: bool = True,
        extract_keywords: bool = True,
        generate_summary: bool = False,
        analyze_sentiment: bool = False,
        max_keywords: int = 20
    ) -> NLPResult:
        """
        Process text with various NLP techniques.
        
        Args:
            text: Text to process
            extract_entities: Whether to extract named entities
            extract_keywords: Whether to extract keywords
            generate_summary: Whether to generate summary
            analyze_sentiment: Whether to analyze sentiment
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            NLPResult: Processing results
            
        Raises:
            ValueError: If text is empty or NLP model not available
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if self.nlp is None:
            raise ValueError("NLP model not available")
        
        self.logger.info(
            "Starting NLP processing",
            text_length=len(text),
            extract_entities=extract_entities,
            extract_keywords=extract_keywords
        )
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract entities
            entities = []
            if extract_entities:
                entities = self._extract_entities(doc)
            
            # Extract keywords
            keywords = []
            if extract_keywords:
                keywords = self._extract_keywords(doc, max_keywords)
            
            # Generate summary
            summary = None
            if generate_summary:
                summary = self._generate_summary(doc)
            
            # Analyze sentiment
            sentiment = None
            if analyze_sentiment:
                sentiment = self._analyze_sentiment(doc)
            
            # Detect language
            language = self._detect_language(doc)
            
            self.logger.info(
                "NLP processing completed",
                entities_count=len(entities),
                keywords_count=len(keywords),
                language=language
            )
            
            return NLPResult(
                text=text,
                entities=entities,
                keywords=keywords,
                summary=summary,
                sentiment=sentiment,
                language=language
            )
            
        except Exception as e:
            self.logger.error(f"NLP processing failed: {e}")
            raise
    
    def _extract_entities(self, doc) -> List[Dict]:
        """
        Extract named entities from spaCy doc.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'score', 1.0)  # Default confidence if not available
            })
        
        # Also extract pattern-based entities
        pattern_entities = self._extract_pattern_entities(doc.text)
        entities.extend(pattern_entities)
        
        # Remove duplicates and sort by start position
        unique_entities = []
        seen_spans = set()
        
        for entity in sorted(entities, key=lambda x: x['start']):
            span = (entity['start'], entity['end'])
            if span not in seen_spans:
                unique_entities.append(entity)
                seen_spans.add(span)
        
        return unique_entities
    
    def _extract_pattern_entities(self, text: str) -> List[Dict]:
        """
        Extract entities using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of pattern-based entities
        """
        entities = []
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': pattern_name.upper(),
                    'description': f'{pattern_name.title()} pattern match',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8  # Pattern matches have lower confidence
                })
        
        return entities
    
    def _extract_keywords(self, doc, max_keywords: int) -> List[str]:
        """
        Extract keywords from spaCy doc.
        
        Args:
            doc: spaCy Doc object
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        # Filter tokens: remove stop words, punctuation, spaces, and short words
        keywords = []
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and 
                len(token.text) > 2 and
                token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']):
                
                # Use lemmatized form
                keyword = token.lemma_.lower()
                if keyword not in keywords:
                    keywords.append(keyword)
        
        # Get noun phrases as additional keywords
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3:  # Only longer phrases
                phrase = chunk.text.lower()
                if phrase not in keywords:
                    keywords.append(phrase)
        
        # Count frequency and return most common
        word_freq = Counter(keywords)
        top_keywords = [word for word, freq in word_freq.most_common(max_keywords)]
        
        return top_keywords
    
    def _generate_summary(self, doc, max_sentences: int = 3) -> Optional[str]:
        """
        Generate extractive summary from spaCy doc.
        
        Args:
            doc: spaCy Doc object
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Summary text or None if not possible
        """
        try:
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            
            if len(sentences) <= max_sentences:
                return ' '.join(sentences)
            
            # Simple extractive summarization based on sentence position and length
            # In a production system, you might use more sophisticated algorithms
            
            # Score sentences (prefer longer sentences and those appearing earlier)
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                # Position score (earlier sentences get higher scores)
                position_score = 1.0 - (i / len(sentences)) * 0.5
                
                # Length score (normalized)
                length_score = min(len(sentence.split()) / 20.0, 1.0)
                
                total_score = position_score + length_score
                sentence_scores.append((sentence, total_score))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent for sent, score in sentence_scores[:max_sentences]]
            
            # Maintain original order
            summary_sentences = []
            for sentence in sentences:
                if sentence in top_sentences:
                    summary_sentences.append(sentence)
                    if len(summary_sentences) >= max_sentences:
                        break
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            self.logger.warning(f"Summary generation failed: {e}")
            return None
    
    def _analyze_sentiment(self, doc) -> Optional[str]:
        """
        Analyze sentiment of the text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Sentiment label (positive, negative, neutral) or None
        """
        try:
            # Simple rule-based sentiment analysis
            # In production, you might use a dedicated sentiment model
            
            positive_words = {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied',
                'success', 'successful', 'perfect', 'outstanding', 'brilliant'
            }
            
            negative_words = {
                'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
                'angry', 'frustrated', 'disappointed', 'unsatisfied', 'poor',
                'fail', 'failure', 'wrong', 'error', 'problem', 'issue'
            }
            
            positive_count = 0
            negative_count = 0
            
            for token in doc:
                word = token.lemma_.lower()
                if word in positive_words:
                    positive_count += 1
                elif word in negative_words:
                    negative_count += 1
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return None
    
    def _detect_language(self, doc) -> Optional[str]:
        """
        Detect language of the text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Language code or None
        """
        try:
            # Simple language detection based on spaCy model
            # The loaded model is for English, so we return 'en'
            # In production, you might use langdetect or similar
            return 'en'
            
        except Exception as e:
            self.logger.warning(f"Language detection failed: {e}")
            return None
    
    def extract_structured_data(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from text based on a schema.
        
        Args:
            text: Input text
            schema: Schema defining what to extract
            
        Returns:
            Dictionary of extracted structured data
        """
        if self.nlp is None:
            raise ValueError("NLP model not available")
        
        doc = self.nlp(text)
        extracted_data = {}
        
        try:
            # Process schema fields
            for field_name, field_config in schema.items():
                field_type = field_config.get('type', 'text')
                pattern = field_config.get('pattern')
                entity_types = field_config.get('entity_types', [])
                
                if pattern:
                    # Extract using regex pattern
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    extracted_data[field_name] = matches[0] if matches else None
                
                elif entity_types:
                    # Extract using named entity recognition
                    entities = [ent.text for ent in doc.ents if ent.label_ in entity_types]
                    extracted_data[field_name] = entities[0] if entities else None
                
                elif field_type == 'date':
                    # Extract dates
                    date_matches = re.findall(self.patterns['date'], text)
                    extracted_data[field_name] = date_matches[0] if date_matches else None
                
                elif field_type == 'currency':
                    # Extract currency amounts
                    currency_matches = re.findall(self.patterns['currency'], text)
                    extracted_data[field_name] = currency_matches[0] if currency_matches else None
                
                elif field_type == 'email':
                    # Extract emails
                    email_matches = re.findall(self.patterns['email'], text)
                    extracted_data[field_name] = email_matches[0] if email_matches else None
                
                elif field_type == 'phone':
                    # Extract phone numbers
                    phone_matches = re.findall(self.patterns['phone'], text)
                    extracted_data[field_name] = phone_matches[0] if phone_matches else None
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Structured data extraction failed: {e}")
            return {}
    
    def classify_document_type(self, text: str) -> Tuple[str, float]:
        """
        Classify document type based on text content.
        
        Args:
            text: Document text
            
        Returns:
            Tuple of (document_type, confidence)
        """
        if not text:
            return 'other', 0.0
        
        text_lower = text.lower()
        
        # Simple rule-based classification
        # In production, you might use a trained ML model
        
        classification_rules = {
            'invoice': ['invoice', 'bill', 'amount due', 'payment terms', 'invoice number'],
            'receipt': ['receipt', 'total', 'paid', 'change', 'thank you'],
            'contract': ['agreement', 'contract', 'terms and conditions', 'party', 'signature'],
            'email': ['from:', 'to:', 'subject:', 'dear', '@'],
            'report': ['report', 'analysis', 'summary', 'findings', 'conclusion'],
            'form': ['form', 'application', 'please fill', 'name:', 'date:']
        }
        
        scores = {}
        for doc_type, keywords in classification_rules.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[doc_type] = score / len(keywords)  # Normalize by number of keywords
        
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = scores[best_type]
            return best_type, confidence
        
        return 'other', 0.0
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform NLP service health check.
        
        Returns:
            Health check results
        """
        health = {
            "service": "NLP",
            "status": "healthy",
            "model_loaded": self.nlp is not None,
            "model_name": None,
            "model_version": None,
            "errors": []
        }
        
        try:
            if self.nlp:
                health["model_name"] = self.nlp.meta.get("name", "unknown")
                health["model_version"] = self.nlp.meta.get("version", "unknown")
                
                # Test basic functionality
                test_doc = self.nlp("This is a test sentence.")
                if len(test_doc) == 0:
                    health["status"] = "unhealthy"
                    health["errors"].append("Model not processing text correctly")
            else:
                health["status"] = "unhealthy"
                health["errors"].append("NLP model not loaded")
                
        except Exception as e:
            health["status"] = "unhealthy"
            health["errors"].append(f"Health check failed: {str(e)}")
        
        return health


# Global NLP service instance
nlp_service = NLPService()