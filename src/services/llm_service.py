"""
Large Language Model service for ClerkAI.
"""

from typing import Dict, List, Optional, Any, Union
import json
import asyncio
from datetime import datetime
import openai
from openai import AsyncOpenAI
import logging

from config.settings import settings
from config.logging import LoggerMixin, log_performance

logger = logging.getLogger(__name__)


class LLMResponse:
    """LLM response container."""
    
    def __init__(
        self,
        content: str,
        model: str,
        usage: Optional[Dict] = None,
        finish_reason: Optional[str] = None,
        processing_time: Optional[float] = None
    ):
        self.content = content
        self.model = model
        self.usage = usage
        self.finish_reason = finish_reason
        self.processing_time = processing_time
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "processing_time": self.processing_time
        }


class LLMService(LoggerMixin):
    """Large Language Model service for text generation and analysis."""
    
    def __init__(self):
        """Initialize LLM service."""
        self.client = None
        self.async_client = None
        self._initialize_client()
        
        # Common system prompts
        self.system_prompts = {
            'document_analysis': """You are an AI assistant specialized in document analysis. 
            Analyze the provided text and extract relevant information accurately. 
            Be concise and factual in your responses.""",
            
            'data_extraction': """You are an AI assistant specialized in structured data extraction. 
            Extract the requested information from the provided text and format it as specified. 
            If information is not available, return null or indicate it's missing.""",
            
            'summarization': """You are an AI assistant specialized in text summarization. 
            Create concise, informative summaries that capture the key points of the text. 
            Focus on the most important information.""",
            
            'classification': """You are an AI assistant specialized in document classification. 
            Classify documents based on their content and provide confidence scores. 
            Be accurate and explain your reasoning.""",
            
            'question_answering': """You are an AI assistant that answers questions based on provided context. 
            Only answer based on the information given. If you cannot answer from the context, 
            say so clearly."""
        }
        
        self.logger.info("LLM service initialized")
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        if settings.openai_api_key:
            try:
                self.client = openai.OpenAI(api_key=settings.openai_api_key)
                self.async_client = AsyncOpenAI(api_key=settings.openai_api_key)
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            self.logger.warning("OpenAI API key not provided")
    
    @log_performance("llm_generate_text")
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate text using LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt to set context
            model: Model to use (defaults to settings)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Response format ('json' or None)
            
        Returns:
            LLMResponse: Generated text and metadata
            
        Raises:
            ValueError: If client not initialized or prompt empty
            Exception: If API call fails
        """
        if not self.async_client:
            raise ValueError("LLM client not initialized")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        model = model or settings.openai_model
        max_tokens = max_tokens or settings.max_tokens
        temperature = temperature or settings.temperature
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        self.logger.info(
            "Generating text with LLM",
            model=model,
            prompt_length=len(prompt),
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            
            response = await self.async_client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            self.logger.info(
                "Text generation completed",
                response_length=len(content),
                tokens_used=usage["total_tokens"]
            )
            
            return LLMResponse(
                content=content,
                model=model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise
    
    async def analyze_document(
        self,
        text: str,
        analysis_type: str = "general",
        specific_questions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze document content using LLM.
        
        Args:
            text: Document text to analyze
            analysis_type: Type of analysis (general, financial, legal, etc.)
            specific_questions: Specific questions to answer about the document
            
        Returns:
            Dictionary with analysis results
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        prompt = f"""Analyze the following document text:

{text}

Please provide a comprehensive analysis including:
1. Document type and purpose
2. Key information and entities
3. Main topics or themes
4. Important dates, numbers, or amounts
5. Any notable patterns or insights

Analysis type: {analysis_type}
"""
        
        if specific_questions:
            prompt += f"\nAlso answer these specific questions:\n"
            for i, question in enumerate(specific_questions, 1):
                prompt += f"{i}. {question}\n"
        
        prompt += "\nProvide your analysis in JSON format with clear categories."
        
        response = await self.generate_text(
            prompt=prompt,
            system_prompt=self.system_prompts['document_analysis'],
            response_format="json"
        )
        
        try:
            analysis_result = json.loads(response.content)
            return analysis_result
        except json.JSONDecodeError:
            # Fallback to text response if JSON parsing fails
            return {"analysis": response.content}
    
    async def extract_structured_data(
        self,
        text: str,
        schema: Dict[str, Any],
        examples: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using LLM.
        
        Args:
            text: Text to extract data from
            schema: Schema defining the expected structure
            examples: Example extractions to guide the model
            
        Returns:
            Dictionary with extracted structured data
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        prompt = f"""Extract structured data from the following text according to the provided schema:

Text:
{text}

Schema:
{json.dumps(schema, indent=2)}
"""
        
        if examples:
            prompt += f"\nExamples:\n{json.dumps(examples, indent=2)}\n"
        
        prompt += """
Extract the data and return it in JSON format matching the schema. 
If a field cannot be found in the text, use null for the value.
"""
        
        response = await self.generate_text(
            prompt=prompt,
            system_prompt=self.system_prompts['data_extraction'],
            response_format="json"
        )
        
        try:
            extracted_data = json.loads(response.content)
            return extracted_data
        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON response from LLM")
            return {}
    
    async def summarize_text(
        self,
        text: str,
        summary_type: str = "general",
        max_length: Optional[int] = None,
        key_points: Optional[int] = None
    ) -> str:
        """
        Summarize text using LLM.
        
        Args:
            text: Text to summarize
            summary_type: Type of summary (general, executive, technical, etc.)
            max_length: Maximum length of summary in words
            key_points: Number of key points to extract
            
        Returns:
            Summary text
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        prompt = f"""Summarize the following text:

{text}

Summary requirements:
- Type: {summary_type}
"""
        
        if max_length:
            prompt += f"- Maximum length: {max_length} words\n"
        
        if key_points:
            prompt += f"- Include {key_points} key points\n"
        
        prompt += "\nProvide a clear, concise summary that captures the essential information."
        
        response = await self.generate_text(
            prompt=prompt,
            system_prompt=self.system_prompts['summarization']
        )
        
        return response.content
    
    async def classify_document(
        self,
        text: str,
        categories: List[str],
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Classify document using LLM.
        
        Args:
            text: Document text to classify
            categories: List of possible categories
            include_confidence: Whether to include confidence scores
            
        Returns:
            Classification result with category and confidence
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        if not categories:
            raise ValueError("Categories cannot be empty")
        
        prompt = f"""Classify the following document into one of these categories:

Categories: {', '.join(categories)}

Document text:
{text}

"""
        
        if include_confidence:
            prompt += """Provide your classification in JSON format with:
- category: the selected category
- confidence: confidence score from 0.0 to 1.0
- reasoning: brief explanation of your decision
"""
        else:
            prompt += "Provide only the category name."
        
        response_format = "json" if include_confidence else None
        
        response = await self.generate_text(
            prompt=prompt,
            system_prompt=self.system_prompts['classification'],
            response_format=response_format
        )
        
        if include_confidence:
            try:
                classification_result = json.loads(response.content)
                return classification_result
            except json.JSONDecodeError:
                return {
                    "category": response.content.strip(),
                    "confidence": 0.5,
                    "reasoning": "Could not parse detailed response"
                }
        else:
            return {"category": response.content.strip()}
    
    async def answer_question(
        self,
        question: str,
        context: str,
        include_citations: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question based on provided context.
        
        Args:
            question: Question to answer
            context: Context text to base answer on
            include_citations: Whether to include text citations
            
        Returns:
            Answer with metadata
        """
        if not question or not context:
            raise ValueError("Question and context cannot be empty")
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

"""
        
        if include_citations:
            prompt += """Provide your answer in JSON format with:
- answer: the answer to the question
- confidence: confidence score from 0.0 to 1.0
- citations: relevant quotes from the context that support your answer
- reasoning: explanation of how you arrived at the answer
"""
        else:
            prompt += "Provide a clear, direct answer based only on the information in the context."
        
        response_format = "json" if include_citations else None
        
        response = await self.generate_text(
            prompt=prompt,
            system_prompt=self.system_prompts['question_answering'],
            response_format=response_format
        )
        
        if include_citations:
            try:
                qa_result = json.loads(response.content)
                return qa_result
            except json.JSONDecodeError:
                return {
                    "answer": response.content.strip(),
                    "confidence": 0.5,
                    "citations": [],
                    "reasoning": "Could not parse detailed response"
                }
        else:
            return {"answer": response.content.strip()}
    
    async def batch_process(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple LLM requests concurrently.
        
        Args:
            requests: List of request dictionaries
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_request(request: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    method = request.get('method', 'generate_text')
                    args = request.get('args', {})
                    
                    if method == 'generate_text':
                        response = await self.generate_text(**args)
                        return {"success": True, "response": response.to_dict()}
                    elif method == 'analyze_document':
                        response = await self.analyze_document(**args)
                        return {"success": True, "response": response}
                    elif method == 'extract_structured_data':
                        response = await self.extract_structured_data(**args)
                        return {"success": True, "response": response}
                    elif method == 'summarize_text':
                        response = await self.summarize_text(**args)
                        return {"success": True, "response": response}
                    elif method == 'classify_document':
                        response = await self.classify_document(**args)
                        return {"success": True, "response": response}
                    elif method == 'answer_question':
                        response = await self.answer_question(**args)
                        return {"success": True, "response": response}
                    else:
                        return {"success": False, "error": f"Unknown method: {method}"}
                        
                except Exception as e:
                    return {"success": False, "error": str(e)}
        
        tasks = [process_single_request(request) for request in requests]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform LLM service health check.
        
        Returns:
            Health check results
        """
        health = {
            "service": "LLM",
            "status": "healthy",
            "client_initialized": self.client is not None,
            "api_key_configured": bool(settings.openai_api_key),
            "model": settings.openai_model,
            "errors": []
        }
        
        if not settings.openai_api_key:
            health["status"] = "unhealthy"
            health["errors"].append("OpenAI API key not configured")
        
        if not self.client:
            health["status"] = "unhealthy"
            health["errors"].append("OpenAI client not initialized")
        
        # Test API connectivity (optional, can be expensive)
        # This would require an actual API call
        
        return health


# Global LLM service instance
llm_service = LLMService()