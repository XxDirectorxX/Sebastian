"""
Structured knowledge management system for Sebastian assistant.

Provides unified access to various knowledge sources with semantic 
search capabilities and confidence scoring.
"""
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import asyncio
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Structured knowledge management system.
    
    Features:
    - Vector-based semantic search
    - Knowledge categorization and organization
    - Dynamic knowledge updates
    - Confidence scoring for retrieved information
    - Integration with multiple knowledge sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize knowledge base with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Knowledge directories
        self.base_dir = Path(self.config.get("base_dir", "data/knowledge"))
        self.static_kb_dir = self.base_dir / "static"
        self.dynamic_kb_dir = self.base_dir / "dynamic"
        self.personal_kb_dir = self.base_dir / "personal"
        
        # Create directories if they don't exist
        for directory in [self.base_dir, self.static_kb_dir, self.dynamic_kb_dir, self.personal_kb_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Knowledge categories
        self.categories = [
            "butler_duties",
            "household_management", 
            "etiquette",
            "cooking",
            "history",
            "user_preferences",
            "master_details",
            "security",
            "schedule",
            "general"
        ]
        
        # Confidence thresholds
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.fallback_threshold = self.config.get("fallback_threshold", 0.5)
        
        # Search parameters
        self.max_results = self.config.get("max_results", 5)
        
        # Vector store for semantic search
        self._initialize_vector_store()
        
        # Load knowledge bases
        self._load_knowledge()
        
        logger.info("Knowledge base initialized")
        
    def _initialize_vector_store(self):
        """Initialize vector store for semantic search."""
        vector_engine = self.config.get("vector_engine", "faiss")
        
        try:
            # Initialize vector store based on configuration
            if vector_engine == "faiss":
                try:
                    import faiss
                    self.vector_engine = "faiss"
                    logger.info("Using FAISS for vector storage")
                except ImportError:
                    logger.warning("FAISS not available, falling back to numpy")
                    self.vector_engine = "numpy"
            else:
                self.vector_engine = "numpy"
                logger.info("Using numpy for vector storage")
                
            # Initialize sentence transformer for embeddings
            try:
                from sentence_transformers import SentenceTransformer
                
                model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
                self.embedding_model = SentenceTransformer(model_name)
                self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded: {model_name} ({self.vector_size} dimensions)")
                
            except ImportError:
                logger.warning("Sentence transformers not available, using fallback embedding")
                self.embedding_model = None
                self.vector_size = 384  # Default dimension
                
            # Initialize vector index
            self._initialize_vector_index()
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
            # Set fallback mode
            self.vector_engine = "fallback"
            self.vector_size = 0
            
    def _initialize_vector_index(self):
        """Initialize vector index for semantic search."""
        if self.vector_engine == "faiss":
            import faiss
            # Create FAISS index
            self.index = faiss.IndexFlatL2(self.vector_size)
            self.vectors = []
            self.knowledge_entries = []
        elif self.vector_engine == "numpy":
            # Use numpy arrays
            self.vectors = np.empty((0, self.vector_size), dtype=np.float32)
            self.knowledge_entries = []
        else:
            # Fallback mode - no vectors
            self.vectors = None
            self.knowledge_entries = []
            
    def _load_knowledge(self):
        """Load knowledge from all sources."""
        # Track loading metrics
        start_time = time.time()
        entries_loaded = 0
        
        # Load static knowledge base (core facts that don't change)
        static_entries = self._load_knowledge_dir(self.static_kb_dir)
        entries_loaded += len(static_entries)
        
        # Load dynamic knowledge (facts that update periodically)
        dynamic_entries = self._load_knowledge_dir(self.dynamic_kb_dir)
        entries_loaded += len(dynamic_entries)
        
        # Load personal knowledge (user-specific information)
        personal_entries = self._load_knowledge_dir(self.personal_kb_dir)
        entries_loaded += len(personal_entries)
        
        # Add all entries to the index
        all_entries = static_entries + dynamic_entries + personal_entries
        
        if all_entries:
            self._add_entries_to_index(all_entries)
            
        load_time = time.time() - start_time
        logger.info(f"Loaded {entries_loaded} knowledge entries in {load_time:.2f}s")
        
    def _load_knowledge_dir(self, directory: Path) -> List[Dict[str, Any]]:
        """Load knowledge from a directory."""
        entries = []
        
        if not directory.exists():
            return entries
            
        # Process all JSON files in the directory
        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle different file formats
                if isinstance(data, list):
                    # List of entries
                    for entry in data:
                        if self._validate_entry(entry):
                            # Add source info
                            entry["source"] = str(file_path)
                            entries.append(entry)
                elif isinstance(data, dict):
                    # Single entry or dict of entries
                    if "content" in data:
                        # Single entry
                        if self._validate_entry(data):
                            data["source"] = str(file_path)
                            entries.append(data)
                    else:
                        # Dict of entries
                        for key, entry in data.items():
                            if isinstance(entry, dict) and self._validate_entry(entry):
                                entry["source"] = str(file_path)
                                entry["id"] = key
                                entries.append(entry)
                                
            except Exception as e:
                logger.error(f"Error loading knowledge file {file_path}: {e}")
                
        return entries
        
    def _validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate knowledge entry format."""
        # Must have content
        if "content" not in entry:
            return False
            
        # Category must be valid if specified
        if "category" in entry and entry["category"] not in self.categories:
            entry["category"] = "general"  # Default to general
            
        # Ensure required fields
        if "id" not in entry:
            entry["id"] = f"kb_{len(self.knowledge_entries)}"
            
        if "category" not in entry:
            entry["category"] = "general"
            
        if "confidence" not in entry:
            entry["confidence"] = 1.0
            
        return True
        
    def _add_entries_to_index(self, entries: List[Dict[str, Any]]):
        """Add knowledge entries to vector index."""
        if not entries:
            return
            
        # Generate embeddings for new entries
        texts = [entry["content"] for entry in entries]
        
        try:
            if self.embedding_model:
                # Use sentence transformer model
                embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            else:
                # Fallback: random embeddings (for testing only)
                embeddings = np.random.random((len(texts), self.vector_size)).astype(np.float32)
                logger.warning("Using random embeddings as fallback - search will be unreliable")
                
            # Add to appropriate index
            if self.vector_engine == "faiss":
                self.index.add(embeddings.astype(np.float32))
                self.vectors.extend(embeddings)
                self.knowledge_entries.extend(entries)
            elif self.vector_engine == "numpy":
                self.vectors = np.vstack([self.vectors, embeddings])
                self.knowledge_entries.extend(entries)
            else:
                # Fallback mode - just store entries without vectors
                self.knowledge_entries.extend(entries)
                
        except Exception as e:
            logger.error(f"Error adding entries to index: {e}", exc_info=True)
            # Store entries without embeddings as fallback
            self.knowledge_entries.extend(entries)
            
    async def query(self, query: str, category: Optional[str] = None, 
                  max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query the knowledge base asynchronously.
        
        Args:
            query: Query text
            category: Optional category to search in
            max_results: Maximum number of results to return
            
        Returns:
            List of matching knowledge entries with scores
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.query_sync(query, category, max_results)
        )
        
    def query_sync(self, query: str, category: Optional[str] = None,
                 max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query the knowledge base synchronously.
        
        Args:
            query: Query text
            category: Optional category to search in
            max_results: Maximum number of results to return
            
        Returns:
            List of matching knowledge entries with scores
        """
        if not query:
            return []
            
        if max_results is None:
            max_results = self.max_results
            
        start_time = time.time()
        
        try:
            # Generate query embedding
            if self.embedding_model:
                query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
                
                if self.vector_engine == "faiss":
                    # FAISS search
                    import faiss
                    distances, indices = self.index.search(
                        np.array([query_embedding], dtype=np.float32), 
                        len(self.knowledge_entries)  # Search all entries
                    )
                    results = []
                    
                    for i, idx in enumerate(indices[0]):
                        if idx != -1 and idx < len(self.knowledge_entries):
                            entry = self.knowledge_entries[idx].copy()
                            # Convert distance to similarity score (0-1)
                            # FAISS uses L2 distance, smaller is better
                            score = max(0, 1 - (distances[0][i] / 10))  # Normalize
                            
                            # Skip entries below threshold
                            if score < self.fallback_threshold:
                                continue
                                
                            # Filter by category if specified
                            if category and entry.get("category") != category:
                                continue
                                
                            entry["score"] = score
                            results.append(entry)
                    
                elif self.vector_engine == "numpy":
                    # Numpy-based search
                    if len(self.vectors) > 0:
                        # Calculate cosine similarity
                        similarities = np.dot(self.vectors, query_embedding) / (
                            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_embedding)
                        )
                        
                        # Get indices sorted by similarity (highest first)
                        sorted_indices = np.argsort(-similarities)
                        
                        results = []
                        for idx in sorted_indices:
                            if idx < len(self.knowledge_entries):
                                entry = self.knowledge_entries[idx].copy()
                                score = float(similarities[idx])
                                
                                # Skip entries below threshold
                                if score < self.fallback_threshold:
                                    continue
                                    
                                # Filter by category if specified
                                if category and entry.get("category") != category:
                                    continue
                                    
                                entry["score"] = score
                                results.append(entry)
                    else:
                        results = []
                        
                else:
                    # Fallback: keyword search
                    results = self._fallback_keyword_search(query, category)
                    
            else:
                # Fallback: keyword search
                results = self._fallback_keyword_search(query, category)
                
            # Limit results
            results = results[:max_results]
            
            process_time = time.time() - start_time
            logger.info(f"Knowledge query completed in {process_time:.3f}s with {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during knowledge query: {e}", exc_info=True)
            return self._fallback_keyword_search(query, category)[:max_results]
            
    def _fallback_keyword_search(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform fallback keyword-based search."""
        query_terms = query.lower().split()
        results = []
        
        for entry in self.knowledge_entries:
            # Filter by category if specified
            if category and entry.get("category") != category:
                continue
                
            # Simple term matching
            content = entry["content"].lower()
            matches = sum(term in content for term in query_terms)
            score = matches / len(query_terms) if query_terms else 0
            
            # Add entry if score above threshold
            if score >= self.fallback_threshold:
                result = entry.copy()
                result["score"] = score
                results.append(result)
                
        # Sort by score descending
        return sorted(results, key=lambda x: x["score"], reverse=True)
        
    async def add_knowledge_entry(self, entry: Dict[str, Any], 
                                dynamic: bool = True,
                                personal: bool = False) -> bool:
        """
        Add a new knowledge entry asynchronously.
        
        Args:
            entry: Knowledge entry to add
            dynamic: Whether entry is dynamic (vs static)
            personal: Whether entry is personal
            
        Returns:
            Success status
        """
        if not self._validate_entry(entry):
            return False
            
        # Determine target directory
        if personal:
            target_dir = self.personal_kb_dir
            file_base = "personal_knowledge"
        elif dynamic:
            target_dir = self.dynamic_kb_dir
            file_base = "dynamic_knowledge"
        else:
            target_dir = self.static_kb_dir
            file_base = "static_knowledge"
            
        # Generate filename if needed
        if "id" in entry:
            file_name = f"{file_base}_{entry['id']}.json"
        else:
            entry["id"] = f"kb_{len(self.knowledge_entries)}"
            file_name = f"{file_base}_{int(time.time())}.json"
            
        try:
            # Add source information
            file_path = target_dir / file_name
            entry["source"] = str(file_path)
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(entry, f, indent=2)
                
            # Add to index
            self._add_entries_to_index([entry])
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge entry: {e}", exc_info=True)
            return False
            
    async def update_knowledge_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing knowledge entry asynchronously.
        
        Args:
            entry_id: ID of entry to update
            updates: Updates to apply
            
        Returns:
            Success status
        """
        try:
            # Find the entry
            found = False
            for i, entry in enumerate(self.knowledge_entries):
                if entry.get("id") == entry_id:
                    found = True
                    # Update entry
                    updated_entry = entry.copy()
                    updated_entry.update(updates)
                    
                    # Re-validate
                    if not self._validate_entry(updated_entry):
                        return False
                        
                    # Update the entry in memory
                    self.knowledge_entries[i] = updated_entry
                    
                    # Update the file
                    file_path = updated_entry.get("source")
                    if file_path and os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                            except:
                                data = updated_entry
                                
                        # Handle different file formats
                        if isinstance(data, list):
                            # Find entry in list
                            for j, item in enumerate(data):
                                if item.get("id") == entry_id:
                                    data[j] = updated_entry
                                    break
                        elif isinstance(data, dict):
                            if "content" in data:
                                # Single entry file
                                data = updated_entry
                            else:
                                # Dict of entries
                                data[entry_id] = updated_entry
                                
                        # Write updated file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                            
                    # Update vector index if content changed
                    if "content" in updates and self.embedding_model:
                        # Generate new embedding
                        embedding = self.embedding_model.encode(
                            updated_entry["content"], 
                            convert_to_numpy=True
                        )
                        
                        # Update in the appropriate index
                        if self.vector_engine == "faiss":
                            # FAISS doesn't support updates, would need to rebuild
                            # This is a simple approximation - replace the vector in our list
                            self.vectors[i] = embedding
                        elif self.vector_engine == "numpy":
                            self.vectors[i] = embedding
                            
                    break
                    
            return found
            
        except Exception as e:
            logger.error(f"Error updating knowledge entry: {e}", exc_info=True)
            return False
            
    async def get_knowledge_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all knowledge entries in a category asynchronously.
        
        Args:
            category: Category to retrieve
            
        Returns:
            List of knowledge entries in the category
        """
        if category not in self.categories:
            return []
            
        results = [
            entry.copy() for entry in self.knowledge_entries 
            if entry.get("category") == category
        ]
        
        return results
        
    async def get_butler_knowledge(self) -> Dict[str, Any]:
        """
        Get Sebastian butler-specific knowledge asynchronously.
        
        Returns:
            Butler-related knowledge entries organized by area
        """
        butler_categories = ["butler_duties", "etiquette", "household_management"]
        
        results = {}
        for category in butler_categories:
            entries = await self.get_knowledge_by_category(category)
            results[category] = entries
            
        return results
        
    async def search_multiple_categories(self, query: str, 
                                       categories: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search multiple categories asynchronously.
        
        Args:
            query: Search query
            categories: Categories to search
            
        Returns:
            Dictionary with results for each category
        """
        results = {}
        
        # Search each category
        for category in categories:
            if category in self.categories:
                category_results = await self.query(query, category=category)
                results[category] = category_results
                
        return results
        
    async def rebuild_index(self) -> bool:
        """
        Rebuild the vector index asynchronously.
        
        Returns:
            Success status
        """
        try:
            # Save entries
            entries = self.knowledge_entries.copy()
            
            # Reinitialize index
            self._initialize_vector_index()
            
            # Reload entries
            self._add_entries_to_index(entries)
            
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}", exc_info=True)
            return False