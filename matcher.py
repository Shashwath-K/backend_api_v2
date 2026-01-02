import numpy as np
import psycopg2
from psycopg2.extras import Json
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime
from psycopg2.extensions import connection, cursor

class FaceMatcher:
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize PostgreSQL connection for face matching
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config
        self.connection: Optional[connection] = None
        self.cursor: Optional[cursor] = None
        self.connect()
        self.create_table()
    
    def connect(self) -> None:
        """Establish PostgreSQL connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("Connected to PostgreSQL database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def check_connection(self) -> None:
        """Check if connection and cursor are available"""
        if self.connection is None or self.cursor is None:
            raise ConnectionError("Database connection not established")
    
    def create_table(self) -> None:
        """Create faces table if not exists"""
        self.check_connection()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            person_id VARCHAR(100) NOT NULL,
            person_name VARCHAR(255),
            embedding vector(256) NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_person_id ON face_embeddings(person_id);
        CREATE INDEX IF NOT EXISTS idx_created_at ON face_embeddings(created_at);
        CREATE INDEX IF NOT EXISTS idx_embedding ON face_embeddings 
        USING ivfflat (embedding vector_cosine_ops);
        """
        
        try:
            self.cursor.execute(create_table_query)  # type: ignore
            self.connection.commit()  # type: ignore
            print("Table created or already exists")
        except Exception as e:
            print(f"Error creating table: {e}")
            self.connection.rollback()  # type: ignore
    
    def store_embedding(self, person_id: str, person_name: str, 
                       embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store a face embedding in the database"""
        self.check_connection()
        
        # Convert numpy array to list for PostgreSQL
        embedding_list = embedding.tolist()
        
        insert_query = """
        INSERT INTO face_embeddings 
        (person_id, person_name, embedding, metadata, updated_at)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        try:
            self.cursor.execute(  # type: ignore
                insert_query,
                (person_id, person_name, embedding_list, 
                 Json(metadata or {}), datetime.now())
            )
            record_id = self.cursor.fetchone()[0]  # type: ignore
            self.connection.commit()  # type: ignore
            return record_id
        except Exception as e:
            print(f"Error storing embedding: {e}")
            self.connection.rollback()  # type: ignore
            raise
    
    def cosine_similarity(self, vec1: Union[List[float], np.ndarray], 
                         vec2: Union[List[float], np.ndarray]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_matches_bruteforce(self, embedding: np.ndarray, threshold: float = 0.6,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Find matches using brute-force comparison (for small datasets)"""
        self.check_connection()
        
        select_query = """
        SELECT id, person_id, person_name, embedding, metadata 
        FROM face_embeddings;
        """
        
        self.cursor.execute(select_query)  # type: ignore
        records = self.cursor.fetchall()  # type: ignore
        
        matches = []
        embedding_list = embedding.tolist()
        
        for record in records:
            db_id, person_id, person_name, db_embedding, metadata = record
            similarity = self.cosine_similarity(embedding_list, db_embedding)
            
            if similarity >= threshold:
                matches.append({
                    'id': db_id,
                    'person_id': person_id,
                    'person_name': person_name,
                    'similarity': similarity,
                    'metadata': metadata
                })
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:limit]
    
    def find_matches_vector(self, embedding: np.ndarray, threshold: float = 0.6,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Find matches using PostgreSQL vector similarity search"""
        self.check_connection()
        
        embedding_list = embedding.tolist()
        
        # Using PostgreSQL vector cosine similarity operator
        # Note: This requires the pgvector extension
        similarity_query = """
        SELECT id, person_id, person_name, metadata,
               (embedding <=> %s) as distance
        FROM face_embeddings
        WHERE (embedding <=> %s) <= %s
        ORDER BY distance
        LIMIT %s;
        """
        
        # Convert similarity threshold to distance
        # For cosine similarity: distance = 1 - similarity
        max_distance = 1.0 - threshold
        
        try:
            self.cursor.execute(  # type: ignore
                similarity_query,
                (embedding_list, embedding_list, max_distance, limit)
            )
            records = self.cursor.fetchall()  # type: ignore
            
            matches = []
            for record in records:
                db_id, person_id, person_name, metadata, distance = record
                similarity = 1.0 - distance  # Convert distance back to similarity
                matches.append({
                    'id': db_id,
                    'person_id': person_id,
                    'person_name': person_name,
                    'similarity': similarity,
                    'distance': distance,
                    'metadata': metadata
                })
            
            return matches
        except Exception as e:
            print(f"Error in vector similarity search: {e}")
            # Fall back to brute-force method
            print("Falling back to brute-force search...")
            return self.find_matches_bruteforce(embedding, threshold, limit)
    
    def find_matches(self, embedding: np.ndarray, threshold: float = 0.6,
                    limit: int = 10, use_vector_search: bool = True) -> List[Dict[str, Any]]:
        """
        Find matches with optional vector search
        
        Args:
            embedding: Face embedding vector
            threshold: Similarity threshold (0.0 to 1.0)
            limit: Maximum number of matches to return
            use_vector_search: Use PostgreSQL vector search if available
        """
        if use_vector_search:
            return self.find_matches_vector(embedding, threshold, limit)
        else:
            return self.find_matches_bruteforce(embedding, threshold, limit)
    
    def delete_person(self, person_id: str) -> bool:
        """Delete all embeddings for a person"""
        self.check_connection()
        
        delete_query = "DELETE FROM face_embeddings WHERE person_id = %s;"
        
        try:
            self.cursor.execute(delete_query, (person_id,))  # type: ignore
            self.connection.commit()  # type: ignore
            return self.cursor.rowcount > 0  # type: ignore
        except Exception as e:
            print(f"Error deleting person: {e}")
            self.connection.rollback()  # type: ignore
            return False
    
    def get_person_count(self) -> int:
        """Get total number of distinct persons in database"""
        self.check_connection()
        
        count_query = "SELECT COUNT(DISTINCT person_id) FROM face_embeddings;"
        
        try:
            self.cursor.execute(count_query)  # type: ignore
            result = self.cursor.fetchone()  # type: ignore
            return result[0] if result else 0
        except Exception as e:
            print(f"Error getting person count: {e}")
            return 0
    
    def get_all_persons(self) -> List[Dict[str, Any]]:
        """Get all unique persons from database"""
        self.check_connection()
        
        query = """
        SELECT DISTINCT person_id, person_name, 
               COUNT(*) as embedding_count,
               MAX(created_at) as last_updated
        FROM face_embeddings
        GROUP BY person_id, person_name
        ORDER BY person_name;
        """
        
        try:
            self.cursor.execute(query)  # type: ignore
            records = self.cursor.fetchall()  # type: ignore
            
            persons = []
            for record in records:
                person_id, person_name, count, last_updated = record
                persons.append({
                    'person_id': person_id,
                    'person_name': person_name,
                    'embedding_count': count,
                    'last_updated': last_updated
                })
            
            return persons
        except Exception as e:
            print(f"Error getting all persons: {e}")
            return []
    
    def close(self) -> None:
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("Database connection closed")