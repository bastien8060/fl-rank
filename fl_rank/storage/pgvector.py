# fl_rank/storage/pgvector.py
"""
PostgreSQL/pgvector storage implementation.
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg2
from psycopg2.extras import Json, execute_values

from fl_rank.core.base import BaseStorage
from fl_rank.storage.base import StorageConfig


class PgVectorStorage(BaseStorage):
    """
    PostgreSQL/pgvector storage backend.
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize with configuration.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        if not config.connection_string:
            raise ValueError("connection_string is required for PgVectorStorage")
        
        # Initialize connection
        self._conn = None
        self._connect()
        
        # Create tables if they don't exist
        self._init_schema()
    
    def _connect(self) -> None:
        """
        Establish database connection.
        """
        self._conn = psycopg2.connect(self.config.connection_string)
    
    def _ensure_connection(self) -> None:
        """
        Ensure database connection is active.
        """
        if self._conn is None or self._conn.closed:
            self._connect()
    
    def _init_schema(self) -> None:
        """
        Initialize database schema.
        """
        self._ensure_connection()
        
        with self._conn.cursor() as cur:
            # Check if pgvector extension exists
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone() is None:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create vectors table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                    {self.config.id_column} VARCHAR PRIMARY KEY,
                    {self.config.vector_column} vector(768),
                    {self.config.metadata_column} JSONB
                )
            """)
            
            # Create index table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.table_name}_indexes (
                    name VARCHAR PRIMARY KEY,
                    index_data BYTEA,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            self._conn.commit()
    
    def store_vectors(self, ids: List[Any], vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Store vectors with their IDs and optional metadata.
        
        Args:
            ids: IDs of the vectors
            vectors: Embedding vectors
            metadata: Optional metadata for each vector
        """
        if len(ids) != vectors.shape[0]:
            raise ValueError("Length of ids must match number of vectors")
        
        if metadata is not None and len(ids) != len(metadata):
            raise ValueError("Length of ids must match length of metadata")
        
        self._ensure_connection()
        
        # Prepare data for bulk insert
        data = []
        for i, id_ in enumerate(ids):
            vector_data = vectors[i].tolist()
            meta_data = Json(metadata[i] if metadata is not None else {})
            data.append((str(id_), vector_data, meta_data))
        
        with self._conn.cursor() as cur:
            # Use ON CONFLICT to handle updates
            sql = f"""
                INSERT INTO {self.config.table_name} 
                ({self.config.id_column}, {self.config.vector_column}, {self.config.metadata_column})
                VALUES %s
                ON CONFLICT ({self.config.id_column}) DO UPDATE SET
                {self.config.vector_column} = EXCLUDED.{self.config.vector_column},
                {self.config.metadata_column} = EXCLUDED.{self.config.metadata_column}
            """
            
            execute_values(
                cur, sql, data, 
                template=f"(%s, %s::vector, %s)"
            )
            
            self._conn.commit()
    
    def get_vectors(self, ids: Optional[List[Any]] = None) -> Tuple[List[Any], np.ndarray, List[Dict[str, Any]]]:
        """
        Retrieve vectors by IDs.
        
        Args:
            ids: IDs of vectors to retrieve (None for all)
            
        Returns:
            Tuple[List[Any], np.ndarray, List[Dict[str, Any]]]: IDs, vectors, and metadata
        """
        self._ensure_connection()
        
        with self._conn.cursor() as cur:
            if ids is None:
                # Retrieve all vectors
                cur.execute(f"""
                    SELECT {self.config.id_column}, {self.config.vector_column}, {self.config.metadata_column}
                    FROM {self.config.table_name}
                """)
            else:
                # Retrieve specific vectors
                id_list = ",".join([f"'{id_}'" for id_ in ids])
                cur.execute(f"""
                    SELECT {self.config.id_column}, {self.config.vector_column}, {self.config.metadata_column}
                    FROM {self.config.table_name}
                    WHERE {self.config.id_column} IN ({id_list})
                """)
            
            rows = cur.fetchall()
            
            if not rows:
                return [], np.array([]), []
            
            result_ids = []
            result_vectors = []
            result_metadata = []
            
            for row in rows:
                result_ids.append(row[0])
                result_vectors.append(np.array(row[1]))
                result_metadata.append(row[2])
            
            return result_ids, np.array(result_vectors), result_metadata
    
    def delete_vectors(self, ids: List[Any]) -> None:
        """
        Delete vectors by IDs.
        
        Args:
            ids: IDs of vectors to delete
        """
        self._ensure_connection()
        
        with self._conn.cursor() as cur:
            id_list = ",".join([f"'{id_}'" for id_ in ids])
            cur.execute(f"""
                DELETE FROM {self.config.table_name}
                WHERE {self.config.id_column} IN ({id_list})
            """)
            
            self._conn.commit()
    
    def store_index(self, index: Any, name: str = "default") -> None:
        """
        Store an index.
        
        Args:
            index: Index to store
            name: Name of the index
        """
        self._ensure_connection()
        
        # Serialize the index
        index_data = pickle.dumps(index)
        
        with self._conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {self.config.table_name}_indexes (name, index_data)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET
                index_data = EXCLUDED.index_data,
                created_at = NOW()
            """, (name, psycopg2.Binary(index_data)))
            
            self._conn.commit()
    
    def get_index(self, name: str = "default") -> Any:
        """
        Retrieve an index.
        
        Args:
            name: Name of the index
            
        Returns:
            Any: The retrieved index
        """
        self._ensure_connection()
        
        with self._conn.cursor() as cur:
            cur.execute(f"""
                SELECT index_data FROM {self.config.table_name}_indexes
                WHERE name = %s
            """, (name,))
            
            row = cur.fetchone()
            
            if row is None:
                return None
            
            # Deserialize the index
            return pickle.loads(row[0])
    
    def __del__(self):
        """
        Clean up resources.
        """
        if hasattr(self, '_conn') and self._conn is not None and not self._conn.closed:
            self._conn.close()