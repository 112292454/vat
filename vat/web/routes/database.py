"""
数据库可视化 API

提供 SQLite 数据库的只读浏览功能：
- 列出所有表及其 schema
- 分页浏览表数据
- 全文搜索（跨所有文本列）
- JSON 字段展开查看
"""
import json
import sqlite3
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from vat.web.deps import get_web_config

router = APIRouter(prefix="/api/database", tags=["database"])


def _get_db_path() -> str:
    """获取数据库路径"""
    config = get_web_config()
    return str(config.storage.database_path)


def _get_connection() -> sqlite3.Connection:
    """获取只读数据库连接"""
    db_path = _get_db_path()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/tables")
async def list_tables():
    """列出所有表及其行数和列信息"""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row['name'] for row in cursor.fetchall()]
        
        result = []
        for table in tables:
            # 行数
            cursor.execute(f"SELECT COUNT(*) as cnt FROM [{table}]")
            count = cursor.fetchone()['cnt']
            
            # 列信息
            cursor.execute(f"PRAGMA table_info([{table}])")
            columns = [
                {
                    'name': col['name'],
                    'type': col['type'],
                    'notnull': bool(col['notnull']),
                    'pk': bool(col['pk']),
                    'default': col['dflt_value'],
                }
                for col in cursor.fetchall()
            ]
            
            result.append({
                'name': table,
                'row_count': count,
                'columns': columns,
            })
        
        return {'tables': result}
    finally:
        conn.close()


@router.get("/tables/{table_name}")
async def query_table(
    table_name: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: Optional[str] = Query(None, description="全文搜索（跨所有文本列）"),
    sort_by: Optional[str] = Query(None, description="排序列名"),
    sort_dir: str = Query("asc", pattern="^(asc|desc)$"),
    column_filter: Optional[str] = Query(None, description="列名=值 格式的精确筛选"),
):
    """
    分页查询表数据
    
    支持搜索、排序、筛选。JSON 字段在返回时自动解析。
    """
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        
        # 验证表存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"表不存在: {table_name}")
        
        # 获取列信息
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = [col['name'] for col in cursor.fetchall()]
        
        # 构建查询
        where_clauses = []
        params = []
        
        # 全文搜索：对所有列做 LIKE 匹配
        if search:
            search_conditions = []
            for col in columns:
                search_conditions.append(f"CAST([{col}] AS TEXT) LIKE ?")
                params.append(f"%{search}%")
            where_clauses.append(f"({' OR '.join(search_conditions)})")
        
        # 精确列筛选
        if column_filter and isinstance(column_filter, str) and '=' in column_filter:
            col_name, col_value = column_filter.split('=', 1)
            col_name = col_name.strip()
            col_value = col_value.strip()
            if col_name in columns:
                where_clauses.append(f"CAST([{col_name}] AS TEXT) LIKE ?")
                params.append(f"%{col_value}%")
        
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        # 排序
        if sort_by and sort_by in columns:
            order_sql = f"ORDER BY [{sort_by}] {sort_dir}"
        else:
            order_sql = ""
        
        # 计数
        count_sql = f"SELECT COUNT(*) as cnt FROM [{table_name}] {where_sql}"
        cursor.execute(count_sql, params)
        total = cursor.fetchone()['cnt']
        
        # 分页查询
        offset = (page - 1) * page_size
        data_sql = f"SELECT * FROM [{table_name}] {where_sql} {order_sql} LIMIT ? OFFSET ?"
        cursor.execute(data_sql, params + [page_size, offset])
        
        rows = []
        for row in cursor.fetchall():
            row_dict = {}
            for col in columns:
                value = row[col]
                # 尝试解析 JSON 字段
                if isinstance(value, str) and value.startswith(('{', '[')):
                    try:
                        value = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        pass
                row_dict[col] = value
            rows.append(row_dict)
        
        return {
            'table': table_name,
            'columns': columns,
            'rows': rows,
            'total': total,
            'page': page,
            'page_size': page_size,
            'total_pages': (total + page_size - 1) // page_size,
        }
    finally:
        conn.close()


@router.get("/tables/{table_name}/row")
async def get_row(
    table_name: str,
    key_column: str = Query(..., description="主键列名"),
    key_value: str = Query(..., description="主键值"),
):
    """获取单行数据的详细信息（所有字段展开）"""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"表不存在: {table_name}")
        
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = [col['name'] for col in cursor.fetchall()]
        
        if key_column not in columns:
            raise HTTPException(status_code=400, detail=f"列不存在: {key_column}")
        
        cursor.execute(f"SELECT * FROM [{table_name}] WHERE [{key_column}] = ?", (key_value,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"未找到: {key_column}={key_value}")
        
        row_dict = {}
        for col in columns:
            value = row[col]
            if isinstance(value, str) and value.startswith(('{', '[')):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
            row_dict[col] = value
        
        return {'table': table_name, 'row': row_dict}
    finally:
        conn.close()
