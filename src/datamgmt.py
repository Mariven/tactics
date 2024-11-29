from __future__ import annotations
from .utilities import *
import requests
from collections import deque
from urllib.parse import urlparse, urljoin

# Initialize logging
logger = logging.getLogger('tree_scrape')

def normalize_url(url: str) -> str:
    """Normalize URL to a standard format."""
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme if parsed_url.scheme in {'http', 'https'} else 'https'
    netloc = parsed_url.netloc
    path = parsed_url.path or '/'
    query = f"?{parsed_url.query}" if parsed_url.query else ''
    fragment = f"#{parsed_url.fragment}" if parsed_url.fragment else ''
    normalized_url = f"{scheme}://{netloc}{path}{query}{fragment}"
    return normalized_url

def get_domain(url: str) -> str:
    """
    Extract the domain from a given URL.
    :param url: The URL to extract the domain from.
    :return: The extracted domain.
    """
    return urlparse(url).netloc

def cache_get(cache_file: str, values: Union[str, List[str]], interpret_as_regex: bool = False, key_column: str = "url", value_column: str = "content") -> Union[str, Dict[str, str]]:
    """
    Retrieve values from the cache.
    :param cache_file: The path to the cache file.
    :param values: The key(s) to retrieve from the cache.
    :param interpret_as_regex: If True, interpret the keys as regex patterns.
    :return: The retrieved value(s) from the cache.
    """
    ext = cache_file.split('.')[-1].lower()
    if ext == 'json':
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    elif ext == 'sqlite3':
        conn = sqlite3.connect(cache_file)
        cursor = conn.cursor()
        # cursor.execute(f"CREATE TABLE IF NOT EXISTS cache ({key_column} TEXT PRIMARY KEY, {value_column} TEXT)")
        cache = dict(cursor.execute(f"SELECT {key_column}, {value_column} FROM cache").fetchall())
    else:
        raise ValueError("Unsupported cache file format")

    if isinstance(values, str):
        values = [values]

    result = {}
    for value in values:
        if interpret_as_regex:
            pattern = re.compile(value)
            result.update({k: v for k, v in cache.items() if pattern.search(k)})
        elif value in cache:
            result[value] = cache[value]
    if len(result) > 1:
        return result
    return next(iter(result.values()), None)

def cache_del(cache_file: str, values: Union[str, List[str]], interpret_as_regex: bool = False, key_column: str = "url", value_column: str = "content") -> None:
    """
    Delete values from the cache.
    :param cache_file: The path to the cache file.
    :param values: The key(s) to delete from the cache.
    :param interpret_as_regex: If True, interpret the keys as regex patterns.
    """
    ext = cache_file.split('.')[-1].lower()
    if ext == 'json':
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    elif ext == 'sqlite3':
        conn = sqlite3.connect(cache_file)
        cursor = conn.cursor()
        # cursor.execute(f"CREATE TABLE IF NOT EXISTS cache ({key_column} TEXT PRIMARY KEY, {value_column} TEXT)")
        cache = dict(cursor.execute(f"SELECT {key_column}, {value_column} FROM cache").fetchall())
    else:
        raise ValueError("Unsupported cache file format")

    if isinstance(values, str):
        values = [values]

    for value in values:
        if interpret_as_regex:
            pattern = re.compile(value)
            keys_to_delete = [k for k in cache if pattern.search(k)]
            for key in keys_to_delete:
                del cache[key]
        elif value in cache:
            del cache[value]

    if ext == 'json':
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    elif ext == 'sqlite3':
        cursor.execute("DELETE FROM cache")
        cursor.executemany("INSERT INTO cache (key, value) VALUES (?, ?)", cache.items())
        conn.commit()
        conn.close()

def setup_cache(cache_file: str) -> (Callable[[str], Optional[str]], Callable[[str, str], None]):
    """Setup cache using JSON or sqlite3 based on the file extension."""
    ext = os.path.splitext(cache_file)[-1].lower()
    if ext == '.json':
        if not os.path.exists(cache_file):
            with open(cache_file, 'w') as f:
                json.dump({}, f)

        def load_cache(url: str) -> Optional[str]:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            return cache.get(normalize_url(url))

        def save_cache(url: str, content: str) -> None:
            with open(cache_file, 'r+') as f:
                cache = json.load(f)
                cache[normalize_url(url)] = content
                f.seek(0)
                json.dump(cache, f, indent=4)
                f.truncate()
    elif ext == '.sqlite3':
        conn = sqlite3.connect(cache_file)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS cache (url TEXT PRIMARY KEY, content TEXT)')
        conn.commit()

        def load_cache(url: str) -> Optional[str]:
            c.execute('SELECT content FROM cache WHERE url = ?', (normalize_url(url),))
            result = c.fetchone()
            return result[0] if result else None

        def save_cache(url: str, content: str) -> None:
            c.execute('INSERT OR REPLACE INTO cache (url, content) VALUES (?, ?)', (normalize_url(url), content))
            conn.commit()
    else:
        raise ValueError("Cache file must be a .json or .sqlite3 file.")
    return load_cache, save_cache

def tree_scrape(
    base_urls: str | List[str],
    allowed_domains: List[str],
    link_filter: Callable[[str], bool] = lambda link: True,
    cache_file: Optional[str] = None,
    scrape_limit: int = 0,
    log_file: Optional[str] = None) -> Dict[str, str]:
    """
    Scrape web content starting from base_urls, following links that match the link_filter's criteria
    and are within the allowed_domains.
    :param base_urls: The starting URLs for scraping.
    :param allowed_domains: List of domains to allow during scraping.
    :param link_filter: Filter function to determine which links to follow.
    :param cache_file: Path to the cache file (either JSON or SQLite3).
    :param scrape_limit: Maximum number of URLs to scrape (0 for no limit).
    :param log_file: Path to the log file.
    :return: Dictionary of URL to content mappings.
    """
    from bs4 import BeautifulSoup
    if log_file:
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
    if isinstance(base_urls, str):
        base_urls = [base_urls]

    stack = deque(base_urls)
    visited = set()
    scraped_content = {}
    if not cache_file:
        load_cache, save_cache = (lambda x: None, lambda x, y: None)
    else:
        load_cache, save_cache = setup_cache(cache_file)

    while stack and (not scrape_limit or len(scraped_content) < scrape_limit):
        url = stack.popleft()
        url = normalize_url(url)
        domain = urlparse(url).netloc
        if domain not in allowed_domains:
            logger.debug(f"Skipping URL not in allowed domains: {url}")
            continue
        visited.add(url)

        content = load_cache(url)
        if content:
            logger.debug(f"Loaded content from cache for URL: {url}")
            scraped_content[url] = content
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            content = response.text
        except requests.RequestException as e:
            msg = str(e)
            logger.exception(f"Failed to fetch URL: {url} - {msg}")
            continue

        save_cache(url, content)
        scraped_content[url] = content
        logger.info(f"Scraped URL: {url}")

        soup = BeautifulSoup(content, 'html.parser')
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if not href.startswith(('http://', 'https://')):
                href = urljoin(url, href)
            href = normalize_url(href)
            if link_filter(href):
                parsed_href = urlparse(href)
                if parsed_href.netloc in allowed_domains and href not in visited:
                    stack.append(href)
                    logger.debug(f"Added link to stack: {href}")
    return scraped_content

def struct_sqlite_db(sql_path: str, table_name: str, columns: Dict[str, str], primary_key: str) -> None:
    """
    Create a SQLite database with columns of indicated (JSON) types.
    :param sql_path: Path to the SQLite database file
    :param table_name: Name of the table to create
    :param columns: Dict of column names and JSON types to include in the database
    :param primary_key: Name of the primary key column
    """
    conn = sqlite3.connect(sql_path)
    cursor = conn.cursor()
    columns_sql = []
    for col, typ in columns.items():
        json_type = typ
        sql_type = "TEXT"
        # the only things that aren't (serialized to) strings are integers, booleans, and floats
        if json_type in {"integer", "boolean"}:
            sql_type = "INTEGER"
        if json_type == "number":
            sql_type = "REAL"
        columns_sql.append(f"{col} {sql_type}")
    columns_sql_str = ', '.join(columns_sql)
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {columns_sql_str},
        PRIMARY KEY ({primary_key})
    )
    """)
    conn.commit()
    conn.close()

def jsonl_to_sqlite(jsonl_path: str, sql_path: str, table_name: str, column_fns: Dict[str, Callable]) -> None:
    """
    Import data from JSONL file to SQLite database using JSON schema.
    :param jsonl_path: Path to the JSONL file
    :param sql_path: Path to the SQLite database file
    :param table_name: Name of the table to insert data into
    :param column_fns: Dictionary mapping column names to functions that extract values from the JSONL entries
    """
    conn = sqlite3.connect(sql_path)
    cursor = conn.cursor()
    idx = 0
    with open(jsonl_path) as jsonl_file:
        for line in jsonl_file:
            print(line, idx)
            idx += 1
            if not line.strip():
                continue
            print(line)
            entry = json_loads(line)
            values = []
            for col_fn in column_fns.values():
                value = col_fn(entry)
                if isinstance(value, (dict, list)):
                    value = json_dumps(value)
                if isinstance(value, bool):
                    value = int(value)
                values.append(value)
            placeholders = ', '.join(['?' for _ in column_fns])
            cursor.execute(f"""
                INSERT OR REPLACE INTO {table_name} ({', '.join(column_fns.keys())})
                VALUES ({placeholders})
                """, tuple(values))
    conn.commit()
    conn.close()

def query_sqlite_db(sql_path: str, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
    """
    Execute a query on the SQLite database and return results.
    :param sql_path: Path to the SQLite database file
    :param query: SQL query to execute
    :param params: Parameters to pass to the query
    :returns: List of dictionaries representing the query results
    """
    conn = sqlite3.connect(sql_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results
