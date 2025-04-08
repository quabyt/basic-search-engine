import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
from tika import parser, initVM
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import nltk

# Apache Tika için Java VM başlat
initVM()

# NLTK bileşenlerini indir
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def get_kwic(text, query_terms, window=50):
    """
    Belge metni içerisinde query_terms listesinde bulunan ilk terimin etrafından KWIC benzeri bir snippet döndürür.
    Eğer terim bulunamazsa boş string döner.
    """
    lower_text = text.lower()
    snippet = ""
    for term in query_terms:
        index = lower_text.find(term)
        if index != -1:
            start = max(0, index - window)
            end = min(len(text), index + len(term) + window)
            snippet = text[start:end].strip()
            snippet = snippet.replace(text[index:index+len(term)], text[index:index+len(term)].upper())
            break
    return snippet

class SearchEngine:
    def __init__(self):
        self.documents = {}  # Belgeler {belge_id: {isim, içerik}}

    def process_text(self, text):
        tokens = word_tokenize(text.lower())
        filtered = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
        return filtered

    def index_document(self, file_path):
        try:
            parsed = parser.from_file(file_path)
            content = parsed.get('content', '') or ''
            if not content.strip():
                print(f"Uyarı: {file_path} boş veya okunamıyor.")
                return

            doc_id = len(self.documents) + 1
            self.documents[doc_id] = {
                'name': os.path.basename(file_path),
                'content': content
            }
            print(f"İndekslendi: {file_path}")

        except Exception as e:
            print(f"Hata {file_path} dosyası için: {e}")

    def search(self, query):
        docs = [doc['content'] for doc in self.documents.values()]
        doc_ids = list(self.documents.keys())
        if not docs:
            return []

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(docs)
        query_terms = self.process_text(query)
        y = [1 if any(term in doc.lower() for term in query_terms) else 0 for doc in docs]

        if sum(y) == 0:
            return []

        clf = LinearSVC()
        clf.fit(X, y)
        scores = clf.decision_function(X)

        results = []
        for doc_id, score, doc_text in zip(doc_ids, scores, docs):
            if score > 0:  # Sadece pozitif skorlu sonuçları ekle
                snippet = get_kwic(doc_text, query_terms, window=50)
                results.append((self.documents[doc_id]['name'], score, snippet))
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return sorted_results

class SearchHandler(BaseHTTPRequestHandler):
    def __init__(self, engine, *args, **kwargs):
        self.engine = engine
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''
                <html><body>
                    <h1>Search Engine</h1>
                    <form method="post">
                        <input type="text" name="query" size="50">
                        <input type="submit" value="Search">
                    </form>
                </body></html>
            '''
            self.wfile.write(html.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode()
        query = parse_qs(post_data).get('query', [''])[0]
        results = self.engine.search(query)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        result_items = ""
        for name, score, snippet in results:
            result_items += f'<li><strong>{name}</strong> (Score: {score:.4f})<br><em>... {snippet} ...</em></li>'

        html = f'''
            <html><body>
                <h1>Search Results for "{query}"</h1>
                <a href="/">New Search</a>
                <ul>
                    {result_items}
                </ul>
            </body></html>
        '''
        self.wfile.write(html.encode())

if __name__ == '__main__':
    engine = SearchEngine()
    doc_dir = r'C:\\Users\\hsngu\\Downloads\\documents'
    if not os.path.exists(doc_dir):
        print(f"Hata: Belirtilen klasör bulunamadı: {doc_dir}")
    else:
        for filename in os.listdir(doc_dir):
            path = os.path.join(doc_dir, filename)
            if os.path.isfile(path):
                engine.index_document(path)
    server_address = ('', 8000)
    handler = lambda *args: SearchHandler(engine, *args)
    httpd = HTTPServer(server_address, handler)
    print(f"Server çalışıyor: http://localhost:{server_address[1]}")
    httpd.serve_forever()
