import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
from tika import parser
from tika import initVM
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Apache Tika için Java VM başlat
initVM()

# NLTK bileşenlerini indir
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


class SearchEngine:
    def __init__(self):
        self.index = {}  # Ters indeksleme: {kelime: {belge_id: sayı}}
        self.documents = {}  # Belgeler {belge_id: {isim, içerik}}

    def process_text(self, text):
        """Tokenize et, stopwords kaldır ve kökleri al"""
        tokens = word_tokenize(text.lower())
        filtered = [stemmer.stem(t) for t in tokens if t.isalnum() and t not in stop_words]
        return filtered

    def index_document(self, file_path):
        """Apache Tika ile belgeyi ayrıştır ve indeksle"""
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

            terms = self.process_text(content)
            term_counts = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1

            for term, count in term_counts.items():
                if term not in self.index:
                    self.index[term] = {}
                self.index[term][doc_id] = count

            print(f"İndekslendi: {file_path}")

        except Exception as e:
            print(f"Hata {file_path} dosyası için: {e}")

    def search(self, query):
        query_terms = self.process_text(query)
        results = {}

        for term in query_terms:
            if term in self.index:
                for doc_id, count in self.index[term].items():
                    results[doc_id] = results.get(doc_id, 0) + count

        # Sonuçları en çok eşleşen belgeden başlayarak sırala
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [(self.documents[doc_id]['name'], score) for doc_id, score in sorted_results]

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

        html = f'''
            <html><body>
                <h1>Search Results for "{query}"</h1>
                <a href="/">New Search</a>
                <ul>
                    {"".join(f'<li>{name} (Score: {score})</li>' for name, score in results)}
                </ul>
            </body></html>
        '''
        self.wfile.write(html.encode())


if __name__ == '__main__':
    # Arama motorunu başlat
    engine = SearchEngine()

    # Belgeleri dizine ekleyin (kendi dizininize göre değiştirin)
    doc_dir = r'C:\\Users\\hsngu\\Downloads\\documents'  # Windows yolu
    if not os.path.exists(doc_dir):
        print(f"Hata: Belirtilen klasör bulunamadı: {doc_dir}")
    else:
        for filename in os.listdir(doc_dir):
            path = os.path.join(doc_dir, filename)
            if os.path.isfile(path):
                engine.index_document(path)

    # HTTP sunucusunu başlat
    server_address = ('', 8000)
    handler = lambda *args: SearchHandler(engine, *args)
    httpd = HTTPServer(server_address, handler)
    """print(engine.documents)  # Kaydedilen belgeleri gösterir
    print(engine.index)  # Kelime indeksini gösterir"""

    print(f"Server çalışıyor: http://localhost:{server_address[1]}")
    httpd.serve_forever()
