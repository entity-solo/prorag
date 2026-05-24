# Đánh giá & Kiểm tra Kiến trúc Toàn diện (ProRAG Architectural Review)

Tài liệu này ghi nhận kết quả kiểm tra kiến trúc hệ thống **ProRAG**, chỉ ra các điểm nghẽn (bottlenecks), rủi ro vận hành (architectural risks) và đề xuất các giải pháp nâng cấp để hệ thống đạt chuẩn production.

---

## 1. Kết Quả Kiểm Tra Thực Tế (Test Suite Audit)

Khi chạy kiểm tra toàn bộ suite test bằng `pytest` lúc đầu, hệ thống đã bị **treo (hang)** ở test thứ 8: `test_alias_bridging` do `EmbeddingStore` cố tải động mô hình `all-MiniLM-L6-v2` từ Hugging Face Hub trực tuyến (đặc biệt trong điều kiện mạng bị chặn/giới hạn).

### Khắc phục & Giải pháp (Đã hoàn thành):
1.  **Tải cục bộ (Local Download)**: Chúng ta đã tải trực tiếp trọng số của mô hình `all-MiniLM-L6-v2` về lưu trữ nội bộ tại đường dẫn `models/all-MiniLM-L6-v2/` trong repo.
2.  **Cấu hình nạp offline**: Refactor hàm `_load` trong [embeddings.py](file:///c:/Users/hanng/Downloads/prorag-repo/prorag/embeddings.py#L36) để tự động kiểm tra và ưu tiên nạp mô hình từ đường dẫn cục bộ trước khi thử gọi trực tuyến.
3.  **Kết quả xác minh**: Chạy lại toàn bộ test suite `pytest` thành công ngoại tuyến (offline) 100% — **26/26 tests passed** (trong thời gian ~65 giây).

---

## 2. Các Điểm Nghẽn & Rủi Ro Kiến Trúc Lớn

Qua việc rà soát mã nguồn chi tiết, hệ thống hiện tại đang tồn tại 5 điểm nghẽn kiến trúc lớn cần cải thiện trước khi triển khai thực tế:

### 2.1. Khả năng mở rộng của Đồ thị (Graph Scalability)
*   **Hiện trạng**: Đồ thị [ProRAGGraph](file:///c:/Users/hanng/Downloads/prorag-repo/prorag/graph.py#L36) được lưu trữ hoàn toàn trên RAM bằng `networkx.MultiDiGraph` và lưu/tải thủ công thông qua file JSON cục bộ (`save`/`load`).
*   **Điểm nghẽn**: Khi số lượng thực thể lên tới hàng chục nghìn hoặc hàng triệu node, việc tải toàn bộ đồ thị vào RAM sẽ làm cạn kiệt tài nguyên máy chủ. Ngoài ra, việc đọc/ghi file JSON liên tục khi có thay đổi sẽ cực kỳ chậm và dễ mất mát dữ liệu nếu ứng dụng bị crash giữa chừng.

### 2.2. Khả năng mở rộng của Tìm kiếm Vector (Vector Search Bottleneck)
*   **Hiện trạng**: Trong hàm `query_vector`, hệ thống lấy tất cả các node trong đồ thị: `all_nodes = list(self.g.nodes)`, sau đó tính tích vô hướng (Dot Product) với từng node một bằng thư viện `numpy`.
*   **Điểm nghẽn**: Đây là thuật toán tìm kiếm tuyến tính $O(N)$ chạy hoàn toàn bằng CPU của Python. Nếu đồ thị có quy mô lớn, thao tác tính toán vector cho từng node này sẽ chiếm dụng CPU rất lớn và làm tăng độ trễ (latency) của câu trả lời lên mức không thể chấp nhận được.

### 2.3. Rủi ro về Đồng thời & Luồng (Concurrency & Thread Safety)
*   **Hiện trạng**: Đối tượng đồ thị `ProRAGGraph` được ghi đè trực tiếp mà không có bất kỳ cơ chế khóa (Lock) hay đồng bộ luồng nào. Thư viện NetworkX mặc định **không an toàn khi ghi đồng thời (not thread-safe for writes)**.
*   **Rủi ro**: Nếu hệ thống nhận nhiều yêu cầu nạp văn bản (`ingest_text`) cùng một lúc từ nhiều luồng khác nhau, đồ thị NetworkX sẽ bị lỗi tranh chấp tài nguyên (race conditions), dẫn đến hỏng cấu trúc đồ thị (data corruption) hoặc crash tiến trình.

### 2.4. Sự trôi lệch ngữ nghĩa của Quan hệ (Relation Semantic Drift)
*   **Hiện trạng**: Các quan hệ trong bộ ba (triples) được trích xuất dưới dạng chuỗi tự do (free-text) do LLM tự quyết định (ví dụ: `"founded"`, `"was established by"`, `"created"`, `"started"`).
*   **Điểm nghẽn**: Mặc dù hệ thống có cơ chế `alias_threshold` để liên kết các thực thể đồng nghĩa (ví dụ: *"Steve Jobs"* và *"ông Jobs"*), nhưng **không có cơ chế chuẩn hóa quan hệ tương đương**. Việc tồn tại quá nhiều từ đồng nghĩa cho cùng một kiểu quan hệ sẽ làm suy giảm hiệu năng của thuật toán lan tỏa (Spreading Activation) và làm giảm chất lượng liên kết 2-hop.

### 2.5. Cơ chế phát hiện mâu thuẫn đơn giản (Simplistic Contradiction Detection)
*   **Hiện trạng**: Hàm `_find_contradicting_edge` chỉ phát hiện mâu thuẫn khi có hai cạnh trùng nhau hoàn toàn về mọi mặt (Subject, Object, Relation, Condition...) nhưng **ngược nhau về cờ phủ định (`negated`)**.
*   **Điểm nghẽn**: Trong thực tế, mâu thuẫn dữ liệu thường xuất hiện dưới dạng khác biệt giá trị của cùng một thuộc tính (Ví dụ: *"Steve Jobs sinh ra ở San Francisco"* và *"Steve Jobs sinh ra ở Green Bay"*). Do hai đối tượng khác nhau (`"san francisco"` và `"green bay"`), hệ thống sẽ không phát hiện ra mâu thuẫn mà sẽ lưu cả hai, tạo ra thông tin rác.

---

## 3. Khuyến Nghị Cải Tiến Kiến Trúc (Architectural Recommendations)

Để nâng cấp ProRAG thành một hệ thống sẵn sàng cho production, chúng ta cần triển khai các giải pháp sau:

```mermaid
graph TD
    subgraph Hiện Tại (Local & In-Memory)
        NetX[NetworkX MultiDiGraph]
        NumpyScan[Linear Numpy Vector Scan]
        SyncCalls[Sequential LLM Calls]
    end

    subgraph Đề Xuất Production-Grade
        Neo4j[(Graph DB - Neo4j / Memgraph)]
        VectorDB[(Vector DB - Qdrant / Milvus)]
        AsyncPipeline[Asynchronous Pipeline / Lock Control]
        Ontology[Predefined Relation Ontology]
    end

    NetX --> |Di chuyển dữ liệu| Neo4j
    NumpyScan --> |Index tìm kiếm| VectorDB
    SyncCalls --> |Tối ưu độ trễ & Luồng| AsyncPipeline
```

### 3.1. Chuyển đổi sang Graph Database và Vector Database chuyên dụng
*   Thay thế NetworkX bằng một Graph Database thực sự như **Neo4j** hoặc **Memgraph**. Các DB này hỗ trợ lưu trữ bền vững (persistence), tối ưu truy vấn đồ thị phức tạp và hỗ trợ ACID transaction.
*   Thay thế tìm kiếm tuyến tính của `EmbeddingStore` bằng một **Vector Database** chuyên dụng (như **Qdrant**, **Milvus**, hoặc **Chroma**). Vector DB hỗ trợ lập chỉ mục (HNSW index) giúp tìm kiếm thực thể tương đồng chỉ mất thời gian $O(\log N)$ thay vì $O(N)$.

### 3.2. Quản lý mô hình Embedding Offline & Cài đặt Timeout
*   Cấu hình `EmbeddingStore` để tải mô hình từ thư mục cục bộ (được tải sẵn khi build Docker image hoặc deploy) thay vì tải động từ Internet:
    ```python
    # Tải offline từ thư mục lưu sẵn
    self._model = SentenceTransformer("./models/all-MiniLM-L6-v2")
    ```
*   Thiết lập thời gian chờ (Timeout) ngắn cho các kết nối Hugging Face để chuyển sang cơ chế fallback nhanh chóng nếu mạng lỗi, tránh treo luồng xử lý.

### 3.3. Đảm bảo an toàn luồng (Thread Safety)
*   Thêm khóa ghi (`threading.Lock` hoặc `asyncio.Lock`) xung quanh phương thức `add_triple` và `add_node` của đồ thị nếu hệ thống chạy đa luồng:
    ```python
    from threading import Lock
    self._write_lock = Lock()

    def add_triple(self, ...):
        with self._write_lock:
            # logic thêm bộ ba an toàn
    ```

### 3.4. Định nghĩa Ontology/Schema cho Quan Hệ
*   Xây dựng một danh mục quan hệ chuẩn hóa (Ontology). Khi trích xuất, yêu cầu LLM phân loại quan hệ trích xuất được vào một tập hợp các quan hệ cố định, hoặc sử dụng một bước LLM phụ trợ để ánh xạ quan hệ thô về quan hệ chuẩn hóa (ví dụ: `"was established by"` $\rightarrow$ `"founded"`).

### 3.5. Nâng cấp bộ lọc mâu thuẫn ngữ nghĩa (Semantic Contradiction)
*   Cải tiến hàm phát hiện mâu thuẫn để nhận diện trường hợp cùng một Subject và Relation nhưng có Object khác nhau đối với các thuộc tính mang tính duy nhất (Single-valued relations như *ngày sinh, nơi sinh, cha mẹ ruột...*).

---

## 4. Kết Quả Thực Nghiệm & Đánh Giá Thừa/Thiếu (Empirical Verification)

Hệ thống đã được chạy kiểm tra end-to-end thực tế bằng tập lệnh benchmark [demo_superpowers.py](file:///c:/Users/hanng/Downloads/prorag-repo/scripts/demo_superpowers.py) sử dụng Groq API Key thật. Kết quả cho thấy cả 4 siêu năng lực (Multi-hop, Conflict, Update, Temporal/Passive) đều hoạt động xuất sắc.

### 4.1. Đánh giá tính dư thừa (Redundancy)
*   **Không có phần dư thừa**: Tất cả các điểm nghẽn được cảnh báo trong báo cáo kiến trúc (đồng thời, khả năng mở rộng bộ nhớ, tải mô hình mạng, mâu thuẫn thô) đều cực kỳ thực tế và phản ánh chính xác cấu trúc mã nguồn hiện tại.

### 4.2. Các điểm bổ sung quan trọng (Thiếu sót đã nhận diện):
1.  **Thiếu cơ chế quản lý tốc độ gọi API (Backpressure & Rate Limiting)**:
    *   *Chi tiết*: Trong quá trình benchmark, hệ thống thực hiện nhiều cuộc gọi LLM liên tiếp. Nếu nạp văn bản lớn, cơ chế gọi tuần tự hiện tại sẽ nhanh chóng chạm hạn mức TPM (Tokens Per Minute) hoặc RPM (Requests Per Minute) của nhà cung cấp API (ví dụ Groq Free Tier giới hạn 6000 TPM).
    *   *Đề xuất*: Cần bổ sung cơ chế hàng đợi (Rate-limited Queue) và cơ chế tự động dừng/thử lại thông minh (Exponential Backoff) ở cấp độ pipeline chứ không chỉ ở cấp độ hàm LLM đơn lẻ.
2.  **Đánh giá cao thiết kế Zero-Rule Multilingual**:
    *   *Chi tiết*: Việc lược bỏ hoàn toàn các bộ lọc ngôn ngữ tiếng Việt cứng nhắc ở tầng Python và chuyển sang sử dụng mô hình embedding đa ngôn ngữ kết hợp LLM là một bước đi rất đúng đắn. Quá trình benchmark thực tế cho thấy hệ thống tự động xử lý tốt các thực thể tiếng Việt/tiếng Anh đan xen nhờ khả năng tự căn chỉnh ngữ nghĩa của mô hình AI.

