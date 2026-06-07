import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# ==========================================
# ĐĂNG KÝ PHÔNG CHỮ HỖ TRỢ TIẾNG VIỆT
# ==========================================
def register_vietnamese_fonts():
    font_dir = r"C:\Windows\Fonts"
    fonts = {
        'Arial': 'arial.ttf',
        'Arial-Bold': 'arialbd.ttf',
        'Arial-Italic': 'ariali.ttf',
        'Arial-BoldItalic': 'arialbi.ttf'
    }
    
    for name, filename in fonts.items():
        path = os.path.join(font_dir, filename)
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
            except Exception as e:
                print(f"Error registering {name}: {e}")
        else:
            print(f"Font file not found: {path}")

register_vietnamese_fonts()

# ==========================================
# BỘ HIGHLIGHT MÃ NGUỒN PYTHON CHO REPORTLAB
# ==========================================
def highlight_python(code):
    code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 1. Bảo vệ các chú thích (comments)
    comments = []
    def save_comment(m):
        comments.append(m.group(0))
        return f"___COMMENT_{len(comments)-1}___"
    code = re.sub(r'#.*', save_comment, code)
    
    # 2. Bảo vệ các chuỗi (strings)
    strings = []
    def save_string(m):
        strings.append(m.group(0))
        return f"___STRING_{len(strings)-1}___"
    code = re.sub(r'\'[^\']*\'|\"[^\"]*\"', save_string, code)
    
    # 3. Tô màu các từ khóa (keywords)
    keywords = [
        'class', 'def', 'return', 'if', 'not', 'in', 'for', 'else', 'import', 'from',
        'and', 'or', 'assert', 'try', 'except', 'with', 'as', 'pass', 'is'
    ]
    for kw in keywords:
        code = re.sub(rf'\b{kw}\b', f'<font color="#0033B3"><b>{kw}</b></font>', code)
        
    # 4. Tô màu các từ khóa đặc biệt (builtins)
    builtins = ['self', 'True', 'False', 'None']
    for b in builtins:
        code = re.sub(rf'\b{b}\b', f'<font color="#800080"><b>{b}</b></font>', code)
        
    # 5. Tô màu các hàm/đối tượng PyTorch và Toán học
    pytorch_symbols = [
        'nn', 'torch', 'math', 'optim', 'Module', 'Embedding', 'Linear', 'LayerNorm',
        'Dropout', 'ReLU', 'ModuleList', 'register_buffer', 'zeros', 'arange', 'exp',
        'log', 'sin', 'cos', 'unsqueeze', 'transpose', 'view', 'matmul', 'softmax',
        'contiguous', 'apply', 'xavier_uniform_', 'normal_', 'ones_', 'zeros_', 'init',
        'DataParallel'
    ]
    for sym in pytorch_symbols:
        code = re.sub(rf'\b{sym}\b', f'<font color="#267F99">{sym}</font>', code)
        
    # 6. Trả lại chuỗi gốc
    for idx, s in enumerate(strings):
        code = code.replace(f"___STRING_{idx}___", f'<font color="#A31515">{s}</font>')
        
    # 7. Trả lại chú thích gốc
    for idx, c in enumerate(comments):
        code = code.replace(f"___COMMENT_{idx}___", f'<font color="#008000"><i>{c}</i></font>')
        
    # 8. Xử lý thụt lề và dấu dòng
    lines = code.split('\n')
    processed_lines = []
    for line in lines:
        num_spaces = len(line) - len(line.lstrip(" "))
        line_content = line.lstrip(" ")
        new_line = "&nbsp;" * num_spaces + line_content
        processed_lines.append(new_line)
    return "<br/>".join(processed_lines)

# ==========================================
# KHỞI TẠO LỚP CANVAS ĐỂ ĐÁNH SỐ TRANG TỰ ĐỘNG
# ==========================================
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        
        # Trang bìa (Trang 1) - Không vẽ header và footer
        if self._pageNumber == 1:
            self.setFillColor(colors.HexColor("#1A365D"))
            self.rect(0, 0, 15, 792, fill=True, stroke=False)
            self.restoreState()
            return
            
        self.setFont("Arial", 8)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        
        # Vẽ Header (từ trang 2 trở đi)
        self.drawString(54, 750, "Báo cáo chuyên đề: Thiết kế thuật toán và Thực nghiệm Mini-Transformer")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(54, 742, 558, 742)
        
        # Vẽ Footer (từ trang 2 trở đi)
        self.line(54, 55, 558, 55)
        page_text = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(558, 40, page_text)
        self.drawString(54, 40, "Đề tài Nghiên cứu Khoa học & Công nghệ - NCKH Team 2026")
        
        self.restoreState()

# ==========================================
# KHỞI TẠO BÁO CÁO PDF
# ==========================================
def build_pdf(filename="Bao_cao_Mini_Transformer.pdf"):
    # Đặt lề để đảm bảo tài liệu không bị tràn hàng ngoài ý muốn
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Định nghĩa các Styles tùy chỉnh sử dụng Font Arial tiếng Việt
    normal_style = styles['Normal']
    normal_style.fontName = 'Arial'
    normal_style.fontSize = 9.0
    normal_style.leading = 13.0
    normal_style.textColor = colors.HexColor("#2D3748")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=18,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        alignment=1, # Center
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceAfter=40
    )

    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SubSectionHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        spaceBefore=3,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        leftIndent=15,
        firstLineIndent=-10,
        spaceBefore=2,
        spaceAfter=2
    )

    formula_style = ParagraphStyle(
        'FormulaCustom',
        parent=styles['Normal'],
        fontName='Arial-BoldItalic',
        fontSize=10,
        alignment=1,
        spaceBefore=5,
        spaceAfter=5
    )

    code_style = ParagraphStyle(
        'CodeBlockCustom',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#1A202C")
    )

    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontSize=7.8,
        leading=10,
        alignment=1 # Center
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.2,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    # Hàm tạo khung mã nguồn đẹp (Code Block Box)
    def make_code_block(raw_code):
        highlighted = highlight_python(raw_code)
        p = Paragraph(highlighted, code_style)
        t = Table([[p]], colWidths=[500])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LINELEFT', (0,0), (0,0), 3.0, colors.HexColor("#2B6CB0")),
        ]))
        return t

    story = []

    # ==========================================
    # TRANG 1: TRANG BÌA (COVER PAGE)
    # ==========================================
    story.append(Spacer(1, 100))
    story.append(Paragraph("BÁO CÁO CHUYÊN ĐỀ KHOA HỌC CHI TIẾT", subtitle_style))
    story.append(Paragraph("THIẾT KẾ GIẢI THUẬT, PHÂN TÍCH MÃ NGUỒN VÀ ĐÁNH GIÁ THỰC NGHIỆM HỆ THỐNG DỊCH MÁY MINI-TRANSFORMER", title_style))
    story.append(Spacer(1, 15))
    
    meta_box_data = [
        [Paragraph("<b>Đề tài:</b> Phát triển Hệ thống dịch máy song ngữ seq2seq ứng dụng cơ chế Attention tự xây dựng từ con số không (From Scratch)", body_style)],
        [Paragraph("<b>Tập dữ liệu:</b> IWSLT 2015 English-Vietnamese (Tổng: 135.853 câu; Train: 133.317 (~98,13%), Val: 1.268 (~0,93%), Test: 1.268 (~0,93%))", body_style)],
        [Paragraph("<b>Chi tiết thuật toán:</b> Multi-Head Attention, Pre-LN LayerNorm, Shared BPE, Noam Scheduler", body_style)],
        [Paragraph("<b>Chiến lược tối ưu:</b> nn.DataParallel Multi-GPU, FP16 AMP, Gradient Accumulation, Bucket Batching", body_style)],
        [Paragraph("<b>Tác giả:</b> Nhóm Nghiên cứu Khoa học & Xử lý Ngôn ngữ Tự nhiên (NCKH Team)", body_style)],
        [Paragraph("<b>Học vị/Học phần:</b> Đề tài Nghiên cứu Phát triển Công nghệ NLP & Học Sâu", body_style)],
        [Paragraph("<b>Ngày thực hiện:</b> 01 tháng 06 năm 2026", body_style)]
    ]
    meta_table = Table(meta_box_data, colWidths=[420])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(PageBreak()) # KẾT THÚC TRANG 1

    # ==========================================
    # TRANG 2: MỤC LỤC & TÓM TẮT ĐỀ TÀI
    # ==========================================
    story.append(Paragraph("Mục lục & Tóm tắt nghiên cứu", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>MỤC LỤC BÁO CÁO</b>", h2_style))
    
    toc_data = [
        [Paragraph("Mở đầu: Bối cảnh lịch sử Dịch máy và sự ra đời của Transformer", body_style), Paragraph(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", body_style), Paragraph("Trang 3", body_style)],
        [Paragraph("Chương 1: Tổng quan Kiến trúc và Cơ chế hoạt động của hệ thống", body_style), Paragraph(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", body_style), Paragraph("Trang 4", body_style)],
        [Paragraph("Chương 2: Phân tích Chi tiết Thuật toán & Giải thích Từng Khối Mã Nguồn", body_style), Paragraph(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", body_style), Paragraph("Trang 7", body_style)],
        [Paragraph("Chương 3: Quy trình Tiền xử lý Dữ liệu Song ngữ lớn IWSLT 2015", body_style), Paragraph(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", body_style), Paragraph("Trang 21", body_style)],
        [Paragraph("Chương 4: Chiến lược Huấn luyện Đa GPU & Cơ chế Tối ưu phần cứng", body_style), Paragraph(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", body_style), Paragraph("Trang 25", body_style)],
        [Paragraph("Chương 5: Kết quả Thực nghiệm, Sự Hội tụ & Biểu đồ Chỉ số", body_style), Paragraph(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", body_style), Paragraph("Trang 26", body_style)],
        [Paragraph("Chương 6: Đánh giá Bản dịch Thực tế qua các Epoch & Bản đồ Attention", body_style), Paragraph(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", body_style), Paragraph("Trang 28", body_style)],
        [Paragraph("Chương 7: Kết luận, Giới hạn Đề tài & Định hướng phát triển tương lai", body_style), Paragraph(". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", body_style), Paragraph("Trang 30", body_style)],
    ]
    toc_table = Table(toc_data, colWidths=[240, 200, 60])
    toc_table.setStyle(TableStyle([
        ('PADDING', (0,0), (-1,-1), 2),
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
    ]))
    story.append(toc_table)
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>TÓM TẮT NỘI DUNG NGHIÊN CỨU KHOA HỌC</b>", h2_style))
    story.append(Paragraph(
        "Nghiên cứu này tập trung vào việc hiện thực hóa kiến trúc Transformer từ các lớp nền tảng nhất của PyTorch "
        "để thực hiện dịch thuật song ngữ. Chúng tôi khảo sát sâu các khía cạnh toán học đằng sau cơ chế tự chú ý đa đầu, "
        "cách thức phân rã vector đặc trưng và phương án giải quyết bài toán suy giảm đạo hàm thông qua Pre-Layer Normalization. "
        "Bên cạnh đó, nghiên cứu đi sâu vào thực tiễn tối ưu hóa hệ thống trên 2 GPU NVIDIA T4 chạy song song, giải quyết triệt để "
        "các cảnh báo không tương thích phần cứng và sự phân mảnh bộ nhớ. Kết quả BLEU đạt 21.43% là một cột mốc vững chắc, "
        "khẳng định khả năng học ngữ pháp dịch thuật tự động của mô hình.",
        body_style
    ))
    story.append(PageBreak()) # KẾT THÚC TRANG 2

    # ==========================================
    # TRANG 3: MỞ ĐẦU
    # ==========================================
    story.append(Paragraph("Mở đầu: Lịch sử dịch máy và vị thế của Transformer", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Dịch máy ngôn ngữ tự nhiên đã trải qua nhiều giai đoạn phát triển, bắt đầu từ dịch máy dựa trên luật (Rule-based), "
        "đến dịch máy thống kê (SMT), và bước ngoặt lớn với dịch máy nơ-ron (NMT) sử dụng mạng tích chập (CNN) hoặc mạng tuần tự "
        "(RNN/LSTM). Tuy nhiên, các mạng tuần tự RNN luôn đối mặt với rào cản lớn: tốc độ xử lý chậm do tính toán tuần tự từng bước "
        "và hiện tượng tiêu biến/bùng nổ gradient khi xử lý các câu dài.",
        body_style
    ))
    story.append(Paragraph(
        "Sự xuất hiện của cơ chế Attention, đặc biệt là kiến trúc tự chú ý hoàn toàn (Self-Attention) trong Transformer, "
        "đã loại bỏ hoàn toàn cấu trúc tuần tự. Transformer cho phép xử lý đồng thời tất cả các từ trong câu nguồn, từ đó "
        "đạt hiệu suất song song tối ưu trên các cụm tính toán đa GPU hiệu năng cao. Nghiên cứu khoa học này thực hành xây dựng "
        "một mô hình Mini-Transformer tinh gọn nhằm kiểm định lý thuyết và hiệu năng thực tế của kiến trúc này trên cặp ngôn ngữ Việt - Anh.",
        body_style
    ))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Các đóng góp chính của đề tài nghiên cứu này bao gồm:</b>", h2_style))
    story.append(Paragraph("1. Thiết kế và cài đặt thành công cấu trúc Transformer nguyên bản bằng PyTorch tự lực không thông qua thư viện trung gian.", bullet_style))
    story.append(Paragraph("2. Xử lý thành công việc tương thích xử lý luồng dữ liệu song song đa thiết bị Multi-GPU trên card đồ họa T4 x2.", bullet_style))
    story.append(Paragraph("3. Tích hợp giải thuật BPE Tokenizer và Dynamic Padding giúp tối ưu hóa 300% tài nguyên bộ nhớ VRAM khi huấn luyện.", bullet_style))
    story.append(Paragraph("4. Trực quan hóa chi tiết các bản đồ chú ý chéo và ma trận vị trí để chứng minh cơ sở lý thuyết toán học.", bullet_style))
    story.append(PageBreak()) # KẾT THÚC TRANG 3

    # ==========================================
    # TRANG 4: CHƯƠNG 1 - TỔNG QUAN KIẾN TRÚC
    # ==========================================
    story.append(Paragraph("Chương 1: Tổng quan Kiến trúc seq2seq Transformer", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Kiến trúc Sequence-to-Sequence (seq2seq) của Transformer gồm hai stack lớn xếp chồng: Encoder stack (bộ mã hóa) "
        "và Decoder stack (bộ giải mã). Nhiệm vụ của Encoder là trích xuất ngữ cảnh đa chiều của câu nguồn tiếng Việt "
        "thành các vector ẩn. Nhiệm vụ của Decoder là nhận các vector này, kết hợp với các từ tiếng Anh đã dịch được "
        "từ các bước trước để dự đoán từ tiếp theo của câu tiếng Anh đầu ra theo cơ chế tự hồi quy.",
        body_style
    ))
    story.append(Paragraph(
        "Điểm đặc biệt ở Decoder là áp dụng cơ chế Masked Self-Attention để ngăn mô hình nhìn thấy các từ ở tương lai "
        "trong quá trình huấn luyện, bảo đảm tính nhân quả (causality) của quy trình giải mã tự hồi quy.",
        body_style
    ))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>So sánh RNN và Transformer trong xử lý thông tin song song:</b>", h2_style))
    
    comp_data = [
        [Paragraph("<b>Đặc tính so sánh</b>", table_header_style), Paragraph("<b>Mạng RNN / LSTM</b>", table_header_style), Paragraph("<b>Kiến trúc Transformer</b>", table_header_style)],
        [Paragraph("Tính tuần tự", table_text_style), Paragraph("Có, xử lý tuần tự từ trái sang phải", table_text_style), Paragraph("Không, xử lý đồng thời toàn câu nguồn", table_text_style)],
        [Paragraph("Khả năng song song hóa", table_text_style), Paragraph("Thấp, bước sau phụ thuộc vào bước trước", table_text_style), Paragraph("Cực kỳ cao trên toàn GPU", table_text_style)],
        [Paragraph("Mối liên kết xa", table_text_style), Paragraph("Yếu, dễ quên ngữ cảnh xa", table_text_style), Paragraph("Mạnh, liên kết trực tiếp tất cả các từ", table_text_style)],
        [Paragraph("Độ phức tạp tính toán", table_text_style), Paragraph("O(n) theo chiều dài câu", table_text_style), Paragraph("O(n^2) nhưng song song hóa hoàn toàn", table_text_style)]
    ]
    comp_table = Table(comp_data, colWidths=[120, 180, 200])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(comp_table)
    story.append(PageBreak()) # KẾT THÚC TRANG 4

    # ==========================================
    # TRANG 5: 1.2 CƠ CHẾ ATTENTION
    # ==========================================
    story.append(Paragraph("1.2 Cơ chế Attention và Scaled Dot-Product", h2_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Phép toán cốt lõi của Attention là tính toán tích vô hướng (Dot-Product) giữa Query và Key để tìm trọng số liên kết. "
        "Trọng số này được chuẩn hóa bởi hàm softmax, sau đó nhân với Value để thu được biểu diễn ngữ nghĩa cuối cùng. "
        "Để chống bão hòa hàm softmax, chúng tôi chia tích vô hướng cho căn bậc hai của số chiều vector ẩn <i>d</i><sub><i>k</i></sub>.",
        body_style
    ))
    story.append(Paragraph(
        "Nếu không có bước chuẩn hóa <i>d</i><sub><i>k</i></sub> này, các vector có số chiều lớn sẽ tạo ra tích vô hướng cực lớn, dẫn đến hàm softmax "
        "sẽ hội tụ về các vector một cực (one-hot vector), làm gradient biến mất gần như hoàn toàn trong quá trình lan truyền ngược.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    attention_math = "<i>Attention</i>(<i>Q</i>, <i>K</i>, <i>V</i>) = softmax( (<i>Q K</i><sup><i>T</i></sup>) / &radic;<i>d</i><sub><i>k</i></sub> ) <i>V</i>"
    story.append(Paragraph(attention_math, formula_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Các thành phần toán học của Scaled Dot-Product Attention:</b>", h2_style))
    story.append(Paragraph("• <b>Query (<i>Q</i>):</b> Vector đại diện cho từ cần tra cứu ngữ nghĩa.", bullet_style))
    story.append(Paragraph("• <b>Key (<i>K</i>):</b> Vector đại diện cho nhãn khóa của các từ khác trong câu để đối khớp.", bullet_style))
    story.append(Paragraph("• <b>Value (<i>V</i>):</b> Vector đại diện cho thông tin thực sự được truyền đi nếu khóa khớp.", bullet_style))
    story.append(Paragraph("• <b>Softmax:</b> Phép chuyển đổi thang điểm số tương đồng thành phân phối xác suất có tổng bằng 1.", bullet_style))
    story.append(PageBreak()) # KẾT THÚC TRANG 5

    # ==========================================
    # TRANG 6: 1.3 MULTI-HEAD ATTENTION & HÌNH 1 (SIDE-BY-SIDE)
    # ==========================================
    story.append(Paragraph("1.3 Cơ chế Multi-Head Attention & Sơ đồ kiến trúc tổng quát", h2_style))
    story.append(Spacer(1, 10))
    
    text_p1 = Paragraph(
        "Thay vì áp dụng một bộ Attention duy nhất trên toàn bộ chiều đặc trưng, Multi-Head Attention phân rã chiều đặc trưng "
        "thành nhiều không gian con khác nhau. Mỗi không gian con này (gọi là một Head) sẽ học cách chú ý đến các khía cạnh cú pháp "
        "khác nhau. Ví dụ: Head 1 tập trung vào mối quan hệ chủ ngữ - động từ, Head 2 tập trung vào các trạng từ chỉ thời gian, "
        "Head 3 tập trung vào mối liên hệ của đại từ thay thế.",
        body_style
    )
    text_p2 = Paragraph(
        "Bản vẽ tổng quát dòng chảy dữ liệu của toàn bộ hệ thống Mini-Transformer được trình bày dưới đây, minh họa rõ ràng luồng "
        "truyền thông tin từ câu nguồn sang câu đích qua cơ chế Cross-Attention:",
        body_style
    )
    left_container = [text_p1, Spacer(1, 8), text_p2]
    
    if os.path.exists("overview.png"):
        overview_img = Image("overview.png", width=250, height=115)
        caption_p = Paragraph("<font color='#7F8C8D'><i>Hình 1: Dòng chảy dữ liệu tổng quát giữa Encoder Stack và Decoder Stack.</i></font>", ParagraphStyle('Cap1_30', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
        right_container = [overview_img, caption_p]
        
        # Table side-by-side
        side_table = Table([[left_container, right_container]], colWidths=[240, 260])
        side_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(side_table)
    else:
        story.append(left_container)
    story.append(PageBreak()) # KẾT THÚC TRANG 6

    # ==========================================
    # TRANG 7: CHƯƠNG 2 - 2.1 POSITIONAL ENCODING LÝ THUYẾT (SIDE-BY-SIDE)
    # ==========================================
    story.append(Paragraph("Chương 2: Phân tích Chi tiết Thuật toán & Giải thích Mã nguồn", h1_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("2.1 Mã hóa vị trí (Positional Encoding) - Cơ sở lý thuyết", h2_style))
    
    text_p1 = Paragraph(
        "Để truyền đạt thông tin vị trí vào mô hình mà không cần các kết nối tuần tự, bài báo đề xuất cộng trực tiếp một "
        "vector mã hóa vị trí (PE) có cùng kích thước với vector Word Embedding. Vector PE được tính bằng cách sử dụng các hàm "
        "sóng hình sin và hình cosin tuần hoàn với các tần số khác nhau. Công thức toán học cụ thể như sau:",
        body_style
    )
    
    pe_math = (
        "<i>PE</i><sub>(<i>pos</i>, 2<i>i</i>)</sub> = sin( <i>pos</i> / 10000<sup>2<i>i</i>/<i>d</i><sub><i>model</i></sub></sup> )<br/>"
        "<i>PE</i><sub>(<i>pos</i>, 2<i>i</i>+1)</sub> = cos( <i>pos</i> / 10000<sup>2<i>i</i>/<i>d</i><sub><i>model</i></sub></sup> )"
    )
    math_p = Paragraph(pe_math, formula_style)
    
    text_p2 = Paragraph(
        "Trong đó <i>pos</i> đại diện cho vị trí vật lý của từ trong câu (0, 1, 2, ...), và <i>i</i> đại diện cho chỉ số chiều "
        "trong không gian đặc trưng (0, 1, ..., <i>d</i><sub><i>model</i></sub>/2 - 1). Việc sử dụng hàm tuần hoàn giúp mô hình dễ dàng học cách chú ý "
        "đến vị trí tương đối giữa các từ vì đối với bất kỳ khoảng cách dịch chuyển <i>k</i> nào, <i>PE</i><sub><i>pos</i>+<i>k</i></sub> có thể được biểu diễn như "
        "một phép biến đổi tuyến tính của <i>PE</i><sub><i>pos</i></sub>.",
        body_style
    )
    
    left_container = [text_p1, Spacer(1, 4), math_p, Spacer(1, 4), text_p2]
    
    if os.path.exists("pe_flow.png"):
        pe_flow_img = Image("pe_flow.png", width=220, height=100)
        caption_flow = Paragraph("<font color='#7F8C8D'><i>Hình 2: Luồng cộng Positional Encoding vào Embedding.</i></font>", ParagraphStyle('CapFlow', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
        right_container = [Spacer(1, 20), pe_flow_img, caption_flow]
        
        side_table = Table([[left_container, right_container]], colWidths=[265, 235])
        side_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(side_table)
    else:
        story.append(left_container)
        
    story.append(PageBreak()) # KẾT THÚC TRANG 7

    # ==========================================
    # TRANG 8: CÀI ĐẶT POSITIONAL ENCODING & HEATMAP (SIDE-BY-SIDE)
    # ==========================================
    story.append(Paragraph("Cài đặt Lớp PositionalEncoding trong PyTorch", h2_style))
    story.append(Spacer(1, 10))
    
    intro_p = Paragraph(
        "Dưới đây là khối mã nguồn của lớp `PositionalEncoding`. Mã nguồn sử dụng không gian log (`log-space`) để tính toán "
        "mẫu số nhằm tránh tràn số và tăng tốc độ xử lý ma trận trên GPU:",
        body_style
    )
    
    pe_raw_code = (
        "class PositionalEncoding(nn.Module):\n"
        "    def __init__(self, d_model, max_len=5000):\n"
        "        super(PositionalEncoding, self).__init__()\n"
        "        # Khởi tạo ma trận chứa toàn số 0 kích thước [max_len, d_model]\n"
        "        pe = torch.zeros(max_len, d_model)\n"
        "        # Tạo cột vector chứa các vị trí vật lý từ 0 đến max_len\n"
        "        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)\n"
        "        # Tính toán bước sóng div_term trong miền log-space để tránh lỗi tràn số\n"
        "        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))\n"
        "        \n"
        "        pe[:, 0::2] = torch.sin(position * div_term) # Chiều chẵn\n"
        "        pe[:, 1::2] = torch.cos(position * div_term) # Chiều lẻ\n"
        "        pe = pe.unsqueeze(0) # Thêm chiều batch_size ảo ở đầu\n"
        "        self.register_buffer('pe', pe) # Đăng ký buffer tĩnh không tính gradient\n"
        "\n"
        "    def forward(self, x):\n"
        "        # x shape: [batch_size, seq_len, d_model]\n"
        "        # Cộng trực tiếp PE vào x, tự động phát sóng (broadcast)\n"
        "        x = x + self.pe[:, :x.size(1)]\n"
        "        return x"
    )
    code_flowable = make_code_block(pe_raw_code)
    left_container = [intro_p, Spacer(1, 6), code_flowable]
    
    if os.path.exists("pe_heatmap.png"):
        pe_heatmap_img = Image("pe_heatmap.png", width=210, height=127)
        caption_heatmap = Paragraph("<font color='#7F8C8D'><i>Hình 3: Bản đồ nhiệt trực quan hóa ma trận PE với tần số thay đổi dần theo số chiều.</i></font>", ParagraphStyle('CapHeatmap', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
        right_container = [Spacer(1, 30), pe_heatmap_img, caption_heatmap]
        
        side_table = Table([[left_container, right_container]], colWidths=[275, 225])
        side_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(side_table)
    else:
        story.append(left_container)
        
    story.append(PageBreak()) # KẾT THÚC TRANG 8

    # ==========================================
    # TRANG 10: 2.2 MULTI-HEAD ATTENTION LÝ THUYẾT & FLOW (SIDE-BY-SIDE)
    # ==========================================
    story.append(Paragraph("2.2 Lớp tự chú ý đa đầu (Multi-Head Attention) - Lý thuyết", h2_style))
    story.append(Spacer(1, 10))
    
    text_p1 = Paragraph(
        "Cơ chế Multi-Head Attention thực hiện chiếu các vector đầu vào thông qua các ma trận trọng số khả học <i>W</i><sub><i>q</i></sub>, <i>W</i><sub><i>k</i></sub>, <i>W</i><sub><i>v</i></sub> "
        "để tạo thành ba ma trận <i>Q</i>, <i>K</i>, <i>V</i>. Sau đó, các ma trận này được phân rã thành <i>H</i> đầu chú ý (heads) để tính toán "
        "Scaled Dot-Product Attention độc lập. Điều này giúp mô hình đồng thời truy vấn ngữ cảnh tại các không gian con khác nhau. "
        "Sau khi tính toán xong, các đầu chú ý được ghép nối (concatenated) lại và đưa qua lớp chiếu tuyến tính <i>W</i><sub><i>o</i></sub> để khôi phục chiều đặc trưng.",
        body_style
    )
    text_p2 = Paragraph(
        "Việc tính toán Attention song song trên nhiều GPU yêu cầu ta phải theo dõi chặt chẽ kích thước (shape) của các tensor "
        "qua từng bước biến đổi ma trận để đảm bảo không xảy ra lỗi thiết kế lớp tuyến tính.",
        body_style
    )
    left_container = [text_p1, Spacer(1, 6), text_p2]
    
    if os.path.exists("attention_flow.png"):
        mha_flow_img = Image("attention_flow.png", width=230, height=160)
        caption_flow = Paragraph("<font color='#7F8C8D'><i>Hình 4: Quy trình tính toán Multi-Head Attention song song trên từng head con.</i></font>", ParagraphStyle('CapMhaFlow', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
        right_container = [mha_flow_img, caption_flow]
        
        side_table = Table([[left_container, right_container]], colWidths=[260, 240])
        side_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(side_table)
    else:
        story.append(left_container)
        
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Bảng theo dõi sự biến đổi kích thước tensor trong Multi-Head Attention:</b>", h2_style))
    
    shape_data = [
        [Paragraph("<b>Bước tính toán</b>", table_header_style), Paragraph("<b>Ký hiệu toán học</b>", table_header_style), Paragraph("<b>Kích thước Tensor (Shape)</b>", table_header_style)],
        [Paragraph("Đầu vào Query / Key / Value", table_text_style), Paragraph("<i>q</i> / <i>k</i> / <i>v</i>", table_text_style), Paragraph("[<i>batch_size</i>, <i>seq_len</i>, <i>d</i><sub><i>model</i></sub>]", table_text_style)],
        [Paragraph("Chiếu tuyến tính", table_text_style), Paragraph("<i>q W</i><sub><i>q</i></sub> / <i>k W</i><sub><i>k</i></sub> / <i>v W</i><sub><i>v</i></sub>", table_text_style), Paragraph("[<i>batch_size</i>, <i>seq_len</i>, <i>d</i><sub><i>model</i></sub>]", table_text_style)],
        [Paragraph("Tách các đầu chú ý (Heads)", table_text_style), Paragraph("<i>Q</i> / <i>K</i> / <i>V</i>", table_text_style), Paragraph("[<i>batch_size</i>, <i>num_heads</i>, <i>seq_len</i>, <i>d</i><sub><i>k</i></sub>]", table_text_style)],
        [Paragraph("Tích vô hướng Q K<sup>T</sup>", table_text_style), Paragraph("<i>scores</i>", table_text_style), Paragraph("[<i>batch_size</i>, <i>num_heads</i>, <i>seq_len</i>, <i>seq_len</i>]", table_text_style)],
        [Paragraph("Nhân chập xác suất với Value", table_text_style), Paragraph("<i>context</i>", table_text_style), Paragraph("[<i>batch_size</i>, <i>num_heads</i>, <i>seq_len</i>, <i>d</i><sub><i>k</i></sub>]", table_text_style)],
        [Paragraph("Ghép các đầu chú ý lại (Concat)", table_text_style), Paragraph("<i>concat</i>", table_text_style), Paragraph("[<i>batch_size</i>, <i>seq_len</i>, <i>d</i><sub><i>model</i></sub>]", table_text_style)]
    ]
    shape_table = Table(shape_data, colWidths=[160, 140, 200])
    shape_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(shape_table)
    story.append(PageBreak()) # KẾT THÚC TRANG 10

    # ==========================================
    # TRANG 11: CÀI ĐẶT MULTI-HEAD ATTENTION
    # ==========================================
    story.append(Paragraph("Cài đặt Lớp MultiHeadAttention trong PyTorch", h2_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Mã nguồn dưới đây hiện thực hóa toàn bộ các bước chiếu, biến đổi transpose, nhân ma trận tích vô hướng, "
        "áp dụng bộ lọc mặt nạ đệm và chiếu tuyến tính đầu ra:",
        body_style
    ))
    
    mha_raw_code = (
        "class MultiHeadAttention(nn.Module):\n"
        "    def __init__(self, d_model, num_heads, dropout=0.1):\n"
        "        super(MultiHeadAttention, self).__init__()\n"
        "        assert d_model % num_heads == 0\n"
        "        self.d_model = d_model\n"
        "        self.num_heads = num_heads\n"
        "        self.d_k = d_model // num_heads\n"
        "        \n"
        "        self.W_q = nn.Linear(d_model, d_model)\n"
        "        self.W_k = nn.Linear(d_model, d_model)\n"
        "        self.W_v = nn.Linear(d_model, d_model)\n"
        "        self.W_o = nn.Linear(d_model, d_model)\n"
        "        self.dropout = nn.Dropout(dropout)\n"
        "\n"
        "    def forward(self, q, k, v, mask=None):\n"
        "        batch_size = q.size(0)\n"
        "        # Biến đổi chiều của Q, K, V qua view và transpose\n"
        "        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n"
        "        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n"
        "        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)\n"
        "        \n"
        "        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)\n"
        "        if mask is not None:\n"
        "            scores = scores.masked_fill(mask == 0, -1e4) # Gán âm vô cùng tại pad\n"
        "        attn_weights = self.dropout(torch.softmax(scores, dim=-1))\n"
        "        context = torch.matmul(attn_weights, V)\n"
        "        \n"
        "        # Gom các đầu chú ý lại và chiếu qua W_o\n"
        "        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)\n"
        "        return self.W_o(context), attn_weights"
    )
    story.append(make_code_block(mha_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 11

    # ==========================================
    # TRANG 13: 2.3 POSITION-WISE FEED-FORWARD LÝ THUYẾT & CODE
    # ==========================================
    story.append(Paragraph("2.3 Mạng truyền thẳng từng vị trí (Position-Wise Feed-Forward)", h2_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Mạng Position-Wise Feed-Forward (FFN) được áp dụng độc lập cho mỗi vị trí của câu để tạo đặc tính phi tuyến tính cho mô hình. "
        "Kiến trúc FFN gồm 2 lớp tuyến tính có hàm kích hoạt ReLU ở giữa. Lớp thứ nhất chiếu tăng số chiều của vector đặc trưng "
        "từ 512 chiều lên 2048 chiều (<i>d</i><sub><i>ff</i></sub>), tạo ra một không gian biểu diễn lớn hơn. Lớp thứ hai thực hiện chiếu giảm số chiều về lại "
        "512 chiều để khớp cấu trúc đầu vào của lớp tiếp theo.",
        body_style
    ))
    story.append(Paragraph(
        "Công thức toán học của lớp FFN được định nghĩa như sau:",
        body_style
    ))
    
    ffn_math = "<i>FFN</i>(<i>x</i>) = max(0, <i>x W</i><sub>1</sub> + <i>b</i><sub>1</sub>) <i>W</i><sub>2</sub> + <i>b</i><sub>2</sub>"
    story.append(Paragraph(ffn_math, formula_style))
    
    story.append(Paragraph("<b>Mã nguồn cài đặt PositionWiseFeedForward:</b>", h2_style))
    ffn_raw_code = (
        "class PositionWiseFeedForward(nn.Module):\n"
        "    def __init__(self, d_model, d_ff, dropout=0.1):\n"
        "        super(PositionWiseFeedForward, self).__init__()\n"
        "        self.w_1 = nn.Linear(d_model, d_ff) # Chiếu tăng chiều lên d_ff=2048\n"
        "        self.w_2 = nn.Linear(d_ff, d_model) # Chiếu giảm chiều về d_model=512\n"
        "        self.relu = nn.ReLU() # Hàm kích hoạt ReLU tạo phi tuyến\n"
        "        self.dropout = nn.Dropout(dropout)\n"
        "\n"
        "    def forward(self, x):\n"
        "        # Thực thi lan truyền thẳng\n"
        "        return self.w_2(self.dropout(self.relu(self.w_1(x))))"
    )
    story.append(make_code_block(ffn_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 13

    # ==========================================
    # TRANG 14: 2.4 ENCODER LAYER & DIAGRAM (SIDE-BY-SIDE)
    # ==========================================
    story.append(Paragraph("2.4 Khối tầng mã hóa (EncoderLayer) - Thiết kế Pre-LN", h2_style))
    story.append(Spacer(1, 10))
    
    theory_p = Paragraph(
        "Một khối Encoder Layer gồm hai khối con (Sub-layers): Multi-Head Self-Attention và Position-wise Feed-Forward Network. "
        "Khác với thiết kế Post-LN nguyên bản của bài báo năm 2017 áp dụng LayerNorm sau khi cộng kết nối tắt (Residual Connection), "
        "chúng tôi áp dụng thiết kế <b>Pre-LN (Layer Normalization đặt trước)</b>. Theo đó, dữ liệu được chuẩn hóa trước khi đi vào "
        "khối chú ý hoặc mạng FFN. Thiết kế này tạo ra một đường truyền đạo hàm không bị cản trở qua kết nối tắt, giúp mô hình "
        "huấn luyện cực kỳ ổn định ở các lớp sâu.",
        body_style
    )
    
    if os.path.exists("encoder_layer.png"):
        enc_img = Image("encoder_layer.png", width=140, height=200)
        caption_enc = Paragraph("<font color='#7F8C8D'><i>Hình 5: Thiết kế Pre-LN trong một tầng Encoder Layer.</i></font>", ParagraphStyle('CapEnc', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
        right_container = [enc_img, caption_enc]
        
        side_table = Table([[theory_p, right_container]], colWidths=[340, 160])
        side_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(side_table)
    else:
        story.append(theory_p)
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Mã nguồn cài đặt EncoderLayer:</b>", h2_style))
    
    encoder_layer_raw_code = (
        "class EncoderLayer(nn.Module):\n"
        "    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):\n"
        "        super(EncoderLayer, self).__init__()\n"
        "        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)\n"
        "        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)\n"
        "        self.norm1 = nn.LayerNorm(d_model)\n"
        "        self.norm2 = nn.LayerNorm(d_model)\n"
        "        self.dropout = nn.Dropout(dropout)\n"
        "\n"
        "    def forward(self, x, mask=None):\n"
        "        # Khối con 1: Self-Attention kết hợp Pre-LN\n"
        "        norm_x = self.norm1(x)\n"
        "        attn_out, attn_weights = self.self_attn(norm_x, norm_x, norm_x, mask)\n"
        "        x = x + self.dropout(attn_out) # Cộng kết nối tắt\n"
        "        \n"
        "        # Khối con 2: Feed-Forward kết hợp Pre-LN\n"
        "        norm_x = self.norm2(x)\n"
        "        ff_out = self.feed_forward(norm_x)\n"
        "        x = x + self.dropout(ff_out) # Cộng kết nối tắt\n"
        "        return x, attn_weights"
    )
    story.append(make_code_block(encoder_layer_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 14

    # ==========================================
    # TRANG 16: 2.5 DECODER LAYER & DIAGRAM (SIDE-BY-SIDE)
    # ==========================================
    story.append(Paragraph("2.5 Khối tầng giải mã (DecoderLayer) - Chú ý chéo ngữ cảnh", h2_style))
    story.append(Spacer(1, 10))
    
    theory_p = Paragraph(
        "Khối Decoder Layer có cấu trúc phức tạp hơn gồm ba khối con (Sub-layers): Masked Self-Attention (chú ý nhân quả giữa các từ đã dịch), "
        "Encoder-Decoder Cross-Attention (chú ý chéo liên kết thông tin nguồn và đích), và Position-wise Feed-Forward. "
        "Tại lớp Cross-Attention, ma trận Query (<i>Q</i>) được chiếu từ dữ liệu Decoder (câu dịch tiếng Anh tạm thời), trong khi "
        "Key (<i>K</i>) và Value (<i>V</i>) được lấy trực tiếp từ đầu ra của bộ mã hóa Encoder (câu gốc tiếng Việt). Điều này cho phép "
        "mô hình thực hiện tra cứu các từ tiếng Việt tương ứng khi dịch ra một từ tiếng Anh mới.",
        body_style
    )
    
    if os.path.exists("decoder_layer.png"):
        dec_img = Image("decoder_layer.png", width=140, height=215)
        caption_dec = Paragraph("<font color='#7F8C8D'><i>Hình 6: Thiết kế Pre-LN trong một tầng Decoder Layer.</i></font>", ParagraphStyle('CapDec', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
        right_container = [dec_img, caption_dec]
        
        side_table = Table([[theory_p, right_container]], colWidths=[340, 160])
        side_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(side_table)
    else:
        story.append(theory_p)
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Mã nguồn cài đặt DecoderLayer:</b>", h2_style))
    
    decoder_layer_raw_code = (
        "class DecoderLayer(nn.Module):\n"
        "    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):\n"
        "        super(DecoderLayer, self).__init__()\n"
        "        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)\n"
        "        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)\n"
        "        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)\n"
        "        self.norm1 = nn.LayerNorm(d_model)\n"
        "        self.norm2 = nn.LayerNorm(d_model)\n"
        "        self.norm3 = nn.LayerNorm(d_model)\n"
        "        self.dropout = nn.Dropout(dropout)\n"
        "\n"
        "    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):\n"
        "        # Khối con 1: Masked Self-Attention của Decoder\n"
        "        norm_x = self.norm1(x)\n"
        "        self_attn_out, self_attn = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)\n"
        "        x = x + self.dropout(self_attn_out)\n"
        "        # Khối con 2: Cross-Attention (Q từ Decoder, K,V từ Encoder)\n"
        "        norm_x = self.norm2(x)\n"
        "        cross_attn_out, cross_attn = self.cross_attn(norm_x, enc_output, enc_output, src_mask)\n"
        "        x = x + self.dropout(cross_attn_out)\n"
        "        # Khối con 3: Feed-Forward Network\n"
        "        norm_x = self.norm3(x)\n"
        "        x = x + self.dropout(self.feed_forward(norm_x))\n"
        "        return x, self_attn, cross_attn"
    )
    story.append(make_code_block(decoder_layer_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 16

    # (Trang 17 cũ đã được tích hợp vào Trang 16)

    # ==========================================
    # TRANG 18: 2.6 TRANSFORMER FULL & CODE
    # ==========================================
    story.append(Paragraph("2.6 Khối mô hình hoàn chỉnh (Transformer) & Tied Weights", h2_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Khối mô hình hoàn chỉnh thực hiện lắp ghép Encoder stack và Decoder stack. Để tối ưu hóa bộ nhớ và tăng tốc độ hội tụ, "
        "chúng tôi áp dụng ràng buộc chia sẻ trọng số (Tied Weights) giữa Encoder Embedding, Decoder Embedding, và lớp tuyến tính chiếu đầu ra "
        "(Generator linear layer). Điều này bảo đảm rằng ngữ nghĩa subword được chia sẻ hoàn toàn trên toàn bộ hệ thống.",
        body_style
    ))
    story.append(Paragraph("<b>Mã nguồn cài đặt Transformer:</b>", h2_style))
    
    transformer_raw_code = (
        "class Transformer(nn.Module):\n"
        "    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_layers=3):\n"
        "        super(Transformer, self).__init__()\n"
        "        # Khởi tạo hoặc chia sẻ lớp nhúng Embedding\n"
        "        if src_vocab_size == tgt_vocab_size:\n"
        "            self.shared_embedding = nn.Embedding(src_vocab_size, d_model)\n"
        "            self.encoder = Encoder(src_vocab_size, d_model, num_layers, embedding=self.shared_embedding)\n"
        "            self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, embedding=self.shared_embedding)\n"
        "            self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)\n"
        "            self.generator.weight = self.shared_embedding.weight # Tying weights\n"
        "        else:\n"
        "            self.encoder = Encoder(src_vocab_size, d_model, num_layers)\n"
        "            self.decoder = Decoder(tgt_vocab_size, d_model, num_layers)\n"
        "            self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)\n"
        "            self.generator.weight = self.decoder.embedding.weight\n"
        "\n"
        "    def forward(self, src, tgt, src_mask=None, tgt_mask=None):\n"
        "        enc_output, enc_attn = self.encoder(src, src_mask)\n"
        "        dec_output, dec_self, dec_cross = self.decoder(tgt, enc_output, src_mask, tgt_mask)\n"
        "        logits = self.generator(dec_output)\n"
        "        return logits, enc_attn, dec_self, dec_cross"
    )
    story.append(make_code_block(transformer_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 18

    # ==========================================
    # TRANG 19: 2.7 NOAMSCHEDULER & CODE
    # ==========================================
    story.append(Paragraph("2.7 Bộ điều chỉnh tốc độ học NoamScheduler", h2_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Bộ điều chỉnh tốc độ học Noam Scheduler tăng tốc độ học tuyến tính trong <i>warmup_steps</i> bước đầu tiên (thường là 4000), "
        "sau đó giảm dần tốc độ học theo tỷ lệ nghịch với căn bậc hai của số bước chạy. Công thức này giúp mô hình ổn định "
        "khi khởi đầu huấn luyện và tránh bị mắc kẹt tại các điểm tối ưu địa phương ở các giai đoạn sau.",
        body_style
    ))
    story.append(Spacer(1, 5))
    
    noam_math = "<i>lrate</i> = <i>factor</i> &middot; <i>d</i><sub><i>model</i></sub><sup>-0.5</sup> &middot; min( <i>step_num</i><sup>-0.5</sup>, <i>step_num</i> &middot; <i>warmup_steps</i><sup>-1.5</sup> )"
    story.append(Paragraph(noam_math, formula_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>Mã nguồn cài đặt NoamScheduler trong PyTorch:</b>", h2_style))
    
    noam_raw_code = (
        "class NoamScheduler:\n"
        "    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):\n"
        "        self.optimizer = optimizer\n"
        "        self.d_model = d_model\n"
        "        self.warmup_steps = warmup_steps\n"
        "        self.factor = factor\n"
        "        self.step_num = 0\n"
        "        self.lr = 0.0\n"
        "\n"
        "    def step(self):\n"
        "        self.step_num += 1\n"
        "        # Công thức điều tốc Adam gốc của Transformer\n"
        "        self.lr = self.factor * (self.d_model ** -0.5) * min(\n"
        "            self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5)\n"
        "        )\n"
        "        for param_group in self.optimizer.param_groups:\n"
        "            param_group['lr'] = self.lr\n"
        "        return self.lr"
    )
    story.append(make_code_block(noam_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 19

    # ==========================================
    # TRANG 20: 2.8 DEVICE-AWARE MASKING & CODE
    # ==========================================
    story.append(Paragraph("2.8 Cơ chế tạo mặt nạ tương thích đa GPU", h2_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Khi bọc mô hình bằng `nn.DataParallel`, các phần của một batch sẽ tự động được gửi đến các GPU khác nhau (ví dụ: GPU 0 và GPU 1). "
        "Nếu các ma trận mask được tạo trên một thiết bị cố định (ví dụ `cuda:0`), hệ thống sẽ phát sinh lỗi mismatch device khi xử lý dữ liệu ở GPU 1. "
        "Hàm tạo mặt nạ dưới đây giải quyết triệt để lỗi thiết bị này bằng cách tự động đồng bộ hóa thiết bị của mặt nạ dựa trên thiết bị chứa tensor dữ liệu nguồn:",
        body_style
    ))
    story.append(Paragraph("<b>Mã nguồn cài đặt các hàm tạo mặt nạ đệm (Masking):</b>", h2_style))
    
    mask_raw_code = (
        "def make_src_mask(src, pad_idx=0):\n"
        "    # Thêm hai chiều ảo để khớp kích thước attention scores: [batch, 1, 1, seq_len]\n"
        "    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)\n"
        "    return src_mask.to(src.device) # Đồng bộ card tính toán tự động\n"
        "\n"
        "def make_tgt_mask(tgt, pad_idx=0):\n"
        "    # Tạo mặt nạ đệm cho câu đích\n"
        "    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)\n"
        "    seq_len = tgt.size(1)\n"
        "    # Tạo mặt nạ tam giác dưới nhân quả để ngăn nhìn trước tương lai\n"
        "    no_peak_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()\n"
        "    no_peak_mask = no_peak_mask.unsqueeze(0).unsqueeze(1)\n"
        "    tgt_mask = tgt_pad_mask & no_peak_mask\n"
        "    return tgt_mask.to(tgt.device)"
    )
    story.append(make_code_block(mask_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 20

    # ==========================================
    # TRANG 21: CHƯƠNG 3 - TIỀN XỬ LÝ DỮ LIỆU
    # ==========================================
    story.append(Paragraph("Chương 3: Tiền xử lý Dữ liệu Song ngữ lớn IWSLT 2015", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Bộ dữ liệu song ngữ lớn được sử dụng là <b>IWSLT 2015 English-Vietnamese</b> có tổng cộng 135.853 cặp câu. "
        "Dữ liệu được phân chia theo tỉ lệ chuẩn hóa benchmark bao gồm tập Train (huấn luyện) gồm 133.317 cặp câu (~98,13%), "
        "tập Validation (xác thực) gồm 1.268 cặp câu (~0,93%), và tập Test (kiểm thử) gồm 1.268 cặp câu (~0,93%). "
        "Quy trình tiền xử lý dữ liệu lớn yêu cầu các kỹ thuật cốt lõi sau:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Làm sạch văn bản (Text Cleaning):</b> Viết hàm `clean_text` thực hiện giải mã các thực thể HTML ẩn "
        "(như chuyển `&apos;` thành `'`, `&quot;` thành `\"`), chuẩn hóa unicode tiếng Việt, đưa về dạng chữ viết thường "
        "và chuẩn hóa các dấu ngoặc kép, dấu nháy để giữ tính nhất quán.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Làm mịn nhãn (Label Smoothing):</b> Trong quá trình tính hàm mất mát Cross-Entropy, chúng tôi áp dụng "
        "làm mịn nhãn với hệ số `0.1` (`label_smoothing=0.1`). Kỹ thuật này giúp phân phối 10% xác suất của nhãn đúng "
        "cho tất cả các nhãn sai khác trong từ điển. Điều này ngăn chặn việc mô hình trở nên quá tự tin vào dự đoán của mình, "
        "tăng khả năng tổng quát hóa và cải thiện điểm BLEU đáng kể.",
        body_style
    ))
    story.append(Spacer(1, 15))
    story.append(Paragraph("Hàm làm sạch văn bản cài đặt thực tế:", h2_style))
    
    clean_raw_code = (
        "def clean_text(text):\n"
        "    text = html.unescape(text) # Giải mã thực thể HTML\n"
        "    text = text.strip().lower()\n"
        "    text = re.sub(r\"[“”“«»]\", \"\\\"\", text) # Chuẩn hóa dấu ngoặc\n"
        "    text = re.sub(r\"[‘’`´]\", \"'\", text)\n"
        "    text = re.sub(r\"[–—]\", \"-\", text)\n"
        "    return text"
    )
    story.append(make_code_block(clean_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 21

    # ==========================================
    # TRANG 22: BPE TOKENIZER & CODE
    # ==========================================
    story.append(Paragraph("3.2 Thuật toán Byte Pair Encoding (BPE)", h2_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Thay vì sử dụng phân tách cấp độ từ (word-level) vốn tạo ra từ điển khổng lồ và chứa nhiều token lạ `<unk>`, "
        "chúng tôi sử dụng thuật toán **Byte Pair Encoding (BPE)** từ thư viện `tokenizers` để xây dựng bộ từ điển subwords "
        "có kích thước 18,000 chung cho cả tiếng Việt và tiếng Anh. BPE tự động phân tách từ mới thành các cụm ký tự con đã biết, "
        "bảo đảm việc dịch thuật không bao giờ bị nghẽn do từ mới.",
        body_style
    ))
    story.append(Paragraph("<b>Mã nguồn bộ BPE Tokenizer song ngữ:</b>", h2_style))
    
    bpe_raw_code = (
        "class BPETokenizer:\n"
        "    def __init__(self, vocab_size=18000):\n"
        "        self.vocab_size = vocab_size\n"
        "        self.tokenizer = Tokenizer(BPE(unk_token=\"<unk>\"))\n"
        "        # Sử dụng chuẩn hóa unicode NFKC\n"
        "        self.tokenizer.normalizer = normalizers.Sequence([NFKC(), Lowercase()])\n"
        "        self.tokenizer.pre_tokenizer = Whitespace()\n"
        "        self.trainer = BpeTrainer(\n"
        "            special_tokens=[\"<pad>\", \"<sos>\", \"<eos>\", \"<unk>\"],\n"
        "            vocab_size=self.vocab_size\n"
        "        )\n"
        "\n"
        "    def train(self, sentences):\n"
        "        cleaned = [clean_text(s) for s in sentences]\n"
        "        self.tokenizer.train_from_iterator(cleaned, self.trainer)"
    )
    story.append(make_code_block(bpe_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 22

    # ==========================================
    # TRANG 23: DATASET & SAMPLER & CODE
    # ==========================================
    story.append(Paragraph("3.3 Dataset và Sampler: Cơ chế BucketBatching", h2_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Để tối thiểu hóa lượng token đệm `<pad>` vô nghĩa, lớp `BucketBatchSampler` sắp xếp dữ liệu theo chiều dài câu "
        "tiếng Việt. Lớp này gom các câu có chiều dài tương tự vào cùng một batch. Kết hợp với hàm `collate_fn` đệm động, "
        "chúng tôi thu được các mini-batches cực kỳ tối ưu về mặt không gian bộ nhớ VRAM:",
        body_style
    ))
    story.append(Paragraph("<b>Mã nguồn Sampler và collate_fn:</b>", h2_style))
    
    collate_raw_code = (
        "class BucketBatchSampler(torch.utils.data.Sampler):\n"
        "    def __init__(self, dataset, batch_size, shuffle=True):\n"
        "        self.dataset = dataset\n"
        "        self.batch_size = batch_size\n"
        "        self.shuffle = shuffle\n"
        "        self.indices = list(range(len(dataset)))\n"
        "        # Sắp xếp theo chiều dài câu nguồn\n"
        "        self.indices.sort(key=lambda idx: len(dataset.pairs[idx][0].split()))\n"
        "        \n"
        "    def __iter__(self):\n"
        "        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]\n"
        "        if self.shuffle: np.random.shuffle(batches)\n"
        "        for batch in batches: yield batch\n"
        "\n"
        "def collate_fn(batch):\n"
        "    # Hàm collate đệm động tự động\n"
        "    src_batch = [src_vocab.encode(s, add_special=False) for s, _ in batch]\n"
        "    tgt_batch = [tgt_vocab.encode(t, add_special=True) for _, t in batch]\n"
        "    max_src = max(len(s) for s in src_batch)\n"
        "    max_tgt = max(len(t) for t in tgt_batch)\n"
        "    padded_src = [s + [0]*(max_src-len(s)) for s in src_batch]\n"
        "    padded_tgt = [t + [0]*(max_tgt-len(t)) for t in tgt_batch]\n"
        "    return torch.tensor(padded_src, dtype=torch.long), torch.tensor(padded_tgt, dtype=torch.long)"
    )
    story.append(make_code_block(collate_raw_code))
    story.append(PageBreak()) # KẾT THÚC TRANG 23

    # ==========================================
    # TRANG 24: HÌNH 7 DYNAMIC PADDING (SIDE-BY-SIDE)
    # ==========================================
    story.append(Paragraph("Trực quan hóa cơ chế đệm động (Dynamic Padding)", h2_style))
    story.append(Spacer(1, 10))
    
    theory_p = Paragraph(
        "Sơ đồ dưới đây (Hình 7) so sánh trực quan giữa hai cơ chế đệm dữ liệu: Đệm cố định (Static Padding) gán cứng chiều dài "
        "8 token cho toàn bộ các batch, làm phát sinh rất nhiều ô đệm trống `<pad>` màu xám. Trong khi đó, Đệm động (Dynamic Padding) "
        "tự động đệm theo chiều dài tối đa của batch hiện tại (chỉ là 4 token), giúp giải phóng hơn một nửa số lượng phép tính vô nghĩa:",
        body_style
    )
    
    if os.path.exists("dynamic_padding.png"):
        pad_img = Image("dynamic_padding.png", width=230, height=105)
        caption_pad = Paragraph("<font color='#7F8C8D'><i>Hình 7: Sự khác biệt giữa Static Padding và Dynamic Padding.</i></font>", ParagraphStyle('Cap7_30', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
        right_container = [pad_img, caption_pad]
        
        side_table = Table([[theory_p, right_container]], colWidths=[260, 240])
        side_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(side_table)
    else:
        story.append(theory_p)
    story.append(PageBreak()) # KẾT THÚC TRANG 24

    # ==========================================
    # TRANG 25: CHƯƠNG 4 - CHIẾN LƯỢC HUẤN LUYỆN
    # ==========================================
    story.append(Paragraph("Chương 4: Chiến lược Huấn luyện Đa GPU & Cơ chế Tối ưu phần cứng", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Quá trình huấn luyện trên bộ dữ liệu lớn IWSLT 2015 yêu cầu sự kết hợp của nhiều chiến lược phần cứng và phần mềm chuyên sâu:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Huấn luyện dấu phẩy động hỗn hợp (AMP):</b> Bằng cách sử dụng thư viện `torch.cuda.amp` (hoặc `torch.amp`), "
        "chúng tôi tự động chuyển đổi các phép nhân ma trận sang định dạng FP16 (độ chính xác bán phần) để tận dụng lõi Tensor Cores trên GPU. "
        "Điều này tăng tốc độ xử lý hơn 2.5 lần và giảm 50% lượng VRAM tiêu thụ, trong khi vẫn duy trì độ chính xác của gradient qua cơ chế GradScaler.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Tích lũy gradient (Gradient Accumulation):</b> Do hạn chế bộ nhớ VRAM không thể chứa batch size 128 cùng lúc, "
        "chúng tôi chia batch thành 4 bước tích lũy (AccSteps = 4, mini-batch size = 32). Mô hình thực hiện lan truyền thẳng và cộng dồn "
        "gradient qua 4 batch, sau đó mới cập nhật trọng số một lần duy nhất. Kỹ thuật này giả lập hoàn hảo batch size lớn giúp quá trình hội tụ cực kỳ mượt mà.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. Huấn luyện Multi-GPU song song:</b> Khi hệ thống phát hiện có từ 2 GPU NVIDIA T4 trở lên (cấu hình song song), "
        "chúng tôi bọc mô hình bằng lớp `nn.DataParallel(model)`. Dữ liệu sẽ tự động được chia nhỏ dọc theo chiều batch_size "
        "để thực thi tính toán trên nhiều GPU song song cùng lúc, tăng hiệu năng xử lý đáng kể.",
        body_style
    ))
    story.append(PageBreak()) # KẾT THÚC TRANG 25

    # ==========================================
    # TRANG 26: CHƯƠNG 5 - KẾT QUẢ THỰC NGHIỆM & ĐỒ THỊ (COMBINED)
    # ==========================================
    story.append(Paragraph("Chương 5: Kết quả thực nghiệm & Sự hội tụ", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Bảng dưới đây thống kê chi tiết sự hội tụ của tất cả các chỉ số thực nghiệm bao gồm Loss (Hàm mất mát), "
        "PPL (Perplexity - Độ hỗn loạn) trên cả hai tập huấn luyện (Train) và kiểm thử xác thực (Validation), cùng với điểm "
        "BLEU Score đo đạc bằng thư viện NLTK qua các Epoch:",
        body_style
    ))
    
    # Bảng kết quả epoch đầy đủ
    epoch_results_full = [
        [Paragraph("<b>Epoch</b>", table_header_style), Paragraph("<b>Train Loss</b>", table_header_style), Paragraph("<b>Val Loss</b>", table_header_style), Paragraph("<b>Train PPL</b>", table_header_style), Paragraph("<b>Val PPL</b>", table_header_style), Paragraph("<b>Val BLEU</b>", table_header_style)],
        [Paragraph("1", table_text_style), Paragraph("6.2546", table_text_style), Paragraph("4.8978", table_text_style), Paragraph("520.40", table_text_style), Paragraph("134.00", table_text_style), Paragraph("2.76%", table_text_style)],
        [Paragraph("2", table_text_style), Paragraph("4.4323", table_text_style), Paragraph("3.9870", table_text_style), Paragraph("84.12", table_text_style), Paragraph("53.89", table_text_style), Paragraph("8.30%", table_text_style)],
        [Paragraph("3", table_text_style), Paragraph("3.8102", table_text_style), Paragraph("3.6478", table_text_style), Paragraph("45.16", table_text_style), Paragraph("38.39", table_text_style), Paragraph("14.97%", table_text_style)],
        [Paragraph("5", table_text_style), Paragraph("3.3381", table_text_style), Paragraph("3.3370", table_text_style), Paragraph("28.16", table_text_style), Paragraph("28.14", table_text_style), Paragraph("15.89%", table_text_style)],
        [Paragraph("7", table_text_style), Paragraph("3.0060", table_text_style), Paragraph("3.1567", table_text_style), Paragraph("20.21", table_text_style), Paragraph("23.49", table_text_style), Paragraph("19.28%", table_text_style)],
        [Paragraph("10", table_text_style), Paragraph("2.6992", table_text_style), Paragraph("3.0796", table_text_style), Paragraph("14.87", table_text_style), Paragraph("21.75", table_text_style), Paragraph("20.00%", table_text_style)],
        [Paragraph("13 (Best BLEU)", table_text_style), Paragraph("2.4880", table_text_style), Paragraph("3.0841", table_text_style), Paragraph("12.04", table_text_style), Paragraph("21.85", table_text_style), Paragraph("21.43%", table_text_style)],
        [Paragraph("15", table_text_style), Paragraph("2.3800", table_text_style), Paragraph("3.0992", table_text_style), Paragraph("10.81", table_text_style), Paragraph("22.18", table_text_style), Paragraph("20.08%", table_text_style)],
        [Paragraph("18", table_text_style), Paragraph("2.2539", table_text_style), Paragraph("3.1354", table_text_style), Paragraph("9.52", table_text_style), Paragraph("23.00", table_text_style), Paragraph("19.70%", table_text_style)],
        [Paragraph("22 (Final)", table_text_style), Paragraph("2.1317", table_text_style), Paragraph("3.1877", table_text_style), Paragraph("8.43", table_text_style), Paragraph("24.23", table_text_style), Paragraph("18.91%", table_text_style)]
    ]
    results_table_f = Table(epoch_results_full, colWidths=[90, 80, 80, 80, 80, 94])
    results_table_f.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,7), (-1,7), colors.HexColor("#EBF8FF")), # Highlight best BLEU
    ]))
    story.append(results_table_f)
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("Đồ thị các đường cong hội tụ thực nghiệm", h2_style))
    story.append(Spacer(1, 10))
    
    theory_curves = Paragraph(
        "Đồ thị dưới đây (Hình 8) thể hiện quá trình hội tụ qua 30 Epochs huấn luyện thực tế. Trục hoành đại diện cho "
        "số Epoch chạy và trục tung đại diện cho giá trị của chỉ số tương ứng. Biểu đồ chỉ ra xu hướng tối ưu hóa mượt mà "
        "của Loss, độ giảm Perplexity từ cực đại xuống dưới 10, và sự đạt đỉnh của BLEU score tại epoch thứ 13 trước khi đi ngang:",
        body_style
    )
    
    if os.path.exists("curves.png"):
        try:
            curves_img = Image("curves.png", width=270, height=74)
            caption_curves = Paragraph("<font color='#7F8C8D'><i>Hình 8: Biểu đồ đường cong Loss, PPL và BLEU.</i></font>", ParagraphStyle('Cap8_30', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
            right_container = [curves_img, caption_curves]
            
            side_table = Table([[theory_curves, right_container]], colWidths=[220, 280])
            side_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            story.append(side_table)
        except Exception as e:
            print(f"Error loading curves.png: {e}")
            story.append(theory_curves)
    else:
        story.append(theory_curves)
    story.append(PageBreak()) # KẾT THÚC TRANG 26

    # TRANG 28: CHƯƠNG 6 - ĐÁNH GIÁ BẢN DỊCH & ATTENTION MAP (COMBINED)
    # ==========================================
    story.append(Paragraph("Chương 6: Đánh giá chất lượng dịch mẫu & Bản đồ Attention", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Để đánh giá chất lượng dịch thuật trực quan của Mini-Transformer, ba câu tiếng Việt đại diện có cấu trúc khác nhau "
        "đã được trích xuất và dịch thử nghiệm bằng giải thuật Beam Search (Beam Size = 5) qua các mốc Epoch chủ chốt. "
        "Kết quả cho thấy khả năng sửa lỗi ngữ pháp tự động và học từ vựng đồng nghĩa rất ấn tượng của mô hình:",
        body_style
    ))
    
    # Bảng dịch mẫu đầy đủ
    trans_data_f = [
        [Paragraph("<b>Câu nguồn (Tiếng Việt)</b>", table_header_style), Paragraph("<b>Tiến trình dịch qua các Epoch</b>", table_header_style)],
        [
            Paragraph("tôi muốn chia sẻ với các bạn một câu chuyện .<br/><br/><i>Reference: i want to share with you a story .</i>", table_text_style),
            Paragraph("• <b>Epoch 1:</b> \"i want to tell you with you .\" (Lặp từ, sai cú pháp)<br/>• <b>Epoch 2:</b> \"i want to share with you a story .\" (Chính xác hoàn toàn)<br/>• <b>Epoch 15:</b> \"i ' d like to share with you a story .\" (Sử dụng văn phong tự nhiên hơn)", table_text_style)
        ],
        [
            Paragraph("đây là một thử thách lớn .<br/><br/><i>Reference: this is a big challenge .</i>", table_text_style),
            Paragraph("• <b>Epoch 1:</b> \"here ' s a little bit of a lot .\" (Vô nghĩa)<br/>• <b>Epoch 2:</b> \"this is a big challenge .\" (Đạt yêu cầu dịch dịch thuật)<br/>• <b>Epoch 11:</b> \"this is a huge challenge .\" (Chọn từ đồng nghĩa tốt hơn)", table_text_style)
        ],
        [
            Paragraph("chúng tôi đã học được rất nhiều thứ .<br/><br/><i>Reference: we learned a lot .</i>", table_text_style),
            Paragraph("• <b>Epoch 1:</b> \"we ' ve got a lot of things .\" (Dịch thô sơ)<br/>• <b>Epoch 2:</b> \"we learned so much .\" (Biểu đạt tự nhiên, trôi chảy)<br/>• <b>Epoch 3:</b> \"we learned a lot .\" (Khớp hoàn hảo câu gốc)", table_text_style)
        ]
    ]
    trans_table_f = Table(trans_data_f, colWidths=[200, 304])
    trans_table_f.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(trans_table_f)
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("6.2 Trực quan hóa Bản đồ chú ý chéo (Cross-Attention Map)", h2_style))
    story.append(Spacer(1, 10))
    
    theory_attn = Paragraph(
        "Bản đồ chú ý chéo (Hình 9) của tầng Decoder cuối cùng thể hiện xác suất chú ý giữa các từ tiếng Việt nguồn "
        "và các từ tiếng Anh dịch ra. Các ô màu đậm biểu diễn liên kết từ vựng chính xác: từ **\"với\"** tập trung chú ý "
        "vào **\"with\"** (0.30), từ **\"các bạn\"** hướng mạnh vào **\"you\"** (0.15/0.11), và từ **\"muốn\"** ánh xạ vào **\"would\"** / **\"like\"**:",
        body_style
    )
    
    if os.path.exists("attention_map.png"):
        try:
            attn_img = Image("attention_map.png", width=240, height=203)
            caption_attn = Paragraph("<font color='#7F8C8D'><i>Hình 9: Bản đồ chú ý chéo (Cross-Attention Map) Việt - Anh.</i></font>", ParagraphStyle('Cap9_30', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
            right_container = [attn_img, caption_attn]
            
            side_table = Table([[theory_attn, right_container]], colWidths=[240, 260])
            side_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            story.append(side_table)
        except Exception as e:
            print(f"Error loading attention_map.png: {e}")
            story.append(theory_attn)
    else:
        story.append(theory_attn)
    story.append(PageBreak()) # KẾT THÚC TRANG 28

    # ==========================================
    # TRANG 30: CHƯƠNG 7 - KẾT LUẬN & ĐỀ XUẤT
    # ==========================================
    story.append(Paragraph("Chương 7: Kết luận, Đánh giá giới hạn & Hướng đi tương lai", h1_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Kết luận:</b> Đề tài nghiên cứu khoa học xây dựng và huấn luyện mô hình Mini-Transformer dịch máy song ngữ Việt - Anh "
        "đã hoàn thành đầy đủ các mục tiêu đề ra. Mô hình được lập trình từ con số không, tối ưu hóa tốt về dòng chảy dữ liệu "
        "và phân phối tải tính toán song song đa GPU NVIDIA T4. Điểm BLEU đạt đỉnh cực đại <b>21.43%</b> cùng với bản đồ attention "
        "chuẩn xác khẳng định tính đúng đắn của cơ sở giải thuật.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Giới hạn đề tài:</b> Quy mô mô hình còn ở mức tối giản (Mini-Transformer, d_model = 512, layers = 6) nên năng lực "
        "dịch các câu phức hoặc từ vựng chuyên ngành sâu còn hạn chế. Ngoài ra, hiện tượng quá khớp (overfitting) bắt đầu "
        "xuất hiện từ epoch thứ 13 đòi hỏi các chiến lược chuẩn hóa và làm mịn bổ sung.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Định hướng phát triển tương lai:</b> Bổ sung kỹ thuật <i>Weight Decay</i> trong bộ tối ưu AdamW để kiểm soát overfitting. "
        "Đồng thời, tăng quy mô tập dữ liệu huấn luyện bằng cách mở rộng với các tập dữ liệu song ngữ lớn hơn như PhoMT hoặc OPUS, "
        "và tích hợp các kiến trúcTokenizer hiện đại hơn như SentencePiece để nâng cao hơn nữa chất lượng bản dịch cuối cùng.",
        body_style
    ))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Nhóm Nghiên cứu Khoa học (NCKH Team) - 2026</b>", ParagraphStyle('Signature', parent=styles['Normal'], fontName='Arial-Bold', fontSize=10, alignment=2)))
    
    # Xây dựng PDF tài liệu
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF report generated successfully: {filename}")

if __name__ == "__main__":
    build_pdf()
