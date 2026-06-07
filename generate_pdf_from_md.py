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
        'Arial-BoldItalic': 'arialbi.ttf',
        'CourierNew': 'cour.ttf',
        'CourierNew-Bold': 'courbd.ttf'
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
        self.drawString(54, 750, "Tài liệu chuyên khảo: Khảo sát Kiến trúc Mini-Transformer Dịch máy Việt - Anh")
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
# CHUYỂN ĐỔI CÔNG THỨC TOÁN HỌC LATEX -> UNICODE
# ==========================================
def parse_matrix_to_unicode(content):
    raw_rows = [r.strip() for r in content.split('\\\\') if r.strip()]
    if not raw_rows:
        raw_rows = [r.strip() for r in content.split('\\') if r.strip()]
    rows = []
    for r in raw_rows:
        cols = [c.strip() for c in r.split('&')]
        rows.append(cols)
    if not rows:
        return []
    num_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < num_cols:
            r.append('')
    col_widths = []
    for c_idx in range(num_cols):
        max_w = max(len(r[c_idx]) for r in rows)
        col_widths.append(max_w)
    formatted_rows = []
    for r in rows:
        formatted_cells = []
        for c_idx, cell in enumerate(r):
            w = col_widths[c_idx]
            formatted_cells.append(cell.center(w))
        formatted_rows.append('  '.join(formatted_cells))
    height = len(formatted_rows)
    matrix_lines = []
    for idx, row in enumerate(formatted_rows):
        if height == 1:
            left, right = '[ ', ' ]'
        elif idx == 0:
            left, right = '┌ ', ' ┐'
        elif idx == height - 1:
            left, right = '└ ', ' ┘'
        else:
            left, right = '│ ', ' │'
        matrix_lines.append(f'{left}{row}{right}')
    return matrix_lines

def parse_math_to_clean_text(math_str):
    math_str = math_str.strip('$').strip()
    if 'begin{bmatrix}' in math_str:
        return parse_math_line(math_str)
    if 'begin{cases}' in math_str:
        return parse_cases_to_unicode(math_str)
        
    cleaned = math_str
    
    # 1. Remove formatting macros first to prevent nested braces matching failures
    cleaned = re.sub(r'\\text\s*\{([^{}]+)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\mathcal\s*\{([^{}]+)\}', r'\1', cleaned)
    cleaned = re.sub(r'\\vec\s*\{([^{}]+)\}', r'\1', cleaned)
    
    # 2. Replace \frac and \sqrt
    frac_pattern = re.compile(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
    while frac_pattern.search(cleaned):
        cleaned = frac_pattern.sub(r'(\1)/(\2)', cleaned)
        
    sqrt_pattern = re.compile(r'\\sqrt\s*\{([^{}]+)\}')
    while sqrt_pattern.search(cleaned):
        cleaned = sqrt_pattern.sub(r'√(\1)', cleaned)
        
    # 3. Replace subscripts and superscripts with ReportLab HTML tags
    cleaned = re.sub(r'_\{([^{}]+)\}', r'<sub>\1</sub>', cleaned)
    cleaned = re.sub(r'\^\{([^{}]+)\}', r'<sup>\1</sup>', cleaned)
    cleaned = re.sub(r'_([a-zA-Z0-9]+)', r'<sub>\1</sub>', cleaned)
    cleaned = re.sub(r'\^([a-zA-Z0-9]+)', r'<sup>\1</sup>', cleaned)
        
    # 4. Replacements with proper ordering (\left( and \right) before \le and \ge)
    replacements = {
        r'\leftrightarrow': '↔',
        r'\left(': '(',
        r'\right)': ')',
        r'\sigma': 'σ',
        r'\sum': 'Σ',
        r'\approx': '≈',
        r'\leqq': '≤',
        r'\leq': '≤',
        r'\le': '≤',
        r'\geq': '≥',
        r'\ge': '≥',
        r'\times': '×',
        r'\cdot': '·',
        r'\log': 'log',
        r'\ln': 'ln',
        r'\exp': 'exp',
        r'\hat{y}': 'y_hat',
        r'\hat{Y}': 'Y_hat',
        r'\bar{y}': 'y_bar',
        r'\alpha': 'α',
        r'\omega': 'ω',
        r'\theta': 'θ',
        r'\lambda': 'λ',
        r'\dots': '...',
        r'\quad': '    ',
        r'\rightarrow': '→',
        r'\pi': 'π',
    }
    for k, v in replacements.items():
        cleaned = cleaned.replace(k, v)
        
    cleaned = cleaned.replace('\\\\', '\\')
    return cleaned

def parse_cases_to_unicode(math_str):
    prefix_match = re.match(r'(.*?)=\s*\\begin\{cases\}', math_str, re.DOTALL)
    prefix = ""
    if prefix_match:
        prefix = parse_math_to_clean_text(prefix_match.group(1)).strip() + " = "
        
    cases_match = re.search(r'\\begin\{cases\}(.*?)\\end\{cases\}', math_str, re.DOTALL)
    if not cases_match:
        return math_str
    
    content = cases_match.group(1)
    raw_rows = [r.strip() for r in content.split('\\\\') if r.strip()]
    if not raw_rows:
        raw_rows = [r.strip() for r in content.split('\\') if r.strip()]
    rows = []
    for r in raw_rows:
        cols = [c.strip() for c in r.split('&')]
        rows.append(cols)
        
    if not rows:
        return math_str
        
    cleaned_rows = []
    for r in rows:
        val = parse_math_to_clean_text(r[0])
        cond = parse_math_to_clean_text(r[1]) if len(r) > 1 else ""
        cleaned_rows.append((val, cond))
        
    max_val_len = max(len(r[0]) for r in cleaned_rows)
    
    height = len(cleaned_rows)
    lines = []
    for idx, (val, cond) in enumerate(cleaned_rows):
        val_padded = val.ljust(max_val_len)
        if height == 2:
            bracket = "┌ " if idx == 0 else "└ "
        else:
            if idx == 0:
                bracket = "┌ "
            elif idx == height - 1:
                bracket = "└ "
            else:
                bracket = "│ "
                
        if idx == 0:
            lines.append(f"{prefix}{bracket}{val_padded}   {cond}")
        else:
            indent = " " * len(prefix)
            lines.append(f"{indent}{bracket}{val_padded}   {cond}")
            
    return "\n".join(lines)

def parse_math_line(line):
    matrix_pattern = re.compile(r'\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}', re.DOTALL)
    matches = list(matrix_pattern.finditer(line))
    if not matches:
        return line
    segments = []
    last_idx = 0
    for match in matches:
        start, end = match.span()
        segments.append(line[last_idx:start])
        matrix_content = match.group(1)
        matrix_lines = parse_matrix_to_unicode(matrix_content)
        segments.append(matrix_lines)
        last_idx = end
    segments.append(line[last_idx:])
    
    max_height = 1
    for seg in segments:
        if isinstance(seg, list):
            max_height = max(max_height, len(seg))
            
    output_lines = [""] * max_height
    for seg in segments:
        if isinstance(seg, list):
            for r_idx in range(max_height):
                output_lines[r_idx] += seg[r_idx]
        else:
            clean_seg = parse_math_to_clean_text(seg)
            middle_idx = max_height // 2
            for r_idx in range(max_height):
                if r_idx == middle_idx:
                    output_lines[r_idx] += clean_seg
                else:
                    output_lines[r_idx] += " " * len(clean_seg)
    return "\n".join(output_lines)

# ==========================================
# CHUYỂN ĐỔI CHỮ LƯỢNG GIÁC MATH/MARKDOWN
# ==========================================
def clean_markdown_inline(text):
    # Thay thế các định dạng cơ bản của markdown sang thẻ ReportLab HTML
    # 1. Bold: **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # 2. Italic: *text* -> <i>text</i> hoặc _text_ -> <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'\b_(.*?)_\b', r'<i>\1</i>', text)
    # 3. Math $formula$ -> <i>cleaned_formula</i>
    text = re.sub(r'\$(.*?)\$', lambda m: f"<i>{parse_math_to_clean_text(m.group(1))}</i>", text)
    # 4. Inline code `code` -> <font face="Courier">code</font>
    text = re.sub(r'`(.*?)`', r'<font face="Arial" color="#1A202C" size="8.5"><b>\1</b></font>', text)
    # 5. Clean LaTeX syntax symbols like \rightarrow, \mathcal
    text = text.replace(r'\rightarrow', '->')
    text = text.replace(r'\approx', '≈')
    text = text.replace(r'\le', '≤')
    text = text.replace(r'\ge', '≥')
    text = text.replace(r'\times', 'x')
    text = text.replace(r'\mathcal{L}', 'L')
    text = text.replace(r'\hat{y}', 'y_hat')
    text = text.replace(r'\hat{Y}', 'Y_hat')
    text = text.replace(r'\bar{y}', 'y_bar')
    text = text.replace(r'\alpha', 'α')
    text = text.replace(r'\omega', 'ω')
    text = text.replace(r'\theta', 'θ')
    text = text.replace(r'\lambda', 'λ')
    text = text.replace(r'\sigma', 'σ')
    text = text.replace(r'\dots', '...')
    text = text.replace(r'\call', '...')
    text = text.replace(r'\cdot', '·')
    text = text.replace(r'\sum', 'Σ')
    text = text.replace(r'\ln', 'ln')
    text = text.replace(r'\log', 'log')
    text = text.replace(r'\max', 'max')
    text = text.replace(r'\text{BLEU}', 'BLEU')
    text = text.replace(r'\text{BP}', 'BP')
    text = text.replace(r'\text{PPL}', 'PPL')
    text = text.replace(r'\text{FFN}', 'FFN')
    text = text.replace(r'\\mathcal{L}', 'L')
    text = text.replace(r'\\text{BLEU}', 'BLEU')
    text = text.replace(r'\\text{BP}', 'BP')
    text = text.replace(r'\\text{PPL}', 'PPL')
    text = text.replace(r'\\text{FFN}', 'FFN')
    
    # 6. Loại bỏ markdown links [text](url) -> <b>text</b>
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'<b>\1</b>', text)
    
    return text

# ==========================================
# HÀM RENDER CÔNG THỨC TOÁN HỌC LATEX THÀNH ẢNH DÙNG MATPLOTLIB
# ==========================================
def render_math_to_flowable(formula, pdf_file_path):
    import hashlib
    from PIL import Image as PILImage
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import re
    
    # Tạo thư mục lưu trữ ảnh toán học
    math_dir = os.path.join(os.path.dirname(os.path.abspath(pdf_file_path)), "math_images")
    os.makedirs(math_dir, exist_ok=True)
    
    # Tạo mã băm duy nhất cho mỗi công thức để đặt tên tệp
    h = hashlib.md5(formula.encode('utf-8')).hexdigest()
    img_path = os.path.join(math_dir, f"eq_{h}.png")
    
    if not os.path.exists(img_path):
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "text.usetex": False,
            "mathtext.fontset": "dejavusans"
        })
        
        if 'begin{bmatrix}' in formula:
            # Helper: Parse matrix content
            def parse_matrix_content(matrix_str):
                raw_rows = [r.strip() for r in matrix_str.split('\\\\') if r.strip()]
                if not raw_rows:
                    raw_rows = [r.strip() for r in matrix_str.split('\\') if r.strip()]
                matrix_data = []
                for r in raw_rows:
                    cols = [c.strip() for c in r.split('&')]
                    matrix_data.append(cols)
                return matrix_data

            # Create a single temporary figure just for measuring text widths
            meas_fig, meas_ax = plt.subplots(figsize=(8.5, 1.2))
            meas_ax.axis('off')
            meas_ax.set_xlim(0, 8.5)
            meas_ax.set_ylim(0, 1)
            meas_fig.canvas.draw()
            meas_renderer = meas_fig.canvas.get_renderer()

            def get_text_width_inches(text):
                val = text.strip()
                if not val:
                    return 0.0
                val = val.replace(r'\quad', '   ')
                text_pattern = re.compile(r'\\text\s*\{([^{}]+)\}')
                val = text_pattern.sub(r'\1', val)
                val = val.replace(r'\times', r'\times ')
                math_str = f"${val}$"
                
                t_obj = meas_ax.text(0, 0.5, math_str, size=13)
                meas_fig.canvas.draw()
                bbox = t_obj.get_window_extent(meas_renderer)
                width_inches = bbox.width / meas_fig.dpi
                t_obj.remove()
                return width_inches

            # Segment formula by matrices
            pos = 0
            segments = []
            pattern = re.compile(r'\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}', re.DOTALL)
            
            for match in pattern.finditer(formula):
                start, end = match.span()
                if start > pos:
                    segments.append(('text', formula[pos:start]))
                segments.append(('matrix', match.group(1)))
                pos = end
            if pos < len(formula):
                segments.append(('text', formula[pos:]))
                
            # Measure all segments and determine widths
            segment_layouts = []
            for seg_type, seg_val in segments:
                if seg_type == 'text':
                    w = get_text_width_inches(seg_val)
                    if w > 0:
                        segment_layouts.append(('text', w, seg_val))
                elif seg_type == 'matrix':
                    matrix_data = parse_matrix_content(seg_val)
                    num_rows = len(matrix_data)
                    num_cols = len(matrix_data[0])
                    
                    col_widths = []
                    for c in range(num_cols):
                        max_len = max(len(matrix_data[r][c]) for r in range(num_rows))
                        col_widths.append(max(0.3, max_len * 0.08 + 0.18))
                        
                    matrix_w = 0.12 + sum(col_widths) + 0.12
                    segment_layouts.append(('matrix', matrix_w, (matrix_data, col_widths)))
            
            plt.close(meas_fig)

            # Determine final figure size with uniform padding
            padding = 0.3  # uniform padding between elements
            total_elements_width = sum(layout[1] for layout in segment_layouts)
            total_padding = padding * (len(segment_layouts) - 1) if segment_layouts else 0
            fig_width = 0.2 + total_elements_width + total_padding + 0.2
            
            fig, ax = plt.subplots(figsize=(fig_width, 1.2))
            ax.axis('off')
            ax.set_xlim(0, fig_width)
            ax.set_ylim(0, 1)
            
            x_curr = 0.2
            for idx, (seg_type, w, data) in enumerate(segment_layouts):
                if idx > 0:
                    x_curr += padding
                
                if seg_type == 'text':
                    val = data.strip()
                    val = val.replace(r'\quad', '   ')
                    text_pattern = re.compile(r'\\text\s*\{([^{}]+)\}')
                    val = text_pattern.sub(r'\1', val)
                    val = val.replace(r'\times', r'\times ')
                    math_str = f"${val}$"
                    ax.text(x_curr, 0.5, math_str, size=13, va="center", ha="left", color="#1A365D")
                    x_curr += w
                elif seg_type == 'matrix':
                    matrix_data, col_widths = data
                    num_rows = len(matrix_data)
                    num_cols = len(matrix_data[0])
                    
                    row_height = 0.22
                    y_center = 0.5
                    y_coords = []
                    if num_rows % 2 == 1:
                        mid = num_rows // 2
                        for r in range(num_rows):
                            y_coords.append(y_center + (mid - r) * row_height)
                    else:
                        half = num_rows / 2
                        for r in range(num_rows):
                            y_coords.append(y_center + (half - 0.5 - r) * row_height)
                            
                    col_x = []
                    curr_x = x_curr + 0.12
                    for w_col in col_widths:
                        col_x.append(curr_x + w_col/2)
                        curr_x += w_col
                    x_end = curr_x + 0.12
                    
                    for r in range(num_rows):
                        for c in range(num_cols):
                            cell_val = matrix_data[r][c].strip()
                            cell_math = f"${cell_val}$"
                            ax.text(col_x[c], y_coords[r], cell_math, size=12, va="center", ha="center")
                            
                    # Draw brackets
                    bracket_w = 0.06
                    # Left bracket
                    ax.plot([x_curr + bracket_w, x_curr, x_curr, x_curr + bracket_w],
                            [y_coords[0] + row_height/2 - 0.02, y_coords[0] + row_height/2 - 0.02, y_coords[-1] - row_height/2 + 0.02, y_coords[-1] - row_height/2 + 0.02],
                            color='#1A365D', lw=1.2)
                    # Right bracket
                    ax.plot([x_end - bracket_w, x_end, x_end, x_end - bracket_w],
                            [y_coords[0] + row_height/2 - 0.02, y_coords[0] + row_height/2 - 0.02, y_coords[-1] - row_height/2 + 0.02, y_coords[-1] - row_height/2 + 0.02],
                            color='#1A365D', lw=1.2)
                    
                    x_curr = x_end
            
            plt.savefig(img_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.03)
            plt.close()
        else:
            # Standard single-line formula rendering
            fig_width = max(3.0, len(formula) * 0.11)
            fig_height = 0.6
            
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            clean_formula = formula.replace('\\\\', '\\')
            math_str = clean_formula if clean_formula.startswith('$') else f"${clean_formula}$"
            
            fig.text(0.5, 0.5, math_str, size=13, va="center", ha="center", color="#1A365D")
            
            plt.savefig(img_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.03)
            plt.close()
            
    with PILImage.open(img_path) as pil_img:
        width_px, height_px = pil_img.size
        
    scale = 0.22
    width_pt = width_px * scale
    height_pt = height_px * scale
    
    max_w = 480
    if width_pt > max_w:
        ratio = max_w / width_pt
        width_pt = max_w
        height_pt = height_pt * ratio
        
    img = Image(img_path, width=width_pt, height=height_pt)
    
    t_math = Table([[img]], colWidths=[500])
    t_math.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    return t_math

# ==========================================
# KHỞI TẠO BÁO CÁO PDF TỪ FILE MD
# ==========================================
def build_pdf_from_md(md_file_path, pdf_file_path="README.pdf"):
    doc = SimpleDocTemplate(
        pdf_file_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()
    
    # Định nghĩa các Styles
    normal_style = styles['Normal']
    normal_style.fontName = 'Arial'
    normal_style.fontSize = 8.5
    normal_style.leading = 12.0
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
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceAfter=40
    )

    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SubSectionHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=9.0,
        leading=12,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'SubSubSectionHeader',
        parent=styles['Normal'],
        fontName='Arial-BoldItalic',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4A5568"),
        spaceBefore=8,
        spaceAfter=4,
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

    code_style = ParagraphStyle(
        'CodeBlockCustom',
        parent=styles['Normal'],
        fontName='CourierNew',
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor("#1A202C")
    )

    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontSize=7.2,
        leading=9,
        alignment=0 # Left
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=7.6,
        leading=10,
        textColor=colors.white,
        alignment=1 # Center
    )

    math_style = ParagraphStyle(
        'MathStyle',
        parent=styles['Normal'],
        alignment=1, # Center
        fontName='Arial-BoldItalic',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=6,
        spaceAfter=6
    )

    def make_code_block(raw_code, is_python=True):
        if is_python:
            highlighted = highlight_python(raw_code)
        else:
            # Đối với ma trận văn bản, giữ nguyên khoảng trắng và ký tự vẽ khung bằng &nbsp;
            highlighted = raw_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;").replace("\n", "<br/>")
            
        p_code = Paragraph(highlighted, code_style)
        t = Table([[p_code]], colWidths=[500], splitByRow=0)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LINELEFT', (0,0), (-1,-1), 3.0, colors.HexColor("#2B6CB0") if is_python else colors.HexColor("#7F8C8D")),
        ]))
        return KeepTogether([t])

    story = []

    # Read the Markdown file content
    with open(md_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ==========================================
    # TRANG BÌA (COVER PAGE)
    # ==========================================
    story.append(Spacer(1, 100))
    story.append(Paragraph("TÀI LIỆU KHẢO SÁT CHUYÊN SÂU", subtitle_style))
    story.append(Paragraph("KIẾN TRÚC MINI-TRANSFORMER DỊCH MÁY VIỆT - ANH", title_style))
    story.append(Spacer(1, 25))
    
    meta_box_data = [
        [Paragraph("<b>Đề tài:</b> Phát triển Hệ thống dịch máy song ngữ seq2seq ứng dụng cơ chế Attention tự xây dựng từ con số không (From Scratch)", body_style)],
        [Paragraph("<b>Tập dữ liệu:</b> IWSLT 2015 English-Vietnamese (Tổng: 135.853 câu; Train: 133.317 (~98,13%), Val: 1.268 (~0,93%), Test: 1.268 (~0,93%))", body_style)],
        [Paragraph("<b>Chi tiết giải thuật:</b> Multi-Head Attention, Pre-LN LayerNorm, Shared BPE, Noam Scheduler", body_style)],
        [Paragraph("<b>Chiến lược tối ưu:</b> Multi-GPU DataParallel, FP16 AMP, Gradient Accumulation, Bucket Batching", body_style)],
        [Paragraph("<b>Tác giả:</b> Nhóm Nghiên cứu Khoa học & Xử lý Ngôn ngữ Tự nhiên (NCKH Team)", body_style)],
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
    story.append(PageBreak())

    # ==========================================
    # PARSING THE MD FILE
    # ==========================================
    in_code_block = False
    code_lines = []
    code_block_lang = "python"
    
    in_table = False
    table_rows = []
    
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        stripped = line.strip()
        
        # 0. Xử lý Display Math Block ($$)
        if stripped.startswith("$$"):
            if stripped.endswith("$$") and len(stripped) > 4:
                formula = stripped[2:-2].strip()
                line_idx += 1
            else:
                formula_lines = [stripped[2:]]
                line_idx += 1
                while line_idx < len(lines):
                    next_line = lines[line_idx]
                    next_stripped = next_line.strip()
                    if next_stripped.endswith("$$"):
                        formula_lines.append(next_stripped[:-2])
                        line_idx += 1
                        break
                    else:
                        formula_lines.append(next_line.rstrip('\n'))
                    line_idx += 1
                formula = "\n".join(formula_lines).strip()
            
            try:
                img_flowable = render_math_to_flowable(formula, pdf_file_path)
                story.append(img_flowable)
                story.append(Spacer(1, 4))
            except Exception as e:
                print(f"Fallback to text math for: {formula}. Error: {e}")
                cleaned_math = parse_math_to_clean_text(formula)
                story.append(Paragraph(cleaned_math, math_style))
                story.append(Spacer(1, 4))
            continue
            
        # 1. Xử lý Code Block
        if stripped.startswith("```"):
            if in_code_block:
                # Kết thúc code block
                code_text = "\n".join(code_lines)
                story.append(make_code_block(code_text, code_block_lang == "python"))
                story.append(Spacer(1, 5))
                in_code_block = False
                code_lines = []
            else:
                # Bắt đầu code block
                in_code_block = True
                code_block_lang = stripped[3:].strip().lower()
            line_idx += 1
            continue
            
        if in_code_block:
            code_lines.append(line.rstrip('\n'))
            line_idx += 1
            continue

        # 2. Xử lý Bảng biểu Markdown
        if stripped.startswith("|"):
            if "---" in stripped and not any(c.isalnum() for c in stripped.replace("-", "").replace("|", "")):
                line_idx += 1
                continue
                
            in_table = True
            cols = [col.strip() for col in stripped.split("|")[1:-1]]
            table_rows.append(cols)
            line_idx += 1
            continue
        else:
            if in_table:
                # Kết thúc và build Table ReportLab
                if table_rows:
                    formatted_rows = []
                    header_cols = [Paragraph(f"<b>{clean_markdown_inline(col)}</b>", table_header_style) for col in table_rows[0]]
                    formatted_rows.append(header_cols)
                    
                    for row in table_rows[1:]:
                        data_cols = []
                        num_cols = len(table_rows[0])
                        col_width = 500 / num_cols
                        for col in row:
                            img_match = re.search(r'!\[(.*?)\]\((.*?)\)', col.strip())
                            if img_match:
                                caption = img_match.group(1)
                                filename = img_match.group(2)
                                if os.path.exists(filename):
                                    img_w = col_width - 12
                                    aspect = 1.0
                                    if "overview" in filename:
                                        aspect = 177 / 440
                                    elif "pe_heatmap" in filename:
                                        aspect = 133 / 220
                                    elif "pe_flow" in filename:
                                        aspect = 100 / 220
                                    elif "encoder_layer" in filename or "decoder_layer" in filename:
                                        aspect = 200 / 140
                                    elif "attention_flow" in filename:
                                        aspect = 194 / 280
                                    elif "attention_map" in filename:
                                        aspect = 203 / 240
                                    elif "curves" in filename:
                                        aspect = 125 / 460
                                    elif "dynamic_padding" in filename:
                                        aspect = 173 / 380
                                    
                                    img_h = img_w * aspect
                                    try:
                                        img_flowable = Image(filename, width=img_w, height=img_h)
                                        cap_p = Paragraph(f"<font color='#7F8C8D'><i>* {caption}</i></font>", ParagraphStyle('TblImgCap', parent=styles['Normal'], fontSize=7.0, alignment=1))
                                        data_cols.append([img_flowable, cap_p])
                                    except Exception as e:
                                        print(f"Error loading table image {filename}: {e}")
                                        data_cols.append(Paragraph(clean_markdown_inline(col), table_text_style))
                                else:
                                    data_cols.append(Paragraph(clean_markdown_inline(col), table_text_style))
                            else:
                                data_cols.append(Paragraph(clean_markdown_inline(col), table_text_style))
                        formatted_rows.append(data_cols)
                    
                    num_cols = len(table_rows[0])
                    col_width = 500 / num_cols
                    
                    t = Table(formatted_rows, colWidths=[col_width] * num_cols)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
                        ('PADDING', (0,0), (-1,-1), 4),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ]))
                    story.append(KeepTogether([t]))
                    story.append(Spacer(1, 6))
                in_table = False
                table_rows = []

        # 3. Xử lý Trích dẫn hoặc Note Block
        if stripped.startswith(">"):
            note_content = stripped.lstrip(">").strip()
            if note_content.startswith("[!"):
                line_idx += 1
                continue
            cleaned_note = clean_markdown_inline(note_content)
            p = Paragraph(f"<i>{cleaned_note}</i>", normal_style)
            t = Table([[p]], colWidths=[500])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0F4F8")),
                ('PADDING', (0,0), (-1,-1), 8),
                ('LINELEFT', (0,0), (0,0), 2.5, colors.HexColor("#4B5563")),
            ]))
            story.append(t)
            story.append(Spacer(1, 5))
            line_idx += 1
            continue

        # 4. Xử lý tiêu đề Chương và Mục
        if stripped.startswith("#"):
            h_level = len(stripped) - len(stripped.lstrip('#'))
            title_text = stripped.lstrip('#').strip()
            title_text_cleaned = clean_markdown_inline(title_text)
            
            if h_level == 1:
                story.append(Paragraph(title_text_cleaned, title_style))
                story.append(Spacer(1, 10))
            elif h_level == 2:
                story.append(PageBreak())
                story.append(Paragraph(title_text_cleaned, h1_style))
                story.append(Spacer(1, 5))
            elif h_level == 3:
                story.append(Paragraph(title_text_cleaned, h2_style))
                story.append(Spacer(1, 4))
            else:
                story.append(Paragraph(title_text_cleaned, h3_style))
                story.append(Spacer(1, 3))
            
            line_idx += 1
            continue

        # 5. Xử lý Ảnh nhúng: ![Caption](Filename)
        img_match = re.match(r'^!\[(.*?)\]\((.*?)\)', stripped)
        if img_match:
            # Check if there is another image immediately following (skipping whitespace)
            next_img_idx = line_idx + 1
            while next_img_idx < len(lines) and not lines[next_img_idx].strip():
                next_img_idx += 1
            
            is_double_img = False
            if next_img_idx < len(lines):
                next_stripped = lines[next_img_idx].strip()
                next_img_match = re.match(r'^!\[(.*?)\]\((.*?)\)', next_stripped)
                if next_img_match:
                    is_double_img = True
            
            if is_double_img:
                caption1 = img_match.group(1)
                filename1 = img_match.group(2)
                caption2 = next_img_match.group(1)
                filename2 = next_img_match.group(2)
                
                if os.path.exists(filename1) and os.path.exists(filename2):
                    common_h = 280
                    # Determine widths based on aspect ratios
                    ratio1 = 0.70 if "encoder_layer" in filename1 else 0.66 if "decoder_layer" in filename1 else 1.0
                    ratio2 = 0.66 if "decoder_layer" in filename2 else 0.70 if "encoder_layer" in filename2 else 1.0
                    
                    img1_w = common_h * ratio1
                    img2_w = common_h * ratio2
                    
                    try:
                        img1 = Image(filename1, width=img1_w, height=common_h)
                        img2 = Image(filename2, width=img2_w, height=common_h)
                        
                        cap1_para = Paragraph(f"<font color='#7F8C8D'><i>* {caption1}</i></font>", ParagraphStyle('Cap1', parent=styles['Normal'], fontSize=7.5, alignment=1))
                        cap2_para = Paragraph(f"<font color='#7F8C8D'><i>* {caption2}</i></font>", ParagraphStyle('Cap2', parent=styles['Normal'], fontSize=7.5, alignment=1))
                        
                        # Use a table to place side-by-side
                        t_data = [
                            [img1, img2],
                            [cap1_para, cap2_para]
                        ]
                        t = Table(t_data, colWidths=[250, 250])
                        t.setStyle(TableStyle([
                            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                            ('VALIGN', (0,0), (-1,-1), 'TOP'),
                            ('TOPPADDING', (0,0), (-1,-1), 2),
                            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                        ]))
                        
                        story.append(KeepTogether([t]))
                        story.append(Spacer(1, 8))
                    except Exception as e:
                        print(f"Error loading side-by-side images: {e}")
                else:
                    print(f"One or both image files not found: {filename1}, {filename2}")
                
                # Advance line_idx past the second image
                line_idx = next_img_idx + 1
                continue

            else:
                # Single image rendering
                caption = img_match.group(1)
                filename = img_match.group(2)
                
                if os.path.exists(filename):
                    width = 380
                    height = 175
                    
                    if "overview" in filename:
                        width = 440
                        height = 177
                    elif "pe_heatmap" in filename:
                        width = 220
                        height = 133
                    elif "pe_flow" in filename:
                        width = 220
                        height = 100
                    elif "encoder_layer" in filename or "decoder_layer" in filename:
                        width = 140
                        height = 200
                    elif "attention_flow" in filename:
                        width = 280
                        height = 194
                    elif "attention_map" in filename:
                        width = 240
                        height = 203
                    elif "curves" in filename:
                        width = 460
                        height = 125
                    elif "dynamic_padding" in filename:
                        width = 380
                        height = 173
                    
                    try:
                        img = Image(filename, width=width, height=height)
                        cap_para = Paragraph(f"<font color='#7F8C8D'><i>* {caption}</i></font>", ParagraphStyle('Cap', parent=styles['Normal'], fontSize=7.5, alignment=1, spaceBefore=3))
                        story.append(KeepTogether([img, cap_para]))
                        story.append(Spacer(1, 8))
                    except Exception as e:
                        print(f"Error loading image {filename}: {e}")
                else:
                    print(f"Image file not found: {filename}")
                line_idx += 1
                continue
            continue

        # 6. Xử lý các dòng danh sách dấu bullet
        if stripped.startswith("* ") or stripped.startswith("- "):
            bullet_text = stripped[2:].strip()
            bullet_text_cleaned = clean_markdown_inline(bullet_text)
            story.append(Paragraph(f"&bull; {bullet_text_cleaned}", bullet_style))
            line_idx += 1
            continue
            
        num_match = re.match(r'^(\d+)\.\s+(.*)', stripped)
        if num_match:
            num = num_match.group(1)
            item_text = num_match.group(2)
            item_text_cleaned = clean_markdown_inline(item_text)
            story.append(Paragraph(f"<b>{num}.</b> {item_text_cleaned}", bullet_style))
            line_idx += 1
            continue

        # 7. Đoạn văn bản thông thường
        if stripped:
            cleaned_body = clean_markdown_inline(stripped)
            story.append(Paragraph(cleaned_body, body_style))
            
        line_idx += 1

    # Xây dựng PDF tài liệu
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF document generated successfully: {pdf_file_path}")

if __name__ == "__main__":
    build_pdf_from_md(
        md_file_path=r"d:\CN12_2024_2028\NCKH\Code\TransformerFormScratch\README.md",
        pdf_file_path=r"d:\CN12_2024_2028\NCKH\Code\TransformerFormScratch\README.pdf"
    )
