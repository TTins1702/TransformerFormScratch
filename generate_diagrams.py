import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Helper functions để vẽ sơ đồ bằng matplotlib
def draw_box(ax, x, y, w, h, text, fc, ec, text_color='#2C3E50', font_size=10, bold=True):
    box = patches.FancyBboxPatch(
        (x, y), w, h, 
        boxstyle="round,pad=0.02,rounding_size=0.08", 
        fc=fc, ec=ec, lw=1.5, mutation_scale=1.0
    )
    ax.add_patch(box)
    ax.text(
        x + w/2, y + h/2, text, 
        ha='center', va='center', 
        color=text_color, fontsize=font_size, 
        fontweight='bold' if bold else 'normal'
    )

def draw_arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate(
        text, xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(facecolor='#5D6D7E', edgecolor='#5D6D7E', width=1.5, headwidth=6, shrink=0.05),
        ha='center', va='center', fontsize=8, color='#5D6D7E'
    )

def generate_all_diagrams():
    print("Generating diagrams...")
    
    # 1. Overview Diagram
    fig, ax = plt.subplots(figsize=(11.5, 4.5), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5)
    draw_box(ax, 0.5, 4.0, 2.5, 0.5, "Câu nguồn Tiếng Việt\n(chúng tôi học sâu)", "#EBF5FB", "#2980B9")
    draw_arrow(ax, 1.75, 4.0, 1.75, 3.3)
    draw_box(ax, 0.5, 2.8, 2.5, 0.5, "Embedding +\nPositional Encoding", "#D4E6F1", "#2980B9")
    draw_arrow(ax, 1.75, 2.8, 1.75, 2.1)
    draw_box(ax, 0.5, 1.0, 2.5, 1.1, "ENCODER STACK\n(3 Layers)\n[Trích xuất ngữ nghĩa]", "#D5F5E3", "#27AE60")
    
    draw_box(ax, 5.2, 4.0, 2.5, 0.5, "Câu dịch tạm Tiếng Anh\n(<sos> we learn)", "#FDEDEC", "#CB4335")
    draw_arrow(ax, 6.45, 4.0, 6.45, 3.3)
    draw_box(ax, 5.2, 2.8, 2.5, 0.5, "Embedding +\nPositional Encoding", "#FADBD8", "#CB4335")
    draw_arrow(ax, 6.45, 2.8, 6.45, 2.1)
    draw_box(ax, 5.2, 1.0, 2.5, 1.1, "DECODER STACK\n(3 Layers)\n[Tự hồi quy]", "#FCF3CF", "#F39C12")
    
    draw_arrow(ax, 3.0, 1.55, 5.2, 1.55, "Ngữ cảnh nguồn (K/V)")
    draw_arrow(ax, 7.7, 1.55, 9.0, 1.55)
    draw_box(ax, 9.0, 1.25, 2.0, 0.6, "Generator\n(Linear + Softmax)", "#EBEDEF", "#7F8C8D")
    draw_arrow(ax, 10.0, 1.25, 10.0, 0.7)
    draw_box(ax, 9.0, 0.2, 2.0, 0.5, "Từ kế tiếp (deep)", "#FCF3CF", "#D4AC0D")
    plt.savefig("overview.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # 2. Positional Encoding Flow
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    words = ["tôi", "yêu", "học", "máy"]
    for i, word in enumerate(words):
        x_offset = 0.5 + i * 2.1
        ax.text(x_offset + 0.8, 3.8, f"Vị trí {i}: \"{word}\"", ha='center', va='center', fontsize=9, color='#2C3E50', fontweight='bold')
        draw_box(ax, x_offset, 2.8, 1.6, 0.5, "Word Embedding\n[d_model=256]", "#D4E6F1", "#2980B9", font_size=8)
        draw_arrow(ax, x_offset + 0.8, 2.8, x_offset + 0.8, 2.1)
        draw_box(ax, x_offset, 1.6, 1.6, 0.5, "PE Vector\n(Sine/Cosine)", "#FCF3CF", "#F39C12", font_size=8)
        draw_arrow(ax, x_offset + 0.8, 1.6, x_offset + 0.8, 1.1)
        ax.add_patch(patches.Circle((x_offset + 0.8, 0.9), 0.2, color='#E74C3C', fill=False, lw=1.5))
        ax.text(x_offset + 0.8, 0.9, "+", ha='center', va='center', color='#E74C3C', fontsize=12, fontweight='bold')
        draw_arrow(ax, x_offset + 0.8, 0.7, x_offset + 0.8, 0.4)
        draw_box(ax, x_offset, 0.0, 1.6, 0.4, f"Đầu vào x_{i}", "#D5F5E3", "#27AE60", font_size=8)
    plt.savefig("pe_flow.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 3. Multi-Head Attention Flow
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    draw_box(ax, 2.5, 5.2, 3.0, 0.5, "Đầu vào Q, K, V\n[batch_size, seq_len, d_model]", "#EBEDEF", "#7F8C8D", font_size=9)
    draw_arrow(ax, 3.0, 5.2, 1.5, 4.3)
    draw_arrow(ax, 4.0, 5.2, 4.0, 4.3)
    draw_arrow(ax, 5.0, 5.2, 6.5, 4.3)
    draw_box(ax, 0.5, 3.8, 2.0, 0.5, "Linear Layer (W_q)", "#D4E6F1", "#2980B9", font_size=8)
    draw_box(ax, 3.0, 3.8, 2.0, 0.5, "Linear Layer (W_k)", "#D4E6F1", "#2980B9", font_size=8)
    draw_box(ax, 5.5, 3.8, 2.0, 0.5, "Linear Layer (W_v)", "#D4E6F1", "#2980B9", font_size=8)
    draw_arrow(ax, 1.5, 3.8, 1.5, 3.1)
    draw_arrow(ax, 4.0, 3.8, 4.0, 3.1)
    draw_arrow(ax, 6.5, 3.8, 6.5, 3.1)
    draw_box(ax, 0.2, 2.6, 7.6, 0.5, "Phân tách thành nhiều Heads và d_k chiều (d_k = d_model / num_heads)", "#FCF3CF", "#F39C12", font_size=8)
    draw_arrow(ax, 4.0, 2.6, 4.0, 1.9)
    draw_box(ax, 0.5, 0.9, 7.0, 1.0, "Scaled Dot-Product Attention (song song trên từng head)\nAttention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) * V", "#D5F5E3", "#27AE60", font_size=9)
    draw_arrow(ax, 4.0, 0.9, 4.0, 0.6)
    draw_box(ax, 1.5, 0.1, 5.0, 0.5, "Gộp heads (Concat) + Chi chiếu tuyến tính cuối (W_o)", "#D4E6F1", "#2980B9", font_size=8)
    plt.savefig("attention_flow.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 4. Encoder Layer
    fig, ax = plt.subplots(figsize=(5.5, 8.0), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 9)
    ax.add_patch(patches.Rectangle((0.1, 0.1), 4.8, 8.8, fill=False, edgecolor='#BDC3C7', linestyle='--', lw=1.5))
    ax.text(2.5, 8.6, "ENCODER LAYER", ha='center', va='center', fontsize=11, fontweight='bold', color='#7F8C8D')
    draw_box(ax, 1.5, 7.8, 2.0, 0.5, "Đầu vào: x", "#EBF5FB", "#2980B9")
    draw_arrow(ax, 2.5, 7.8, 2.5, 7.0)
    draw_box(ax, 1.2, 6.5, 2.6, 0.5, "LayerNorm 1 (Pre-LN)", "#EBEDEF", "#7F8C8D", font_size=9)
    draw_arrow(ax, 2.5, 6.5, 2.5, 5.7)
    draw_box(ax, 1.0, 5.2, 3.0, 0.5, "Multi-Head Self-Attention", "#D5F5E3", "#27AE60")
    draw_arrow(ax, 2.5, 5.2, 2.5, 4.6)
    ax.text(2.5, 4.9, "Dropout", ha='center', va='center', fontsize=8, color='#7F8C8D')
    ax.add_patch(patches.Circle((2.5, 4.3), 0.25, color='#E74C3C', fill=False, lw=1.5))
    ax.text(2.5, 4.3, "+", ha='center', va='center', color='#E74C3C', fontsize=14, fontweight='bold')
    draw_arrow(ax, 2.5, 4.05, 2.5, 3.3)
    ax.plot([2.5, 4.4, 4.4, 2.75], [8.05, 8.05, 4.3, 4.3], color='#E74C3C', linestyle='-', lw=1.5)
    ax.text(4.4, 6.1, "Residual", ha='center', va='center', rotation=-90, fontsize=8, color='#E74C3C', fontweight='bold')
    draw_box(ax, 1.2, 2.8, 2.6, 0.5, "LayerNorm 2 (Pre-LN)", "#EBEDEF", "#7F8C8D", font_size=9)
    draw_arrow(ax, 2.5, 2.8, 2.5, 2.0)
    draw_box(ax, 1.0, 1.5, 3.0, 0.5, "Position-Wise Feed-Forward", "#D4E6F1", "#2980B9")
    draw_arrow(ax, 2.5, 1.5, 2.5, 0.9)
    ax.text(2.5, 1.2, "Dropout", ha='center', va='center', fontsize=8, color='#7F8C8D')
    ax.add_patch(patches.Circle((2.5, 0.65), 0.25, color='#E74C3C', fill=False, lw=1.5))
    ax.text(2.5, 0.65, "+", ha='center', va='center', color='#E74C3C', fontsize=14, fontweight='bold')
    draw_arrow(ax, 2.5, 0.4, 2.5, 0.2)
    ax.plot([2.5, 4.4, 4.4, 2.75], [4.05, 4.05, 0.65, 0.65], color='#E74C3C', linestyle='-', lw=1.5)
    ax.text(4.4, 2.3, "Residual", ha='center', va='center', rotation=-90, fontsize=8, color='#E74C3C', fontweight='bold')
    ax.text(2.5, 0.15, "Đầu ra Layer", ha='center', va='center', fontsize=10, fontweight='bold', color='#2C3E50')
    plt.savefig("encoder_layer.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 5. Decoder Layer
    fig, ax = plt.subplots(figsize=(6.0, 9.5), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 11)
    ax.add_patch(patches.Rectangle((0.1, 0.1), 5.8, 10.8, fill=False, edgecolor='#BDC3C7', linestyle='--', lw=1.5))
    ax.text(3.0, 10.6, "DECODER LAYER", ha='center', va='center', fontsize=11, fontweight='bold', color='#7F8C8D')
    draw_box(ax, 2.0, 9.8, 2.0, 0.4, "Đầu vào Decoder: x", "#FDEDEC", "#CB4335", font_size=9)
    draw_arrow(ax, 3.0, 9.8, 3.0, 9.1)
    draw_box(ax, 1.7, 8.7, 2.6, 0.4, "LayerNorm 1 (Pre-LN)", "#EBEDEF", "#7F8C8D", font_size=8)
    draw_arrow(ax, 3.0, 8.7, 3.0, 8.0)
    draw_box(ax, 1.5, 7.6, 3.0, 0.4, "Masked Self-Attention", "#FCF3CF", "#F39C12", font_size=9)
    draw_arrow(ax, 3.0, 7.6, 3.0, 7.1)
    ax.text(3.0, 7.35, "Dropout", ha='center', va='center', fontsize=7, color='#7F8C8D')
    ax.add_patch(patches.Circle((3.0, 6.85), 0.2, color='#E74C3C', fill=False, lw=1.5))
    ax.text(3.0, 6.85, "+", ha='center', va='center', color='#E74C3C', fontsize=12, fontweight='bold')
    draw_arrow(ax, 3.0, 6.65, 3.0, 6.0)
    ax.plot([3.0, 5.3, 5.3, 3.2], [10.0, 10.0, 6.85, 6.85], color='#E74C3C', linestyle='-', lw=1.5)
    ax.text(5.3, 8.4, "Residual", ha='center', va='center', rotation=-90, fontsize=8, color='#E74C3C', fontweight='bold')
    draw_box(ax, 1.7, 5.6, 2.6, 0.4, "LayerNorm 2 (Pre-LN)", "#EBEDEF", "#7F8C8D", font_size=8)
    draw_arrow(ax, 3.0, 5.6, 3.0, 4.9)
    draw_box(ax, 1.5, 4.5, 3.0, 0.4, "Encoder-Decoder Cross-Attention", "#FADBD8", "#E74C3C", font_size=9)
    draw_arrow(ax, 0.5, 4.7, 1.5, 4.7, "Từ Encoder (K/V)")
    draw_arrow(ax, 3.0, 4.5, 3.0, 4.0)
    ax.text(3.0, 4.25, "Dropout", ha='center', va='center', fontsize=7, color='#7F8C8D')
    ax.add_patch(patches.Circle((3.0, 3.75), 0.2, color='#E74C3C', fill=False, lw=1.5))
    ax.text(3.0, 3.75, "+", ha='center', va='center', color='#E74C3C', fontsize=12, fontweight='bold')
    draw_arrow(ax, 3.0, 3.55, 3.0, 2.9)
    ax.plot([3.0, 5.3, 5.3, 3.2], [6.65, 6.65, 3.75, 3.75], color='#E74C3C', linestyle='-', lw=1.5)
    draw_box(ax, 1.7, 2.5, 2.6, 0.4, "LayerNorm 3 (Pre-LN)", "#EBEDEF", "#7F8C8D", font_size=8)
    draw_arrow(ax, 3.0, 2.5, 3.0, 1.8)
    draw_box(ax, 1.5, 1.4, 3.0, 0.4, "Position-Wise Feed-Forward", "#D4E6F1", "#2980B9", font_size=9)
    draw_arrow(ax, 3.0, 1.4, 3.0, 0.9)
    ax.text(3.0, 1.15, "Dropout", ha='center', va='center', fontsize=7, color='#7F8C8D')
    ax.add_patch(patches.Circle((3.0, 0.65), 0.2, color='#E74C3C', fill=False, lw=1.5))
    ax.text(3.0, 0.65, "+", ha='center', va='center', color='#E74C3C', fontsize=12, fontweight='bold')
    draw_arrow(ax, 3.0, 0.45, 3.0, 0.2)
    ax.plot([3.0, 5.3, 5.3, 3.2], [3.55, 3.55, 0.65, 0.65], color='#E74C3C', linestyle='-', lw=1.5)
    ax.text(3.0, 0.15, "Đầu ra Decoder Layer", ha='center', va='center', fontsize=10, fontweight='bold', color='#2C3E50')
    plt.savefig("decoder_layer.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 6. Dynamic Padding
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.text(5, 4.2, "STATIC PADDING (Đệm cố định max_len = 8)", ha='center', va='center', fontsize=9, fontweight='bold', color='#2C3E50')
    c1 = ["tôi", "yêu", "học", "máy", "<pad>", "<pad>", "<pad>", "<pad>"]
    c2 = ["bạn", "khỏe", "không", "<pad>", "<pad>", "<pad>", "<pad>", "<pad>"]
    for j, token in enumerate(c1):
        fc = "#EAEDED" if token == "<pad>" else "#D4E6F1"
        ec = "#BDC3C7" if token == "<pad>" else "#2980B9"
        draw_box(ax, 0.5 + j * 1.1, 3.5, 1.0, 0.4, token, fc, ec, font_size=7, bold=False)
    for j, token in enumerate(c2):
        fc = "#EAEDED" if token == "<pad>" else "#D4E6F1"
        ec = "#BDC3C7" if token == "<pad>" else "#2980B9"
        draw_box(ax, 0.5 + j * 1.1, 2.9, 1.0, 0.4, token, fc, ec, font_size=7, bold=False)
    ax.text(5, 2.0, "DYNAMIC PADDING (Đệm động theo batch - max_len_batch = 4)", ha='center', va='center', fontsize=9, fontweight='bold', color='#2C3E50')
    c1_d = ["tôi", "yêu", "học", "máy"]
    c2_d = ["bạn", "khỏe", "không", "<pad>"]
    for j, token in enumerate(c1_d):
        fc = "#EAEDED" if token == "<pad>" else "#D5F5E3"
        ec = "#BDC3C7" if token == "<pad>" else "#27AE60"
        draw_box(ax, 0.5 + j * 1.1, 1.3, 1.0, 0.4, token, fc, ec, font_size=7, bold=False)
    for j, token in enumerate(c2_d):
        fc = "#EAEDED" if token == "<pad>" else "#D5F5E3"
        ec = "#BDC3C7" if token == "<pad>" else "#27AE60"
        draw_box(ax, 0.5 + j * 1.1, 0.7, 1.0, 0.4, token, fc, ec, font_size=7, bold=False)
    plt.savefig("dynamic_padding.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print("All diagrams generated successfully.")

if __name__ == "__main__":
    generate_all_diagrams()
