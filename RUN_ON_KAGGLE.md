# HƯỚNG DẪN CHẠY NOTEBOOK TRÊN KAGGLE GIAO DIỆN GPU

Tài liệu này hướng dẫn chi tiết cách tải, import và chạy hai file Jupyter Notebook (`mini_transformer_demo.ipynb` và `load_model_demo.ipynb`) trên môi trường đám mây miễn phí **Kaggle** để tận dụng phần cứng GPU tốc độ cao (NVIDIA T4).

---

## 📋 Yêu cầu chuẩn bị
1. Một tài khoản [Kaggle](https://www.kaggle.com/) đã được xác thực số điện thoại (để kích hoạt quyền sử dụng GPU miễn phí và bật Internet).
2. Tải sẵn hai file notebook về máy tính của bạn:
   * [mini_transformer_demo.ipynb](mini_transformer_demo.ipynb) (Notebook huấn luyện chính)
   * [load_model_demo.ipynb](load_model_demo.ipynb) (Notebook nạp mô hình và dịch thử nghiệm)

---

## 🚀 Bước 1: Khởi tạo và Import Notebook lên Kaggle

1. Truy cập vào trang chủ **Kaggle** và đăng nhập.
2. Ở thanh công cụ bên trái, bấm vào nút **Create** (biểu tượng dấu cộng) ➔ Chọn **New Notebook**.
3. Tại giao diện notebook mới mở, chọn tab **File** trên thanh menu góc trên bên trái ➔ Chọn **Import notebook**.
4. Kéo thả file `mini_transformer_demo.ipynb` từ máy tính của bạn vào khu vực tải lên ➔ Bấm **Import**.

*(Thực hiện tương tự các bước trên cho notebook thứ hai `load_model_demo.ipynb`)*.

---

## ⚙️ Bước 2: Cấu hình Môi trường Huấn luyện (Quan trọng)

Để chạy mô hình thành công, bạn **bắt buộc** phải cấu hình hai cài đặt phần cứng và mạng ở bảng điều khiển bên phải (tab **Notebook options**):

1. **Bật GPU tăng tốc (Accelerator):**
   * Tìm mục **Accelerator** (mặc định là *None*).
   * Bấm vào và chọn **GPU T4 x2** (ưu tiên chạy song song đa GPU để khớp với luồng xử lý `nn.DataParallel` trong code) hoặc **GPU T4 x1**.
   * Bấm **Confirm** để khởi động lại máy ảo với GPU.

2. **Bật kết nối Internet (Internet on):**
   * Tìm mục **Internet on** (mặc định tắt).
   * Gạt công tắc sang trạng thái **On** (màu xanh lá).
   * *Lưu ý:* Cài đặt này là bắt buộc để Python có thể tải bộ dữ liệu dịch máy `thainq107/iwslt2015-en-vi` từ Hugging Face.

---

## 🏃 Bước 3: Thực thi Notebook Huấn luyện (`mini_transformer_demo.ipynb`)

1. Chọn **Run** ➔ **Run All** trên thanh menu hoặc chạy từng ô code (Cell) từ trên xuống dưới bằng tổ hợp phím `Shift + Enter`.
2. **Tiến trình huấn luyện:**
   * Hệ thống sẽ tự động cài đặt các thư viện cần thiết (`datasets`, `tokenizers`, `nltk`,...).
   * Tokenizer BPE sẽ tự động được huấn luyện trên kho từ vựng tiếng Việt - Anh và lưu lại file cấu hình `tokenizer.json`.
   * Vòng lặp huấn luyện sẽ chạy qua 30 Epochs. Ở mỗi Epoch, hệ thống sẽ in ra thông số Loss, Perplexity, điểm BLEU Score và in thử 3 mẫu dịch thực tế để bạn theo dõi trực tiếp độ hội tụ.
   * Sau khi huấn luyện hoàn tất, các tệp tin trọng số mô hình tốt nhất (`mini_transformer_best_bleu.pt`, `mini_transformer_vi_en.pt`) và tệp cấu hình tokenizer (`tokenizer.json`) sẽ được lưu vào thư mục làm việc `/kaggle/working`.

---

## 📥 Bước 4: Tải xuống các Checkpoint và Cấu hình Tokenizer

Sau khi chạy xong notebook huấn luyện, bạn cần tải các tệp tin đầu ra về để chuẩn bị cho việc nạp lại mô hình ở notebook demo:

1. Nhìn sang cột bên phải, mở rộng thư mục **Output** ➔ **/kaggle/working**.
2. Tìm các file sau:
   * `tokenizer.json`
   * `mini_transformer_best_bleu.pt`
3. Di chuột vào từng file, bấm vào biểu tượng 3 chấm ở bên phải tên file ➔ Chọn **Download** để tải về máy tính.

---

## 🧪 Bước 5: Chạy thử nghiệm nạp mô hình (`load_model_demo.ipynb`)

Notebook này được dùng để chứng minh khả năng nạp lại các trọng số đã huấn luyện và chạy dịch các câu tùy ý một cách nhanh chóng mà không cần phải huấn luyện lại từ đầu.

1. Mở Notebook `load_model_demo.ipynb` đã import ở Bước 1 trên Kaggle.
2. **Tải lên các file trọng số:**
   * Ở bảng điều khiển bên phải, bấm vào nút **Upload** (biểu tượng mũi tên đi lên) hoặc kéo thả hai file `tokenizer.json` và `mini_transformer_best_bleu.pt` bạn vừa tải về ở Bước 4 vào khu vực **Output** / `/kaggle/working`.
3. Cấu hình **Accelerator** (có thể chọn *None* / *CPU* hoặc *GPU T4*) và bật **Internet on** (nếu cần).
4. Chạy toàn bộ các Cell trong Notebook.
5. Giao diện dịch máy tương tác sẽ xuất hiện ở cuối Notebook. Bạn chỉ cần nhập một câu tiếng Việt bất kỳ vào ô nhập liệu và bấm **Translate** để xem kết quả dịch máy từ mô hình đã huấn luyện của mình.

---

## ⚠️ Các lỗi thường gặp khi chạy trên Kaggle

* **Lỗi `RuntimeError: CUDA out of memory`:**
  ➔ Đảm bảo bạn đã chạy ô cấu hình môi trường có lệnh `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"` ở ngay đầu notebook huấn luyện để tối ưu hóa cấp phát bộ nhớ VRAM.
* **Lỗi `ConnectionError / HTTPError` khi tải dataset:**
  ➔ Do chưa bật cài đặt **Internet on** ở bảng điều khiển bên phải. Vui lòng bật lại Internet và chạy lại.
* **Không lưu được file pt sau khi tắt tab:**
  ➔ Các tệp tin trong `/kaggle/working` sẽ bị xóa khi phiên làm việc (session) kết thúc. Vui lòng tải các file checkpoint về máy tính cá nhân ngay sau khi huấn luyện xong.
