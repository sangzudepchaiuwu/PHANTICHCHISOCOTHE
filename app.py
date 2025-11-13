# app.py
import json
import os
import sqlite3
import pickle
import random
import re
from datetime import datetime, timedelta # <--- ĐÃ THÊM timedelta

from flask import (
    Flask, jsonify, render_template, request, redirect, url_for,
    session, flash, g
)
from flask_login import login_required
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import google.generativeai as genai

# ========== CẤU HÌNH ==========
DATABASE = "database.db"
MODEL_PATH = r"D:\chuyendoiso\body_status_model_5classes.pkl"

API_KEYS = [
    "AIzaSyB91mWnouaa05PJFm0Wq7UpTL27y44bz00",
    "AIzaSyBBvzcepUKCHBg1bN4gC0kVMcVSX4nNG-k",
    "AIzaSyB4pyI-JH7dz4WIoWifllS-an8PStEVG_A"
]

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key_change_me")

# ========== LOAD MODEL ==========
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    try:
        FEATURE_NAMES = list(model.feature_names_in_)
    except Exception:
        FEATURE_NAMES = ['chieu_cao_cm', 'can_nang_kg', 'calo_nap', 'calo_tieu_hao', 'thoi_gian_ngu', 'bmi']
    print("[INFO] Model loaded. FEATURE_NAMES:", FEATURE_NAMES)
except Exception as e:
    model = None
    FEATURE_NAMES = ['chieu_cao_cm', 'can_nang_kg', 'calo_nap', 'calo_tieu_hao', 'thoi_gian_ngu', 'bmi']
    print("[WARN] Không load được model. Lỗi:", e)

LABEL_MAP = {
    0: "Thiếu cân nghiêm trọng",
    1: "Thiếu cân",
    2: "Bình thường",
    3: "Thừa cân",
    4: "Béo phì"
}

STYLE_MAP = {
    "Thiếu cân nghiêm trọng": "bg-secondary text-white",
    "Thiếu cân": "bg-info text-dark",
    "Bình thường": "bg-success text-white",
    "Thừa cân": "bg-warning text-dark",
    "Béo phì": "bg-danger text-white"
}

# ========== DB HELPERS ==========
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
        
# Hàm helper để lấy kết nối DB (Được giữ lại để dễ dàng thay thế các hàm get_db_connection cũ)
def get_db_connection():
    return get_db()

# ========== GEMINI HELPERS ==========
def configure_genai_with_key(key):
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-2.5-flash")

def try_generate_content_with_failover(prompt):
    keys = API_KEYS.copy()
    random.shuffle(keys)
    for key in keys:
        try:
            model_gemini = configure_genai_with_key(key)
            print(f"[INFO] Thử key: {key}")
            response = model_gemini.generate_content(prompt)
            print(f"[OK] Key {key} thành công.")
            return response
        except Exception as e:
            print(f"[ERROR] Key {key} lỗi: {e}")
            continue
    return None

def extract_text_from_response(response):
    if response is None:
        return ""
    try:
        if hasattr(response, "text") and response.text:
            return str(response.text)
    except Exception:
        pass
    try:
        if hasattr(response, "candidates") and response.candidates:
            parts = []
            cont = response.candidates[0].content
            if hasattr(cont, "parts"):
                for p in cont.parts:
                    parts.append(getattr(p, "text", "") or "")
            if parts:
                return "\n".join(parts)
    except Exception:
        pass
    return str(response)

# ========== UTILITIES ==========
def markdown_like_to_html(text):
    if not text:
        return ""
    s = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    # 1. Xử lý in đậm **...**
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    lines = s.split("\n")
    out = []
    in_list = False
    
    # Biểu thức chính quy cho danh sách, thêm dấu cách sau ký tự list
    LIST_PATTERN = re.compile(r"^(\*|-|•|\d+\.)\s+") 
    
    for raw in lines:
        line = raw.strip()
        if not line:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        
        # 2. Chỉ chuyển đổi thành <li> nếu khớp với mẫu danh sách
        if LIST_PATTERN.match(line):
            if not in_list:
                out.append("<ul>")
            in_list = True
            li = LIST_PATTERN.sub("", line, 1) # Chỉ thay thế lần xuất hiện đầu tiên
            out.append(f"<li>{li}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{line}</p>")
            
    if in_list:
        out.append("</ul>")
        
    # Loại bỏ dấu ** thừa nếu có
    html = "\n".join(out).replace("**", "") 
    return html.strip()

def extract_and_format_section(content, start_marker, end_marker):
    """Trích xuất một khối nội dung giữa hai tiêu đề."""
    # Tìm vị trí bắt đầu và kết thúc của khối nội dung
    start_match = re.search(re.escape(start_marker), content, re.IGNORECASE | re.DOTALL)
    if not start_match:
        return ""
    
    start_pos = start_match.end()
    
    # Tìm tiêu đề tiếp theo (end_marker)
    end_match = re.search(re.escape(end_marker), content[start_pos:], re.IGNORECASE | re.DOTALL)
    
    if end_match:
        end_pos = start_pos + end_match.start()
        block = content[start_pos:end_pos].strip()
    else:
        # Nếu không tìm thấy end_marker, lấy đến hết chuỗi
        block = content[start_pos:].strip()
        
    return block

import re

def parse_day_details_to_todos(nutrition_html, workout_html):
    """Phân tích chi tiết ngày thành To-do List (dinh dưỡng + bài tập) và phần thông tin gợi ý."""

    def parse_section(html, pattern, section_name):
        """Phân tích 1 phần (dinh dưỡng hoặc bài tập)."""
        todos = []
        info_table = ""
        clean_text = re.sub(r'<[^>]*>', '', html).strip()
        last_end = 0

        for match in pattern.finditer(clean_text):
            title = match.group(1).strip()
            content = match.group(2).strip()
            todos.append(f"{title}: {content}")
            last_end = match.end()

        remaining = clean_text[last_end:].strip()
        if remaining:
            info_table = (
                f"<h6>Thông tin {section_name} khác:</h6>"
                f"<div class='alert alert-secondary'>{remaining}</div>"
            )
        return todos, info_table

    # =======================
    # 1️⃣ DINH DƯỠNG
    # =======================
    nutrition_pattern = re.compile(
        r'(?i)^\s*(Sáng|Phụ sáng|Trưa|Phụ chiều|Tối|Phụ tối|Bữa sáng|Bữa phụ 1|Bữa trưa|Bữa phụ 2|Bữa tối|Bữa phụ 3|Bữa phụ tối)\s*[:\.]\s*(.*?)(?=\n\S|$)',
        re.MULTILINE | re.DOTALL
    )
    nutrition_todos, nutrition_info_table = parse_section(
        nutrition_html, nutrition_pattern, "dinh dưỡng"
    )

    # =======================
    # 2️⃣ BÀI TẬP
    # =======================
    # Cập nhật: cho phép nhận cả Khởi động, Giãn cơ, Bài tập và từng động tác
    workout_text = re.sub(r'<[^>]*>', '', workout_html).strip()

    workout_todos = []
    workout_info_lines = []

    
    exercise_pattern = re.compile(
        r'(?i)^\s*([A-ZĐa-zÀ-ỹ0-9\s\-\(\)]+?)\s*[:\.]\s*(.+)$',
        re.MULTILINE
    )

    lines = [line.strip() for line in workout_text.splitlines() if line.strip()]

    for line in lines:
        # Nếu là tiêu đề phần (VD: "Tập Toàn thân", "Bài tập:", "Strength")
        if re.match(r'(?i)^(tập|strength|bài tập|lưu ý)', line):
            workout_info_lines.append(line)
        # Nếu khớp pattern "Tên: mô tả" → To-do
        elif exercise_pattern.match(line):
            match = exercise_pattern.match(line)
            title = match.group(1).strip()
            content = match.group(2).strip()
            workout_todos.append(f"{title}: {content}")
        # Nếu dòng thông tin thêm
        else:
            workout_info_lines.append(line)

    workout_info_table = ""
    if workout_info_lines:
        workout_info_table = (
            "<h6>Khác:</h6>"
            "<div class='alert alert-secondary'>" +
            " ".join(workout_info_lines) +
            "</div>"
        )

 
    final_todos = nutrition_todos + workout_todos
    return final_todos, nutrition_info_table, workout_info_table


def parse_full_plan_sections(raw_text):
    """
    Tách raw_text thành 3 phần chính (Dinh dưỡng, Tập luyện, Lưu ý)
    và trích xuất chi tiết hàng ngày từ các khối chính đó.
    """
    if not raw_text:
        return "", "", "", []
    
    # Chuẩn hóa văn bản
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n').strip()
    
    # 1. Tách các phần chính I, II, III (Dùng regex mạnh mẽ hơn)
    parts = re.split(r'\n---\n\n*(I\.\s*Kế hoạch Dinh dưỡng|II\.\s*Kế hoạch Tập luyện|III\.\s*Lưu ý chung)', text, flags=re.IGNORECASE)
    
    header_text = parts[0].strip() if len(parts) > 0 else ""
    
    # Ghép các phần lại thành khối lớn, vì regex split sẽ tách tiêu đề ra khỏi nội dung
    nutrition_block = ""
    workout_block = ""
    notes_block = ""
    
    for i in range(1, len(parts)):
        if "I. Kế hoạch Dinh dưỡng" in parts[i]:
            if i + 1 < len(parts):
                nutrition_block = "I. Kế hoạch Dinh dưỡng\n\n" + parts[i+1].strip()
        elif "II. Kế hoạch Tập luyện" in parts[i]:
            if i + 1 < len(parts):
                workout_block = "II. Kế hoạch Tập luyện\n\n" + parts[i+1].strip()
        elif "III. Lưu ý chung" in parts[i]:
            if i + 1 < len(parts):
                notes_block = "III. Lưu ý chung\n\n" + parts[i+1].strip()

    # 2. Tách nội dung tổng quan (Khối Dinh dưỡng và Tập luyện)
    
    # --- Dinh dưỡng: Tách phần Nguyên tắc/Mục tiêu chung khỏi Thực đơn chi tiết
    # Giả định: Thực đơn chi tiết bắt đầu từ "Ngày 1:" hoặc mục số 4.
    nutrition_general_content = re.split(r'(?i)\n*(Ngày\s*1\s*:|4\.\s*Thực đơn gợi ý từng ngày)', nutrition_block)[0].replace("I. Kế hoạch Dinh dưỡng\n\n", "").strip()
    
    # --- Tập luyện: Tách phần Nguyên tắc chung khỏi Lịch trình chi tiết
    # Giả định: Lịch trình chi tiết bắt đầu từ "Ngày 1:" hoặc mục số 2.
    workout_general_content = re.split(r'(?i)\n*(Ngày\s*1\s*:|2\.\s*Lịch trình tập luyện)', workout_block)[0].replace("II. Kế hoạch Tập luyện\n\n", "").strip()
    
    # 3. Trích xuất chi tiết từng ngày
    days_data = {}
    
    # Pattern 1: Tìm "Ngày X:" và nội dung tương ứng (Tiêu đề ngày đơn)
    # LƯU Ý: Đã cải thiện pattern để bao gồm dấu xuống dòng sau tiêu đề ngày.
    day_pattern_single = re.compile(r'(?i)\bNgày\s*([0-9]{1,2})\s*:\s*\n*(.*?)(?=\bNgày\s*[0-9]{1,2}\s*:|\bNgày\s*[0-9]{1,2}\s*đến|$|\n---\n)', re.DOTALL)
    
    # Pattern tìm Khối lặp (Ngày X đến Ngày Y:)
    repeat_pattern = re.compile(r'(?i)\bNgày\s*([0-9]{1,2})\s*đến\s*Ngày\s*([0-9]{1,2})\s*:\s*\n*(.*?)(?=\bNgày\s*[0-9]{1,2}\s*:|\n---\n|$)', re.DOTALL)

    last_single_day_data = {'nutrition': "", 'workout': ""}

    # --- Trích xuất phần chi tiết ngày từ khối Dinh dưỡng ---
    # PHIÊN BẢN SỬA LỖI: Tìm nội dung chi tiết bắt đầu từ mục 4 hoặc Ngày 1:
    nutrition_daily_part = re.search(r'(?i)(4\.\s*Thực đơn gợi ý từng ngày\s*\n*|\bNgày\s*1\s*:\s*\n*)(.*?)$', nutrition_block, re.DOTALL)
    if nutrition_daily_part:
        # Nếu tìm thấy mục 4, lấy nội dung sau đó. Nếu không (tức là chỉ tìm thấy Ngày 1:), lấy nội dung sau Ngày 1:
        daily_text = nutrition_daily_part.group(2).strip()
        
        # 1. Xử lý các ngày đơn (Ngày 1:, Ngày 2:, ...)
        matches = day_pattern_single.findall(daily_text)
        for day_num_str, content in matches:
            day_num = int(day_num_str)
            days_data.setdefault(day_num, {'day': day_num, 'nutrition_html': '', 'workout_html': ''})
            
            nutrition_daily_content = content.strip()
            days_data[day_num]['nutrition_html'] = markdown_like_to_html(nutrition_daily_content)
            
            # Lưu lại nội dung của ngày đơn gần nhất
            last_single_day_data['nutrition'] = nutrition_daily_content

        # 2. Xử lý các khối lặp (Ngày X đến Ngày Y:)
        repeat_matches = repeat_pattern.findall(daily_text)
        for start_num_str, end_num_str, content in repeat_matches:
            start_num = int(start_num_str)
            end_num = int(end_num_str)
            
            for day_num in range(start_num, end_num + 1):
                days_data.setdefault(day_num, {'day': day_num, 'nutrition_html': '', 'workout_html': ''})
                
                # Tự động tìm ngày cơ sở (thường là ngày 1 nếu là khối 8-14)
                base_day_num = 1 
                if start_num > 7:
                    base_day_num = start_num - 7
                
                base_day_content = days_data.get(base_day_num, {}).get('nutrition_html', "")

                repeat_note = f"*LƯU Ý: Đây là lịch dinh dưỡng lặp lại theo nguyên tắc/thực đơn của Ngày {base_day_num} như đã đề cập trong mục **Ngày {start_num} đến Ngày {end_num}**:\n"
                
                if base_day_content:
                    # Nếu Ngày cơ sở đã có nội dung, dùng nó
                    days_data[day_num]['nutrition_html'] = markdown_like_to_html(repeat_note) + base_day_content
                elif content:
                    # Nếu không, dùng nội dung mô tả khối lặp
                    days_data[day_num]['nutrition_html'] = markdown_like_to_html(repeat_note + content.strip())


    # --- Trích xuất phần chi tiết ngày từ khối Tập luyện ---
    # PHIÊN BẢN SỬA LỖI: Tìm nội dung chi tiết bắt đầu từ mục 2 hoặc Ngày 1:
    workout_daily_part = re.search(r'(?i)(2\.\s*Lịch trình tập luyện\s*\n*|\bNgày\s*1\s*:\s*\n*)(.*?)$', workout_block, re.DOTALL)
    if workout_daily_part:
        daily_text = workout_daily_part.group(2).strip()

        # 1. Xử lý các ngày đơn (Ngày 1:, Ngày 2:, ...)
        matches = day_pattern_single.findall(daily_text)
        for day_num_str, content in matches:
            day_num = int(day_num_str)
            days_data.setdefault(day_num, {'day': day_num, 'nutrition_html': '', 'workout_html': ''})
            
            workout_daily_content = content.strip()
            days_data[day_num]['workout_html'] = markdown_like_to_html(workout_daily_content)
            
            # Lưu lại nội dung của ngày đơn gần nhất
            last_single_day_data['workout'] = workout_daily_content
            
        # 2. Xử lý các khối lặp (Ngày X đến Ngày Y:)
        repeat_matches = repeat_pattern.findall(daily_text)
        for start_num_str, end_num_str, content in repeat_matches:
            start_num = int(start_num_str)
            end_num = int(end_num_str)
            
            for day_num in range(start_num, end_num + 1):
                days_data.setdefault(day_num, {'day': day_num, 'nutrition_html': '', 'workout_html': ''})
                
                base_day_num = 1
                if start_num > 7:
                    base_day_num = start_num - 7
                
                base_day_content = days_data.get(base_day_num, {}).get('workout_html', "")

                repeat_note = f"*LƯU Ý: Đây là lịch tập luyện lặp lại theo Ngày {base_day_num} như đã đề cập trong mục **Ngày {start_num} đến Ngày {end_num}**:\n"
                
                if base_day_content:
                    days_data[day_num]['workout_html'] = markdown_like_to_html(repeat_note) + base_day_content
                elif content:
                    days_data[day_num]['workout_html'] = markdown_like_to_html(repeat_note + content.strip())


    days_list = sorted(days_data.values(), key=lambda x: x['day'])
    
    return (
        markdown_like_to_html(nutrition_general_content), 
        markdown_like_to_html(workout_general_content), 
        markdown_like_to_html(notes_block.replace("III. Lưu ý chung\n\n", "").strip()), # Chỉ lấy nội dung Lưu ý chung
        days_list
    )

# Hàm helper để tạo prompt cho Gemini (từ logic cũ)
def create_gemini_prompt(user_data):
    # Lấy dữ liệu từ user_data
    ho_va_ten = user_data["ho_va_ten"]
    tuoi = int(user_data["tuoi"])
    gioi_tinh = user_data["gioi_tinh"]
    chieu_cao = float(user_data["chieu_cao_cm"])
    can_nang = float(user_data["can_nang_kg"])
    calo_nap = float(user_data["calo_nap"])
    calo_tieu_hao = float(user_data["calo_tieu_hao"])
    thoi_gian_ngu = float(user_data["thoi_gian_ngu"])
    so_ngay = int(user_data["so_ngay"])
    lo_trinh = user_data["lo_trinh"]
    
    # Tính BMI và dự đoán trạng thái
    bmi = can_nang / ((chieu_cao / 100) ** 2)
    # Giả định model đã load và FEATURE_NAMES được định nghĩa
    if model is not None:
        X = pd.DataFrame([[chieu_cao, can_nang, calo_nap, calo_tieu_hao, thoi_gian_ngu, bmi]], columns=FEATURE_NAMES)
        X = X.reindex(columns=FEATURE_NAMES, fill_value=0).astype(float)
        pred = int(model.predict(X)[0])
        status_label = LABEL_MAP.get(pred, str(pred))
    else:
        status_label = "Không có model"
    
    # Tạo prompt
    prompt = f"""
Bạn là chuyên gia dinh dưỡng và huấn luyện viên thể hình cá nhân chuyên nghiệp.

Mục tiêu:
Hãy tạo **kế hoạch Dinh dưỡng và Tập luyện chi tiết trong {so_ngay} ngày** cho người dùng, theo đúng cấu trúc bên dưới.
Không thêm lời chào, tóm tắt, phần mở đầu hoặc kết luận dư thừa. 
Trả về nội dung theo **định dạng rõ ràng, đánh số, có tiêu đề và chia thành 3 phần** như sau:

---

Kế hoạch AI
Kế hoạch Dinh dưỡng và Tập luyện Chi tiết trong {so_ngay} ngày

---

I. Kế hoạch Dinh dưỡng

1. Giải thích mục tiêu calo, macro (Protein, Fat, Carb).
2. Nguyên tắc dinh dưỡng chung.
3. Lịch trình bữa ăn mẫu (sáng, phụ, trưa, phụ, tối).
4. Thực đơn gợi ý từng ngày trong {so_ngay} ngày. **(ghi rõ Ngày 1:, Ngày 2:, ...), có định lượng thực phẩm cụ thể.**
5. Gợi ý thay thế nhóm thực phẩm.

---

II. Kế hoạch Tập luyện

1. Nguyên tắc tập luyện chung.
2. Lịch trình tập luyện trong {so_ngay} ngày, ghi rõ từng **Ngày 1: → Ngày {so_ngay}:**, gồm:
    - Tên buổi tập (ví dụ: Toàn thân, Cardio, Lưng - Tay, Nghỉ, v.v.)
    - Các bài tập cụ thể (có số hiệp x số lần lặp).
    - Phần khởi động và giãn cơ nếu cần.

---

III. Lưu ý chung

- Các lời khuyên tổng quát để đạt hiệu quả tốt.
- Nhấn mạnh việc kiên trì, phục hồi, ngủ đủ, và điều chỉnh linh hoạt kế hoạch.

---

Thông tin người dùng:

- Tên: {ho_va_ten}
- Tuổi: {tuoi}
- Giới tính: {gioi_tinh}
- Chiều cao: {chieu_cao} cm
- Cân nặng: {can_nang} kg
- Calo nạp: {calo_nap}
- Calo tiêu hao: {calo_tieu_hao}
- Thời gian ngủ trung bình: {thoi_gian_ngu} giờ/ngày
- Tình trạng cơ thể: {status_label}
- Mục tiêu cá nhân: {lo_trinh}

---

Yêu cầu về định dạng đầu ra:
- Giữ nguyên cấu trúc, tiêu đề và các mục như ví dụ trên.
- Dùng các đoạn gạch đầu dòng, đánh số, chia ngày rõ ràng (ví dụ “Ngày 1:”, “Ngày 2:”...).
- Không thêm bất kỳ lời giải thích, phần tóm tắt, hoặc lời khuyên ngoài cấu trúc đã quy định.
- Trả về **toàn bộ nội dung dạng văn bản thuần (plain text)**, không dùng JSON.

Ví dụ mẫu định dạng:

Kế hoạch AI
Kế hoạch Dinh dưỡng và Tập luyện Chi tiết trong 7 ngày

---

I. Kế hoạch Dinh dưỡng
[...]
Ngày 1: [...]
Ngày 2: [...]
...
---
II. Kế hoạch Tập luyện
[...]
Ngày 1: [...]
Ngày 2: [...]
...
---
III. Lưu ý chung
[...]

---
"""
    return prompt


# ========== ROUTES ==========
@app.route("/")
def index():
    return render_template("index.html", logged_in=("user_id" in session))

# --- FORM BMI (thêm thanh tiến trình) ---
@app.route("/bmi-form")
def bmi_form():
    if "user_id" not in session:
        flash("Vui lòng đăng nhập để bắt đầu lộ trình.", "warning")
        return redirect(url_for("login"))

    # Thanh tiến trình mặc định ở bước đầu (step 1)
    current_step = 1
    total_steps = 9 # Đã đếm lại trong form_bmi.html
    progress = int((current_step / total_steps) * 100)
    return render_template("form_bmi.html", progress=progress, current_step=current_step, total_steps=total_steps)

# ========== AUTH ==========
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        ho_va_ten = request.form.get("ho_va_ten")
        email = request.form.get("email")
        password = request.form.get("password")
        if not (email and password and ho_va_ten):
            flash("Vui lòng điền đủ thông tin", "danger")
            return redirect(url_for("register"))
        pw_hash = generate_password_hash(password)
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (ho_va_ten, email, mat_khau) VALUES (?, ?, ?)",
                (ho_va_ten, email, pw_hash)
            )
            conn.commit()
            flash("Đăng ký thành công. Mời bạn đăng nhập.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email đã tồn tại.", "danger")
            return redirect(url_for("register"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user and check_password_hash(user["mat_khau"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["ho_va_ten"]
            flash("Đăng nhập thành công!", "success")
            return redirect(url_for("index"))
        else:
            flash("Sai email hoặc mật khẩu.", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Đã đăng xuất.", "info")
    return redirect(url_for("index"))

# --- Profile ---
@app.route("/profile")
def profile():
    if "user_id" not in session:
        flash("Bạn cần đăng nhập để xem hồ sơ.", "warning")
        return redirect(url_for("login"))
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    # Chú ý: Cần đổi tên bảng ai_results thành results (tùy thuộc vào DB của bạn)
    results = conn.execute("SELECT * FROM ai_results WHERE user_id = ? ORDER BY created_at DESC", (session["user_id"],)).fetchall()
    return render_template("profile.html", user=user, results=results)

@app.route('/update_todo_progress', methods=['POST'])
def update_todo_progress():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.json
    plan_id = data.get('plan_id')
    day_number = data.get('day_number')
    todo_index = data.get('todo_index')
    completed = data.get('completed')
    total_todos = data.get('total_todos')
    
    if any(x is None for x in [plan_id, day_number, todo_index, completed, total_todos]):
        return jsonify({'error': 'Missing data'}), 400
    
    conn = get_db()
    # Check ownership
    plan = conn.execute('SELECT user_id FROM user_plans WHERE id = ?', (plan_id,)).fetchone()
    if not plan or plan['user_id'] != session['user_id']:
        conn.close()
        return jsonify({'error': 'Forbidden'}), 403
    
    # Get or create progress row
    progress = conn.execute('SELECT completed_todos FROM user_plan_progress WHERE user_plan_id = ? AND day_number = ?',
                            (plan_id, day_number)).fetchone()
    
    if progress:
        completed_list = json.loads(progress['completed_todos'])
        if len(completed_list) != total_todos:
            completed_list = [False] * total_todos
    else:
        completed_list = [False] * total_todos
        conn.execute('INSERT INTO user_plan_progress (user_plan_id, day_number, completed_todos, all_completed) VALUES (?, ?, ?, 0)',
                     (plan_id, day_number, json.dumps(completed_list)))
        conn.commit()  # Commit insert
    
    # Update list
    completed_list[todo_index] = completed
    
    # Check all completed
    all_completed = all(completed_list)
    
    # Update DB
    conn.execute('UPDATE user_plan_progress SET completed_todos = ?, all_completed = ? WHERE user_plan_id = ? AND day_number = ?',
                 (json.dumps(completed_list), 1 if all_completed else 0, plan_id, day_number))
    conn.commit()
    conn.close()
    
    return jsonify({'all_completed': all_completed})
# --- Route mới: Chỉnh sửa thông tin ---
@app.route("/edit-info", methods=["GET", "POST"])
def edit_info():
    if "user_id" not in session:
        flash("Bạn cần đăng nhập để chỉnh sửa thông tin.", "warning")
        return redirect(url_for("login"))
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    if request.method == "POST":
        tuoi = request.form.get("tuoi")
        gioi_tinh = request.form.get("gioi_tinh")
        chieu_cao = request.form.get("chieu_cao_cm")
        can_nang = request.form.get("can_nang_kg")
        conn.execute(
            "UPDATE users SET tuoi = ?, gioi_tinh = ?, chieu_cao = ?, can_nang = ? WHERE id = ?",
            (tuoi, gioi_tinh, chieu_cao, can_nang, session["user_id"])
        )
        conn.commit()
        flash("Cập nhật thông tin thành công!", "success")
        return redirect(url_for("profile"))
    return render_template("edit_info.html", user=user)

# --- Feedback ---
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        user_id = session.get("user_id")
        fullname = request.form.get("fullname") or (session.get("user_name") if session.get("user_name") else "")
        rating = int(request.form.get("rating", 5))
        comment = request.form.get("comment")
        conn = get_db()
        conn.execute("INSERT INTO feedbacks (user_id, fullname, rating, comment) VALUES (?, ?, ?, ?)",
                      (user_id, fullname, rating, comment))
        conn.commit()
        flash("Cảm ơn bạn đã đánh giá!", "success")
        return redirect(url_for("feedback"))
    return render_template("feedback.html")


# -----------------------------------------------------
# ** BƯỚC 1: ROUTE MỚI CHO MÀN HÌNH CHỜ (/analyzing) **
# -----------------------------------------------------
@app.route("/analyzing", methods=["POST"])
def analyzing_screen():
    """
    Nhận dữ liệu form, tính BMI nhanh, lưu vào Session, và render màn hình chờ.
    """
    try:
        # 1. Lấy tất cả dữ liệu form
        user_data = request.form.to_dict()

        # 2. Tính toán BMI và dự đoán nhanh trạng thái
        chieu_cao_cm = float(user_data.get("chieu_cao_cm", 0))
        can_nang = float(user_data.get("can_nang_kg", 0))
        calo_nap = float(user_data.get("calo_nap", 0))
        calo_tieu_hao = float(user_data.get("calo_tieu_hao", 0))
        thoi_gian_ngu = float(user_data.get("thoi_gian_ngu", 0))
        
        bmi = can_nang / ((chieu_cao_cm / 100) ** 2)
        user_data["bmi"] = bmi # Lưu BMI vào session

        # Dự đoán nhanh trạng thái (Dùng model đã load)
        input_data = pd.DataFrame([[
            chieu_cao_cm, can_nang, calo_nap, calo_tieu_hao, thoi_gian_ngu, bmi
        ]], columns=FEATURE_NAMES)
        
        X = input_data.reindex(columns=FEATURE_NAMES, fill_value=0).astype(float)
        
        if model is not None:
            status_label_index = model.predict(X)[0]
            status_label = LABEL_MAP.get(status_label_index, "Không xác định")
        else:
            status_label = "Chưa load model"
            
        status_class = STYLE_MAP.get(status_label, "bg-light text-dark")

        # 3. Lưu toàn bộ dữ liệu cần thiết vào Session
        session["user_data_for_analysis"] = user_data

        # 4. Render analyzing.html
        return render_template(
            "analyzing.html", 
            data={
                "ho_va_ten": user_data.get("ho_va_ten"),
                "can_nang": can_nang,
                "can_nang_mong_muon": user_data.get("can_nang_mong_muon"),
                "so_ngay": user_data.get("so_ngay"),
                "status_label": status_label,
                "status_class": status_class,
            }
        )
    except Exception as e:
        flash(f"Lỗi khi nhận dữ liệu: {e}", "danger")
        return redirect(url_for("bmi_form"))


# -----------------------------------------------------------------
# ** BƯỚC 2: ROUTE MỚI CHO XỬ LÝ AI & HIỂN THỊ KẾT QUẢ (/result) **
# -----------------------------------------------------------------
@app.route("/result", methods=["GET"])
def predict_and_show_result():
    """
    Thực hiện logic gọi AI, lưu DB, và render result.html.
    Đây chính là logic của route /predict cũ, đã được đổi tên và dùng Session.
    """
    # 1. Lấy dữ liệu từ Session và xóa ngay lập tức (để tránh xử lý lại)
    user_data = session.pop("user_data_for_analysis", None)
    
    if not user_data:
        flash("Phiên phân tích đã hết hạn hoặc không tìm thấy dữ liệu.", "info")
        return redirect(url_for("bmi_form"))
    
    # Bắt đầu Logic Xử lý AI Tốn Thời Gian
    try:
        # --- LẤY DỮ LIỆU TỪ SESSION ---
        ho_va_ten = user_data.get("ho_va_ten")
        tuoi = int(user_data.get("tuoi"))
        gioi_tinh = user_data.get("gioi_tinh")
        chieu_cao = float(user_data.get("chieu_cao_cm"))
        can_nang = float(user_data.get("can_nang_kg"))
        calo_nap = float(user_data.get("calo_nap"))
        calo_tieu_hao = float(user_data.get("calo_tieu_hao"))
        thoi_gian_ngu = float(user_data.get("thoi_gian_ngu"))
        so_ngay = int(user_data.get("so_ngay"))
        lo_trinh = user_data.get("lo_trinh")
        bmi = user_data.get("bmi")
        
        # 2. Dự đoán trạng thái
        X = pd.DataFrame([user_data]).reindex(columns=FEATURE_NAMES, fill_value=0).astype(float)
        
        if model is not None:
            pred = int(model.predict(X)[0])
            status_label = LABEL_MAP.get(pred, str(pred))
        else:
            status_label = "Không có model"

        status_class = STYLE_MAP.get(status_label, "bg-light text-dark")
        
        # 3. GỌI GEMINI API (QUÁ TRÌNH LÂU NHẤT)
        prompt = create_gemini_prompt(user_data) # Sử dụng user_data từ session
        response = try_generate_content_with_failover(prompt)
        raw_text = extract_text_from_response(response) if response else "API lỗi hoặc không phản hồi."
        
        # 4. Phân tích kết quả
        full_nutrition_html, full_workout_html, notes_html, days_list = parse_full_plan_sections(raw_text)
        
        # 5. Lưu DB
        conn = get_db()
        user_id = session.get("user_id")
        if user_id:
            conn.execute(
                "UPDATE users SET tuoi=?, gioi_tinh=? WHERE id=?",
                (tuoi, gioi_tinh, user_id)
            )
        # Lưu vào DB và lấy ID của kết quả vừa tạo
        cursor = conn.execute('''
            INSERT INTO ai_results (
                user_id, ho_va_ten, tuoi, gioi_tinh, chieu_cao_cm,
                can_nang_kg, calo_nap, calo_tieu_hao, thoi_gian_ngu,
                bmi, so_ngay, tinh_trang, plan_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, ho_va_ten, tuoi, gioi_tinh, chieu_cao, can_nang,
            calo_nap, calo_tieu_hao, thoi_gian_ngu, bmi, so_ngay,
            status_label, raw_text
        ))
        rid = cursor.lastrowid
        conn.commit()
        conn.close() # Đóng kết nối DB
        
        # 6. Xử lý hiển thị tuần/ngày (Không dùng trong result.html, nhưng giữ lại)
        weeks = []
        for i in range(0, len(days_list), 7):
            weeks.append({"week": i // 7 + 1, "days": days_list[i:i + 7]})
        
        # 7. Render kết quả
        return render_template(
            "result.html",
            rid=rid, # Truyền ID của kết quả vừa tạo
            name=ho_va_ten,
            tuoi=tuoi,
            gioi_tinh=gioi_tinh,
            lo_trinh=lo_trinh,
            status=status_label,
            status_class=status_class,
            weeks=weeks,
            days_list=days_list,
            full_nutrition_html=full_nutrition_html,
            full_workout_html=full_workout_html,
            notes_html=notes_html,
            raw_text=raw_text
        )

    except Exception as e:
        flash(f"Lỗi khi xử lý và gọi AI: {e}", "danger")
        return redirect(url_for("bmi_form"))


# --- View kết quả đã lưu (ĐÃ SỬA LỖI days_list|length) ---
@app.route("/result/<int:rid>")
def view_saved_result(rid):
    conn = get_db()
    ai_result = conn.execute("SELECT * FROM ai_results WHERE id = ?", (rid,)).fetchone()
    conn.close()

    if not ai_result:
        flash("Không tìm thấy kết quả", "warning")
        return redirect(url_for("profile"))
    
    raw_text = ai_result['plan_text'] # Lấy plan_text (nội dung thô của AI)

    # Phân tích kết quả để lấy days_list và các phần HTML
    full_nutrition_html, full_workout_html, notes_html, days_list = parse_full_plan_sections(raw_text)

    # Tính toán trạng thái và class hiển thị
    status_class = STYLE_MAP.get(ai_result['tinh_trang'], "bg-light text-dark")
    
    # Render result.html (để sử dụng modal "Xác nhận Lộ trình")
    return render_template(
        "result.html",
        rid=rid, # Truyền ID của kết quả AI
        name=ai_result['ho_va_ten'],
        tuoi=ai_result['tuoi'],
        gioi_tinh=ai_result['gioi_tinh'],
        lo_trinh="Đã lưu", # Không có thông tin mục tiêu cụ thể khi lưu, tạm dùng "Đã lưu"
        status=ai_result['tinh_trang'],
        status_class=status_class,
        days_list=days_list, # <<< ĐÃ KHẮC PHỤC LỖI days_list|length
        full_nutrition_html=full_nutrition_html,
        full_workout_html=full_workout_html,
        notes_html=notes_html,
        raw_text=raw_text
    )

@app.route('/confirm_plan', methods=['POST'])

def confirm_plan():
    # Lấy thông tin từ form xác nhận
    plan_name = request.form.get('plan_name')
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')
    ai_result_id = request.form.get('ai_result_id')
    user_id = session.get('user_id')

    if not all([plan_name, start_date_str, end_date_str, ai_result_id, user_id]):
        flash("Lỗi: Thiếu thông tin lộ trình.", 'danger')
        return redirect(url_for('view_saved_result', rid=ai_result_id))
        
    try:
        # Kiểm tra tính hợp lệ của ngày
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        if start_date > end_date:
            flash("Ngày bắt đầu không thể sau Ngày kết thúc.", 'warning')
            return redirect(url_for('view_saved_result', rid=ai_result_id))

        # Lưu vào DB (ĐÃ SỬA get_db_connection -> get_db)
        conn = get_db()
        conn.execute('INSERT INTO user_plans (user_id, plan_name, start_date, end_date, ai_result_id) VALUES (?, ?, ?, ?, ?)',
                     (user_id, plan_name, start_date_str, end_date_str, ai_result_id))
        conn.commit()
        conn.close()
        
        flash(f"Đã xác nhận và lưu lộ trình '{plan_name}'!", 'success')
        return redirect(url_for('current_plan'))

    except ValueError:
        flash("Định dạng ngày không hợp lệ.", 'danger')
        return redirect(url_for('view_saved_result', rid=ai_result_id))
    except Exception as e:
        flash(f"Lỗi khi lưu lộ trình: {e}", 'danger')
        return redirect(url_for('view_saved_result', rid=ai_result_id))
    

@app.route('/current_plan')
def current_plan():
    user_id = session.get('user_id')
    conn = get_db()
    
    # 1. Lấy lộ trình hiện tại (ví dụ: lộ trình mới nhất)
    current_plan_row = conn.execute('SELECT * FROM user_plans WHERE user_id = ? ORDER BY created_at DESC LIMIT 1', (user_id,)).fetchone()
    
    if not current_plan_row:
        conn.close()
        flash("Bạn chưa có lộ trình nào được xác nhận. Vui lòng tạo một lộ trình.", 'info')
        return render_template('current_plan.html', plan=None, daily_data=[])
        
    # 2. Lấy chi tiết AI Result
    ai_result_id = current_plan_row['ai_result_id']
    ai_result = conn.execute('SELECT * FROM ai_results WHERE id = ?', (ai_result_id,)).fetchone()
    
    if not ai_result:
        conn.close()
        flash("Không tìm thấy chi tiết phân tích AI cho lộ trình này.", 'danger')
        return render_template('current_plan.html', plan=current_plan_row, daily_data=[])
        
    # 3. Phân tích dữ liệu ngày
    full_nutrition_html, full_workout_html, notes_html, days_list = parse_full_plan_sections(ai_result['plan_text'])
    
    plan_start_date = datetime.strptime(current_plan_row['start_date'], '%Y-%m-%d').date()
    plan_end_date = datetime.strptime(current_plan_row['end_date'], '%Y-%m-%d').date()
    
    daily_data = []
    
    for d in days_list:
        day_index = d['day'] # Ngày 1, Ngày 2, ...
        
        # Tính toán ngày thực tế
        actual_date = plan_start_date + timedelta(days=day_index - 1)
        
        if actual_date > plan_end_date:
            continue

        # Phân tích thành To-do List và Bảng gợi ý
        todos, nutri_info, workout_info = parse_day_details_to_todos(d['nutrition_html'], d['workout_html'])
        
        # Get progress
        progress = conn.execute('SELECT completed_todos, all_completed FROM user_plan_progress WHERE user_plan_id = ? AND day_number = ?',
                                (current_plan_row['id'], day_index)).fetchone()
        
        if progress and progress['completed_todos']:
            completed_todos = json.loads(progress['completed_todos'])
            if len(completed_todos) != len(todos):
                completed_todos = [False] * len(todos)
            all_completed = progress['all_completed']
        else:
            completed_todos = [False] * len(todos)
            all_completed = 0
        
        daily_data.append({
            'day': day_index,
            'actual_date': actual_date.strftime('%d/%m/%Y'),
            'todos': todos,
            'nutri_info': nutri_info,
            'workout_info': workout_info,
            'completed_todos': completed_todos,
            'all_completed': bool(all_completed)
        })
        
    conn.close()
    return render_template('current_plan.html', plan=current_plan_row, daily_data=daily_data)

if __name__ == "__main__":
    app.run(debug=True)