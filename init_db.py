# Modified init_db.py
# init_db.py
import sqlite3

DB = "database.db"

conn = sqlite3.connect(DB)
c = conn.cursor()

# Bảng người dùng
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ho_va_ten TEXT,
    email TEXT UNIQUE,
    mat_khau TEXT,
    tuoi INTEGER,
    gioi_tinh TEXT,
    chieu_cao_cm REAL,
    can_nang_kg REAL,
    calo_nap REAL,
    calo_tieu_hao REAL,
    thoi_gian_ngu REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Bảng lưu kết quả AI (lưu nguyên bản plan + các thông số cơ bản)
c.execute('''
CREATE TABLE IF NOT EXISTS ai_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    ho_va_ten TEXT,
    tuoi INTEGER,
    gioi_tinh TEXT,
    chieu_cao_cm REAL,
    can_nang_kg REAL,
    calo_nap REAL,
    calo_tieu_hao REAL,
    thoi_gian_ngu REAL,
    bmi REAL,
    so_ngay INTEGER,
    tinh_trang TEXT,
    plan_text LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
''')

# Bảng feedback
c.execute('''
CREATE TABLE IF NOT EXISTS feedbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    fullname TEXT,
    rating INTEGER,
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

c.execute("""
    CREATE TABLE IF NOT EXISTS user_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        plan_name TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        ai_result_id INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(ai_result_id) REFERENCES ai_results(id)
    );
""")

# New table for progress
c.execute("""
    CREATE TABLE IF NOT EXISTS user_plan_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_plan_id INTEGER NOT NULL,
        day_number INTEGER NOT NULL,
        completed_todos TEXT,  -- JSON list of booleans
        all_completed INTEGER DEFAULT 0,
        FOREIGN KEY(user_plan_id) REFERENCES user_plans(id),
        UNIQUE(user_plan_id, day_number)
    );
""")

conn.commit()
conn.close()
print("✅ Đã tạo database database.db và các bảng cần thiết")