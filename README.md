<h1 align="center">HỆ THỐNG TRÍ TUỆ NHÂN TẠO PHÂN TÍCH
 CHỈ SỐ KHỐI CƠ THỂ VÀ ĐỀ XUẤT LIỆU PHÁP
 CẢI THIỆN SỨC KHỎE</h1>

<div align="center">

<p align="center">
  <img src="images/logoDaiNam.png" alt="DaiNam University Logo" width="200"/>
  <img src="images/LogoAIoTLab.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>

<h2 align="center">Be My trAIner</h2>

<p align="left">
Hệ thống Phân tích BMI và Đề xuất Liệu pháp Sức khỏe Cá nhân
</p>

---
## 🌟 Giới thiệu
# 🤖 Be My trAIner - Hệ thống Phân tích BMI và Đề xuất Liệu pháp Sức khỏe Cá nhân

Hệ thống AI này kết hợp Machine Learning truyền thống và Generative AI (Google Gemini) để phân tích toàn diện các chỉ số cơ thể của người dùng và tự động xây dựng lộ trình dinh dưỡng, tập luyện cá nhân hóa theo từng ngày.

---

## 🚀 Tính năng Chính

* **Phân loại Thể trạng:** Sử dụng thuật toán **Random Forest** để phân loại trạng thái cơ thể (5 lớp) dựa trên 6 chỉ số (BMI, Calo nạp/tiêu hao, Giờ ngủ, v.v.).
* **Đề xuất Cá nhân hóa:** Tận dụng API **Gemini 2.5 Flash** để sinh ra kế hoạch sức khỏe chi tiết, có cấu trúc **JSON**, dựa trên kết quả phân loại từ mô hình ML.
* **Quản lý Lộ trình:** Ứng dụng web cho phép người dùng đăng ký/đăng nhập, kích hoạt kế hoạch, và theo dõi tiến độ hoàn thành các mục tiêu hàng ngày (To-do List) theo thời gian thực.
* **Tính ổn định cao:** Triển khai cơ chế dự phòng API Key (\textit{failover}) cho các lệnh gọi LLM.

---

## 📁 Cấu trúc Dự án
<img width="292" height="602" alt="image" src="https://github.com/user-attachments/assets/af2f160d-822a-49b4-ae62-4ce7194f8030" />


### 1. Cài đặt Môi trường

Tạo và kích hoạt môi trường ảo:

```bash
python -m venv venv
source venv/bin/activate  # Trên Linux/macOS
# hoặc
.\venv\Scripts\activate   # Trên Windows





