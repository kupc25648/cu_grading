# ArchGrade Studio

**ArchGrade Studio** is a Streamlit-based grading exploration app designed for faculty review meetings.  
It helps instructors upload student scores, inspect score distributions, test multiple grading schemes, visualize grade outcomes, and export final grades.

This app was developed for architecture/design studio grading contexts where grading decisions often require both quantitative support and human academic judgment.

---

## Table of Contents

- [1. What This App Does](#1-what-this-app-does)
- [2. Key Features](#2-key-features)
- [3. Recommended Repository Structure](#3-recommended-repository-structure)
- [4. Installation](#4-installation)
- [5. How to Run](#5-how-to-run)
- [6. Input File Format](#6-input-file-format)
- [7. Step-by-Step User Manual](#7-step-by-step-user-manual)
- [8. Grading Modes](#8-grading-modes)
- [9. Excel-Like Quantile-Range Logic](#9-excel-like-quantile-range-logic)
- [10. K-Means and Outlier Handling](#10-k-means-and-outlier-handling)
- [11. Exporting Final Grades](#11-exporting-final-grades)
- [12. Thai Manual / คู่มือภาษาไทย](#12-thai-manual--คู่มือภาษาไทย)
- [13. Development Notes](#13-development-notes)
- [14. Limitations](#14-limitations)
- [15. License](#15-license)

---

## 1. What This App Does

ArchGrade Studio supports grading review by combining:

1. **Score normalization**
2. **Outlier detection**
3. **Visual grade distribution review**
4. **Multiple grading schemes**
5. **Excel-like quantile-range grading**
6. **K-means clustering**
7. **Final grade export**

The app is not intended to replace academic judgment.  
Instead, it provides a transparent visual and numerical interface for comparing grading decisions.

---

## 2. Key Features

### General Features

- Upload CSV or Excel files.
- Select student ID and score columns.
- Normalize raw scores to a 0–4 scale.
- Detect outliers using Z-score and IQR.
- Choose whether outliers are included or excluded from calculations.
- Review score histogram, ranked score plot, grade distribution bar chart, and donut chart.
- Export official grades and full comparison tables.

### Grading Modes

The app currently supports:

1. **Z-score grading**
2. **Simple range grading**
3. **Quantile-range grading**
4. **K-means grading**
5. **Final comparison and export**

### Excel-Like Quantile-Range Mode

The latest version includes Excel-like logic based on the `ตัดGrade` sheet workflow:

- Score cells are created with a fixed cell width, typically `0.025`.
- Central score is based on **Excel GEOMEAN**.
- The central-score anchor is rounded to 3 decimals for display/anchor use.
- Student counting follows Excel `COUNTIFS` style:

```text
score <= upper boundary AND score > lower boundary
```

This means a student exactly at a boundary belongs to the lower cell.

---

## 3. Recommended Repository Structure

For GitHub, the following structure is recommended:

```text
archgrade-studio/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── sample_scores.csv
│   └── grading_schemes_example.xlsx
│
├── docs/
│   ├── user_manual_en.md
│   ├── user_manual_th.md
│   └── grading_logic_notes.md
│
├── screenshots/
│   ├── overview.png
│   ├── quantile_range.png
│   ├── kmeans.png
│   └── export.png
│
└── tests/
    └── test_grading_logic.py
```

For a small first release, you can keep only:

```text
archgrade-studio/
├── app.py
├── README.md
├── requirements.txt
└── data/
    └── sample_scores.csv
```

---

## 4. Installation

### 4.1 Clone the repository

```bash
git clone https://github.com/your-username/archgrade-studio.git
cd archgrade-studio
```

### 4.2 Create a virtual environment

For macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, create one with:

```text
streamlit
pandas
numpy
scipy
scikit-learn
plotly
openpyxl
```

Then install:

```bash
pip install -r requirements.txt
```

---

## 5. How to Run

Run the app with:

```bash
streamlit run app.py
```

The app will open in your browser.

Usually the local address is:

```text
http://localhost:8501
```

---

## 6. Input File Format

The input file should be a `.csv`, `.xlsx`, or `.xls` file.

At minimum, it should contain:

| Column | Description |
|---|---|
| Student ID | Unique student identifier |
| Score | Final numeric score |

Example:

```csv
ID,SCORE
6430046525,2.356486
6430093025,2.446538
6330062425,2.584615
```

The score can be:

- already on a 0–4 scale, or
- raw score, such as 0–100, which can be normalized inside the app.

---

## 7. Step-by-Step User Manual

### Step 1: Upload data

Use the sidebar to upload a CSV or Excel file.

### Step 2: Select columns

Choose:

- Student ID column
- Final score column

### Step 3: Select score scale

Choose one of:

1. **Score is already 0–4**
2. **Normalize by observed min/max to 0–4**
3. **Normalize by custom raw min/max to 0–4**

If your score is already a GPA-like score from 0 to 4, choose:

```text
Score is already 0–4
```

### Step 4: Choose outlier policy

Available options:

1. Use all students
2. Exclude detected outliers from calculation only
3. Manually exclude selected students from calculation only
4. Exclude detected + manually selected from calculation

Important:  
Excluded students are removed only from calculation. They are still included in final output.

### Step 5: Review dataset overview

The overview shows:

- number of students
- calculation students
- central score / GEOMEAN
- median
- standard deviation
- number of outliers

The score distribution chart shows:

- score histogram
- central score line
- quartile lines
- outlier status

### Step 6: Try grading modes

Use the tabs:

- Z-score
- Simple range
- Quantile-range
- K-means
- Final comparison & export

### Step 7: Select final grading result

In the final export tab, choose the final grade column.

The app automatically remembers the most recently adjusted grading scheme, but you should always confirm the selected scheme before export.

### Step 8: Export grades

Available export files:

1. Official grade CSV by student ID
2. Full comparison CSV
3. Ranked review CSV

---

## 8. Grading Modes

## 8.1 Z-Score Grading

Z-score grading assigns grades based on distance from the class mean.

Example default thresholds:

| Grade | Condition |
|---|---|
| A | z ≥ 1.50 |
| B+ | z ≥ 1.00 |
| B | z ≥ 0.50 |
| C+ | z ≥ 0.00 |
| C | z ≥ -0.50 |
| D+ | z ≥ -1.00 |
| D | z ≥ -1.50 |
| F | below previous boundary |

This mode is useful when grading should reflect relative performance.

---

## 8.2 Simple Range Grading

Simple range grading uses direct score cutoffs.

Example:

| Grade | Condition |
|---|---|
| A | score ≥ 3.70 |
| B+ | score ≥ 3.30 |
| B | score ≥ 3.00 |
| C+ | score ≥ 2.50 |
| C | score ≥ 2.00 |
| D+ | score ≥ 1.50 |
| D | score ≥ 1.00 |
| F | below previous boundary |

This mode is useful when official or fixed grade boundaries are required.

---

## 8.3 Quantile-Range Grading

Quantile-range grading divides the score range into small score cells.

Typical setting:

```text
Cell width = 0.025
```

For example:

| Cell | Upper | Lower |
|---|---:|---:|
| 1 | 4.000 | 3.975 |
| 2 | 3.975 | 3.950 |
| 3 | 3.950 | 3.925 |

Each cell counts students using Excel-style logic:

```text
score <= upper AND score > lower
```

The last cell also includes the lower bound.

This mode is useful for faculty review because the grading boundary is adjusted visually and transparently.

---

## 8.4 Central-Score Cell Width Mode

This is the most Excel-like mode.

It follows the logic of the grading worksheet:

1. Create score cells.
2. Compute central score using Excel-like GEOMEAN.
3. Round the central score anchor to 3 decimals.
4. Find the cell containing the central score.
5. Assign grade bands upward and downward from the central cell.
6. Preserve fixed cell bands.
7. Apply Excel-like rim behavior if extra students remain.

For example, if the selected core grades are:

```text
A / B+ / B / C+
```

and lower cells remain, the app can add:

```text
C
```

at the lower rim.

For 5 or 7 grades, the middle grade is placed around the central score cell.

---

## 9. Excel-Like Quantile-Range Logic

The app is designed to approximate the grading worksheet logic.

### 9.1 Central score

The central score uses geometric mean:

```excel
=GEOMEAN(score_range)
```

In the app:

- GEOMEAN is computed from valid positive scores.
- The displayed anchor is rounded to 3 decimals.
- The central-score line in the plot follows this rounded anchor.

### 9.2 Quartiles

Quartiles are based on Excel-like exclusive quartiles:

```excel
=QUARTILE.EXC(score_range,1)
=QUARTILE.EXC(score_range,2)
=QUARTILE.EXC(score_range,3)
```

This may differ from pandas default quantiles.

### 9.3 Cell counting

Excel-style cell counting uses:

```excel
=COUNTIFS(score_range,"<="&upper_cell,score_range,">"&lower_cell)
```

Therefore:

```text
upper boundary is included
lower boundary is excluded
```

Example:

| Score | Cell 3.800–3.775 | Cell 3.775–3.750 |
|---:|---:|---:|
| 3.780 | yes | no |
| 3.775 | no | yes |
| 3.760 | no | yes |

A score exactly on the boundary goes to the lower cell.

### 9.4 Why raw scores are not rounded

The app does **not** round raw student scores before counting.

Example:

```text
score = 3.725384
boundary = 3.725
```

This score remains above the boundary, so it stays in the upper cell.

If the score were rounded first, it would become `3.725` and incorrectly move to the lower cell.

---

## 10. K-Means and Outlier Handling

K-means grading clusters student scores into natural groups.

### 10.1 Basic K-means logic

1. Choose number of clusters.
2. Fit K-means using the calculation set.
3. Sort cluster centers from highest to lowest.
4. Map clusters to grade labels.

Example:

```text
k = 4
labels = A, B+, B, C+
```

The highest cluster becomes A, the next becomes B+, and so on.

### 10.2 Excluded outlier handling

If outliers are excluded from fitting:

- Upper-end excluded students are assigned to the top grade.
- Lower-end excluded students are assigned to one grade lower than the lowest selected K-means grade.

Examples:

| Selected K-means grades | Lower-end excluded outlier becomes |
|---|---|
| A / B+ / B / C+ | C |
| A / B+ / B / C+ / C | D+ |
| A / B+ / B / C+ / C / D+ | D |
| A / B+ / B / C+ / C / D+ / D | F |

This avoids forcing very low excluded students into the lowest fitted cluster.

---

## 11. Exporting Final Grades

The export tab combines all grading results.

The exported official file contains:

| Column | Description |
|---|---|
| student_id | Student identifier |
| raw_score | Original uploaded score |
| score_4 | Normalized score |
| grade | Final selected grade |

The app provides:

1. **Official CSV — student ID order**
2. **Full comparison CSV — student ID order**
3. **Ranked review CSV — score order**

Before exporting, confirm the selected final grade column.

---

# 12. Thai Manual / คู่มือภาษาไทย

## 12.1 แอปนี้ใช้ทำอะไร

ArchGrade Studio เป็นแอปสำหรับช่วยอาจารย์ตรวจสอบและเปรียบเทียบการตัดเกรดจากคะแนนรวมของนิสิต

แอปนี้ไม่ได้ตัดสินเกรดแทนอาจารย์ แต่ช่วยให้เห็นว่าเมื่อปรับเกณฑ์แล้ว:

- จำนวน A เปลี่ยนอย่างไร
- B+ / B / C+ กระจายตัวอย่างไร
- มีนิสิตคะแนนต่ำหรือสูงผิดปกติหรือไม่
- เกณฑ์ที่ใช้สมเหตุสมผลหรือไม่
- ผลลัพธ์สามารถอธิบายในที่ประชุมได้หรือไม่

---

## 12.2 วิธีติดตั้ง

ติดตั้ง Python แล้วสร้าง virtual environment:

```bash
python -m venv .venv
```

เปิดใช้งาน environment:

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

ติดตั้ง package:

```bash
pip install -r requirements.txt
```

---

## 12.3 วิธีเปิดแอป

```bash
streamlit run app.py
```

จากนั้นเปิดใน browser

```text
http://localhost:8501
```

---

## 12.4 รูปแบบไฟล์ข้อมูล

ไฟล์ควรเป็น `.csv` หรือ `.xlsx`

อย่างน้อยต้องมี 2 คอลัมน์:

1. รหัสนิสิต
2. คะแนนรวม

ตัวอย่าง:

```csv
ID,SCORE
6430046525,2.356486
6430093025,2.446538
6330062425,2.584615
```

---

## 12.5 ขั้นตอนการใช้งาน

### ขั้นที่ 1: Upload ไฟล์

เลือกไฟล์คะแนนจาก sidebar

### ขั้นที่ 2: เลือกคอลัมน์

เลือก:

- คอลัมน์รหัสนิสิต
- คอลัมน์คะแนนรวม

### ขั้นที่ 3: เลือกสเกลคะแนน

หากคะแนนอยู่ในช่วง 0–4 แล้ว ให้เลือก:

```text
Score is already 0–4
```

หากเป็นคะแนนดิบ เช่น 0–100 สามารถเลือก normalize ได้

### ขั้นที่ 4: เลือกการจัดการ outlier

เลือกได้ว่า:

- ใช้นิสิตทุกคน
- ตัด outlier ออกจากการคำนวณเท่านั้น
- เลือกตัดบางคนออกจากการคำนวณเอง
- ตัดทั้ง outlier และรายชื่อที่เลือกเอง

หมายเหตุ:  
การตัดออกจาก calculation ไม่ได้ลบนิสิตออกจากผลลัพธ์สุดท้าย

### ขั้นที่ 5: ดูภาพรวมคะแนน

ดูกราฟ histogram, ค่า central score, median, std และ outlier

### ขั้นที่ 6: ทดลองตัดเกรด

เลือก tab ที่ต้องการ:

- Z-score
- Simple range
- Quantile-range
- K-means

### ขั้นที่ 7: เลือกผลลัพธ์สุดท้าย

ไปที่ tab:

```text
Final comparison & export
```

แล้วเลือก grade column ที่ต้องการใช้เป็นผลลัพธ์ทางการ

### ขั้นที่ 8: Export

ดาวน์โหลดไฟล์ CSV สำหรับส่งออก

---

## 12.6 คำอธิบายโหมดต่าง ๆ

### Z-score

ใช้ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานในการดูว่านิสิตอยู่สูงหรือต่ำกว่าค่าเฉลี่ยของห้องมากเพียงใด

เหมาะกับกรณีที่ต้องการตัดเกรดแบบสัมพันธ์กับภาพรวมของห้อง

---

### Simple range

ใช้เกณฑ์คะแนนตรง ๆ เช่น:

```text
A >= 3.70
B+ >= 3.30
B >= 3.00
```

เหมาะกับกรณีที่มีเกณฑ์คงที่หรืออยากปรับเส้นด้วยมือ

---

### Quantile-range

แบ่งคะแนนเป็น cell เล็ก ๆ เช่น cell กว้าง 0.025

ตัวอย่าง:

```text
4.000–3.975
3.975–3.950
3.950–3.925
```

แล้วนับจำนวนนิสิตในแต่ละ cell

เหมาะกับการประชุมที่ต้องการเห็นรายละเอียดการกระจายตัวของคะแนน

---

### Central-score cell width mode

เป็นโหมดที่ใกล้กับ Excel มากที่สุด

หลักการ:

1. แบ่งคะแนนเป็น cell
2. หา central score ด้วย GEOMEAN
3. ปัดค่า central score เป็น 3 ตำแหน่งเพื่อใช้เป็น anchor
4. หา cell ที่ central score อยู่
5. วางช่วงเกรดจาก cell กลางขึ้นและลง
6. หากมีคะแนนเหลือที่ขอบล่าง จะเพิ่มเกรดถัดไปตามลำดับ

ตัวอย่าง:

```text
เลือก A / B+ / B / C+
ถ้ายังมีคะแนนต่ำกว่า C+
จะเพิ่ม C ที่ขอบล่าง
```

---

### K-means

ใช้ clustering เพื่อดูว่าคะแนนแบ่งกลุ่มตามธรรมชาติอย่างไร

หากตัด outlier ออกจากการ fit:

- outlier ด้านบนจะได้เกรดบนสุด
- outlier ด้านล่างจะได้เกรดต่ำกว่าเกรดล่างสุดที่เลือก 1 ขั้น

ตัวอย่าง:

```text
เลือก A / B+ / B / C+
outlier ด้านล่างจะได้ C
```

---

## 13. Development Notes

Suggested improvements for future versions:

1. Add save/load grading configuration.
2. Add editable grade scheme presets.
3. Add Excel export with multiple sheets.
4. Add PDF report generation for faculty meetings.
5. Add automatic comparison with historical grading distributions.
6. Add unit tests for Excel-like grading logic.
7. Add bilingual UI toggle.
8. Add instructor notes for each grading scheme.
9. Add GitHub Actions for basic testing.
10. Add sample datasets.

---

## 14. Limitations

- K-means is a clustering tool, not a grading policy.
- Quantile-range grading depends strongly on cell width and selected grade bands.
- GEOMEAN requires positive scores.
- Excel-like behavior is approximated carefully, but users should verify with known grading sheets when institutional rules are strict.
- Final grades should always be reviewed by instructors.

---

## 15. License

Choose one license depending on how you want to share the project.

Suggested options:

### MIT License

Recommended if you want others to freely use, modify, and build on the app.

### Private / Internal Use

Recommended if the app is only for departmental use.

---

## Citation / Acknowledgment

If used in an academic or teaching context, please cite or acknowledge:

```text
ArchGrade Studio: A Streamlit-based visual grading review tool for faculty assessment and grade distribution analysis.
```

---

## Short Project Description

ArchGrade Studio is a bilingual Streamlit app for visual grading review. It supports Z-score grading, range-based grading, Excel-like quantile-range grading, K-means clustering, outlier-aware calculation, visual distribution review, and final grade export.
