# เตรียม PC Windows สำหรับฝึก AI

เอกสารนี้เป็นขั้นตอนเตรียม PC (Windows + การ์ดจอ RTX 3070) ให้พร้อมฝึก CNN ซึ่งเป็น AI ตัวหลักของโปรเจกต์ ทำตามทีละข้อได้เลย ถ้าเจอคำที่ไม่รู้จัก ดูได้ที่ [อภิธานศัพท์](GLOSSARY.md)

**ทำไมต้องใช้ PC:** การ์ดจอคำนวณหลายอย่างพร้อมกันได้ จึงฝึก AI ได้เร็วกว่า notebook หลายสิบเท่า ฝึกเสร็จแล้วจะได้ไฟล์ AI (ONNX) ที่นำกลับไปใช้บน notebook ได้

**เวลาที่ใช้:** ประมาณ 30–45 นาที (ส่วนใหญ่เป็นเวลาดาวน์โหลด)

---

## ภาพรวม

```
Notebook                         PC Windows
────────                         ──────────
0. ส่งโค้ดขึ้น GitHub   ───────►  4. ดึงโค้ดลงมา
   คัดลอกข้อมูลลง USB  ───────►  5. วางข้อมูล
                                 1–3. ติดตั้งโปรแกรม
                                 6–7. ติดตั้งเครื่องมือ + ตรวจว่าใช้การ์ดจอได้
                                 8. ฝึก AI (วันที่ 4)
   ใช้ไฟล์ AI ต่อ      ◄───────  9. ส่งไฟล์ AI กลับ
```

**Checklist**
- [ ] 0. เตรียมของจาก notebook
- [ ] 1. ตรวจการ์ดจอและไดรเวอร์
- [ ] 2. ติดตั้ง Git
- [ ] 3. ติดตั้ง uv
- [ ] 4. ดึงโค้ดจาก GitHub
- [ ] 5. วางไฟล์ข้อมูลฝึก
- [ ] 6. ติดตั้งเครื่องมือของโปรเจกต์
- [ ] 7. ตรวจว่าใช้การ์ดจอได้จริง

---

## 0. เตรียมของจาก notebook (ทำบน notebook)

**0.1 ส่งโค้ดล่าสุดขึ้น GitHub**
```fish
cd ~/work/psu/LuemPreanParSar
git push
```

**0.2 คัดลอกไฟล์ข้อมูลฝึกลง USB หรือ Google Drive**

ไฟล์ข้อมูลไม่ได้อยู่บน GitHub เพราะใหญ่เกินไป (249 MB) ต้องคัดลอกเอง ไฟล์ที่ต้องใช้อยู่ในโฟลเดอร์ `data/processed/`

| ไฟล์ | ขนาด |
|---|---|
| `train.jsonl` | 199 MB |
| `val.jsonl` | 25 MB |
| `test.jsonl` | 25 MB |
| `stats.json` | 9 KB |

ถ้าจะบีบอัดก่อนเพื่อให้เล็กลงและคัดลอกเร็วขึ้น ใช้คำสั่งนี้ จะได้ไฟล์ประมาณ 60 MB
```fish
tar -czf ~/luem_data.tar.gz -C data processed
```

---

## 1. ตรวจการ์ดจอและไดรเวอร์

เปิด **PowerShell** (กดปุ่ม Windows แล้วพิมพ์ `powershell`) จากนั้นพิมพ์
```powershell
nvidia-smi
```

**สิ่งที่ต้องเห็น:** ชื่อ `NVIDIA GeForce RTX 3070` และมุมขวาบนต้องมี `Driver Version` **ตั้งแต่ 570 ขึ้นไป**

**ถ้าไม่ขึ้น หรือเลขต่ำกว่า 570:** ติดตั้งไดรเวอร์ใหม่ผ่านโปรแกรม NVIDIA App หรือดาวน์โหลดจาก https://www.nvidia.com/Download/index.aspx แล้วรีสตาร์ทเครื่อง

> **ทำไมต้อง 570:** โปรเจกต์ใช้ PyTorch รุ่นที่สร้างกับ CUDA 12.8 ซึ่งต้องใช้ไดรเวอร์รุ่นนี้ขึ้นไป

---

## 2. ติดตั้ง Git

Git คือโปรแกรมดึงโค้ดจาก GitHub พิมพ์ใน PowerShell
```powershell
winget install --id Git.Git -e
```

ติดตั้งเสร็จแล้ว **ปิด PowerShell แล้วเปิดใหม่** จากนั้นตั้งชื่อผู้ใช้ (ใช้ชื่อและอีเมลเดียวกับบน notebook)
```powershell
git config --global user.name "ชื่อของคุณ"
git config --global user.email "อีเมลของคุณ"
```

---

## 3. ติดตั้ง uv

uv คือโปรแกรมติดตั้ง Python และเครื่องมือทั้งหมดของโปรเจกต์ ไม่ต้องติดตั้ง Python เอง
```powershell
winget install --id astral-sh.uv -e
```

**ปิด PowerShell แล้วเปิดใหม่** จากนั้นตรวจว่าใช้ได้
```powershell
uv --version
```

---

## 4. ดึงโค้ดจาก GitHub

```powershell
mkdir $HOME\work -Force
cd $HOME\work
git clone https://github.com/Mhapong/LuemPreanParSar.git
cd LuemPreanParSar
```

ถ้า repo เป็นแบบส่วนตัว (private) จะมีหน้าต่างให้ login GitHub ผ่านเบราว์เซอร์ ให้ login ตามขั้นตอน

> ใช้ลิงก์ `https://` แทน `git@github.com:` ที่ใช้บน notebook เพราะไม่ต้องตั้ง SSH key ใหม่บน PC

---

## 5. วางไฟล์ข้อมูลฝึก

คัดลอกไฟล์จาก USB ไปไว้ที่ `LuemPreanParSar\data\processed\` ให้ได้แบบนี้
```
LuemPreanParSar\
└── data\
    └── processed\
        ├── train.jsonl
        ├── val.jsonl
        ├── test.jsonl
        └── stats.json
```

ถ้าคัดลอกมาเป็นไฟล์บีบอัด `luem_data.tar.gz` ให้แตกไฟล์ด้วยคำสั่ง
```powershell
tar -xzf D:\luem_data.tar.gz -C data
```
(เปลี่ยน `D:\` เป็นตำแหน่งของ USB)

ตรวจว่าไฟล์ครบ
```powershell
dir data\processed
```

<details>
<summary>ทางเลือก: สร้างข้อมูลใหม่บน PC แทนการคัดลอก</summary>

ใช้เวลาประมาณ 5 นาที แต่ข้อมูลอาจไม่ตรงกับบน notebook ทุกตัว ผลการวัดจึงอาจต่างกันเล็กน้อย **แนะนำให้คัดลอก**

```powershell
uv sync --extra data
uv run python scripts/download_corpus.py
uv run python scripts/build_dataset.py
```

</details>

---

## 6. ติดตั้งเครื่องมือของโปรเจกต์

```powershell
uv sync --extra ml
```

คำสั่งนี้ทำทุกอย่างให้เอง ได้แก่ ติดตั้ง Python 3.14, PyTorch **รุ่นที่ใช้การ์ดจอ** และเครื่องมือฝึก AI อื่นๆ

**ใช้เวลา:** 5–15 นาที เพราะ PyTorch รุ่นการ์ดจอมีขนาดประมาณ 3 GB

> **ทำไมไม่ต้องเลือกรุ่นเอง:** โปรเจกต์ตั้งค่าไว้แล้วว่าบน Windows ให้ใช้ PyTorch รุ่นการ์ดจอ (CUDA) ส่วนบน notebook Linux ให้ใช้รุ่นเล็กที่ไม่ใช้การ์ดจอ

**แนะนำ:** ตั้งให้ Python อ่านเขียนภาษาไทยแบบ UTF-8 เสมอ ไม่อย่างนั้น Windows อาจแสดงภาษาไทยเพี้ยน
```powershell
setx PYTHONUTF8 1
```
แล้วปิด PowerShell เปิดใหม่อีกครั้ง

---

## 7. ตรวจว่าใช้การ์ดจอได้จริง

**7.1 ตรวจว่า PyTorch เห็นการ์ดจอ**
```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**ต้องได้ผลแบบนี้**
```
2.11.0+cu128 True NVIDIA GeForce RTX 3070
```

| ถ้าได้ | หมายความว่า | แก้อย่างไร |
|---|---|---|
| `True` + ชื่อการ์ดจอ | ✅ พร้อมแล้ว | — |
| `False` | PyTorch มองไม่เห็นการ์ดจอ | ดูข้อ 1 ว่าไดรเวอร์ ≥ 570 แล้วรัน `uv sync --extra ml --reinstall-package torch` |
| ชื่อรุ่นลงท้าย `+cpu` หรือไม่มี `+cu128` | ได้ PyTorch รุ่นไม่ใช้การ์ดจอ | รัน `uv sync --extra ml --reinstall-package torch` |

**7.2 ตรวจว่าโค้ดทำงานถูกต้องบน Windows**
```powershell
uv run pytest
```

ต้องได้ประมาณ `32 passed, 1 skipped` (ข้อที่ถูกข้ามเป็นการตรวจคีย์บอร์ดของ Linux ซึ่งไม่มีบน Windows ข้ามได้ตามปกติ)

---

## 8. ขั้นต่อไป: ฝึก CNN (วันที่ 4)

เมื่อผ่านข้อ 7 แล้ว PC พร้อมฝึก AI ตอนนี้ยังไม่มีโค้ด CNN ซึ่งเป็นงานวันที่ 4 ที่ต้องเขียนต่อ

**ถ้าใช้ Claude Code บน PC:** เปิด Claude Code ในโฟลเดอร์ `LuemPreanParSar` แล้วบอกว่า "เริ่มวันที่ 4" Claude จะอ่าน `CLAUDE.md` และเอกสารใน `docs/` เพื่อรู้บริบทของโปรเจกต์เอง

> **ข้อควรรู้:** ความจำของ Claude (memory) อยู่แยกในแต่ละเครื่อง ไม่ได้ติดมากับโค้ด สิ่งที่สำคัญจึงเขียนไว้ใน `CLAUDE.md` แล้ว เช่น ให้ตอบเป็นภาษาไทย และให้เขียนเอกสารแบบอ่านง่าย

**งานของวันที่ 4** (รายละเอียดใน [แผนการทำงาน](IMPLEMENTATION_PLAN.md))
1. เขียน CNN (`src/luem/models/cnn.py`) และสคริปต์ฝึก (`scripts/train_cnn.py`)
2. ลองฝึกกับข้อมูลชุดเล็กก่อน เพื่อตรวจว่าโค้ดถูก
3. ฝึกเต็ม แล้วหาค่าตั้งที่ดีที่สุด 2–3 รอบ
4. แปลงเป็นไฟล์ ONNX และตรวจว่าได้ผลเหมือนก่อนแปลง
5. วัดผลด้วย `eval/evaluate.py` แล้วเทียบกับวิธีสถิติกลุ่มตัวอักษร (ต้องลดการถูกตัวอย่างหลอกหลอกจาก 8.2% ให้ได้)

**ระหว่างฝึก อย่าลืมถ่ายรูปหลักฐาน** (รูปที่ 15 ใน [รายการรูป](FIGURES.md)): เปิด PowerShell อีกหน้าต่าง พิมพ์ `nvidia-smi` แล้วกด `Win + Shift + S` เพื่อถ่ายภาพหน้าจอ

---

## 9. ส่งไฟล์ AI กลับไปที่ notebook

ฝึกเสร็จแล้วจะได้ไฟล์ AI ในโฟลเดอร์ `models\` เช่น `models\cnn.onnx` ซึ่งมีขนาดไม่ถึง 1 MB ส่งกลับได้ 2 วิธี

| วิธี | ทำอย่างไร |
|---|---|
| **ผ่าน GitHub (แนะนำ)** | ไฟล์เล็ก จึงตั้งให้ git รับไฟล์ `.onnx` ได้ (จะตั้งตอนเขียนโค้ดวันที่ 4) แล้ว `git push` บน PC และ `git pull` บน notebook |
| ผ่าน USB | คัดลอก `models\cnn.onnx` ไปวางที่ `models/` บน notebook |

---

## แก้ปัญหาที่อาจเจอ

| ปัญหา | สาเหตุ | แก้อย่างไร |
|---|---|---|
| `winget` ไม่รู้จัก | Windows รุ่นเก่า | ติดตั้ง "App Installer" จาก Microsoft Store หรือดาวน์โหลด Git และ uv จากเว็บโดยตรง |
| `uv` หรือ `git` ไม่รู้จัก หลังติดตั้งแล้ว | PowerShell ยังไม่รู้ว่ามีโปรแกรมใหม่ | ปิด PowerShell แล้วเปิดใหม่ |
| ภาษาไทยแสดงเป็นตัวแปลกๆ หรือ error `UnicodeEncodeError` | Windows ไม่ได้ใช้ UTF-8 เป็นค่าเริ่มต้น | `setx PYTHONUTF8 1` แล้วเปิด PowerShell ใหม่ (ข้อ 6) และใช้โปรแกรม Windows Terminal |
| `torch.cuda.is_available()` ได้ `False` | ไดรเวอร์เก่า หรือได้ PyTorch รุ่นไม่ใช้การ์ดจอ | ดูข้อ 7.1 |
| ฝึกแล้ว error `CUDA out of memory` | ใช้หน่วยความจำการ์ดจอเกิน 8 GB | ลดขนาด batch ในคำสั่งฝึก (สคริปต์ฝึกจะมีตัวเลือก `--batch-size`) |
| ฝึกแล้วค้างหรือ error เรื่อง `DataLoader` / `spawn` | Windows เปิดโปรเซสย่อยต่างจาก Linux | สคริปต์ฝึกจะเขียนให้รองรับไว้ ถ้ายังเจอให้ใช้ตัวเลือก `--workers 0` |
| `git clone` ขอรหัสผ่านแล้วไม่ผ่าน | GitHub ไม่รับรหัสผ่านแบบเดิมแล้ว | login ผ่านหน้าต่างเบราว์เซอร์ที่ขึ้นมา (Git Credential Manager) |

<details>
<summary>ทางเลือก: ใช้ WSL2 (Linux ใน Windows) แทน Windows ตรงๆ</summary>

WSL2 ทำให้ PC ทำงานเหมือน Linux จึงใช้คำสั่งชุดเดียวกับ notebook ได้ และใช้การ์ดจอได้ แต่ต้องแก้ `pyproject.toml` ด้วย เพราะตอนนี้ตั้งไว้ว่าเครื่อง Linux ให้ใช้ PyTorch รุ่นไม่ใช้การ์ดจอ (สำหรับ notebook) ถ้าจะใช้ WSL2 ต้องเปลี่ยนให้ Linux บน PC ใช้รุ่น `cu128`

สำหรับโปรเจกต์นี้ **แนะนำ Windows ตรงๆ** เพราะตั้งค่าไว้ให้แล้ว และไม่ต้องติดตั้ง WSL2 เพิ่ม

</details>
