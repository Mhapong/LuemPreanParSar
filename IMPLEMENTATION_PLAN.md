# Implementation Plan: LuemPreanParSar (ลืมเปลี่ยนภาษา)

ระบบตรวจจับและแก้ข้อความที่พิมพ์ผิดแป้น (เกษมณี ↔ US QWERTY) บน Linux

- **กำหนดส่ง:** 8 ต.ค. 2026
- **เริ่ม:** 1 ต.ค. 2026 (มีเวลา 7 วัน)
- **บันทึกงานที่ทำแล้วและเหตุผล:** [WORKLOG.md](WORKLOG.md)
- **Flow กระบวนการ (diagram):** [FLOW.md](FLOW.md)

---

## 1. เป้าหมายและขอบเขต

### สิ่งที่ต้องส่ง (Must have)

1. **Dataset generator** ที่สร้างข้อมูลพิมพ์ผิดแป้นพร้อม label อัตโนมัติจาก corpus ไทยและอังกฤษ
2. **โมเดล 3 ระดับ** สำหรับเปรียบเทียบกัน
   - Dictionary lookup (baseline)
   - Character n-gram language model
   - Neural ขนาดเล็ก (char-CNN) export เป็น ONNX
3. **การวัดผล:** precision/recall, false positive rate บน hard negatives, จำนวนตัวอักษรที่ต้องใช้ก่อนตรวจจับได้, latency ต่อ keystroke
4. **Demo บน Wayland** (Linux Mint Cinnamon, evdev + uinput ซึ่งใช้กับ X11 ได้ด้วย) ที่ดักคีย์จริง ลบข้อความ พิมพ์ใหม่ และสลับ layout ให้
5. **รายงานและสไลด์** พร้อมวิดีโอ demo

### นอกขอบเขต (ทำต่อหลังส่งงาน)

- Fcitx5/IBus addon (วิธีที่ถูกต้องที่สุดบน Wayland และไม่ต้องใช้สิทธิ์ `input` group)
- Layout อื่น เช่น ปัตตะโชติ หรือ Dvorak
- การแก้ข้อความที่พิมพ์ภาษาปนกันในคำเดียว

---

## 2. การนิยามปัญหา

**ข้อสังเกตสำคัญ:** ตัวอักษรที่พิมพ์ออกมาบอกได้อยู่แล้วว่าตอนนี้ layout ไหนทำงานอยู่ (อักษรไทยหมายถึง layout ไทย อักษรละตินหมายถึง layout US) ระบบจึงต้องตอบคำถามเดียวคือ

> **ผู้ใช้ตั้งใจพิมพ์ด้วย layout ที่กำลังทำงานอยู่หรือไม่?**

ปัญหานี้จึงเป็น **binary classification**

| ข้อความที่พิมพ์ | Layout ที่ทำงานอยู่ | ตั้งใจพิมพ์ | Label |
|---|---|---|---|
| `สวัสดี` | th | ไทย | `ok` |
| `hello` | en | อังกฤษ | `ok` |
| `l;ylfu` | en | ไทย | `wrong` |
| `้ำสสน` | th | อังกฤษ | `wrong` |

ทั้งนี้เทียบเท่ากับ 4 คลาสที่คุยกันไว้ก่อนหน้า แต่ทำ metric และ threshold ได้ง่ายกว่า

**Input ของโมเดล:** ข้อความที่พิมพ์ตั้งแต่ขอบเขตล่าสุด ได้แก่ space, Enter, การคลิก หรือการเปลี่ยนหน้าต่าง โดยจำกัดไม่เกิน 32 ตัวอักษร
**Output:** `P(wrong | typed_text)`
**การตัดสินใจ:** แก้ข้อความเมื่อ `P ≥ τ` และพิมพ์ไปแล้วอย่างน้อย `k_min` ตัว ทั้ง τ และ k_min ปรับจาก validation set โดยตั้งเป้า precision ≥ 99%

---

## 3. Dataset

### 3.1 แหล่งข้อมูล

| ชนิด | แหล่ง | หมายเหตุ |
|---|---|---|
| ไทยทั่วไป | `wikimedia/wikipedia` (`20231101.th`) จาก HuggingFace | ภาษาเขียนทางการ |
| ไทยภาษาพูด | `wisesight_sentiment` จาก HuggingFace | ภาษาแชทและโซเชียล ใกล้เคียงสิ่งที่พิมพ์จริง |
| อังกฤษทั่วไป | `wikitext-103` หรือ Wikipedia EN (สุ่มบางส่วน) | |
| สายโปรแกรมเมอร์ | คำสั่ง shell, Python keywords, ชื่อ package, ชื่อตัวแปรที่สุ่มสร้าง | **ห้ามใช้ shell history ส่วนตัวใน dataset ที่ commit** เพราะอาจมีรหัสผ่านหรือ token หลุดไปด้วย |

### 3.2 การสร้างตัวอย่าง

1. ทำความสะอาด corpus: normalize Unicode (NFC), ตัดอักขระที่ไม่อยู่บนแป้นพิมพ์ทิ้ง, แบ่งเป็นช่วงสั้นๆ ตาม space
2. แต่ละช่วงสร้างตัวอย่าง 2 แบบ
   - **`ok`:** ข้อความเดิม
   - **`wrong`:** แปลงด้วย `th_to_en` หรือ `en_to_th` จาก [src/luem/layout.py](src/luem/layout.py)
3. **ตัดเป็น prefix** ความยาว 1..n เพื่อให้วัด "ต้องใช้กี่ตัวอักษร" ได้โดยตรง
4. **Hard negatives (label `ok`)** เพื่อกด false positive:
   - รหัสผ่านสุ่ม เช่น `xK9!q2`, hash, UUID
   - URL, path (`/usr/share/...`), email
   - ชื่อตัวแปรแบบ `snake_case`/`camelCase`, คำย่อ (`ok`, `lol`, `btw`, `pls`)
   - ตัวเลขและสัญลักษณ์ล้วน
5. แบ่ง train/val/test **ตามเอกสารต้นทาง** ไม่ใช่สุ่มตามบรรทัด เพื่อไม่ให้ข้อความเดียวกันไปอยู่ทั้งสองฝั่ง
6. กำหนด random seed ตายตัว เพื่อให้รันซ้ำบน PC แล้วได้ข้อมูลชุดเดิม

### 3.3 รูปแบบไฟล์

`data/processed/{train,val,test}.jsonl`

```json
{"text": "l;ylfu", "active": "en", "label": "wrong", "intended": "สวัสดี", "source": "wisesight", "kind": "normal"}
```

`kind` มีค่า `normal` หรือ `hard_negative` เพื่อแยกรายงานผล

**ขนาดเป้าหมาย:** ประมาณ 1–2 ล้านตัวอย่าง (ก่อนตัด prefix) ซึ่งเพียงพอสำหรับโมเดลขนาดนี้

---

## 4. โมเดล

| # | โมเดล | วิธีการ | จุดเด่น/จุดอ่อนที่คาดไว้ |
|---|---|---|---|
| 1 | **Dictionary** | แปลงข้อความเป็นอีก layout แล้วค้นใน word list ทั้งสองภาษา ภาษาไทยใช้ `pythainlp` ตัดคำก่อน | ทำง่าย แต่แพ้คำนอกพจนานุกรมและ prefix ที่ยังพิมพ์ไม่จบคำ |
| 2 | **Char n-gram LM** | เทรน LM ระดับตัวอักษร (n=5, Kneser-Ney หรือ stupid backoff) แยกภาษาละตัว แล้วเปรียบเทียบ `log P_th(x_as_th) − log P_en(x_as_en)` | เร็วมาก ไม่ต้องใช้ GPU เป็นคู่เทียบที่แข็งแรง |
| 3 | **Char-CNN** | Embedding (ประมาณ 200 ตัวอักษร × 32 มิติ) → Conv1D หลาย kernel → max-pool → linear → sigmoid ขนาดรวมต่ำกว่า 200k พารามิเตอร์ | เรียนรู้ pattern จาก hard negatives ได้ export ONNX แล้วรันบน CPU ได้เร็ว |
| 3b | GRU (ถ้ามีเวลา) | ประมวลผลทีละตัวอักษร เก็บ state ไว้ระหว่าง keystroke | รองรับ streaming โดยธรรมชาติ |

### Hardware

- **Notebook (CPU onboard):** เขียนโค้ด, สร้าง dataset, เทรน n-gram, รัน demo และ **วัด latency** (เพราะเป็นเครื่องที่ใช้งานจริง)
- **PC Windows (Ryzen 5 9600X, RTX 3070 8 GB, RAM 32 GB):** เทรนโมเดล neural
  - ย้ายโค้ดผ่าน git ส่วน dataset ให้สร้างใหม่ด้วย script และ seed เดิม หรือคัดลอกไฟล์ `.jsonl` ไปตรงๆ
  - ย้าย `models/*.onnx` กลับมาที่ notebook เพราะ ONNX ไม่ขึ้นกับ OS
  - VRAM 8 GB เหลือเฟือสำหรับโมเดลขนาดนี้ ข้อจำกัดอยู่ที่ data loading ไม่ใช่ GPU

#### ข้อควรระวังให้โค้ดรันได้ทั้ง Windows และ Linux

- **ทุก `open()` ต้องใส่ `encoding="utf-8"`** เพราะ Windows ที่ตั้ง locale ไทยจะใช้ cp874 เป็นค่าเริ่มต้น และจะอ่านอักขระที่นอกเหนือ cp874 (เช่น emoji) ไม่ได้
- ใช้ `pathlib.Path` แทนการต่อ path ด้วย string
- โค้ดเทรนต้องอยู่ใต้ `if __name__ == "__main__":` เพราะ DataLoader บน Windows ใช้ spawn เมื่อ `num_workers > 0`
- ใส่ `*.jsonl -text` ใน `.gitattributes` เพื่อกัน git แปลง line ending ของไฟล์ข้อมูล
- ติดตั้ง PyTorch แบบ CUDA ด้วย `uv pip install torch --index-url https://download.pytorch.org/whl/cu128` (ตรวจ index ล่าสุดที่ pytorch.org)
- ทางเลือก: ใช้ **WSL2** (Ubuntu) บน PC ซึ่งรองรับ CUDA และได้สภาพแวดล้อมแบบเดียวกับ notebook

---

## 5. การวัดผล

| Metric | นิยาม | เหตุผล |
|---|---|---|
| Precision / Recall / F1 | คิดบน class `wrong` ที่ τ ที่เลือก | ภาพรวม |
| **FPR บน hard negatives** | สัดส่วน hard negative ที่ถูกแก้ผิด | การแก้ผิดคือสิ่งที่ผู้ใช้เกลียดที่สุด |
| **Chars-to-detect** | ความยาว prefix สั้นสุดที่ `P ≥ τ` และคงอยู่จนจบคำ แสดงเป็นกราฟ CDF แยกตามโมเดล | metric หลักของงาน |
| Precision–recall curve | กวาดค่า τ | ใช้เลือก τ |
| Latency | p50/p99 ต่อ keystroke บน notebook ใช้ CPU 1 thread | เป้าหมาย < 5 ms |
| ขนาดโมเดล | ขนาดไฟล์ (KB/MB) | ความเหมาะกับการใช้งานจริง |

**Error analysis:** สุ่มตัวอย่างที่ผิดมา 50 ตัวอย่าง แล้วจัดกลุ่มสาเหตุ เช่น คำสั้น คำย่อ ชื่อเฉพาะ และศัพท์เทคนิค

---

## 6. Demo บน Linux (Wayland ผ่าน evdev + uinput)

**ข้อมูลเครื่อง:** Linux Mint 22.3, Cinnamon 6.6.9 รองรับทั้ง session `cinnamon-wayland` และ X11, layout `us,th,us`, มี IBus ทำงานอยู่

**เหตุผลที่ไม่ใช้ X RECORD/XTest:** บน Wayland โปรแกรมดักหรือส่งคีย์ข้ามหน้าต่างผ่าน display server ไม่ได้ จึงต้องทำงานในระดับ kernel แทน วิธีนี้ใช้ได้ทั้ง Wayland และ X11 ถ้า Cinnamon Wayland (ยังเป็น experimental) มีปัญหาตอน demo ให้ login เป็น X11 แล้วรันโค้ดเดิมได้เลย

```
/dev/input/event*  (evdev: อ่าน keycode ดิบจากคีย์บอร์ด)
   → buffer ของ (keycode, shift) ตั้งแต่ขอบเขตล่าสุด
   → แปลงเป็นข้อความตาม layout ที่ทำงานอยู่ (keycode → US char → en_to_th ถ้าเป็นไทย)
   → model.predict(text) → P(wrong)
   → ถ้า P ≥ τ:
        1. uinput: กด BackSpace × len(buffer)
        2. สลับ layout (en ↔ th)
        3. uinput: กด keycode ชุดเดิมซ้ำ
```

**ข้อดี:** evdev ให้ keycode ดิบมาอยู่แล้ว หลังสลับ layout ก็ส่ง keycode ชุดเดิมซ้ำได้ทันที ไม่ต้องจัดการ Unicode input
**ข้อเสีย:** evdev ไม่รู้ว่า layout ไหนทำงานอยู่ ต้องอ่านหรือติดตามสถานะเอง (ดูด้านล่าง)

**สิทธิ์ที่ต้องตั้ง** (ทำครั้งเดียว):
```fish
sudo usermod -aG input $USER              # อ่าน /dev/input/event* (ต้อง logout/login)
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo udevadm control --reload-rules; and sudo udevadm trigger
```
ทางเลือกคือรัน daemon ด้วย `sudo` ตอน demo ระบุในรายงานด้วยว่ากลุ่ม `input` มีสิทธิ์อ่านทุก keystroke ซึ่งเป็นข้อแลกเปลี่ยนด้านความปลอดภัยของวิธีนี้

**การรู้และสลับ layout:** ผล spike วันที่ 1 ต.ค. บน X11 (`demo/spike_evdev.py`)

| ทดสอบ | ผล |
|---|---|
| อ่านคีย์จาก evdev และแปลงเป็นตัวอักษรทั้งสอง layout (รวม Shift) | ✅ |
| ส่ง `hello` + BackSpace × 2 ผ่าน uinput | ✅ ได้ `hel` |
| `gsettings get ... current` อ่าน layout | ❌ ไม่เปลี่ยนตามเมื่อกด Super+Space |
| `gsettings set ... current` สลับ layout | ❌ ค่าใน gsettings เปลี่ยน แต่ layout จริงไม่เปลี่ยน |
| ส่ง Super+Space ผ่าน uinput | ✅ สลับ layout จริงได้ทั้งไปและกลับ |

**ข้อสรุป:** สลับ layout ด้วยการส่ง Super+Space ผ่าน uinput และ **ติดตามสถานะเอง** โดยนับ Super+Space ที่เห็นจาก evdev รวมกับครั้งที่ daemon ส่งเอง
- ตอนเริ่มทำงาน daemon จะสมมติว่าเป็น `en` และมี hotkey สำหรับ resync
- ถ้าผู้ใช้สลับ layout ด้วยการคลิก applet บน panel สถานะจะเพี้ยน ให้ระบุเป็นข้อจำกัดในรายงาน
- ยังต้องทดสอบซ้ำบน Wayland 1 ครั้งก่อนวันที่ 6

**ความปลอดภัยและ UX:**
- ปุ่ม hotkey เปิด/ปิดระบบ และโหมด "แก้เมื่อกดปุ่มเท่านั้น" (เช่น Pause) สำหรับใช้คู่กับโหมดอัตโนมัติ
- ไม่มี blacklist ตามหน้าต่าง เพราะ Wayland ไม่บอกว่าหน้าต่างไหน focus อยู่ จึงใช้ hotkey เปิด/ปิดแทน
- ล้าง buffer เมื่อคลิกเมาส์ (อ่าน event ของเมาส์จาก evdev ด้วย) หรือกดปุ่มลูกศร/Ctrl/Alt/Super/Tab
- กรองไม่ให้ daemon อ่าน event จากอุปกรณ์ uinput ของตัวเอง เพื่อไม่ให้วนลูป
- **ไม่บันทึก keystroke ลงไฟล์เด็ดขาด**

---

## 7. โครงสร้างโปรเจกต์

```
LuemPreanParSar/
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml          # uv, optional deps แยกตามเฟส
├── src/luem/
│   ├── layout.py           # ✅ mapping เกษมณี ↔ QWERTY (ตรวจกับ xkb แล้ว)
│   ├── dataset.py          # สร้าง prefix, hard negatives, split
│   ├── models/
│   │   ├── dictionary.py
│   │   ├── ngram.py
│   │   └── cnn.py
│   └── predict.py          # interface กลาง: predict(text) -> float
├── scripts/
│   ├── verify_layout.py    # ✅ เทียบ layout.py กับ /usr/share/X11/xkb/symbols/th
│   ├── download_corpus.py
│   ├── build_dataset.py
│   ├── train_ngram.py
│   ├── train_cnn.py        # รันบน PC (GPU)
│   └── export_onnx.py
├── eval/
│   ├── evaluate.py         # metrics ทั้งหมดในข้อ 5
│   ├── latency.py
│   └── plots.py
├── demo/
│   ├── spike_evdev.py      # spike: อ่าน/ส่งคีย์ และสลับ layout
│   └── daemon.py           # evdev + uinput (Wayland/X11)
├── tests/
├── data/{raw,processed}/   # ไม่ commit
├── models/                 # ไม่ commit
└── report/
```

ทุกโมเดลต้องมี interface เดียวกัน (`predict(text) -> float`) เพื่อให้ `evaluate.py` และ demo สลับโมเดลได้โดยไม่ต้องแก้โค้ด

---

## 8. ตารางงานรายวัน

### วันที่ 1 (พุธ 1 ต.ค.): ตั้งโปรเจกต์ + mapping
- [x] ตั้งโปรเจกต์ (uv, git, โครงสร้างโฟลเดอร์)
- [x] `layout.py` ครบ 94 ปุ่ม (รวม Shift) และผ่าน `verify_layout.py`
- [x] Unit test สำหรับการแปลงไปกลับ
- [x] `download_corpus.py`: ดาวน์โหลด Wikipedia TH และ Wisesight
- [x] ตั้งสิทธิ์ `input` group และ udev rule สำหรับ uinput (ข้อ 6)
- [x] **Spike 30 นาที (X11):** อ่านคีย์จาก evdev, ส่งคีย์ผ่าน uinput และสลับ layout ด้วย Super+Space ได้ทั้งหมด ส่วน gsettings ใช้ไม่ได้ (ดูข้อ 6)

### วันที่ 2 (พฤหัส 2 ต.ค.): Dataset
- [ ] ดาวน์โหลด corpus อังกฤษและสร้างรายการศัพท์โปรแกรมเมอร์
- [ ] `dataset.py` + `build_dataset.py` (prefix, hard negatives, split ตามเอกสาร)
- [ ] ตรวจด้วยตาว่าข้อมูลสมเหตุสมผล และดูสถิติ (จำนวนต่อคลาส, ความยาว)
- [ ] Logout แล้วเข้า "Cinnamon on Wayland" จากนั้นรัน `demo/spike_evdev.py` ซ้ำ 1 ครั้ง (ใช้เวลาประมาณ 10 นาที)

### วันที่ 3 (ศุกร์ 3 ต.ค.): Baselines + Evaluation framework
- [ ] `models/dictionary.py`
- [ ] `models/ngram.py` + `train_ngram.py`
- [ ] `eval/evaluate.py` เวอร์ชันแรก ให้ได้ precision/recall และ chars-to-detect
- [ ] **Checkpoint:** มีตัวเลขเปรียบเทียบ 2 โมเดลแล้ว

### วันที่ 4 (เสาร์ 4 ต.ค.): Neural (บน PC)
- [ ] `models/cnn.py` + `train_cnn.py` ทดสอบบน notebook ด้วยข้อมูลชุดเล็กก่อน
- [ ] ย้ายไปเทรนเต็มบน PC (RTX 3070) และปรับ hyperparameter 2–3 รอบ
- [ ] `export_onnx.py` แล้วตรวจว่าผลจาก ONNX ตรงกับผลจาก PyTorch
- [ ] (ถ้ามีเวลา) GRU

### วันที่ 5 (อาทิตย์ 5 ต.ค.): Evaluation เต็มรูป
- [ ] รันทุก metric ในข้อ 5 กับทุกโมเดล
- [ ] เลือก τ และ k_min จาก val แล้วรายงานผลบน test
- [ ] `latency.py` บน notebook
- [ ] กราฟ: PR curve, CDF ของ chars-to-detect, ตาราง latency/ขนาดโมเดล
- [ ] Error analysis

### วันที่ 6 (จันทร์ 6 ต.ค.): Demo
- [ ] `demo/daemon.py` ทำงานครบทั้งสาย
- [ ] ทดสอบใน xed, Firefox, terminal ทั้งบน Wayland และ X11
- [ ] hotkey เปิด/ปิด และโหมดแก้เมื่อกดปุ่ม

### วันที่ 7 (อังคาร 7 ต.ค.): รายงาน + สไลด์
- [ ] รายงาน: บทนำ, งานที่เกี่ยวข้อง (xneur, Punto Switcher), วิธีการ, ผลการทดลอง, ข้อจำกัด, งานในอนาคต
- [ ] สไลด์
- [ ] **อัดวิดีโอ demo** เผื่อ live demo มีปัญหา

### วันที่ 8 (พุธ 8 ต.ค.): ส่งงาน
- [ ] ตรวจทาน, แก้ bug สุดท้าย, เขียนวิธีรันใน README
- [ ] ส่ง

---

## 9. แผนสำรองเมื่อเวลาไม่พอ

ตัดตามลำดับนี้

1. ตัด GRU เหลือ CNN อย่างเดียว
2. ตัด blacklist/hotkey ใน demo
3. ตัด dictionary baseline เหลือ n-gram กับ CNN
4. Demo เปลี่ยนเป็น CLI ที่อ่านจาก stdin แล้วแสดงผลการแก้ แทนการดักคีย์จริง

**ห้ามตัด:** dataset generator, n-gram LM, กราฟ chars-to-detect และ FPR บน hard negatives

---

## 10. ความเสี่ยง

| ความเสี่ยง | ผลกระทบ | วิธีรับมือ |
|---|---|---|
| ดาวน์โหลด corpus ช้าหรือใหญ่เกินไป | ข้อมูลไม่ทันวันที่ 2 | ใช้แค่บางส่วน (เช่น 100k บทความแรก) ด้วย streaming mode ของ `datasets` |
| Cinnamon Wayland ยังเป็น experimental (แอปบางตัวพัง, สลับ layout ด้วยโค้ดไม่ได้) | Demo ไม่ทำงาน | evdev/uinput ใช้ได้กับ X11 ด้วย ถ้า Wayland มีปัญหาให้ demo บน X11 แล้วระบุในรายงาน |
| IBus ชนกับคีย์ที่ส่งผ่าน uinput | ข้อความพิมพ์ซ้ำผิด | ทำ spike ไว้ตั้งแต่วันที่ 1–2 ถ้าไม่ได้ ให้ปิด IBus ตอน demo |
| ตั้งค่า CUDA บน PC ไม่ผ่าน | เทรน neural ไม่ได้ | CNN ขนาดนี้เทรนบน CPU 16 threads ได้ภายในไม่กี่ชั่วโมง |
| False positive สูง | Demo น่ารำคาญ | เพิ่ม hard negatives, ปรับ τ, ใช้โหมดแก้เมื่อกดปุ่ม |
| ภาษาไทยไม่มี space | buffer ยาวและขอบเขตคำไม่ชัด | จำกัด buffer 32 ตัว และตัดสินใจจาก prefix ไม่ต้องรอจบคำ |

---

## 11. คำสั่งที่ใช้บ่อย

```fish
uv sync                                   # dev env
uv sync --extra data --extra ml           # เมื่อเริ่มเฟส dataset/training
uv run pytest                             # tests + doctests
uv run python scripts/verify_layout.py    # เทียบ mapping กับ xkb
```

บน PC ที่มี GPU ให้ติดตั้ง PyTorch แบบ CUDA ตาม https://pytorch.org/get-started/locally/
