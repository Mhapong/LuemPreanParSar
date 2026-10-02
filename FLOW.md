# Flow กระบวนการทำงาน

แผนรายวันอยู่ที่ [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) และเหตุผลของงานที่ทำแล้วอยู่ที่ [WORKLOG.md](WORKLOG.md)

> ดู diagram ใน VS Code: ติดตั้ง extension "Markdown Preview Mermaid Support" แล้วกด `Ctrl+Shift+V` (บน GitHub แสดงผลได้ทันที)

---

## 1. ภาพรวมทั้งโปรเจกต์

ลำดับงานตั้งแต่เริ่มจนส่ง แยกตามเครื่องที่ใช้ทำ สีเขียวคือเสร็จแล้ว

```mermaid
flowchart TD
    subgraph NB["💻 Notebook: Linux Mint"]
        A["ตั้งโปรเจกต์ + layout.py<br/>ตรวจกับ xkb แล้ว"]
        B["download_corpus.py<br/>Wikipedia TH/EN, Wisesight"]
        S["Spike: evdev + uinput<br/>อ่านคีย์ / ส่งคีย์ / สลับ layout"]
        C["build_dataset.py<br/>train / val / test .jsonl"]
        D1["Model 1: Dictionary"]
        D2["Model 2: Char n-gram LM"]
        G["evaluate.py + latency.py<br/>วัดทุกโมเดลด้วยชุดเดียวกัน"]
        H["เลือกโมเดล + ค่า τ, k_min"]
        I["daemon.py<br/>demo บน Wayland / X11"]
    end

    subgraph PC["🖥️ PC Windows: RTX 3070"]
        E["train_cnn.py<br/>Model 3: Char-CNN"]
        F["export_onnx.py"]
    end

    R["📄 รายงาน + สไลด์ + วิดีโอ demo"]

    A --> B --> C
    A --> S
    C --> D1 & D2
    C -- "คัดลอก .jsonl" --> E --> F
    F -- "คัดลอก .onnx" --> G
    D1 & D2 --> G
    G --> H --> I
    S -- "ยืนยันว่าวิธีนี้ใช้ได้" --> I
    G -- "กราฟ + ตาราง" --> R
    I -- "วิดีโอ" --> R

    classDef done fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
    class A,B,S done
```

**จุดที่ต้องระวัง**
- `build_dataset.py` เป็นทางผ่านของทุกอย่าง ถ้าทำช้า งานส่วนอื่นจะช้าตามทั้งหมด จึงอยู่ในวันที่ 2
- ส่วน PC กับ notebook ทำพร้อมกันได้: ระหว่างที่ CNN เทรนบน PC ให้ทำ n-gram และ `evaluate.py` บน notebook
- ทุกโมเดลต้องเข้า `evaluate.py` ผ่าน interface เดียวกัน `predict(text) -> float` ผลจึงเทียบกันได้อย่างยุติธรรม

---

## 2. การสร้าง Dataset (`build_dataset.py`)

```mermaid
flowchart TD
    R1[("data/raw/<br/>wiki_th, wiki_en, wisesight")]
    R1 --> N["ทำความสะอาด<br/>Unicode NFC, ตัดอักขระที่ไม่มีบนแป้นพิมพ์"]
    N --> SP["แบ่ง train / val / test ตาม document id<br/>⚠️ ต้องแบ่งก่อนสร้างตัวอย่าง"]
    SP --> SEG["ตัดเป็นช่วงสั้นตาม space<br/>ยาวไม่เกิน 32 ตัว"]

    SEG --> OK["ตัวอย่าง ok<br/>ข้อความเดิม เช่น สวัสดี"]
    SEG --> WR["ตัวอย่าง wrong<br/>th_to_en / en_to_th เช่น l;ylfu"]

    HN["Hard negatives (label ok)<br/>รหัสผ่าน, URL, path, git, ok, ตัวเลข"]

    OK & WR & HN --> PF["สร้าง prefix ความยาว 1..n<br/>l → l; → l;y → ..."]
    PF --> SH["Shuffle ด้วย seed ตายตัว"]
    SH --> OUT[("data/processed/<br/>train.jsonl, val.jsonl, test.jsonl")]
```

**ทำไมต้องแบ่ง train/test ก่อนสร้างตัวอย่าง:** ประโยคเดียวกันจะกลายเป็นหลายตัวอย่าง (ok, wrong และ prefix ทุกความยาว) ถ้าสร้างตัวอย่างก่อนแล้วค่อยสุ่มแบ่ง ตัวอย่างจากประโยคเดียวกันจะไปอยู่ทั้งใน train และ test ผล test จะดีเกินจริง (data leakage)

**ตัวอย่าง 1 บรรทัดในไฟล์ output**
```json
{"text": "l;ylfu", "active": "en", "label": "wrong", "intended": "สวัสดี", "source": "wisesight", "kind": "normal"}
```

---

## 3. การเทรนและวัดผล

```mermaid
flowchart TD
    TR[("train.jsonl")] --> M1["Dictionary<br/>ไม่ต้องเทรน ใช้ word list"]
    TR --> M2["n-gram LM<br/>นับความถี่บน notebook"]
    TR --> M3["Char-CNN<br/>เทรนบน PC → .onnx"]

    M1 & M2 & M3 --> P["predict(text) → P(wrong)"]

    VA[("val.jsonl")] --> P
    P --> PR["Precision–Recall curve"]
    PR --> TAU["เลือก τ และ k_min<br/>เป้าหมาย precision ≥ 99%"]

    TE[("test.jsonl<br/>ใช้ครั้งเดียวตอนท้าย")] --> FIN
    TAU --> FIN["วัดผลจริงบน test"]

    FIN --> O1["Precision / Recall / F1"]
    FIN --> O2["FPR บน hard negatives"]
    FIN --> O3["Chars-to-detect<br/>กราฟ CDF"]
    FIN --> O4["Latency p50 / p99<br/>วัดบน notebook"]
    FIN --> O5["Error analysis<br/>ดูตัวอย่างที่ผิด 50 ตัวอย่าง"]
```

**ทำไมต้องแยก val กับ test:** ใช้ val เลือกค่า τ ถ้าใช้ test เลือก τ แล้ววัดผลบน test ซ้ำ ก็เหมือนดูข้อสอบก่อนสอบ ตัวเลขจะดีเกินจริง

---

## 4. การทำงานของ Daemon ตอนใช้งานจริง (`daemon.py`)

### 4.1 เมื่อได้รับ event 1 ครั้ง

```mermaid
flowchart TD
    START(["เริ่มทำงาน<br/>สมมติ layout = en, buffer ว่าง<br/>เปิด /dev/input/event*, สร้าง uinput"]) --> EV

    EV["รอ event จาก evdev"] --> K{"เป็นปุ่มอะไร?"}

    K -- "Shift" --> SH["อัปเดตสถานะ Shift"] --> EV
    K -- "Super+Space<br/>(ผู้ใช้สลับเอง)" --> TG["สลับ tracked layout<br/>ล้าง buffer"] --> EV
    K -- "Hotkey เปิด/ปิด" --> EN["สลับ enabled"] --> EV
    K -- "Enter, Tab, ลูกศร,<br/>Ctrl/Alt+ปุ่ม, คลิกเมาส์" --> CL["ล้าง buffer"] --> EV
    K -- "Space" --> CL
    K -- "BackSpace" --> BS["ลบตัวท้ายออกจาก buffer"] --> EV
    K -- "ปุ่มตัวอักษร" --> AP["เพิ่ม (keycode, shift) ลง buffer"]

    AP --> CHK{"enabled และ<br/>ยังไม่เคยแก้คำนี้ และ<br/>len ≥ k_min ?"}
    CHK -- "ไม่" --> EV
    CHK -- "ใช่" --> TXT["text = key_to_char(buffer, tracked layout)"]
    TXT --> PRED["p = model.predict(text)"]
    PRED --> DEC{"p ≥ τ ?"}
    DEC -- "ไม่" --> EV
    DEC -- "ใช่" --> FIX[["แก้ข้อความ (ข้อ 4.2)"]] --> MARK["ทำเครื่องหมายว่าคำนี้แก้แล้ว<br/>กันแก้ซ้ำจนกว่าจะขึ้นคำใหม่"] --> EV
```

**จุดที่ต้องระวัง**
- **ทำเครื่องหมายว่าแก้แล้ว:** หลังแก้ ผู้ใช้จะพิมพ์คำเดิมต่อ ถ้าไม่ทำเครื่องหมาย daemon อาจตัดสินซ้ำแล้วสลับกลับ
- **คลิกเมาส์ต้องล้าง buffer:** ผู้ใช้อาจย้ายเคอร์เซอร์ไปที่อื่นแล้ว ถ้าส่ง BackSpace ตอนนั้นจะลบข้อความผิดที่
- **ไม่อ่าน event จากอุปกรณ์ uinput ของตัวเอง:** ไม่อย่างนั้นคีย์ที่ส่งออกไปจะวนกลับมาเป็น input อีกรอบ

### 4.2 ขั้นตอนแก้ข้อความ

ตัวอย่าง: ผู้ใช้ตั้งใจพิมพ์ "สวัสดี" แต่ layout เป็น en

```mermaid
sequenceDiagram
    actor U as ผู้ใช้
    participant K as คีย์บอร์ด (evdev)
    participant D as daemon.py
    participant M as Model
    participant V as uinput (คีย์บอร์ดเสมือน)
    participant A as แอป (xed)

    U->>K: กดปุ่ม l #59; y l
    K->>A: แอปแสดง "l#59;yl"
    K->>D: keycode 38, 39, 21, 38
    D->>M: predict("l#59;yl")
    M-->>D: p = 0.99 ≥ τ
    D->>V: BackSpace × 4
    V->>A: ลบ "l#59;yl"
    D->>V: Super+Space
    V->>A: layout → th
    Note over D: tracked layout = th
    D->>V: keycode 38, 39, 21, 38 (ชุดเดิม)
    V->>A: แอปแสดง "สวัส"
    U->>K: พิมพ์ต่อ f u
    K->>A: "สวัสดี" (layout เป็น th แล้ว)
```

**ข้อดีของวิธีนี้:** หลังสลับ layout แล้วส่ง keycode **ชุดเดิม** กลับไปได้เลย ไม่ต้องแปลงตัวอักษร ระบบแปลงให้เองตาม layout ใหม่ และผู้ใช้พิมพ์ต่อได้ทันทีเพราะ layout ถูกแล้ว

---

## 5. สถานะปัจจุบัน

| ขั้นตอน | สถานะ |
|---|---|
| ตั้งโปรเจกต์ + `layout.py` + tests | ✅ |
| `download_corpus.py` | ✅ |
| Spike evdev/uinput บน X11 | ✅ |
| Spike บน Wayland | ⏳ ต้อง logout แล้ว login เข้า session Wayland |
| `build_dataset.py` | ⬜ งานถัดไป |
| Models, evaluation, daemon, รายงาน | ⬜ |
