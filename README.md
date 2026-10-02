# LuemPreanParSar (ลืมเปลี่ยนภาษา)

ตรวจจับและแก้ข้อความที่พิมพ์ผิดแป้น (เกษมณี ↔ US QWERTY) บน Linux เช่น `l;ylfu` → `สวัสดี`, `้ำสสน` → `hello`

ดูแผนงานทั้งหมดที่ [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## เริ่มต้น

```fish
uv sync
uv run pytest
uv run python scripts/verify_layout.py
```
# LuemPreanParSar
