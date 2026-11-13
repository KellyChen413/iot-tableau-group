# -*- coding: utf-8 -*-
"""
IoT 組｜Tableau 圖表數據稽核腳本
用法：
  python iot_data_audit.py                         # 預設讀取 ../tim/作業.twbx 內打包的 CSV
  python iot_data_audit.py 資料.csv                # 指定 CSV
  python iot_data_audit.py 某檔案.twbx             # 指定 .twbx（自動讀取內含的 CSV）
需要：pandas（pip install pandas）
每一項檢查會印出 [PASS] 或 [FAIL]，以及實際算出的數字。
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT = Path(__file__).resolve().parent.parent / "tim" / "作業.twbx"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT


def load(src: Path) -> pd.DataFrame:
    """讀取 CSV；若為 .twbx，則讀取其中打包的 iot_sensor_logs CSV。"""
    if src.suffix.lower() == ".twbx":
        with zipfile.ZipFile(src) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv") and "iot_sensor_logs" in n)
            print(f"資料來源：{src.name} → {name}")
            return pd.read_csv(io.BytesIO(z.read(name)), encoding="utf-8-sig")
    print(f"資料來源：{src}")
    return pd.read_csv(src, encoding="utf-8-sig")


df = load(SRC)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour

results = []

def check(name, actual, expected, tol=0.0):
    """數字比對：tol 為允許誤差（百分比用 0.05，代表 ±0.05 個百分點）"""
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        ok = abs(actual - expected) <= tol
    else:
        ok = actual == expected
    tag = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"  [{tag}] {name}：實際 = {actual}｜預期 = {expected}")

pct = lambda part, whole: round(part / whole * 100, 2)

motor = df[df.device_type == "motor"]
pg = df[df.device_type == "pressure_gauge"]
ts = df[df.device_type == "temperature_sensor"]

# =====================================================================
print("\n=== 第 1 題：基礎資料量與缺值 ===")
# =====================================================================
check("總筆數", len(df), 92250)
check("設備台數", df.device_id.nunique(), 45)
n_dev = df.groupby("device_type").device_id.nunique()
check("馬達台數", int(n_dev["motor"]), 10)
check("壓力計台數", int(n_dev["pressure_gauge"]), 15)
check("溫度感測器台數", int(n_dev["temperature_sensor"]), 20)
per_dev = df.groupby("device_id").size()
check("每台都剛好 2,050 筆", bool((per_dev == 2050).all()), True)
check("pressure 缺值", int(df.pressure.isna().sum()), 61500)
check("pressure 有值的都是壓力計", set(df[df.pressure.notna()].device_type), {"pressure_gauge"})
check("vibration_level 缺值", int(df.vibration_level.isna().sum()), 41000)
check("vibration_level 缺值的都是溫度感測器", set(df[df.vibration_level.isna()].device_type), {"temperature_sensor"})
other_na = df.drop(columns=["pressure", "vibration_level"]).isna().sum().sum()
check("其他欄位缺值總數", int(other_na), 0)
check("log_id 無重複", bool(df.log_id.is_unique), True)
zone_types = df.groupby("factory_zone").device_type.nunique()
check("每個廠區只對應 1 種設備", bool((zone_types == 1).all()), True)
print("  廠區對應：", df.groupby("factory_zone").device_type.first().to_dict())

# =====================================================================
print("\n=== 第 2 題：面向一（異常比例）===")
# =====================================================================
vc = df.status.value_counts()
for s, n, p in [("normal", 87354, 94.7), ("offline", 1801, 2.0), ("warning", 1764, 1.9), ("error", 1331, 1.4)]:
    check(f"{s} 筆數", int(vc[s]), n)
    check(f"{s} 比例 %（四捨五入到 1 位）", round(vc[s] / len(df) * 100, 1), p)

df["is_abnormal"] = (df.status != "normal").astype(int)
ab_rate = df.groupby("device_type").is_abnormal.mean() * 100
check("馬達異常率 %", round(ab_rate["motor"], 2), 4.61, 0.005)
check("壓力計異常率 %", round(ab_rate["pressure_gauge"], 2), 5.50, 0.005)
check("溫度感測器異常率 %", round(ab_rate["temperature_sensor"], 2), 5.51, 0.005)
check("馬達 error 比例 %（分母＝馬達 20,500 筆）", pct((motor.status == "error").sum(), len(motor)), 1.88, 0.005)
check("馬達 warning 比例 %（分母＝馬達 20,500 筆）", pct((motor.status == "warning").sum(), len(motor)), 0.75, 0.005)

dev_ab = df.groupby("device_id").is_abnormal.sum().sort_values(ascending=False)
top2 = dev_ab.head(2)
check("前兩名設備", set(top2.index), {"TEM020", "PRE012"})
check("前兩名筆數皆為 132", bool((top2 == 132).all()), True)
check("45 台平均異常筆數（4,896 ÷ 45）", round(dev_ab.mean(), 0), 109)
print("  前 10 名：", dev_ab.head(10).to_dict())

# =====================================================================
print("\n=== 第 3 題：面向二（能耗）===")
# =====================================================================
pw = df.groupby("device_type").power_consumption.agg(["sum", "mean"])
share = pw["sum"] / pw["sum"].sum() * 100
check("馬達耗電占比 %", round(share["motor"], 1), 95.3, 0.05)
check("馬達平均每筆", round(pw.loc["motor", "mean"], 1), 27.5, 0.05)
check("馬達 ÷ 壓力計（倍）", round(pw.loc["motor", "mean"] / pw.loc["pressure_gauge", "mean"], 0), 55)
check("馬達 ÷ 溫度感測器（倍）", round(pw.loc["motor", "mean"] / pw.loc["temperature_sensor", "mean"], 0), 92)

dev_pw = df.groupby(["device_id", "device_type"]).power_consumption.sum().sort_values(ascending=False)
top10 = dev_pw.head(10)
check("前 10 名全是馬達", set(top10.index.get_level_values("device_type")), {"motor"})
check("第 1 名", top10.index[0][0], "MOT007")
check("第 1 名耗電", round(top10.iloc[0], 0), 56929, 1)
check("第 10 名", top10.index[-1][0], "MOT001")
check("第 10 名耗電", round(top10.iloc[-1], 0), 55565, 1)
check("第 1 名比第 10 名高出 %", pct(top10.iloc[0] - top10.iloc[-1], top10.iloc[-1]), 2.5, 0.05)

hr = motor.groupby("hour").power_consumption.mean()
check("最高時段", int(hr.idxmax()), 6)
check("最高平均耗電", round(hr.max(), 1), 28.3, 0.05)
check("最低時段", int(hr.idxmin()), 12)
check("最低平均耗電", round(hr.min(), 1), 26.3, 0.05)
check("最高比最低高出 %（以最低值為分母）", round((hr.max() - hr.min()) / hr.min() * 100, 1), 7.6, 0.1)

# =====================================================================
print("\n=== 第 4 題：面向三（感測值門檻）===")
# =====================================================================
oh = motor[motor.temperature > 80]
check("馬達過熱（> 80°C）筆數", len(oh), 408)
check("過熱占馬達 %", pct(len(oh), len(motor)), 2.0, 0.05)
check("> 100°C 筆數", int((motor.temperature > 100).sum()), 55)
check("馬達最高溫", round(motor.temperature.max(), 2), 109.06, 0.001)

oh_dev = oh.groupby("device_id").size().sort_values(ascending=False)
check("10 台馬達都有過熱", len(oh_dev), 10)
check("過熱最少筆數", int(oh_dev.min()), 31)
check("過熱最多筆數", int(oh_dev.max()), 48)
check("過熱最多的馬達", oh_dev.index[0], "MOT004")
check("資料天數", df.date.nunique(), 29)
check("過熱發生的天數（每天都有）", oh.date.nunique(), 29)

check("壓力計總筆數", len(pg), 30750)
hv = pg[pg.vibration_level >= 15]
check("壓力計震動 ≥ 15 筆數", len(hv), 130)
check("其中 error 筆數", int((hv.status == "error").sum()), 128)
check("error 比例 %", pct((hv.status == "error").sum(), len(hv)), 98.46, 0.005)
check("其中 normal 筆數", int((hv.status == "normal").sum()), 0)
lv_err = pg[(pg.vibration_level < 15) & (pg.status == "error")]
check("壓力計低中震動 error 筆數", len(lv_err), 279)
check("這些紀錄溫度全部 > 55°C", bool((lv_err.temperature > 55).all()), True)
print(f"  這些紀錄的最低溫度：{lv_err.temperature.min()}°C")

# =====================================================================
print("\n=== 第 5 題：交叉稽核（邏輯一致性）===")
# =====================================================================
m_err = motor[motor.status == "error"]
m_err_oh = (m_err.temperature > 80).sum()
m_err_hv = (m_err.vibration_level >= 15).sum()
print(f"  馬達 error 總筆數：{len(m_err)}")
print(f"  ├ 其中溫度 > 80°C：{m_err_oh} 筆（占馬達 error {pct(m_err_oh, len(m_err))}%）")
print(f"  └ 其中震動 ≥ 15：{m_err_hv} 筆（占馬達 error {pct(m_err_hv, len(m_err))}%）")
print(f"  馬達過熱 408 筆的狀態分布：{oh.status.value_counts().to_dict()}")
print(f"  馬達震動 ≥ 15 的狀態分布：{motor[motor.vibration_level >= 15].status.value_counts().to_dict()}")
print(f"  馬達 warning 中溫度 > 80°C 的比例：{pct((motor[motor.status=='warning'].temperature > 80).sum(), (motor.status=='warning').sum())}%")
print("\n  馬達：溫度分組 × 狀態")
print(pd.crosstab(pd.cut(motor.temperature, [0, 80, 100, 200], labels=["≤80", "80–100", ">100"]), motor.status))

# =====================================================================
print("\n=== 總結 ===")
print(f"  通過 {sum(results)} / {len(results)} 項")
