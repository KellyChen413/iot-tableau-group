#!/usr/bin/env python3
"""pip install pandas
用法：python verify_iot_sensor_logs.py [CSV路徑]
PASS/FAIL 比對精確整數或指定顯示位數；不刪資料、不去重、不補值。
結束碼：0 全通過，1 至少一項不符，2 欄位或型別無法稽核。
"""
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(
    'antdata_iot_sensor_logs_20261112_001823.csv')
df = pd.read_csv(src, encoding='utf-8-sig')
required = ['log_id', 'timestamp', 'device_id', 'device_type', 'status',
            'factory_zone', 'temperature', 'pressure', 'vibration_level',
            'power_consumption']
missing = set(required) - set(df.columns)
if missing:
    sys.exit('缺少必要欄位：' + str(missing))
results = []


def check(label, actual, expected, digits=None):
    shown = actual
    if digits is not None:
        if pd.isna(actual):
            shown = '無法計算'
        else:
            shown = Decimal(str(actual)).quantize(
                Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
        ok = shown == Decimal(str(expected))
    else:
        ok = actual == expected
    results.append(bool(ok))
    print(f'[{"PASS" if ok else "FAIL"}] {label}: 原值={actual}; '
          f'比對值={shown}; 預期={expected}')


def pct(n, d):
    return 100 * n / d if d else float('nan')


def rate(label, numerator, denominator, expected=None, digits=2):
    value = pct(numerator, denominator)
    label = f'{label} %（分子={numerator}, 分母={denominator}）'
    if expected is None:
        print(f'[INFO] {label}: {value:.6f}')
    else:
        check(label, value, expected, digits)
    return value


print(f'資料來源：{src.resolve()}')
for c in ['temperature', 'pressure', 'vibration_level', 'power_consumption']:
    parsed = pd.to_numeric(df[c], errors='coerce')
    bad = df[c].notna() & (parsed.isna() | parsed.isin([float('inf'), -float('inf')]))
    if bad.any():
        print(df.loc[bad, ['log_id', c]].head(10))
        sys.exit(2)
    df[c] = parsed
parsed = pd.to_datetime(df.timestamp, errors='coerce')
if (df.timestamp.notna() & parsed.isna()).any():
    sys.exit('timestamp 有無法解析的值；請先核對來源格式。')
df['date'] = parsed.dt.date
df['hour'] = parsed.dt.hour

print('\n=== 第 1 題：資料量、缺值、設備與廠區 ===')
check('總筆數', len(df), 92250)
check('設備總數', df.device_id.nunique(), 45)
check('設備類型集合', set(df.device_type.dropna()),
      {'motor', 'pressure_gauge', 'temperature_sensor'})
for t, n in [('motor', 10), ('pressure_gauge', 15), ('temperature_sensor', 20)]:
    sub = df[df.device_type.eq(t)]
    check(f'{t} 台數', sub.device_id.nunique(), n)
    check(f'{t} 筆數', len(sub), n * 2050)
per_device = df.groupby('device_id', dropna=False).size()
check('每台剛好 2050 筆', bool(per_device.eq(2050).all()), True)
check('每個 device_id 只屬於一種設備類型', bool(df.groupby(
    'device_id', dropna=False).device_type.nunique(dropna=False).eq(1).all()), True)
check('log_id 無缺值且唯一', bool(df.log_id.notna().all() and df.log_id.is_unique), True)
check('pressure 缺值', int(df.pressure.isna().sum()), 61500)
check('pressure 有值若且唯若為壓力計',
      bool(df.pressure.notna().eq(df.device_type.eq('pressure_gauge')).all()), True)
check('vibration_level 缺值', int(df.vibration_level.isna().sum()), 41000)
check('震動缺值若且唯若為溫度感測器', bool(df.vibration_level.isna().eq(
    df.device_type.eq('temperature_sensor')).all()), True)
for c in set(required) - {'pressure', 'vibration_level'}:
    check(f'{c} 缺值', int(df[c].isna().sum()), 0)
check('每廠區只對應一種設備類型', bool(df.groupby(
    'factory_zone', dropna=False).device_type.nunique(dropna=False).eq(1).all()), True)
print(pd.crosstab(df.factory_zone, df.device_type, dropna=False).to_string())

motor = df[df.device_type.eq('motor')].copy()
pg = df[df.device_type.eq('pressure_gauge')].copy()
abnormal = df.status.notna() & df.status.ne('normal')

print('\n=== 第 2 題：異常比例（分母包含該群組全部列）===')
check('status 只有四種已知值', set(df.status.dropna()),
      {'normal', 'offline', 'warning', 'error'})
for status, count, percent in [('normal', 87354, 94.7), ('offline', 1801, 2.0),
                               ('warning', 1764, 1.9), ('error', 1331, 1.4)]:
    n = int(df.status.eq(status).sum())
    check(status + ' 筆數', n, count)
    rate(status, n, len(df), percent, 1)
for t, expected in [('motor', 4.61), ('pressure_gauge', 5.50),
                    ('temperature_sensor', 5.51)]:
    mask = df.device_type.eq(t)
    rate(t + ' 非 normal', int((mask & abnormal).sum()), int(mask.sum()), expected)
for s, expected in [('error', 1.88), ('warning', 0.75)]:
    rate('馬達 ' + s, int(motor.status.eq(s).sum()), len(motor), expected)
dev_ab = abnormal.groupby(df.device_id).sum().reindex(per_device.index, fill_value=0)
leaders = dev_ab[dev_ab.eq(dev_ab.max())]
check('異常並列最高的設備集合', set(leaders.index), {'TEM020', 'PRE012'})
check('異常最高筆數', int(dev_ab.max()), 132)
check('每台平均異常筆數（包含 0 異常設備）', dev_ab.mean(), 109, 0)
print(dev_ab.sort_values(ascending=False).head(10).to_string())

print('\n=== 第 3 題：能耗分布 ===')
power = df.groupby('device_type').power_consumption.agg(['size', 'count', 'sum', 'mean'])
print(power.to_string())
rate('馬達總耗電占全體', power.loc['motor', 'sum'], df.power_consumption.sum(), 95.3, 1)
check('馬達每筆平均耗電', power.loc['motor', 'mean'], 27.5, 1)
for t, expected in [('pressure_gauge', 55), ('temperature_sensor', 92)]:
    check('馬達平均耗電 / ' + t + ' 平均耗電',
          power.loc['motor', 'mean'] / power.loc[t, 'mean'], expected, 0)
dev_pw = df.groupby(['device_id', 'device_type']).power_consumption.sum(min_count=1)
dev_pw = dev_pw.sort_values(ascending=False)
top = dev_pw.head(10)
print('設備耗電前 10 名：\n' + top.to_string())
check('前 10 名皆為馬達', set(top.index.get_level_values('device_type')), {'motor'})
check('最高耗電設備（含並列）', set(dev_pw[dev_pw.eq(dev_pw.max())].index.get_level_values(0)), {'MOT007'})
check('第 10 名設備', top.index[-1][0], 'MOT001')
check('MOT007 耗電（整數顯示）', dev_pw.get(('MOT007', 'motor'), float('nan')), 56929, 0)
check('MOT001 耗電（整數顯示）', dev_pw.get(('MOT001', 'motor'), float('nan')), 55565, 0)
rate('最高較第 10 名高出（以第 10 名為分母）', top.iloc[0] - top.iloc[-1], top.iloc[-1], 2.5, 1)
hourly = motor.groupby('hour').power_consumption.agg(['size', 'count', 'mean']).reindex(range(24))
print('以所有馬達紀錄按小時直接平均：\n' + hourly.to_string())
check('24 小時都有紀錄', bool(hourly['size'].notna().all()), True)
check('最高平均耗電時段（含並列）', set(hourly.index[hourly['mean'].eq(hourly['mean'].max())]), {6})
check('最低平均耗電時段（含並列）', set(hourly.index[hourly['mean'].eq(hourly['mean'].min())]), {12})
hi, lo = hourly['mean'].max(), hourly['mean'].min()
check('最高小時平均', hi, 28.3, 1)
check('最低小時平均', lo, 26.3, 1)
# 「約 7%」未指定分母：兩種皆列出，不擅自定義寬鬆容差。
rate('小時差距 / 最高值', hi - lo, hi)
rate('小時差距 / 最低值', hi - lo, lo)

print('\n=== 第 4 題：溫度、震動門檻 ===')
hot = motor.temperature.gt(80)
check('馬達 >80°C 筆數', int(hot.sum()), 408)
rate('過熱占全部馬達紀錄', int(hot.sum()), len(motor), 2.0, 1)
check('馬達 >100°C 筆數', int(motor.temperature.gt(100).sum()), 55)
check('馬達最高溫', motor.temperature.max(), 109.06, 2)
hot_dev = hot.groupby(motor.device_id).sum()
check('馬達過熱統計涵蓋 10 台', len(hot_dev), 10)
check('每台過熱最少筆數', int(hot_dev.min()), 31)
check('每台過熱最多筆數', int(hot_dev.max()), 48)
check('過熱最多設備（含並列）', set(hot_dev[hot_dev.eq(hot_dev.max())].index), {'MOT004'})
print(hot_dev.to_string())
all_dates = set(df.date.dropna())
check('全資料不同日期數', len(all_dates), 29)
check('每天都有馬達過熱（日期集合相等）', set(motor.loc[hot, 'date'].dropna()), all_dates)
if all_dates:
    calendar = set(pd.date_range(min(all_dates), max(all_dates)).date)
    check('全資料日期連續、無缺日', all_dates, calendar)
hv = pg.vibration_level.ge(15)
check('壓力計總筆數', len(pg), 30750)
check('壓力計震動 >=15 筆數', int(hv.sum()), 130)
check('高震動且 error', int((hv & pg.status.eq('error')).sum()), 128)
rate('高震動中的 error', int((hv & pg.status.eq('error')).sum()), int(hv.sum()), 98.46)
check('高震動且 normal', int((hv & pg.status.eq('normal')).sum()), 0)
low_error = pg[pg.vibration_level.lt(15) & pg.status.eq('error')]
check('低中震動 error 不是空集合', bool(len(low_error) > 0), True)
check('低中震動 error 溫度皆非缺值且 >55°C', bool(len(low_error) > 0 and
      low_error.temperature.notna().all() and low_error.temperature.gt(55).all()), True)
print(f'低中震動 error: {len(low_error)} 筆；最低溫={low_error.temperature.min()}')
print('高震動狀態分布：\n' + pg.loc[hv, 'status'].value_counts(dropna=False).to_string())

print('\n=== 第 5 題：雙向交叉稽核 ===')
def overlap(label, data, signal):
    error = data.status.eq('error')
    a = int((signal & error).sum())
    b = int((signal & ~error).sum())
    c = int((~signal & error).sum())
    d = int((~signal & ~error).sum())
    print(f'\n{label}: error={int(error.sum())}; 門檻命中={int(signal.sum())}')
    print(pd.DataFrame([[a, b], [c, d]], index=['達門檻', '未達門檻（含缺值）'],
                       columns=['error', '非 error（含未知狀態）']).to_string())
    rate('error 中達門檻（涵蓋率）', a, a + c)
    rate('達門檻中為 error（精確率）', a, a + b)
    rate('交集 / 聯集（Jaccard）', a, a + b + c)
    print('達門檻狀態：', data.loc[signal, 'status'].value_counts(dropna=False).to_dict())
    exceptions = data.loc[error & ~signal, ['log_id', 'device_id', 'temperature',
        'vibration_level', 'status']]
    print('error 未達門檻的前 10 筆：\n' + exceptions.head(10).to_string(index=False))
    return a, b, c, d

check('馬達 error 筆數（1.88% 所對應的約 385 筆）', int(motor.status.eq('error').sum()), 385)
overlap('馬達：temperature >80°C', motor, hot)
overlap('壓力計：vibration_level >=15', pg, hv)
pg_error = pg.status.eq('error')
rule_union = hv | pg.temperature.gt(55)
rate('壓力計 error 中，高震動或溫度 >55°C 的涵蓋率',
     int((pg_error & rule_union).sum()), int(pg_error.sum()))
print('\n判讀：精確率與涵蓋率須分開。優先監測門檻可有資料支持，'
      '但只有 error 未達門檻與達門檻非 error 都為 0，才可宣稱門檻與 error 完全等價。'
      '交叉表是描述性關聯，不能單憑它證明因果或完整故障規則。')
print(f'\n總結：PASS={sum(results)}；FAIL={len(results)-sum(results)}；共 {len(results)} 項。')
sys.exit(0 if all(results) else 1)
