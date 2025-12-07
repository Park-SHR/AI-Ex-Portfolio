import json
from pathlib import Path
from datetime import datetime

INPUT_DIR = Path("./datalastic_results")
OUTPUT_DIR = Path("./ais_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def to_arrival_format(ts: str):
    """ISO8601 → YYYYMMDDHHMMSS"""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%d%H%M%S")


def convert_file(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ JSON 파싱 오류: {path.name} / {e}")
        return

    vessel = data.get("data")
    if not vessel:
        print(f"⚠️ data 없음 → {path.name}")
        return

    mmsi = vessel.get("mmsi")
    if not mmsi:
        print(f"⚠️ mmsi 없음 → {path.name}")
        return

    positions = vessel.get("positions", [])
    if len(positions) < 2:
        print(f"⚠️ positions 개수 부족(len={len(positions)}) → {path.name}")
        return

    # ❗ Datalastic 데이터는 최신 → 과거 정렬이므로 오름차순으로 재정렬
    positions = sorted(positions, key=lambda x: x["last_position_epoch"])

    pred_results = []

    for i in range(len(positions) - 1):
        p0 = positions[i]
        p1 = positions[i + 1]

        epoch0 = p0.get("last_position_epoch")
        epoch1 = p1.get("last_position_epoch")
        moving_time = float(epoch1 - epoch0) if epoch0 and epoch1 else None

        arrival_utc = p1.get("last_position_UTC")
        arrival_time = (
            to_arrival_format(arrival_utc) if arrival_utc else None
        )

        pred_results.append({
            "SEQ": i,
            "BEFORE_NODE": float(i),
            "AFTER_NODE": float(i + 1),
            "BEFORE_LAT": p0.get("lat"),
            "BEFORE_LON": p0.get("lon"),
            "AFTER_LAT": p1.get("lat"),
            "AFTER_LON": p1.get("lon"),
            "SOG": p1.get("speed"),
            "MOVINGTIME": moving_time,
            "ARRIVALTIME": arrival_time
        })

    out = {
        "mmsi": mmsi,
        "PredResult": pred_results
    }

    out_path = OUTPUT_DIR / f"aisResult_{mmsi}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ 변환 완료 → {out_path.name}")


def main():
    files = list(INPUT_DIR.glob("*.json"))
    print(f"총 {len(files)}개 파일 변환")

    for f in files:
        convert_file(f)


if __name__ == "__main__":
    main()
