import os
import time
import json
import requests
import pandas as pd
from pathlib import Path

# ==============================
# 0. 기본 설정
# ==============================
API_KEY = "8ac4f5b1-1228-489c-bd73-38c08bb7e0b7"  # <- 여기에 본인 키 넣기 (코드 공유할 땐 절대 올리지 말기!)
BASE_URL = "https://api.datalastic.com/api/v0/vessel_history"

# 입력 파일 & 출력 폴더
INPUT_CSV = "./MMSIList.csv"    
OUTPUT_DIR = Path("datalastic_results2")
OUTPUT_DIR.mkdir(exist_ok=True)

# 요청 사이 딜레이 (rate limit 보호용, 필요 없으면 줄여도 됨)
REQUEST_SLEEP = 0.3  # 초


# ==============================
# 1. MMSI 리스트 읽기
# ==============================
def load_mmsi_list(csv_path: str) -> list[str]:
    """
    MMSIList.csv에서 MMSI 목록만 추출.
    현재 파일 구조:
      - 첫 번째 컬럼: 코드(ACAH, BIGE, …)
      - 두 번째 컬럼: MMSI 숫자
    """
    df = pd.read_csv(csv_path)

    # 두 번째 컬럼을 MMSI로 사용 (헤더가 숫자 '477154400'로 되어 있음)
    mmsi_series = df.iloc[:, 1]

    # 문자열로 변환 후 앞뒤 공백 제거
    mmsi_list = mmsi_series.astype(str).str.strip().tolist()
    return mmsi_list


# ==============================
# 2. MMSI로 IMO 조회
# ==============================
def get_imo_from_mmsi(mmsi: str) -> str | None:
    """
    /vessel_history?mmsi=... 호출해서 IMO를 얻는다.
    응답 예시는 문서 기준으로 data.imo에 들어 있음.
    """
    params = {
        "api-key": API_KEY,
        "mmsi": mmsi,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
    except requests.RequestException as e:
        print(f"[ERROR] MMSI {mmsi} 요청 실패: {e}")
        return None

    if resp.status_code != 200:
        print(f"[ERROR] MMSI {mmsi} 응답 코드 {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        print(f"[ERROR] MMSI {mmsi} JSON 파싱 실패: {e}")
        return None

    # Datalastic 문서 기준 구조:
    # {
    #   "data": {
    #       "imo": "9595539",
    #       "mmsi": "431661000",
    #       ...
    #   },
    #   "meta": { ... }
    # }
    d = data.get("data")

    # 혹시 data가 리스트로 오는 경우도 방어
    if isinstance(d, list):
        if not d:
            print(f"[WARN] MMSI {mmsi}: data 리스트가 비어 있음.")
            return None
        d = d[0]

    if not isinstance(d, dict):
        print(f"[WARN] MMSI {mmsi}: 예상과 다른 data 구조: {type(d)}")
        return None

    imo = d.get("imo")
    if not imo:
        print(f"[WARN] MMSI {mmsi}: IMO를 찾지 못함. data = {d}")
        return None

    return str(imo).strip()


# ==============================
# 3. IMO로 2일치 히스토리 조회 후 JSON 저장
# ==============================
def fetch_history_by_imo(imo: str, days: int = 30) -> dict | None:
    params = {
        "api-key": API_KEY,
        "imo": imo,
        "days": days,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
    except requests.RequestException as e:
        print(f"[ERROR] IMO {imo} 요청 실패: {e}")
        return None

    if resp.status_code != 200:
        print(f"[ERROR] IMO {imo} 응답 코드 {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        return resp.json()
    except json.JSONDecodeError as e:
        print(f"[ERROR] IMO {imo} JSON 파싱 실패: {e}")
        return None


def save_json_for_mmsi(mmsi: str, imo: str, data: dict):
    """
    각 MMSI별 JSON 파일 저장.
    파일 이름 예시: 477154400_9244207.json
    """
    filename = OUTPUT_DIR / f"{mmsi}_{imo}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 저장 완료: {filename}")


# ==============================
# 4. 전체 파이프라인
# ==============================
def main():
    mmsi_list = load_mmsi_list(INPUT_CSV)
    print(f"총 {len(mmsi_list)}개 MMSI 읽음.")

    for idx, mmsi in enumerate(mmsi_list, start=1):
        print(f"\n[{idx}/{len(mmsi_list)}] MMSI {mmsi} 처리 중...")

        # 1단계: MMSI → IMO
        imo = get_imo_from_mmsi(mmsi)
        if not imo:
            print(f"    → MMSI {mmsi}: IMO 없음, 스킵.")
            continue

        print(f"    → IMO = {imo}")

        # 2단계: IMO → history(days=7) 조회
        history = fetch_history_by_imo(imo, days=30)
        if not history:
            print(f"    → IMO {imo}: history 조회 실패, 스킵.")
            continue

        # 3단계: JSON 저장
        save_json_for_mmsi(mmsi, imo, history)

        # API 과사용 방지용 살짝 딜레이
        time.sleep(REQUEST_SLEEP)


if __name__ == "__main__":
    main()
