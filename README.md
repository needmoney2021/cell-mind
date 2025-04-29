# CellMind – **분산 미니-GPT** 프로젝트 가이드 (2025-04-29 rev)

당연하게도 챗지피티와 커서가 다 만들었습니다.   

**아직 테스트 못 해봤어요. 단말기가 없어서....ㅎㅎ**   
**시간 날 때 단계별로 테스트하고 수정하고 있습니다.**   
**토크나이저 부분 테스트 및 수정 완료. 🔨**   

## 0. 목차

1. [사전 준비](#prereq)
2. [데이터·토크나이저 준비](#data)
3. [`shared/config.json`] 설정법
4. [메인 단말 (Orchestrator)](#orch)
5. [워커 단말](#worker)
6. [추론 / 체크포인트 동기화](#infer-sync)
7. [기타 팁](#tips)

---

<a name="prereq"></a>

## 1. 사전 준비

```bash
# 1) 저장소 클론
git clone {repos}
cd cellmind

# 2) 가상환경
python -m venv venv
source venv/bin/activate        # Win: venv\Scripts\activate

# 3) 의존성
pip install -r requirements.txt
```

> **모든 단말(메인·워커)에서 동일하게** 수행해야 합니다.

---

<a name="data"></a>

## 2. 데이터 & 토크나이저

| 단계     | 파일/디렉터리                                                                                                                | 설명                                      |
|--------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| ① 🔨   | `data/raw/*.txt`                                                                                                       | 원본 로 데이터                                |
| ② 🔨   | **토크나이저 학습**<br>`python -m tokenizer`                                                                                  | → `{path}/tokenizer.model`              |
| ③ TODO | **JSONL 변환**<br>각 줄 `{"text": "..."} `                                                                                 | → `data/for-model-training/train.jsonl` |
| ④ TODO | 학습 시 **`TextDataset`** 가 이 JSONL을 읽어 자동으로 `<\|start\|> … <\|endoftext\|>` 토큰을 붙이고, `collate_fn` 으로 배치 패딩(-100 mask) 처리 |

### 2.1 예제. raw_data

```csv
Date,User,Message
2022-10-30 10:35:01,"강XX","이태원 가서 다친 분들 없겠지 여긴????"
2022-10-30 10:35:32,"김YY","없을걸?"
2022-10-30 10:37:48,"강XX","정말 큰일 터졌더라"
2022-10-30 10:37:53,"강XX","조심들 하자"
```

### 2.2 예제. 정제.

```python
import pandas as pd
import random
import re


# timestamp 추출
def timestamp() -> list[str]:
    df = pd.read_csv("data/raw/group_chat_001.csv")
    df["timestamp"] = pd.to_datetime(df["Date"])
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return sorted(random.choices(df["timestamp"].tolist(), k=50))


# 이름 추출
def nameonly() -> list[str]:
    df1 = pd.read_csv("data/raw/group_chat_001.csv")
    df2 = pd.read_csv("data/raw/group_chat_002.csv")
    df3 = pd.read_csv("data/raw/group_chat_003.csv")
    df = pd.concat([df1, df2, df3], ignore_index=True)

    return df["User"].unique().tolist()


# 텍스트 전처리
def preprocess_message(fullname: set[str], firstnames: list[str]) -> list[str]:
    # CSV 파일 로드 및 병합
    df1 = pd.read_csv("data/raw/group_chat_001.csv")
    df2 = pd.read_csv("data/raw/group_chat_002.csv")
    df3 = pd.read_csv("data/raw/group_chat_003.csv")
    df = pd.concat([df1, df2, df3], ignore_index=True)

    # 원본 메시지 소문자 변환
    messages = df["Message"].astype(str).str.lower().tolist()

    # 정규식 패턴 컴파일
    photo_pattern = re.compile(r'^사진(\s*\d+장)?$')
    movie_pattern = re.compile(r'^동영상$')
    url_pattern = re.compile(r'https?://\S+')
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"  # symbols & pictographs
        r"\U0001F680-\U0001F6FF"  # transport & map symbols
        r"\U0001F1E0-\U0001F1FF"  # flags
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        r"\U0001FA00-\U0001FA6F"  # Chess Symbols etc.
        r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        r"\u2600-\u26FF\u2700-\u27BF]+", flags=re.UNICODE)
    # 5개 이상 반복되는 비공백 문자 일반화
    repeat_pattern = re.compile(r'(\S)\1{4,}')
    newline_pattern = re.compile(r'\n+')
    ws_paren_pattern = re.compile(r'\(\s*\)')
    entire_names = list(fullname) + firstnames
    name_pattern = re.compile(r'|'.join(entire_names))

    processed = []
    for msg in messages:
        # ZWSP 제거
        msg = msg.replace('\u200b', '')
        # 연속된 줄바꿈 하나로 축소
        msg = newline_pattern.sub('\n', msg)
        msg = msg.strip()

        # 전체 메시지가 사진/동영상/이모티콘인 경우 치환
        if photo_pattern.match(msg):
            processed.append('<PHOTO>')
            continue
        if movie_pattern.match(msg):
            processed.append('<MOVIE>')
            continue
        if msg == '이모티콘':
            processed.append('<EMOJI>')
            continue

        # URL, 이모지, 반복 문자, 불필요 패턴 치환
        msg = url_pattern.sub('<URL>', msg)
        msg = emoji_pattern.sub('<EMOJI>', msg)
        msg = repeat_pattern.sub(lambda m: m.group(1) * 5, msg)
        msg = ws_paren_pattern.sub('', msg)
        msg = name_pattern.sub('<NAME>', msg)

        processed.append(msg)

    return processed


# 추출한 이름으로 대화 내용에 있는 이름을 <NAME> 토큰으로 변화시킴
names = nameonly()
fullname = set()
firstname_only = []
for name in names:
    processed_name = name.replace(" ", "")
    re_search = re.search("[가-힣]{3}", processed_name)
    if re_search:
        fullname.add(re_search.group())
    else:
        # 가지고 계신 데이터에 맞게 처리하세요.
        pass
```

### 2.3 예제. 토크나이저 학습 실행.

```bash
python -m tokenizer --input data/for-tokenizer-training/textonly.txt --model-prefix minigpt --character-coverage 0.98 --model-type bpe --output-dir tokenizer
```

```bash
> ls -al tokenizer

total 1000tokenizer                                                                                                                                            ─╯
drwxr-xr-x   8 needmoney  staff     256 Apr 29 13:37 .
drwxr-xr-x  20 needmoney  staff     640 Apr 29 13:46 ..
-rw-r--r--@  1 needmoney  staff    3524 Apr 29 13:36 __init__.py
-rw-r--r--   1 needmoney  staff      58 Apr 29 13:06 __main__.py
drwxr-xr-x   5 needmoney  staff     160 Apr 29 13:36 __pycache__
##### 생성 ##### 
-rw-r--r--   1 needmoney  staff  373645 Apr 29 13:36 minigpt.model
-rw-r--r--   1 needmoney  staff  118815 Apr 29 13:36 minigpt.vocab
###############
-rw-r--r--@  1 needmoney  staff    3045 Apr 29 13:37 sentencepiece_tokenizer.py

```

---

## 3. `shared/config.json` 예시

```jsonc
{
  "distributed": {
    "backend": "gloo",                   // CPU-only → gloo, GPU → nccl
    "init_method": "tcp://192.168.0.100:23456",
    "world_size": 3,                     // 1(메인) + 2(워커)
    "worker_ips": [
      "192.168.0.101",
      "192.168.0.102"
    ],
    "usernames": {
      "192.168.0.101": "laptop_user",
      "192.168.0.102": "termux_user"
    }
  },

  "model": {
    "d_model":     512,
    "nhead":       8,
    "num_layers":  6,
    "dropout":     0.1,
    "checkpoint":  "checkpoints/latest.pt"   // 최초 학습 시 비워둬도 됨
  },

  "tokenizer": {
    "model_path":  "tokenizer/tokenizer.model",
    "vocab_size":  8000
  },

  "training": {
    "batch_size":  32,
    "epochs":      3,
    "lr":          3e-4,
    "dataset": {
      "path":      "data/for-model-training/train.jsonl",
      "seq_len":   256
    }
  }
}
```

---

<a name="orch"></a>

## 4. 메인 단말 (Orchestrator)

### 4-1. SSH 준비 (최초 1회)

```bash
ssh-keygen -t rsa -b 4096 -N ""      # 키 없으면 생성
ssh-copy-id laptop_user@192.168.0.101
ssh-copy-id termux_user@192.168.0.102
```

### 4-2. 학습 시작

```bash
python orchestrator/manager.py --mode train
```

* `manager.py` 가 각 워커에 **환경변수(MASTER_ADDR · RANK 등)** 를 포함한
  `python -m worker.processor --mode train` 명령을 SSH로 전송 → 학습 자동 시작.

### 4-3. 실시간 로그

메인 단말 콘솔에 rank 0 로그가 출력되고, 워커 콘솔에는 각자 rank 로그가 뜹니다.

---

<a name="worker"></a>

## 5. 워커 단말

| 항목                     | Linux laptop               | Android (예: Termux)                       |
|------------------------|----------------------------|-------------------------------------------|
| 레포 클론·venv·pip install | 메인과 동일                     | ```pkg install python git openssh``` 후 동일 |
| SSH 서버                 | 보통 이미 `sshd` 실행 중          | ```sshd``` 실행                             |
| **실행 필요 없음**           | Orchestrator가 알아서 프로세스를 띄움 | Orchestrator가 알아서 프로세스를 띄움                |

---

<a name="infer-sync"></a>

## 6. 추론 & 체크포인트 동기화

| 모드                 | 명령                                                                         |
|--------------------|----------------------------------------------------------------------------|
| **추론**             | `python orchestrator/manager.py --mode inference --prompt "안녕 GPT!"`       |
| **파일로 추론**         | `python orchestrator/manager.py --mode inference --prompt-file prompt.txt` |
| **체크포인트만 강제 sync** | `python orchestrator/manager.py --mode sync`                               |

---

<a name="tips"></a>

## 7. 기타 팁

* **GPU** 단말은 PyTorch + CUDA 설치 후 `distributed.backend` 을 `nccl` 로.
* 학습 로그·그래프를 보고 싶으면 `DistributedTrainer`에 **TensorBoard** 코드(네 줄) 끼워 넣으면 됨.
* 데이터셋 크기가 수 GiB 이상이면 `TextDataset` 을 **메모리-매핑(mmap)** 방식으로 바꾸는 것을 추천.  
